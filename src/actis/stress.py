"""Blind end-to-end sensitivity checks on a mask with independently known geometry."""

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

from .analysis import METHODS, analyze
from .pooled import pooled_group_tiles
from .statistics import response_fit
from .synthetic import lattice_basis, render_repeats

UNITS = {"raw": "DN", "flux": "DN·px²"}
ALL_METHODS = ("raw", *METHODS, "harmonic", "pooled")


def _patterns(patterns):
    """Retain distinct region geometries, including optical sampling orientation."""
    used = set()
    for item in patterns:
        if isinstance(item, dict):
            config = dict(item)
        else:
            shape, lattice, pitch, cd = item
            config = dict(shape=shape, lattice=lattice, pitch_px=pitch, cd_nm=cd)
        shape, lattice = config["shape"], config["lattice"]
        if shape not in {"square", "circle", "ellipse"} or lattice not in {
            "square",
            "staggered",
            "oblique",
        }:
            raise ValueError(f"Unsupported synthetic pattern {shape!r}, {lattice!r}")
        pitch, cd = float(config["pitch_px"]), float(config["cd_nm"])
        if not np.isfinite([pitch, cd]).all() or min(pitch, cd) <= 0:
            raise ValueError("Synthetic pitch and CD must be positive and finite")
        angle, ratio = (
            float(config.get("angle_deg", 0.0)),
            float(config.get("ellipse_ratio", 0.75)),
        )
        basis = lattice_basis(lattice, pitch, angle, config.get("basis_xy"))
        region = config.get("region", -1)
        key = (region, shape, lattice, pitch, cd, angle, ratio, *basis.ravel())
        if key not in used:
            used.add(key)
            yield dict(
                region=region,
                shape=shape,
                lattice=lattice,
                pitch_px=pitch,
                cd_nm=cd,
                angle_deg=angle,
                ellipse_ratio=ratio,
                basis_xy=basis.tolist(),
            )


def _challenge_frames(
    shape,
    lattice,
    pitch,
    cd,
    pixel_nm,
    *,
    angle_deg=0.0,
    ellipse_ratio=0.75,
    basis_xy=None,
    small=False,
):
    # Three complete tiles along x. The outside pair is clear of the step boundary.
    basis = lattice_basis(lattice, pitch, angle_deg, basis_xy)
    tile_px = max(
        34, int(np.ceil(5.75 * max(pitch, np.linalg.norm(basis, axis=1).max())))
    )
    image_shape = (tile_px + 6, 3 * tile_px + 8)
    delta = min(0.10 * cd, 0.35 * (pitch * pixel_nm - cd))
    if delta <= 0:
        raise ValueError("Nominal synthetic CD must fit inside its pitch")
    args = dict(
        shape=shape,
        lattice=lattice,
        pitch_px=pitch,
        cd_nm=cd,
        pixel_nm=pixel_nm,
        image_shape=image_shape,
        supersample=8,
        n_repeats=1,
        nuisance=False,
        angle_deg=angle_deg,
        ellipse_ratio=ellipse_ratio,
        basis_xy=basis,
    )

    def expose(**changes):
        return render_repeats(**(args | changes))[0]

    nominal = expose()
    meta = dict(
        delta=delta,
        tile_px=tile_px,
        field_shape_yx_px=list(image_shape),
        cd_levels_nm=[cd, cd - delta, cd + delta],
        minor_ratios=[ellipse_ratio, 0.8 * ellipse_ratio],
    )
    phase_offsets = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5), (0.25, -0.25)]
    if small:
        parts = [nominal, expose(global_delta_nm=-delta), expose(global_delta_nm=delta)]
        phase_offsets = [(0.5, 0.0), (0.0, 0.5), (0.25, -0.25)]
        meta["phase_start"] = len(parts)
        parts.extend(expose(global_shift_xy_px=shift) for shift in phase_offsets)
        return np.concatenate(parts), dict(meta, phase_offsets=phase_offsets)
    local_shift = pixel_nm * np.array([0.35, -0.25])
    ceiling = int(np.quantile(nominal, 0.95))
    clipped = np.minimum(nominal, ceiling)
    parts = [
        nominal,
        expose(global_delta_nm=-delta),
        expose(global_delta_nm=delta),
        expose(local_delta_nm=delta),
        expose(nuisance=True, n_repeats=3),
        clipped,
    ]
    if shape == "ellipse":
        parts.append(expose(ellipse_ratio=meta["minor_ratios"][1]))
    shift_idx = sum(len(part) for part in parts)
    parts.append(expose(local_shift_xy_nm=local_shift))
    flat = nominal.astype(float)
    rng = np.random.default_rng(27017)
    controls = {
        "gain_only": 1400 + 1.20 * (flat - 1400),
        "offset_only": flat + 750.0,
        "blur_only": gaussian_filter(flat, (0, 0.90, 0.50), mode="nearest"),
        "noise_only": 1400
        + 6.0 * rng.poisson(np.maximum(flat - 1400, 0) / 6)
        + rng.normal(0, 40, flat.shape),
    }
    indices = {}
    for idx, (key, control) in enumerate(controls.items(), start=shift_idx + 1):
        parts.append(np.rint(np.clip(control, 0, 65535)).astype(np.uint16))
        indices[key] = idx
    meta["phase_start"] = sum(len(part) for part in parts)
    parts.extend(expose(global_shift_xy_px=shift) for shift in phase_offsets)
    return np.concatenate(parts), dict(
        meta,
        local_shift=local_shift,
        phase_offsets=phase_offsets,
        shift_idx=shift_idx,
        controls=indices,
        ceiling=ceiling,
        clipped_fraction=float(np.mean(nominal > ceiling)),
    )


@dataclass
class Observations:
    matrix: np.ndarray
    x: np.ndarray
    tiles: list
    complete: np.ndarray

    def reduce(self, runs, mask=None, equal_tiles=False):
        values = self.matrix[np.asarray(runs)]
        valid = np.isfinite(values).all(axis=0)
        if mask is not None:
            valid &= mask
        if equal_tiles:
            valid &= self.complete
        indices = np.flatnonzero(valid)
        if not len(indices):
            return np.full(len(runs), np.nan), 0
        if equal_tiles:
            groups = defaultdict(list)
            for index in indices:
                groups[self.tiles[index]].append(index)
            vals = np.mean(
                [values[:, group].mean(axis=1) for group in groups.values()], axis=0
            )
        else:
            vals = values[:, indices].mean(axis=1)
        return vals, len(indices)


def _series(study):
    n = len(study.frames)
    observations = defaultdict(dict)

    def collect(row, field, value, key, x, complete):
        group = observations[row["method"], field]
        tile = (row["region"], row["i"], row["j"])
        if key not in group:
            group[key] = (np.full(n, np.nan), x, tile, complete)
        if not study.registration[row["repeat"]]["ambiguous"]:
            group[key][0][row["repeat"]] = value

    for r in study.holes:
        key = (r["region"], r["site"])
        scale = study.pixel_nm
        values = {
            "cd": r.get("cd_px", np.nan) * scale,
            "x": r.get("x_px", np.nan) * scale,
            "y": r.get("y_px", np.nan) * scale,
            "minor": np.minimum(r.get("width_px", np.nan), r.get("height_px", np.nan))
            * scale,
        }
        if r["method"] == "flux":
            values["cd"] = r.get("flux_dn_px2", np.nan)
        for field, value in values.items():
            collect(
                r,
                field,
                value if r["valid"] else np.nan,
                key,
                r["ref_x_px"],
                r["complete_tile"],
            )
    for r in study.tiles:
        key = (r["region"], r["i"], r["j"])
        collect(
            r, "cd", r["value"] if r["status"] == "ok" else np.nan, key, r["x_px"], True
        )
        if r["method"] in {"harmonic", "pooled"}:
            for field in ("x", "y"):
                value = r.get(f"d{field}_px", np.nan) * study.pixel_nm
                collect(
                    r,
                    field,
                    value if r.get("phase_status") == "ok" else np.nan,
                    key,
                    r["x_px"],
                    True,
                )
            minor = (
                min(r.get("cd_x_px", np.nan), r.get("cd_y_px", np.nan)) * study.pixel_nm
            )
            collect(
                r,
                "minor",
                minor if r["status"] == "ok" else np.nan,
                key,
                r["x_px"],
                True,
            )
    series = {}
    for key, rows in observations.items():
        matrix, x, tiles, complete = zip(*rows.values())
        series[key] = Observations(
            np.stack(matrix, axis=1), np.array(x), tiles, np.array(complete)
        )
    return series


def _fit_row(
    base, method, scenario, unit, known, measured, count, nominal=None, require_levels=2
):
    result = response_fit(known, measured)
    valid = (
        count > 0
        and result["count"] == len(known)
        and np.unique(known).size >= require_levels
    )
    if unit != "nm":
        result["mean_error"] = result["rmse"] = np.nan
    if not valid:
        result["slope"] = result["intercept"] = np.nan
    if count == 0:
        result.update(count=0, mean_error=np.nan, rmse=np.nan)
    normal = nominal is not None and np.isfinite(nominal) and nominal != 0
    fractional = result["slope"] / nominal if normal else np.nan
    return dict(
        base,
        method=method,
        scenario=scenario,
        unit=unit,
        support=count,
        **result,
        status="ok" if valid else "insufficient matched measurements",
        nominal_value=nominal if nominal is not None else np.nan,
        fractional_slope=fractional,
        elasticity=fractional * base["nominal_cd_nm"],
    )


def _matched_pooled_series(study, images, groups):
    """Use a common set of apertures inside each independent challenge."""
    shifts = [r["shift_xy"] for r in study.registration]
    side = study.tile_um * 1000 / study.pixel_nm
    tiles = {g: [] for g in groups}
    complete = {
        (r["region"], r["i"], r["j"]) for r in study.tiles if r["method"] == "raw"
    }
    for region in study.regions:
        matches = pooled_group_tiles(
            images, region, shifts, side, study.bounds[:2], groups
        )
        for group, exposures in matches.items():
            for repeat, rows in zip(group, exposures):
                for r in rows:
                    i, j = r["tile_i"], r["tile_j"]
                    if (region.region_id, i, j) in complete:
                        tiles[group].append(
                            dict(
                                r,
                                method="pooled",
                                region=region.region_id,
                                i=i,
                                j=j,
                                x_px=r["x"],
                                repeat=repeat,
                                value=r["cd_px"] * study.pixel_nm,
                            )
                        )
    return {
        g: _series(replace(study, holes=[], tiles=rows)) for g, rows in tiles.items()
    }


def _score_pattern(study, meta, base, images):
    series = _series(study)
    controls = {"nuisance": [0, 4, 5, 6], "clipping": [0, 7]}
    controls.update({name: [0, run] for name, run in meta["controls"].items()})
    groups = {
        (0, 1, 2),
        (0, 3),
        (0, meta["shift_idx"]),
        *(tuple(ids) for ids in controls.values()),
    }
    if base["shape"] == "ellipse":
        groups.add((0, 8))
    pooled = _matched_pooled_series(study, images, groups)
    empty = Observations(
        np.empty((len(study.frames), 0)), np.array([]), [], np.array([], bool)
    )
    field = np.array(
        [
            f["common_dn"] if not r["ambiguous"] else np.nan
            for f, r in zip(study.frames, study.registration)
        ]
    )
    native = Observations(field[:, None], np.array([0]), [None], np.array([False]))
    delta, cd = meta["delta"], base["nominal_cd_nm"]
    mid = (images.shape[-1] - 1) / 2
    truth = np.array([cd, cd - delta, cd + delta])
    rows = []
    for method in ALL_METHODS:
        unit = UNITS.get(method, "nm")

        def measurements_for(field_name, runs):
            lookup = pooled[tuple(runs)] if method == "pooled" else series
            return lookup.get((method, field_name), empty)

        measurements = measurements_for("cd", (0, 1, 2))
        full = native if method == "raw" else measurements
        for scenario, data, tiles in [
            ("global_cd", full, False),
            ("global_cd_tiles", measurements, True),
        ]:
            observed, count = data.reduce([0, 1, 2], equal_tiles=tiles)
            rows.append(
                _fit_row(
                    base,
                    method,
                    scenario,
                    unit,
                    truth,
                    observed,
                    count,
                    nominal=observed[0],
                    require_levels=3,
                )
            )
        # Differencing the changed half against the unchanged half removes any
        # fitted common translation without using the known centers in fitting.
        guard = (
            meta["tile_px"] / 2
            if method in {"harmonic", "pooled", "raw"}
            else base["pitch_px"] / 2
        )
        for scenario, field_name, run, change, u in [
            ("local_cd", "cd", 3, delta, unit),
            ("placement_x", "x", meta["shift_idx"], meta["local_shift"][0], "nm"),
            ("placement_y", "y", meta["shift_idx"], meta["local_shift"][1], "nm"),
        ]:
            if method == "raw" and field_name != "cd":
                continue
            data = measurements_for(field_name, (0, run))
            left, nleft = data.reduce([0, run], data.x < mid - guard)
            right, nright = data.reduce([0, run], data.x > mid + guard)
            diff = (right[1] - right[0]) - (left[1] - left[0])
            support = nleft + nright if min(nleft, nright) else 0
            row = _fit_row(base, method, scenario, u, [0, change], [0, diff], support)
            row.update(
                truth_change=float(change),
                measured_change=float(diff),
                left_count=nleft,
                right_count=nright,
            )
            rows.append(row)
        for scenario, runs in controls.items():
            data = native if method == "raw" else measurements_for("cd", runs)
            observed, count = data.reduce(runs)
            diffs = observed[1:] - observed[0]
            valid = np.isfinite(diffs).all() and count > 0
            rows.append(
                dict(
                    base,
                    scenario=scenario,
                    method=method,
                    unit=unit,
                    support=count,
                    count=len(diffs) if valid else 0,
                    slope=np.nan,
                    intercept=np.nan,
                    nominal_value=observed[0],
                    fractional_slope=np.nan,
                    elasticity=np.nan,
                    mean_error=float(diffs.mean()) if valid else np.nan,
                    rmse=float(np.sqrt(np.mean(diffs**2))) if valid else np.nan,
                    sigma3=float(3 * np.std(diffs, ddof=1))
                    if valid and len(diffs) > 1
                    else np.nan,
                    status="ok" if valid else "insufficient matched measurements",
                    clipping_dn=meta["ceiling"] if scenario == "clipping" else None,
                    clipped_fraction=meta["clipped_fraction"]
                    if scenario == "clipping"
                    else None,
                )
            )
        if base["shape"] == "ellipse" and method not in {"raw", "flux"}:
            data = measurements_for("minor", (0, 8))
            observed, count = data.reduce([0, 8])
            rows.append(
                _fit_row(
                    base,
                    method,
                    "ellipse_minor",
                    "nm",
                    np.asarray(meta["minor_ratios"]) * cd,
                    observed,
                    count,
                )
            )
    return rows


def _score_sampling(study, meta, base, small):
    """Measure CD leakage and center recovery on native exposures at known pixel phases."""
    series = _series(study)
    n = len(study.frames)
    unavailable = Observations(np.empty((n, 0)), np.array([]), [], np.array([], bool))
    raw = np.array(
        [
            f["common_dn"] if not r["ambiguous"] else np.nan
            for f, r in zip(study.frames, study.registration)
        ]
    )
    native = Observations(raw[:, None], np.array([0]), [None], np.array([False]))
    offsets = np.asarray(meta["phase_offsets"])
    runs = np.r_[0, np.arange(meta["phase_start"], meta["phase_start"] + len(offsets))]
    rows = []
    for method in ALL_METHODS:
        measurements = (
            native if method == "raw" else series.get((method, "cd"), unavailable)
        )
        unit = UNITS.get(method, "nm")
        if small:
            observed, support = measurements.reduce([0, 1, 2])
            rows.append(
                _fit_row(
                    base,
                    method,
                    "small_cd70",
                    unit,
                    meta["cd_levels_nm"],
                    observed,
                    support,
                    nominal=observed[0],
                    require_levels=3,
                )
            )
        for field in ("cd",) if small or method == "raw" else ("cd", "x", "y"):
            data = (
                measurements
                if field == "cd"
                else series.get((method, field), unavailable)
            )
            values, support = data.reduce(runs)
            observed = values[1:] - values[0]
            matrix = data.matrix[runs]
            common = np.isfinite(matrix).all(axis=0)
            location_changes = matrix[1:, common] - matrix[0, common]
            expected = np.zeros(len(observed))
            if field in ("x", "y"):
                axis = 0 if field == "x" else 1
                expected = offsets[:, axis] * base["pixel_nm"]
                if method in {"harmonic", "pooled"}:
                    # Tile positions are residuals after global alignment. Add
                    # that independently estimated component back to score motion.
                    reg = np.array(
                        [study.registration[int(i)]["shift_xy"][axis] for i in runs]
                    )
                    shift = (reg[1:] - reg[0]) * base["pixel_nm"]
                    observed += shift
                    location_changes += shift[:, None]
            error = observed - expected
            location_error = location_changes - expected[:, None]
            valid = support > 0 and np.isfinite(error).all()
            rows.append(
                dict(
                    base,
                    method=method,
                    scenario="small_cd70_phase" if small else "pixel_phase_" + field,
                    unit=unit if field == "cd" else "nm",
                    support=support,
                    count=len(error) if valid else 0,
                    status="ok" if valid else "insufficient matched measurements",
                    slope=np.nan,
                    intercept=np.nan,
                    nominal_value=values[0],
                    fractional_slope=np.nan,
                    elasticity=np.nan,
                    mean_error=float(error.mean()) if valid else np.nan,
                    rmse=float(np.sqrt((error**2).mean())) if valid else np.nan,
                    peak_abs_error=float(np.max(abs(error))) if valid else np.nan,
                    location_rmse=float(np.sqrt(np.mean(location_error**2)))
                    if valid
                    else np.nan,
                    p95_abs_location_error=float(np.percentile(abs(location_error), 95))
                    if valid
                    else np.nan,
                    injected_offsets_xy_px=offsets.tolist(),
                    observed_changes=observed.tolist(),
                    expected_changes=expected.tolist(),
                )
            )
    return rows


def run_stress(patterns, pixel_nm, backend="cpu"):
    """Run blind sensitivity and pixel-phase controls independently for each region.

    Tuple inputs are (shape, lattice, pitch_px, cd_nm). Mapping inputs also
    carry the source region, aperture orientation, axis ratio, and both primitive
    lattice vectors. A separate 70 nm field exercises the small-size limit.
    """
    results = []
    for pattern in _patterns(patterns):
        shape, pitch = pattern["shape"], pattern["pitch_px"]
        source_cd = pattern["cd_nm"]
        for small in (False, True):
            cd = 70.0 if small else source_cd
            options = {
                k: pattern[k] for k in ("angle_deg", "ellipse_ratio", "basis_xy")
            }
            base = dict(
                pattern,
                pixel_nm=float(pixel_nm),
                source_cd_nm=source_cd,
                nominal_cd_nm=cd,
                sampling_cd_px=cd / pixel_nm,
                stress_group="small_aperture" if small else "source_geometry",
            )
            failure_scenarios = (
                ("small_cd70", "small_cd70_phase")
                if small
                else ("global_cd", "pixel_phase_cd")
            )
            if cd >= pitch * pixel_nm:
                for method in ALL_METHODS:
                    for scenario in failure_scenarios:
                        row = _fit_row(
                            base,
                            method,
                            scenario,
                            UNITS.get(method, "nm"),
                            [cd],
                            [np.nan],
                            0,
                        )
                        results.append(dict(row, status="size exceeds available pitch"))
                continue
            images, meta = _challenge_frames(
                shape, pattern["lattice"], pitch, cd, pixel_nm, small=small, **options
            )
            base.update(
                tile_um=meta["tile_px"] * pixel_nm / 1000,
                field_shape_yx_px=meta["field_shape_yx_px"],
                cd_levels_nm=meta["cd_levels_nm"],
                local_delta_nm=0.0 if small else meta["delta"],
            )
            try:
                study = analyze(
                    images, pixel_nm=pixel_nm, tile_um=base["tile_um"], backend=backend
                )
            except ValueError as exc:
                # Detection/registration failure is itself a stress-test outcome.
                if "No periodic hole array" not in str(
                    exc
                ) and "translation" not in str(exc):
                    raise
                for method in ALL_METHODS:
                    for scenario in failure_scenarios:
                        row = _fit_row(
                            base,
                            method,
                            scenario,
                            UNITS.get(method, "nm"),
                            [cd],
                            [np.nan],
                            0,
                        )
                        results.append(dict(row, status=str(exc)))
                continue
            base["discovered_shapes"] = ",".join(
                sorted({r.shape for r in study.regions})
            )
            base["shape_agreement"] = all(r.shape == shape for r in study.regions)
            if not small:
                results.extend(_score_pattern(study, meta, base, images))
            results.extend(_score_sampling(study, meta, base, small))
    return results
