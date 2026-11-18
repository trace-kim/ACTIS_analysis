"""Analyze repeats in native image coordinates; correspondence uses the first field."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .discovery import detect_centers, discover, match_centers, register
from .estimators import empty_measurement, estimate_holes
from .harmonics import harmonic_tiles
from .pooled import pooled_common_tiles

METHODS = ("contour", "model", "differential", "flux")
NAMES = {
    "raw": "Raw brightness",
    "contour": "Local contour",
    "model": "Aperture model",
    "differential": "Differential fit",
    "flux": "Integrated hole intensity",
    "harmonic": "Lattice harmonics",
    "pooled": "Pooled aperture moments",
}


@dataclass
class Study:
    pixel_nm: float
    tile_um: float
    regions: list
    registration: list
    bounds: tuple
    holes: list
    tiles: list
    frames: list
    timing: dict
    notes: list


def rect_mean(image, bounds):
    """Integral of the native piecewise-constant pixel field over a fractional rectangle."""
    x0, y0, x1, y1 = bounds
    h, w = image.shape
    if not (
        -0.500001 <= x0 < x1 <= w - 0.499999 and -0.500001 <= y0 < y1 <= h - 0.499999
    ):
        return np.nan
    x = np.arange(max(0, int(np.floor(x0 + 0.5))), min(w, int(np.ceil(x1 + 0.5))))
    y = np.arange(max(0, int(np.floor(y0 + 0.5))), min(h, int(np.ceil(y1 + 0.5))))
    wx = np.maximum(0, np.minimum(x + 0.5, x1) - np.maximum(x - 0.5, x0))
    wy = np.maximum(0, np.minimum(y + 0.5, y1) - np.maximum(y - 0.5, y0))
    patch = image[y[0] : y[-1] + 1, x[0] : x[-1] + 1]
    return float(
        np.einsum("i,ij,j->", wy, patch, wx, optimize=False) / (wx.sum() * wy.sum())
    )


def frame_metrics(image, bounds, shift):
    dx, dy = shift
    moved = np.array(bounds) + [dx, dy, dx, dy]
    histogram = np.bincount(image.ravel(), minlength=65536)
    levels = np.flatnonzero(histogram)
    return {
        "mean_dn": float(image.mean(dtype=np.float64)),
        "common_dn": rect_mean(image, moved),
        "min_dn": int(image.min()),
        "max_dn": int(image.max()),
        "zero_fraction": float(np.mean(image == 0)),
        "full_scale_fraction": float(np.mean(image == 65535)),
        "occupied_levels": int(len(levels)),
        "observed_min_fraction": float(histogram[levels[0]] / image.size),
        "observed_max_fraction": float(histogram[levels[-1]] / image.size),
        "row_mean_dn": image.mean(axis=1),
        "column_mean_dn": image.mean(axis=0),
    }


def common_bounds(shape, shifts):
    """Intersection of native pixel extents in the reference coordinate system."""
    h, w = shape
    shifts = np.asarray(shifts, float)
    x0, y0 = -0.5 - shifts.min(axis=0)
    x1, y1 = np.array([w - 0.5, h - 0.5]) - shifts.max(axis=0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Reference acquisitions have no common image footprint")
    return x0, y0, x1, y1


def analyze(
    images,
    pixel_nm=21.0,
    tile_um=5.0,
    backend="cpu",
    progress=None,
    allow_unpatterned=False,
):
    if len(images) < 1 or any(
        im.ndim != 2 or im.shape != images[0].shape for im in images
    ):
        raise ValueError("Expected equal-size grayscale acquisitions")
    if (
        not np.isfinite(pixel_nm)
        or not np.isfinite(tile_um)
        or min(pixel_nm, tile_um) <= 0
    ):
        raise ValueError("Pixel pitch and tile size must be finite and positive")
    timing = defaultdict(float)
    start = perf_counter()
    regions = discover(images[0])
    if not regions:
        if allow_unpatterned:
            bounds = common_bounds(images[0].shape, [(0.0, 0.0)])
            registrations = [
                dict(
                    shift_xy=(0.0, 0.0),
                    score=0.0,
                    ambiguous=True,
                    periodic_ambiguity=False,
                    candidates=[],
                    reason="No reference lattice",
                )
                for _ in images
            ]
            timing["discovery_and_alignment"] = perf_counter() - start
            start = perf_counter()
            frames = [frame_metrics(im, bounds, (0.0, 0.0)) for im in images]
            timing["raw"] = perf_counter() - start
            return Study(
                pixel_nm,
                tile_um,
                [],
                registrations,
                bounds,
                [],
                [],
                frames,
                dict(timing),
                [
                    "No periodic hole array was found in the reference acquisition. Only native page brightness is qualified."
                ],
            )
        raise ValueError(
            "No periodic hole array was detected in the reference acquisition"
        )
    pitch = min(r.pitch for r in regions)
    alignment = [
        {
            "shift_xy": (0.0, 0.0),
            "ambiguous": False,
            "periodic_ambiguity": False,
            "score": 1.0,
            "reason": "Reference acquisition",
            "candidates": [],
        }
    ]
    for image in images[1:]:
        reg = register(images[0], image, pitch_px=pitch)
        alignment.append(
            {
                k: getattr(reg, k, None)
                for k in (
                    "shift_xy",
                    "ambiguous",
                    "periodic_ambiguity",
                    "score",
                    "reason",
                    "candidates",
                )
            }
        )
    shifts = np.array([r["shift_xy"] for r in alignment], dtype=float)
    if not np.isfinite(shifts).all():
        raise ValueError(
            "A finite translation could not be estimated for every acquisition"
        )
    bounds = common_bounds(images[0].shape, shifts)
    x0, y0, x1, y1 = bounds
    side = tile_um * 1000 / pixel_nm
    nx, ny = np.floor([(x1 - x0) / side, (y1 - y0) / side]).astype(int)
    timing["discovery_and_alignment"] = perf_counter() - start
    holes, tiles, frames, notes = [], [], [], []
    if nx * ny == 0:
        notes.append(
            "No complete pilot tile fits the common field. Full-field statistics remain available."
        )
    if any(r["ambiguous"] for r in alignment):
        notes.append(
            "At least one acquisition has an unresolved translation inside the search window. Matched repeat scores are unavailable."
        )
    if any(r["periodic_ambiguity"] for r in alignment):
        notes.append(
            "Array periodicity permits shifts by whole lattice periods. Reported placement assumes the same nominal crop."
        )
    for repeat, (im, shift) in enumerate(zip(images, shifts)):
        if progress:
            progress(f"Acquisition {repeat + 1}/{len(images)}")
        t = perf_counter()
        frames.append(frame_metrics(im, bounds, shift))
        timing["raw"] += perf_counter() - t
        t = perf_counter()
        moving_centers = detect_centers(im) if repeat else None
        timing["discovery_and_alignment"] += perf_counter() - t
        dx, dy = shift
        for region in regions:
            rx0, ry0, rx1, ry1 = np.asarray(region.bounds) - 0.5
            centers = np.asarray(region.centers, float)
            in_common = (
                (centers[:, 0] >= x0)
                & (centers[:, 0] < x1)
                & (centers[:, 1] >= y0)
                & (centers[:, 1] < y1)
            )
            ids = np.flatnonzero(in_common)
            seeds = centers[ids] + shift
            found = np.ones(len(ids), bool)
            if moving_centers is not None:
                t = perf_counter()
                pairs = match_centers(centers[ids], moving_centers, shift, region.pitch)
                found[:] = False
                if len(pairs):
                    found[pairs[:, 0]] = True
                    seeds[pairs[:, 0]] = moving_centers[pairs[:, 1]]
                timing["discovery_and_alignment"] += perf_counter() - t
            # A deterministic spatial split keeps placement calibration and scoring sites disjoint.
            rng = np.random.default_rng(1973 + int(region.region_id))
            anchors = np.zeros(len(centers), bool)
            anchors[rng.permutation(len(centers))[: len(centers) // 2]] = True
            good_tiles = set()
            for j in range(ny):
                for i in range(nx):
                    left, top = x0 + i * side, y0 + j * side
                    if left < rx0 or top < ry0 or left + side > rx1 or top + side > ry1:
                        continue
                    good_tiles.add((i, j))
                    t = perf_counter()
                    dn = rect_mean(
                        im, (left + dx, top + dy, left + dx + side, top + dy + side)
                    )
                    timing["raw"] += perf_counter() - t
                    tiles.append(
                        dict(
                            repeat=repeat,
                            region=region.region_id,
                            method="raw",
                            i=i,
                            j=j,
                            x_px=left + side / 2,
                            y_px=top + side / 2,
                            value=dn,
                            unit="DN",
                            status="ok",
                        )
                    )
            for method in METHODS:
                if not len(ids):
                    continue
                t = perf_counter()
                fits = estimate_holes(
                    im,
                    seeds[found],
                    pitch_px=region.pitch,
                    shape=region.shape,
                    angle_rad=region.angle,
                    methods=(method,),
                    backend=backend,
                )[method]
                timing[method] += perf_counter() - t
                if len(fits) != int(found.sum()):
                    raise RuntimeError(
                        f"{method} returned a different number of sites than requested"
                    )
                results = iter(fits)
                for idx, paired in zip(ids, found):
                    row = (
                        next(results)
                        if paired
                        else empty_measurement(
                            idx, centers[idx] + shift, region.shape, "unmatched_center"
                        )
                    )
                    row["site_index"] = int(idx)
                    ref_x, ref_y = centers[idx]
                    i, j = np.floor([(ref_x - x0) / side, (ref_y - y0) / side]).astype(
                        int
                    )
                    row.update(
                        repeat=repeat,
                        region=region.region_id,
                        method=method,
                        site=int(idx),
                        ref_x_px=ref_x,
                        ref_y_px=ref_y,
                        anchor=bool(anchors[idx]),
                        i=int(i),
                        j=int(j),
                        complete_tile=(i, j) in good_tiles,
                    )
                    holes.append(row)
            t = perf_counter()
            native_tiles = harmonic_tiles(
                im, region, shift_xy=tuple(shift), tile_px=side, origin_xy=(x0, y0)
            )
            timing["harmonic"] += perf_counter() - t
            for row in native_tiles:
                i, j = row["tile_i"], row["tile_j"]
                if (i, j) in good_tiles:
                    tiles.append(
                        dict(
                            row,
                            repeat=repeat,
                            region=region.region_id,
                            method="harmonic",
                            i=i,
                            j=j,
                            x_px=row["x"],
                            y_px=row["y"],
                            value=float(row["cd_px"]) * pixel_nm,
                            unit="nm",
                        )
                    )
    raw_grid = {(t["region"], t["i"], t["j"]) for t in tiles if t["method"] == "raw"}
    t = perf_counter()
    for region in regions:
        pooled = pooled_common_tiles(
            images, region, shifts, tile_px=side, origin_xy=(x0, y0)
        )
        for repeat, native_tiles in enumerate(pooled):
            for row in native_tiles:
                i, j = row["tile_i"], row["tile_j"]
                if (region.region_id, i, j) in raw_grid:
                    tiles.append(
                        dict(
                            row,
                            repeat=repeat,
                            region=region.region_id,
                            method="pooled",
                            i=i,
                            j=j,
                            x_px=row["x"],
                            y_px=row["y"],
                            value=float(row["cd_px"]) * pixel_nm,
                            unit="nm",
                        )
                    )
    timing["pooled"] = perf_counter() - t
    # One translation per method and acquisition preserves coherent regional shifts.
    # Use identical valid anchors in every acquisition to avoid changing the datum.
    groups = defaultdict(list)
    anchor_count = Counter()
    for row in holes:
        groups[row["repeat"], row["method"]].append(row)
        if (
            row["valid"]
            and row["anchor"]
            and np.isfinite([row["x_px"], row["y_px"]]).all()
        ):
            anchor_count[row["method"], row["region"], row["site"]] += 1
    for group in groups.values():
        measured = [
            r for r in group if r["valid"] and np.isfinite([r["x_px"], r["y_px"]]).all()
        ]
        offsets = [
            [r["x_px"] - r["ref_x_px"], r["y_px"] - r["ref_y_px"]]
            for r in measured
            if anchor_count[r["method"], r["region"], r["site"]] == len(images)
        ]
        if len(offsets) < 3:
            continue
        tx, ty = np.median(offsets, axis=0)
        for r in measured:
            r["translation_x_px"], r["translation_y_px"] = tx, ty
            if not r["anchor"]:
                r["residual_x_nm"] = (r["x_px"] - r["ref_x_px"] - tx) * pixel_nm
                r["residual_y_nm"] = (r["y_px"] - r["ref_y_px"] - ty) * pixel_nm
    return Study(
        pixel_nm,
        tile_um,
        regions,
        alignment,
        bounds,
        holes,
        tiles,
        frames,
        dict(timing),
        notes,
    )
