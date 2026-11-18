"""Score each acquisition before pooling spatial locations. Missing sites are explicit."""

from collections import defaultdict

import numpy as np

from .statistics import precision, triplet_scores


def _avg(values):
    good = np.asarray(values, float)
    good = good[np.isfinite(good)]
    return float(good.mean()) if len(good) else np.nan


def summarize(study):
    n = len(study.frames)
    scales = {
        "cd_px": ("CD", "nm", study.pixel_nm),
        "width_px": ("Width", "nm", study.pixel_nm),
        "height_px": ("Height", "nm", study.pixel_nm),
        "area_diameter_px": ("Area diameter", "nm", study.pixel_nm),
        "flux_dn_px2": ("Integrated signal", "DN·px²", 1),
        "residual_x_nm": ("Placement X", "nm", 1),
        "residual_y_nm": ("Placement Y", "nm", 1),
    }
    summary, splits, spatial, aggregates = [], [], [], []
    matched = not any(r["ambiguous"] for r in study.registration)

    def add(
        region,
        method,
        metric,
        unit,
        level,
        matrix,
        *,
        labels=(),
        possible=None,
        paired=None,
    ):
        matrix = np.asarray(matrix, dtype=float).reshape(n, -1)
        valid = np.isfinite(matrix).all(axis=0)
        score = precision(matrix)
        entry = dict(
            region=region,
            method=method,
            metric=metric,
            unit=unit,
            level=level,
            mean=_avg(score["mean"]),
            sigma3=_avg(score["sigma3"]),
            percent=_avg(score["percent"])
            if not metric.startswith("Placement")
            else np.nan,
            valid=int(valid.sum()),
            possible=matrix.shape[1] if possible is None else possible,
        )
        if paired is not None:
            base = precision(paired[:, valid])
            entry["paired_raw_percent"] = _avg(base["percent"])
            entry["paired_raw_mean_dn"] = _avg(base["mean"])
        summary.append(entry)
        if level in (
            "Native page",
            "Common field",
            "Pilot tile",
            "Shared sites · tile",
        ):
            for item in triplet_scores(matrix):
                splits.append(
                    dict(
                        region=region,
                        method=method,
                        metric=metric,
                        unit=unit,
                        level=level,
                        **item,
                    )
                )
        if labels:
            for k, label in enumerate(labels):
                spatial.append(
                    dict(
                        region=region,
                        method=method,
                        metric=metric,
                        unit=unit,
                        level=level,
                        i=label[0],
                        j=label[1],
                        sigma3=score["sigma3"][k],
                        percent=score["percent"][k]
                        if not metric.startswith("Placement")
                        else np.nan,
                        count=int(score["count"][k]),
                    )
                )
                for r in range(n):
                    aggregates.append(
                        dict(
                            region=region,
                            method=method,
                            metric=metric,
                            unit=unit,
                            level=level,
                            repeat=r,
                            i=label[0],
                            j=label[1],
                            value=matrix[r, k],
                        )
                    )

    for field, level in [("mean_dn", "Native page"), ("common_dn", "Common field")]:
        arr = np.array([x[field] for x in study.frames])
        if not matched and field == "common_dn":
            arr[:] = np.nan
        add(-1, "raw", "Brightness", "DN", level, arr)
    for axis, metric in enumerate(("Placement X", "Placement Y")):
        shifts = (
            np.array([r["shift_xy"][axis] for r in study.registration]) * study.pixel_nm
        )
        if not matched:
            shifts[:] = np.nan
        add(-1, "alignment", metric, "nm", "Acquisition shift", shifts)
    holes_by_region = defaultdict(list)
    for r in study.holes:
        holes_by_region[r["region"]].append(r)
    tile_by_region = defaultdict(list)
    for t in study.tiles:
        tile_by_region[t["region"]].append(t)
    for region in study.regions:
        rid = region.region_id
        tiles = tile_by_region[rid]
        keys = sorted({(t["i"], t["j"]) for t in tiles if t["method"] == "raw"})
        index = {key: i for i, key in enumerate(keys)}
        baseline = np.full((n, len(keys)), np.nan)
        tile_methods = {
            method: {
                metric: np.full_like(baseline, np.nan)
                for metric in (
                    "CD",
                    "Width",
                    "Height",
                    "Area diameter",
                    "Placement X",
                    "Placement Y",
                )
            }
            for method in ("pooled", "harmonic")
        }
        for t in tiles:
            col = index[t["i"], t["j"]]
            if not matched:
                continue
            if t["method"] == "raw":
                baseline[t["repeat"], col] = t["value"]
            else:
                matrices = tile_methods[t["method"]]
                for metric, field in zip(
                    matrices,
                    [
                        "cd_px",
                        "cd_x_px",
                        "cd_y_px",
                        "area_diameter_px",
                        "dx_px",
                        "dy_px",
                    ],
                ):
                    valid = (
                        t.get("phase_status") == "ok"
                        if metric.startswith("Placement")
                        else t["status"] == "ok"
                    )
                    if valid:
                        matrices[metric][t["repeat"], col] = (
                            t.get(field, np.nan) * study.pixel_nm
                        )
        add(rid, "raw", "Brightness", "DN", "Pilot tile", baseline, labels=keys)
        for method, matrices in tile_methods.items():
            for metric, arr in matrices.items():
                add(
                    rid,
                    method,
                    metric,
                    "nm",
                    "Pilot tile",
                    arr,
                    labels=keys,
                    paired=baseline,
                )
        rows = holes_by_region[rid]
        sites = sorted({r["site"] for r in rows})
        lookup = {site: k for k, site in enumerate(sites)}
        by_tile = defaultdict(set)
        for r in rows:
            if r["complete_tile"]:
                by_tile[r["i"], r["j"]].add(lookup[r["site"]])
        site_idx = [np.array(sorted(by_tile[key]), dtype=int) for key in keys]
        methods = defaultdict(list)
        for r in rows:
            methods[r["method"]].append(r)
        common_cds = {}

        def bin_sites(arr):
            result = np.full((n, len(keys)), np.nan)
            complete = np.isfinite(arr).all(axis=0)
            for k, idx in enumerate(site_idx):
                idx = idx[complete[idx]]
                if len(idx):
                    result[:, k] = arr[:, idx].mean(axis=1)
            return result

        for method, samples in methods.items():
            for field, (metric, unit, scale) in scales.items():
                if field == "flux_dn_px2" and method != "flux":
                    continue
                if method == "flux" and not any(
                    np.isfinite(r.get(field, np.nan)) for r in samples
                ):
                    continue
                arr = np.full((n, len(sites)), np.nan)
                if matched:
                    for r in samples:
                        if r["valid"]:
                            arr[r["repeat"], lookup[r["site"]]] = (
                                r.get(field, np.nan) * scale
                            )
                if metric == "CD" and np.any(np.isfinite(arr).all(axis=0)):
                    common_cds[method] = arr
                support = np.isfinite(arr).all(axis=0)
                possible = (
                    len({r["site"] for r in samples if not r["anchor"]})
                    if metric.startswith("Placement")
                    else len(sites)
                )
                add(
                    rid, method, metric, unit, "Individual hole", arr, possible=possible
                )
                whole = (
                    arr[:, support].mean(axis=1)
                    if support.any()
                    else np.full(n, np.nan)
                )
                add(rid, method, metric, unit, "Common field", whole)
                add(
                    rid,
                    method,
                    metric,
                    unit,
                    "Pilot tile",
                    bin_sites(arr),
                    labels=keys,
                    paired=baseline,
                )
        if len(common_cds) > 1:
            mask = np.logical_and.reduce(
                [np.isfinite(a).all(axis=0) for a in common_cds.values()]
            )
            for method, arr in common_cds.items():
                shared = np.where(mask, arr, np.nan)
                add(
                    rid,
                    method,
                    "CD",
                    "nm",
                    "Shared sites · tile",
                    bin_sites(shared),
                    labels=keys,
                    paired=baseline,
                )
                summary[-1]["shared_holes"] = int(mask.sum())
    return summary, splits, spatial, aggregates
