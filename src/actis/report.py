"""Static, offline report. Display stretching never enters an estimator."""

from collections import Counter, defaultdict
from html import escape
from importlib.resources import files
from time import perf_counter

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from .analysis import NAMES

PALETTE = (colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype("uint8")


def num(x):
    if x is None:
        return "—"
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return f"{x:.4g}" if np.isfinite(x) else "—"
    return escape(str(x))


def table(columns, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{escape(s)}</th>" for s in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{num(v)}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _range(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return (0.0, 1.0)
    lo, hi = np.percentile(v, [2, 98])
    return (float(lo), float(hi)) if hi > lo else (float(lo) - 0.5, float(hi) + 0.5)


def _color(v, limits):
    lo, hi = limits
    return tuple(map(int, PALETTE[int(255 * np.clip((v - lo) / (hi - lo), 0, 1))]))


def _legend(limits, unit):
    return f'<p class="legend">{num(limits[0])} <i></i> {num(limits[1])} {escape(unit)} · color limits at the 2nd and 98th percentiles</p>'


def _base(image, limits):
    lo, hi = limits
    return Image.fromarray(
        np.uint8(np.clip((image.astype(np.float32) - lo) * (255 / (hi - lo)), 0, 255))
    ).convert("RGB")


def _dot(draw, x, y, color, radius=2):
    if np.isfinite([x, y]).all():
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _rect(draw, x, y, side, color):
    draw.rectangle(
        (x - side / 2, y - side / 2, x + side / 2, y + side / 2),
        fill=color,
        outline=(232, 232, 232, 185),
        width=1,
    )


def _ruler(im, pixel_nm):
    target = im.width * pixel_nm / 5
    magnitude = 10 ** np.floor(np.log10(max(target, 1)))
    length = max(
        k * magnitude
        for k in (0.1, 0.2, 0.5, 1, 2, 5, 10)
        if k * magnitude <= target or k == 0.1
    )
    d = ImageDraw.Draw(im)
    px = length / pixel_nm
    label = f"{length / 1000:g} um" if length >= 1000 else f"{length:g} nm"
    d.rectangle((5, im.height - 28, max(53, px + 15), im.height - 4), fill="#181818")
    d.text((8, im.height - 27), label, fill="white")
    d.line((8, im.height - 9, 8 + px, im.height - 9), fill="white", width=2)


def _figure(path, caption):
    return f'<figure><a href="{path}"><img loading="lazy" src="{path}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


def _curves(frame, path):
    im = Image.new("RGB", (460, 115), "white")
    d = ImageDraw.Draw(im)
    low, high = (
        min(min(frame[k]) for k in ("row_mean_dn", "column_mean_dn")),
        max(max(frame[k]) for k in ("row_mean_dn", "column_mean_dn")),
    )
    for field, color in (("row_mean_dn", "#157473"), ("column_mean_dn", "#ab5128")):
        y = frame[field]
        xy = np.column_stack(
            (np.linspace(5, 454, len(y)), 90 - 83 * (y - low) / max(high - low, 1))
        )
        d.line(list(map(tuple, xy)), fill=color, width=1)
    d.text(
        (4, 96),
        f"rows (teal) / columns (orange)    {low:.0f} to {high:.0f} DN",
        fill="#333333",
    )
    im.save(path)


def make_report(
    out,
    images,
    info,
    study,
    summary,
    splits,
    spatial,
    stress,
    comparisons,
    performance,
    synthetic,
):
    start = perf_counter()
    assets = out / "images"
    assets.mkdir(exist_ok=True)
    limits = _range(np.concatenate([a[::4, ::4].ravel() for a in images]))
    bases = [_base(a, limits) for a in images]
    rows_by = defaultdict(list)
    tiles_by = defaultdict(list)
    for row in study.holes:
        rows_by[row["method"], row["repeat"]].append(row)
    for row in study.tiles:
        tiles_by[row["method"], row["repeat"]].append(row)
    all_cd = _range(
        [
            r["cd_px"] * study.pixel_nm
            for r in study.holes
            if r["valid"] and np.isfinite(r["cd_px"])
        ]
        + [
            r["value"]
            for r in study.tiles
            if r["method"] in ("harmonic", "pooled") and r["status"] == "ok"
        ]
    )
    side = study.tile_um * 1000 / study.pixel_nm
    h, w = images[0].shape
    body = [
        "<header><div>Photomask metrology</div><h1>ACTIS reference-image benchmark</h1><p>"
        + (
            "Synthetic demonstration · simulated apertures and detector conditions"
            if synthetic
            else "Reference data · eight ACTIS detector positions"
        )
        + "</p></header>"
    ]
    body.append(
        f"<p>{len(images)} acquisitions · {w} × {h} pixels · {study.pixel_nm:g} nm/pixel nominal mask scale · "
        f"{w * study.pixel_nm / 1000:.3f} × {h * study.pixel_nm / 1000:.3f} µm per acquisition. "
        f"Complete {study.tile_um:g} µm pilot tiles share a fixed reference grid. "
        "The 100 × 100 µm historical precision map requires a larger footprint.</p>"
    )
    body.append(
        "<p>CD values describe apparent mask-scale dimensions. The eight measurements vary detector position and acquisition time together. "
        "The synthetic size sweep below tests whether lower scatter accompanies sensitivity to a change in aperture size.</p>"
    )
    body.extend(f'<p class="note">{escape(note)}</p>' for note in study.notes)
    body.append(
        '<nav><a href="#scores">Precision</a><a href="#originals">Acquisitions</a><a href="#overlays">Overlays</a>'
        '<a href="#placement">Placement</a><a href="#response">Size response</a><a href="#timing">Throughput</a></nav>'
    )
    body.append(
        '<h2 id="scores">Precision across detector positions</h2><p>Compute one value per acquisition and location, then '
        "calculate sample 3σ and 300s/|mean|. Average the binwise results with equal weights. A paired raw percentage uses "
        "the same valid tiles. DN and integrated intensity have different responses to CD. Placement percentages are undefined.</p>"
    )
    headline = [
        r
        for r in summary
        if (
            r["level"] in ("Pilot tile", "Native page")
            and r["metric"] in ("CD", "Brightness", "Integrated signal")
        )
    ]

    def score_table(values):
        return table(
            [
                "Region",
                "Method",
                "Quantity",
                "Support",
                "Mean",
                "3σ",
                "Unit",
                "3σ %",
                "Raw % on paired tiles",
                "Valid / eligible",
            ],
            (
                [
                    r["region"] if r["region"] != -1 else "All",
                    NAMES.get(r["method"], "Image registration"),
                    r["metric"],
                    r["level"],
                    r["mean"],
                    r["sigma3"],
                    r["unit"],
                    r["percent"],
                    r.get("paired_raw_percent"),
                    f"{r['valid']} / {r['possible']}",
                ]
                for r in values
            ),
        )

    body.append(score_table(headline))
    body.append(
        "<details><summary>Full-field, common-site, per-hole, axis and placement results</summary>"
        + score_table(summary)
        + "</details>"
    )
    body.append(
        "<p>All 56 overlapping three-acquisition subsets use the same eight-acquisition support. "
        'The downloadable table preserves each subset and bin count. <a href="triplets.csv">Three-acquisition scores</a></p>'
    )
    grouped = defaultdict(list)
    for row in splits:
        if (
            row["level"] in ("Native page", "Pilot tile")
            and row["metric"] in ("Brightness", "CD", "Integrated signal")
            and np.isfinite(row["percent"])
        ):
            grouped[(row["region"], row["method"], row["level"], row["metric"])].append(
                row["percent"]
            )
    body.append(
        "<details><summary>Precision across choices of three acquisitions</summary>"
        + table(
            [
                "Region",
                "Method",
                "Support",
                "Quantity",
                "Minimum %",
                "Median %",
                "Maximum %",
                "Triplets",
            ],
            (
                [
                    rid if rid != -1 else "All",
                    NAMES[m],
                    lev,
                    metric,
                    min(vals),
                    np.median(vals),
                    max(vals),
                    len(vals),
                ]
                for (rid, m, lev, metric), vals in grouped.items()
            ),
        )
        + "</details>"
    )
    body.append(
        "<h3>Discovered patterns</h3>"
        + table(
            [
                "Region",
                "Shape",
                "Best shape hypothesis",
                "Orientation °",
                "Array",
                "Pitch px",
                "Candidate sites",
            ],
            (
                [
                    r.region_id,
                    r.shape,
                    r.best_shape,
                    np.degrees(r.angle),
                    r.lattice,
                    r.pitch,
                    len(r.centers),
                ]
                for r in study.regions
            ),
        )
    )
    body.append(
        '<h2 id="originals">Individual acquisitions</h2><p>All images share one grayscale stretch. Open an image for native pixels. '
        'The teal and orange curves show mean brightness across rows and columns.</p><div class="gallery">'
    )
    for i, (base, frame, meta) in enumerate(zip(bases, study.frames, info)):
        im = base.copy()
        _ruler(im, study.pixel_nm)
        im.save(assets / f"native-{i}.png")
        _curves(frame, assets / f"profile-{i}.png")
        caption = f"{i + 1}. {meta['file']} · mean {frame['mean_dn']:.2f} DN · range {frame['min_dn']}–{frame['max_dn']}"
        body.append(
            "<div>"
            + _figure(f"images/native-{i}.png", caption)
            + f'<img src="images/profile-{i}.png" alt="Row and column intensity profiles"></div>'
        )
    body.append(
        '</div><h2 id="overlays">Measurements on each acquisition</h2><p>Colored outlines mark measured contours or fitted apertures. '
        "Dots mark measurements without contours. Crosses mark excluded sites. Each detail crop is enlarged using nearest-neighbor pixels. "
        "Raw brightness, lattice harmonics and pooled aperture moments are displayed at tile resolution.</p>"
    )
    for method, name in NAMES.items():
        hole_method = method not in ("raw", "harmonic", "pooled")
        flux_units = method == "flux" and not any(
            np.isfinite(r.get("cd_px", np.nan)) for r in rows_by[method, 0]
        )
        unit = "DN·px²" if flux_units else ("DN" if method == "raw" else "nm")
        scale = (
            _range(
                [
                    r.get("flux_dn_px2", np.nan)
                    for i in range(len(images))
                    for r in rows_by[method, i]
                    if r["valid"]
                ]
            )
            if flux_units
            else (
                _range(
                    [
                        r["value"]
                        for i in range(len(images))
                        for r in tiles_by[method, i]
                    ]
                )
                if method == "raw"
                else all_cd
            )
        )
        body.append(
            f"<details open><summary>{escape(name)}</summary>"
            + _legend(scale, unit)
            + '<div class="gallery">'
        )
        for i, base in enumerate(bases):
            im = base.copy()
            d = ImageDraw.Draw(im, "RGBA")
            dx, dy = study.registration[i]["shift_xy"]
            if hole_method:
                for r in rows_by[method, i]:
                    x, y = r["ref_x_px"] + dx, r["ref_y_px"] + dy
                    val = (
                        r.get("flux_dn_px2", np.nan)
                        if flux_units
                        else r["cd_px"] * study.pixel_nm
                    )
                    if not r["valid"] or not np.isfinite(val):
                        d.line((x - 1, y - 1, x + 1, y + 1), fill=(250, 135, 95, 190))
                        d.line((x + 1, y - 1, x - 1, y + 1), fill=(250, 135, 95, 190))
                        continue
                    color = _color(val, scale)
                    outline = r.get("contour_xy")
                    if outline is not None and len(outline) > 2:
                        xy = [tuple(pt) for pt in outline]
                        d.polygon(xy, fill=color + (70,))
                        d.line(xy + [xy[0]], fill=color + (255,), width=1)
                    else:
                        _dot(d, r["x_px"], r["y_px"], color + (220,))
            else:
                for t in tiles_by[method, i]:
                    if t["status"] == "ok" and np.isfinite(t["value"]):
                        _rect(
                            d,
                            t["x_px"] + dx,
                            t["y_px"] + dy,
                            side,
                            _color(t["value"], scale) + (122,),
                        )
            if study.regions:
                chosen = max(study.regions, key=lambda region: len(region.centers))
                cx, cy = np.median(chosen.centers, axis=0) + [dx, dy]
            else:
                cx, cy = w / 2, h / 2
            size = min(80, w, h)
            x = int(np.clip(cx - size / 2, 0, w - size))
            y = int(np.clip(cy - size / 2, 0, h - size))
            crop = im.crop((x, y, x + size, y + size)).resize(
                (240, 240), Image.Resampling.NEAREST
            )
            crop.save(assets / f"crop-{method}-{i}.png")
            _ruler(im, study.pixel_nm)
            im.save(assets / f"overlay-{method}-{i}.png")
            body.append(
                "<div>"
                + _figure(f"images/overlay-{method}-{i}.png", info[i]["file"])
                + _figure(
                    f"images/crop-{method}-{i}.png", f"{size} × {size} native pixels"
                )
                + "</div>"
            )
        body.append("</div></details>")
    body.append(
        "<h3>Spatial 3σ maps</h3><p>Each rectangle is a complete pilot tile. Blank areas have no qualified value. "
        'CD maps use one shared scale per unit across methods.</p><div class="gallery">'
    )
    common_map_scale = {
        key: _range(
            [
                r[key]
                for r in spatial
                if r["level"] == "Pilot tile" and r["metric"] == "CD"
            ]
        )
        for key in ("sigma3", "percent")
    }
    for method, name in NAMES.items():
        entries = [
            r
            for r in spatial
            if r["method"] == method
            and r["level"] == "Pilot tile"
            and r["metric"] in ("Brightness", "CD", "Integrated signal")
        ]
        for metric in sorted({r["metric"] for r in entries}):
            layer = [r for r in entries if r["metric"] == metric]
            for key, unit in (("sigma3", layer[0]["unit"]), ("percent", "%")):
                v = [r[key] for r in layer if np.isfinite(r[key])]
                if not v:
                    continue
                scale = common_map_scale[key] if metric == "CD" else _range(v)
                im = bases[0].copy()
                d = ImageDraw.Draw(im, "RGBA")
                for r in layer:
                    if np.isfinite(r[key]):
                        x = study.bounds[0] + (r["i"] + 0.5) * side
                        y = study.bounds[1] + (r["j"] + 0.5) * side
                        _rect(d, x, y, side, _color(r[key], scale) + (185,))
                _ruler(im, study.pixel_nm)
                path = f"images/map-{method}-{metric.replace(' ', '-')}-{key}.png"
                im.save(out / path)
                body.append(
                    "<div>"
                    + _figure(path, f"{name} · {metric} · 3σ ({unit})")
                    + _legend(scale, unit)
                    + "</div>"
                )
    body.append(
        '</div><h2 id="placement">Relative placement</h2><p>X increases to the right and Y downward. Global shift is moving minus reference. '
        "Local per-hole residuals use centers held out of the translation fit. Lattice phase measures tile-average residual displacement. "
        "Whole-pitch alternatives can persist in periodic fields.</p>"
    )
    body.append(
        table(
            [
                "Acquisition",
                "Global X (nm)",
                "Global Y (nm)",
                "Score",
                "Registration status",
            ],
            (
                [
                    i + 1,
                    reg["shift_xy"][0] * study.pixel_nm,
                    reg["shift_xy"][1] * study.pixel_nm,
                    reg["score"],
                    reg["reason"],
                ]
                for i, reg in enumerate(study.registration)
            ),
        )
    )
    body.append(
        table(
            ["Global acquisition shift", "3σ (nm)"],
            (
                [r["metric"], r["sigma3"]]
                for r in summary
                if r["level"] == "Acquisition shift"
            ),
        )
    )
    body.append('<div class="gallery">')
    for i, base in enumerate(bases):
        im = base.copy()
        d = ImageDraw.Draw(im)
        samples = [
            r
            for r in rows_by["contour", i]
            if np.isfinite(r.get("residual_x_nm", np.nan))
        ]
        for r in samples[:: max(1, len(samples) // 250)]:
            x, y = r["x_px"], r["y_px"]
            ex, ey = (
                x + 10 * r["residual_x_nm"] / study.pixel_nm,
                y + 10 * r["residual_y_nm"] / study.pixel_nm,
            )
            d.line((x, y, ex, ey), fill="#ffb95d", width=2)
            _dot(d, ex, ey, "#f17a42", 1)
        _ruler(im, study.pixel_nm)
        path = f"images/placement-{i}.png"
        im.save(out / path)
        body.append(
            _figure(
                path,
                f"Acquisition {i + 1} · contour displacement ×10 · {len(samples)} held-out centers",
            )
        )
    body.append(
        '</div><h2 id="response">Response to known size and placement changes</h2><p>Independent simulated arrays exercise the same blind pipeline. '
        "The simulated optical response and digitizer gain are illustrative. Local steps test that spatial variation survives processing.</p>"
    )
    body.append(
        "<details><summary>All synthetic controls, including the 70 nm sampling test</summary>"
        + table(
            [
                "Region",
                "True shape",
                "Simulated CD (nm)",
                "Discovered",
                "Method",
                "Scenario",
                "Unit",
                "Slope",
                "Mean error",
                "RMSE",
                "Samples",
                "Status",
            ],
            (
                [
                    r.get("region"),
                    r.get("shape"),
                    r.get("nominal_cd_nm"),
                    r.get("discovered_shapes"),
                    r.get("method"),
                    r.get("scenario"),
                    r.get("unit"),
                    r.get("slope"),
                    r.get("mean_error"),
                    r.get("rmse"),
                    r.get("count"),
                    r.get("status"),
                ]
                for r in stress
            ),
        )
        + "</details>"
    )
    body.append(
        "<h3>Comparison with sensitivity taken into account</h3><p>Each matched raw or candidate percentage is divided by its "
        "simulated fractional response per nm. This is a model-dependent comparison. Digitizer offset, contrast and optical transfer can "
        "change that response on an actual mask.</p>"
    )
    body.append(
        table(
            [
                "Region",
                "Method",
                "Paired tiles",
                "3σ %",
                "Raw 3σ %",
                "Modeled 3σ (nm)",
                "Modeled raw 3σ (nm)",
                "Finding",
            ],
            (
                [
                    r["region"],
                    r["method"],
                    r["tiles"],
                    r["percent"],
                    r["raw_percent"],
                    r["equivalent_nm"],
                    r["raw_equivalent_nm"],
                    r["finding"],
                ]
                for r in comparisons
            ),
        )
    )
    reasons = Counter(
        (r["method"], r.get("flag", "")) for r in study.holes if not r["valid"]
    )
    reasons.update(
        (r["method"], r["status"]) for r in study.tiles if r["status"] != "ok"
    )
    if reasons:
        body.append(
            "<h3>Excluded measurements</h3>"
            + table(
                ["Method", "Reason", "Count"],
                ([m, f, n] for (m, f), n in reasons.most_common()),
            )
        )
    body.append(
        '<h2 id="timing">Workstation throughput</h2><p>Input read and decode took '
        + num(performance["input_seconds"])
        + " s. "
        "Method times include native-pixel processing and fit setup. Projections scale observed pixel throughput to "
        "1,500 pages of 2,048 × 1,376 pixels at this pattern density, on this device. TIFF decoding, disk contention and production stitching require separate measurement.</p>"
    )
    body.append(
        table(
            ["Stage", "Elapsed s", "Full board pages/s", "1,500-page projection (s)"],
            (
                [k, v["seconds"], v["pages_per_s"], v["projected_seconds"]]
                for k, v in performance["stages"].items()
            ),
        )
    )
    if performance.get("disk"):
        disk = performance["disk"]
        body.append(
            f"<p>Sequential read of {escape(disk['file'])} took {num(disk['seconds'])} s at {num(disk['mib_s'])} MiB/s. "
            "The operating-system file cache may affect this measurement.</p>"
        )
    body.append(
        "<h2>Exports</h2><p>"
        + " · ".join(
            f'<a href="{p}">{escape(label)}</a>'
            for p, label in (
                ("run.json", "Acquisition settings and diagnostics"),
                ("precision.csv", "Precision by method and scale"),
                ("tile_values.csv", "Matched tile values"),
                ("tile_scores.csv", "Tile precision maps"),
                ("holes.csv.gz", "Individual holes and contours"),
                ("native_tiles.csv", "Harmonic fits and raw tiles"),
                ("stress.csv", "Synthetic response checks"),
                ("performance.json", "Timings and hardware"),
                ("comparison.csv", "Method comparison"),
            )
        )
        + "</p>"
    )
    css = files("actis").joinpath("report.css").read_text(encoding="utf-8")
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>ACTIS reference-image benchmark</title><style>"
        + css
        + "</style><body>"
        + "".join(body)
        + "</body></html>",
        encoding="utf-8",
    )
    return perf_counter() - start
