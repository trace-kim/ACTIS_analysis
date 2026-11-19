"""Precision and acquisition diagnostics for the offline comparison report."""

from collections import Counter, defaultdict
from html import escape

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .analysis import NAMES

PRIMARY = {"CD", "Brightness", "Integrated signal"}
PANEL = "#1c2025"
INK = "#ccd2d8"
EXPORTS = (
    ("holes.csv.gz", "Individual holes and fitted contours"),
    ("native_tiles.csv", "Native tile measurements and fits"),
    ("tile_values.csv", "Matched values by acquisition and tile"),
    ("tile_scores.csv", "Absolute and relative precision for every tile"),
    ("precision.csv", "Precision by method and spatial support"),
    ("triplets.csv", "All three-acquisition subsets"),
    ("comparison.csv", "Sensitivity-adjusted comparisons"),
    ("stress.csv", "Synthetic response and sampling controls"),
    ("run.json", "Acquisition metadata, profiles and registration"),
    ("performance.json", "Hardware, software versions and timings"),
)


def _text(value):
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4g}" if np.isfinite(value) else "—"
    return str(value)


def _method(key):
    return NAMES.get(key, "Image registration" if key == "alignment" else key)


def _region(value):
    return "All" if value == -1 else value


def _table(headers, rows, empty="No measurements available."):
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_text(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    if not body:
        return f"<p>{escape(empty)}</p>"
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _card(title, content, intro=""):
    lead = f"<p>{escape(intro)}</p>" if intro else ""
    return f'<section class="detail-card"><h3>{escape(title)}</h3>{lead}{content}</section>'


def _fold(title, content):
    return f"<details><summary>{escape(title)}</summary>{content}</details>"


def _scores(rows, shared=False):
    headers = [
        "Region",
        "Method",
        "Quantity",
        "Support",
        "Mean",
        "3σ",
        "Unit",
        "3σ (%)",
        "Paired raw (%)",
        "Valid / eligible",
    ]
    if shared:
        headers.append("Shared holes")
    return _table(
        headers,
        (
            [
                _region(r["region"]),
                _method(r["method"]),
                r["metric"],
                r["level"],
                r["mean"],
                r["sigma3"],
                r["unit"],
                r["percent"],
                r.get("paired_raw_percent"),
                f"{r['valid']:,} / {r['possible']:,}",
            ]
            + ([r.get("shared_holes")] if shared else [])
            for r in rows
        ),
        "No eligible measurements at this level.",
    )


def _shared_comparison(rows):
    regions = defaultdict(list)
    for row in rows:
        regions[row["region"]].append(row)
    panels = []
    for rid, candidates in regions.items():
        qualified = [r for r in candidates if r["valid"] and np.isfinite(r["sigma3"])]
        if not qualified:
            continue
        upper = max(r["sigma3"] for r in qualified)
        bars = []
        for row in sorted(qualified, key=lambda r: r["sigma3"]):
            length = 100 * row["sigma3"] / upper if upper else 0
            bars.append(
                f'<tr><th scope="row" style="position:static;background:transparent">{escape(_method(row["method"]))}</th>'
                f'<td>{_text(row["sigma3"])} nm</td><td style="width:48%">'
                f'<span aria-hidden="true" style="display:block;margin-top:.4em;height:5px;'
                f'background:var(--accent);width:{length:.3f}%"></span></td></tr>'
            )
        panels.append(
            f"<h4>Region {escape(_text(rid))} · tile CD repeatability</h4>"
            '<div class="table-wrap"><table aria-label="Absolute three sigma on shared sites in nanometers">'
            + "<tbody>"
            + "".join(bars)
            + "</tbody></table></div>"
        )
    return "".join(panels) + _fold(
        "Shared holes, tile coverage and relative scores", _scores(rows, shared=True)
    )


def _axis_style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=INK, labelsize=8, length=2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.spines[:].set_visible(False)


def _plot_image(fig, assets, filename, label, caption):
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / filename, dpi=135)
    fig.clear()
    path = escape(f"{assets.name}/{filename}", quote=True)
    return (
        f'<figure><a href="{path}"><img loading="lazy" src="{path}" alt="{escape(label, quote=True)}"></a>'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def _tile_maps(tiles, methods, rid, metric, unit, assets):
    if assets is None:
        return ""
    indices = list(tiles)
    j, i = np.asarray(indices).T
    xy = np.column_stack((i, j))
    corners = xy[:, None, :] + np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    )
    out = []
    for field, display_unit in (("sigma3", unit), ("percent", "%")):
        series = [
            np.array(
                [tiles[t].get(key, {}).get(field, np.nan) for t in indices], dtype=float
            )
            for key in methods
        ]
        complete = np.concatenate([v[np.isfinite(v)] for v in series])
        if not complete.size:
            continue
        scale = Normalize(0, float(complete.max()) or 1)
        cols = min(3, len(methods))
        rows = (len(methods) + cols - 1) // cols
        fig = Figure(
            figsize=(3.25 * cols, 2.4 * rows + 0.7),
            facecolor=PANEL,
            layout="constrained",
        )
        axes = fig.subplots(rows, cols, squeeze=False).ravel()
        for ax, key, values in zip(axes, methods, series):
            _axis_style(ax)
            ax.set_facecolor("#101315")
            ax.set_title(_method(key), color=INK, fontsize=10)
            valid = np.isfinite(values)
            polygons = PolyCollection(
                corners[valid],
                array=values[valid],
                cmap="cividis",
                norm=scale,
                edgecolors=PANEL,
                linewidths=0.4,
            )
            ax.add_collection(polygons)
            if (~valid).any():
                ax.add_collection(
                    PolyCollection(
                        corners[~valid],
                        facecolors="#333940",
                        hatch="//",
                        edgecolors="#858b94",
                        linewidths=0.4,
                    )
                )
            ax.set(
                xlim=(i.min() - 0.5, i.max() + 0.5),
                ylim=(j.max() + 0.5, j.min() - 0.5),
                aspect="equal",
                xlabel="Tile X",
                ylabel="Tile Y",
            )
            for axis, coord in ((ax.xaxis, i), (ax.yaxis, j)):
                if coord.min() == coord.max():
                    axis.set_ticks([coord[0]])
                else:
                    axis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        for ax in axes[len(methods) :]:
            ax.remove()
        bar = fig.colorbar(
            polygons,
            ax=axes[: len(methods)].tolist(),
            orientation="horizontal",
            aspect=45,
            fraction=0.10,
            shrink=0.8,
        )
        bar.set_label(f"3σ ({display_unit})", color=INK, fontsize=9)
        bar.ax.tick_params(colors=INK, labelsize=8)
        bar.outline.set_visible(False)
        name = f"precision-{rid}-{metric.replace(' ', '-')}-{field}.png"
        title = f"Region {rid} · {metric} · 3σ ({display_unit})"
        out.append(
            _fold(
                title,
                _plot_image(
                    fig,
                    assets,
                    name,
                    title,
                    "Each square is one complete pilot tile. Hatching marks unavailable scores. "
                    "All panels in this map share the color scale.",
                ),
            )
        )
    return "".join(out)


def _spatial_tables(spatial, assets):
    """Put methods in adjacent columns for each repeat-matched tile."""
    tables = defaultdict(dict)
    for row in spatial:
        if row["level"] == "Pilot tile" and row["metric"] in PRIMARY:
            key = row["region"], row["metric"], row["unit"]
            tile = row["j"], row["i"]
            tables[key].setdefault(tile, {})[row["method"]] = row
    sections = []
    for (rid, metric, unit), tiles in sorted(
        tables.items(), key=lambda item: (item[0][0], item[0][1] != "CD", item[0][1])
    ):
        methods = [key for key in NAMES if any(key in vals for vals in tiles.values())]
        sections.append(_tile_maps(tiles, methods, rid, metric, unit, assets))
        rows = []
        for (j, i), values in sorted(tiles.items()):
            cells = []
            for key in methods:
                r = values.get(key, {})
                cells.append(f"{_text(r.get('sigma3'))} / {_text(r.get('percent'))}")
            rows.append([i, j, *cells])
        sections.append(
            _fold(
                f"Region {rid} · {metric} · values for {len(tiles):,} tiles",
                _table(["Tile X", "Tile Y", *map(_method, methods)], rows),
            )
        )
    return "".join(sections) or "<p>No complete pilot tiles.</p>"


def _triplets(splits):
    groups = defaultdict(list)
    for row in splits:
        if row["level"] in ("Native page", "Pilot tile") and row["metric"] in PRIMARY:
            groups[row["region"], row["level"], row["method"], row["metric"]].append(
                row
            )
    rows = []
    for (rid, level, method, metric), group in groups.items():
        values = [r["percent"] for r in group if np.isfinite(r["percent"])]
        limits = (
            (min(values), np.median(values), max(values)) if values else (None,) * 3
        )
        rows.append(
            [
                _region(rid),
                _method(method),
                metric,
                level,
                *limits,
                f"{len(values)} / {len(group)}",
            ]
        )
    return _table(
        [
            "Region",
            "Method",
            "Quantity",
            "Support",
            "Minimum (%)",
            "Median (%)",
            "Maximum (%)",
            "Valid / subsets",
        ],
        rows,
    )


def _response_plot(comparisons, assets):
    if assets is None:
        return ""
    valid = [
        r
        for r in comparisons
        if r["tiles"]
        and np.isfinite(r["equivalent_nm"])
        and np.isfinite(r["raw_equivalent_nm"])
    ]
    if not valid:
        return ""
    groups = defaultdict(list)
    for r in valid:
        groups[r["region"]].append(r)
    charts = []
    for rid, group in groups.items():
        fig = Figure(
            figsize=(7, 1.1 + 0.42 * len(group)), facecolor=PANEL, layout="constrained"
        )
        ax = fig.subplots()
        _axis_style(ax)
        y = np.arange(len(group))
        raw = [r["raw_equivalent_nm"] for r in group]
        modeled = [r["equivalent_nm"] for r in group]
        ax.hlines(
            y,
            np.minimum(raw, modeled),
            np.maximum(raw, modeled),
            color="#72808b",
            linewidth=1.5,
        )
        ax.scatter(raw, y, marker="o", color="#e8bb78", label="Paired raw", zorder=3)
        ax.scatter(modeled, y, marker="D", color="#78d5d5", label="Method", zorder=3)
        ax.set_yticks(y, [_method(r["method"]) for r in group])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel("Modeled 3σ (nm)")
        ax.grid(axis="x", color="#434b55", linewidth=0.5)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1),
            ncols=2,
            frameon=False,
            labelcolor=INK,
            fontsize=9,
        )
        title = f"Region {rid} · modeled method and paired raw response"
        charts.append(
            _fold(
                title,
                _plot_image(
                    fig,
                    assets,
                    f"response-comparison-{rid}.png",
                    title,
                    "Each pair uses the same valid tiles. The conversion to nm comes from the simulated size sweep.",
                ),
            )
        )
    return "".join(charts)


def _precision(summary, splits, spatial, stress, comparisons, assets):
    headline = [
        r
        for r in summary
        if r["level"] in ("Native page", "Pilot tile") and r["metric"] in PRIMARY
    ]
    shared = [r for r in summary if r["level"].startswith("Shared sites")]
    scores = [
        _card(
            "Repeatability by method",
            _scores(headline),
            "3σ is three times the sample standard deviation across acquisitions. Each complete tile contributes equally. "
            "Relative 3σ is expressed as a percentage of the absolute mean. The paired raw column uses the same valid tiles.",
        )
    ]
    if shared:
        scores.append(
            _card(
                "Same-hole CD comparison",
                _shared_comparison(shared),
                "Within each region, the methods use the same holes in every acquisition. "
                "Shorter bars indicate less repeat-to-repeat scatter on the matched tiles.",
            )
        )
    scores.append(
        _card(
            "Precision by location",
            _spatial_tables(spatial, assets),
            "Each cell shows absolute 3σ in the stated unit, followed by relative 3σ (%). "
            "Tile coordinates start at zero in the common reference field.",
        )
    )
    scores.append(
        _card(
            "Spatial support and acquisition subsets",
            _fold("All spatial scales, axes and placement results", _scores(summary))
            + _fold("Three-acquisition subset ranges", _triplets(splits)),
            "Subset ranges use locations measured in every acquisition. The triplets overlap.",
        )
    )
    checks = _table(
        [
            "Region",
            "Method",
            "Paired tiles",
            "3σ (%)",
            "Raw (%)",
            "Modeled 3σ (nm)",
            "Modeled raw (nm)",
            "Finding",
        ],
        (
            [
                r["region"],
                _method(r["method"]),
                r["tiles"],
                r["percent"],
                r["raw_percent"],
                r["equivalent_nm"],
                r["raw_equivalent_nm"],
                r["finding"],
            ]
            for r in comparisons
        ),
        "No comparison qualified for sensitivity adjustment.",
    )
    checks += _response_plot(comparisons, assets)
    grouped = defaultdict(list)
    for r in stress:
        grouped[r.get("scenario", "Unspecified")].append(r)
    for scenario, group in grouped.items():
        checks += _fold(
            str(scenario).replace("_", " ").capitalize(),
            _table(
                [
                    "Region",
                    "Shape",
                    "Nominal CD (nm)",
                    "Discovered",
                    "Method",
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
                        _method(r.get("method", "")),
                        r.get("unit"),
                        r.get("slope"),
                        r.get("mean_error"),
                        r.get("rmse"),
                        r.get("count"),
                        r.get("status"),
                    ]
                    for r in group
                ),
            ),
        )
    scores.append(
        _card(
            "Response to simulated changes",
            checks,
            "The modeled comparison converts relative scatter to nm using the simulated fractional "
            "response per nm. Its interpretation depends on the assumed optical and detector response.",
        )
    )
    return "".join(scores)


def _profiles(frames, assets):
    if assets is None or not frames:
        return ""
    panel_color, label_color = "#1c2025", "#ccd2d8"
    fig = Figure(
        figsize=(10, 1.05 * len(frames) + 1.05),
        layout="constrained",
        facecolor=panel_color,
    )
    axes = fig.subplots(len(frames), 2, squeeze=False, sharey=True, sharex="col")
    axes[0, 0].set_title("Row means", loc="left", fontsize=11, color=label_color)
    axes[0, 1].set_title("Column means", loc="left", fontsize=11, color=label_color)
    for i, (frame, pair) in enumerate(zip(frames, axes)):
        pair[0].set_ylabel(f"{i + 1:02d}  ·  DN", fontsize=9, color=label_color)
        for ax, field, color in zip(
            pair, ("row_mean_dn", "column_mean_dn"), ("#a1c5ee", "#c0d7a7")
        ):
            ax.plot(frame[field], color=color, linewidth=0.9)
            ax.set_facecolor(panel_color)
            ax.grid(axis="y", color="#424b54", linewidth=0.5)
            ax.spines[:].set_visible(False)
            ax.tick_params(labelsize=8, length=0, colors=label_color)
            ax.margins(x=0)
    axes[-1, 0].set_xlabel("Native row (px)", color=label_color)
    axes[-1, 1].set_xlabel("Native column (px)", color=label_color)
    assets.mkdir(exist_ok=True)
    name = "intensity-profiles.png"
    fig.savefig(assets / name, dpi=110)
    fig.clear()
    src = escape(f"{assets.name}/{name}", quote=True)
    return _fold(
        "Row and column intensity profiles",
        f'<figure><img loading="lazy" src="{src}" style="max-width:100%;height:auto" '
        'alt="Per-acquisition profiles of mean signal by native row and column, with a shared DN axis.">'
        "<figcaption>Every panel uses the same intensity range. Exact profiles are in run.json.</figcaption></figure>",
    )


def _shift_plot(study, assets):
    if assets is None or not study.registration:
        return ""
    xy = (
        np.asarray([r["shift_xy"] for r in study.registration], dtype=float)
        * study.pixel_nm
    )
    xy[[bool(r.get("ambiguous")) for r in study.registration]] = np.nan
    if not np.any(np.isfinite(xy)):
        return ""
    fig = Figure(figsize=(7, 2.3), facecolor=PANEL, layout="constrained")
    ax = fig.subplots()
    _axis_style(ax)
    repeats = np.arange(1, len(xy) + 1)
    for i, (color, name, marker) in enumerate(
        (("#e8bb78", "X shift", "o"), ("#78d5d5", "Y shift", "s"))
    ):
        ax.plot(
            repeats,
            xy[:, i],
            marker=marker,
            color=color,
            linewidth=1.25,
            markersize=4,
            label=name,
        )
    ax.set(xticks=repeats, xlabel="Acquisition", ylabel="Shift (nm)")
    ax.axhline(0, color="#64717b", linewidth=0.75, zorder=0)
    ax.grid(axis="y", linewidth=0.4, color="#424b54")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=2,
        frameon=False,
        labelcolor=INK,
        fontsize=9,
    )
    return _plot_image(
        fig,
        assets,
        "global-shifts.png",
        "Global X and Y shift by acquisition",
        "Gaps indicate ambiguous registration.",
    )


def _registration(study, summary, assets):
    global_rows = [
        [
            i + 1,
            reg["shift_xy"][0] * study.pixel_nm,
            reg["shift_xy"][1] * study.pixel_nm,
            reg["score"],
            reg["reason"],
        ]
        for i, reg in enumerate(study.registration)
    ]
    local = defaultdict(list)
    for row in study.holes:
        x = row.get("residual_x_nm", np.nan)
        y = row.get("residual_y_nm", np.nan)
        if np.isfinite(x) and np.isfinite(y) and not row.get("anchor", False):
            local[row["repeat"], row["method"], row["region"]].append((x, y))
    residuals = []
    for (repeat, method, rid), vectors in sorted(local.items()):
        dx, dy = np.asarray(vectors).T
        dist = np.hypot(dx, dy)
        residuals.append(
            [
                repeat + 1,
                rid,
                _method(method),
                len(vectors),
                dx.mean(),
                dy.mean(),
                np.median(dist),
                np.percentile(dist, 95),
            ]
        )
    return (
        _shift_plot(study, assets)
        + _table(
            [
                "Acquisition",
                "Global X (nm)",
                "Global Y (nm)",
                "Score",
                "Registration status",
            ],
            global_rows,
        )
        + _fold(
            "Global shift precision",
            _scores(r for r in summary if r["level"] == "Acquisition shift"),
        )
        + _fold(
            "Held-out placement residuals",
            _table(
                [
                    "Acquisition",
                    "Region",
                    "Method",
                    "Held-out holes",
                    "Mean X (nm)",
                    "Mean Y (nm)",
                    "Median distance (nm)",
                    "95th percentile (nm)",
                ],
                residuals,
            ),
        )
    )


def details_views(
    study,
    info,
    summary,
    splits,
    spatial,
    stress,
    comparisons,
    performance,
    synthetic,
    *,
    assets=None,
):
    """Return HTML fragments for the Precision and Run details tabs."""
    precision_view = _precision(summary, splits, spatial, stress, comparisons, assets)
    meta = info[0] if info else {}
    source = "Synthetic" if synthetic else "Reference"
    inputs = _table(
        [
            "Acquisition",
            "File",
            "Mean (DN)",
            "Common field (DN)",
            "Min (DN)",
            "Max (DN)",
            "At 0 (%)",
            "At 65,535 (%)",
            "Distinct levels",
        ],
        (
            [
                i + 1,
                row["file"],
                f["mean_dn"],
                f["common_dn"],
                f["min_dn"],
                f["max_dn"],
                100 * f["zero_fraction"],
                100 * f["full_scale_fraction"],
                f["occupied_levels"],
            ]
            for i, (row, f) in enumerate(zip(info, study.frames))
        ),
    )
    notes = "".join(f"<li>{escape(note)}</li>" for note in study.notes)
    parts = [
        _card(
            "Acquisition signal",
            inputs + _profiles(study.frames, assets),
            f"{source} input · {len(info)} acquisitions · {meta.get('width', '—')} × "
            f"{meta.get('height', '—')} px · {meta.get('bit_depth', 16)}-bit grayscale · "
            f"{study.pixel_nm:g} nm/px at mask scale.",
        )
    ]
    if notes:
        parts.append(_card("Run notes", f"<ul>{notes}</ul>"))
    patterns = _table(
        [
            "Region",
            "Shape",
            "Best hypothesis",
            "Orientation (°)",
            "Lattice",
            "Pitch (px)",
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
        "No periodic pattern was identified.",
    )
    counts = Counter(
        (r["method"], r.get("flag", "Excluded")) for r in study.holes if not r["valid"]
    )
    counts.update(
        (r["method"], r["status"]) for r in study.tiles if r["status"] != "ok"
    )
    reasons = _fold(
        "Excluded measurements",
        _table(
            ["Method", "Reason", "Observations"],
            ([_method(m), flag, count] for (m, flag), count in counts.most_common()),
            "No excluded hole or tile observations.",
        ),
    )
    x0, y0, x1, y1 = study.bounds
    parts.append(
        _card(
            "Discovered patterns",
            patterns + reasons,
            f"The common field spans {(x1 - x0) * study.pixel_nm / 1000:.3f} × "
            f"{(y1 - y0) * study.pixel_nm / 1000:.3f} µm. Each complete pilot tile is "
            f"{study.tile_um:g} × {study.tile_um:g} µm.",
        )
    )
    parts.append(
        _card(
            "Registration and placement",
            _registration(study, summary, assets),
            "Shifts are relative to acquisition 01, with positive X to the right and positive Y down. "
            "Local residuals use centers held out of the translation fit. Lattice phase measures "
            "tile-level displacement. Periodic arrays can admit shifts differing by a whole pitch.",
        )
    )
    stages = _table(
        ["Stage", "Elapsed (s)", "Full board pages/s", "1,500-page projection (s)"],
        (
            [
                _method(k).replace("_", " "),
                v["seconds"],
                v["pages_per_s"],
                v["projected_seconds"],
            ]
            for k, v in performance["stages"].items()
        ),
    )
    disk = performance.get("disk")
    if disk:
        stages += f"<p>Sequential read of {escape(disk['file'])} reached {_text(disk['mib_s'])} MiB/s in {_text(disk['seconds'])} s.</p>"
    parts.append(
        _card(
            "Workstation throughput",
            stages,
            f"Input read and decode took {_text(performance['input_seconds'])} s. "
            "Projections use 2,048 × 1,376 pixels per board page and the measured pattern density.",
        )
    )
    exports = list(EXPORTS)
    if (
        synthetic
        and assets is not None
        and (assets.parent / "synthetic-truth.npz").is_file()
    ):
        exports.append(("synthetic-truth.npz", "Simulation ground truth"))
    links = "".join(
        f'<li><a download href="{name}">{label}</a> <small>{name}</small></li>'
        for name, label in exports
    )
    parts.append(
        _card(
            "Data exports",
            f'<ul class="export-list">{links}</ul>',
            "Acquisition numbers in the viewer begin at 1. CSV acquisition indices begin at 0.",
        )
    )
    return precision_view, "".join(parts)
