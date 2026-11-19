"""Write native image planes and independently loadable measurement layers."""

import base64
import json
import math
from array import array
from collections import defaultdict

import numpy as np
from PIL import Image, PngImagePlugin

from .analysis import METHODS, NAMES


def _number(value, scale=1):
    if value is None:
        return None
    x = float(value) * scale
    return x if math.isfinite(x) else None


def _display_transfer(images):
    """One DN lookup table for the run, with headroom for sparse hot/dark pixels."""
    histogram = np.zeros(65536, dtype=np.int64)
    for image in images:
        histogram += np.bincount(image.ravel(), minlength=len(histogram))
    occupied = np.flatnonzero(histogram)
    low, high = int(occupied[0]), int(occupied[-1])
    levels = np.arange(len(histogram))
    if low == high:
        return np.full(len(histogram), 128, np.uint8), [low, high], [[low, 128]]
    cumulative = np.cumsum(histogram)
    p_low, p_high = np.searchsorted(
        cumulative, 1 + (cumulative[-1] - 1) * np.array([0.001, 0.999])
    )
    # Keep both tails monotone. Even a true maximum is a light gray on screen.
    knots = [[low, 6]]
    if p_high > p_low:
        if p_low > low:
            knots.append([int(p_low), 18])
        if p_high < high:
            knots.append([int(p_high), 230])
    knots.append([high, 249])
    dn, gray = np.asarray(knots).T
    lut = np.rint(np.interp(levels, dn, gray)).astype(np.uint8)
    return lut, [low, high], knots


def _pack(data):
    return base64.b64encode(data.tobytes()).decode("ascii")


def _outlines(measurements, rows):
    """Encode centers' relative vertex offsets, bounding display error to half a step."""
    lengths = np.zeros(len(rows), dtype=np.uint32)
    polygons, centers = [], []
    for i, (measurement, row) in enumerate(zip(measurements, rows)):
        points = measurement.get("contour_xy")
        if row[3] is None or points is None:
            continue
        vertices = np.asarray(points, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or len(vertices) < 3
            or not np.isfinite(vertices).all()
        ):
            continue
        lengths[i] = len(vertices)
        polygons.append(vertices)
        centers.append(row[1:3])
    offsets = np.empty(len(rows) + 1, dtype="<u4")
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    vertices = np.empty((0, 2), dtype="<i2")
    scale = 256
    if polygons:
        relative = np.concatenate(polygons)
        relative -= np.repeat(centers, lengths[lengths != 0], axis=0)
        maximum = np.max(np.abs(relative))
        if maximum * scale > 32767:
            scale = 32767.0 / maximum
        vertices = np.rint(relative * scale).astype("<i2")
    return [_pack(offsets), _pack(vertices)], float(scale)


def _holes(measurements, site_index, metric, pixel_nm, shift):
    rows = []
    for m in measurements:
        key = (m["region"], m["site"])
        value = (
            _number(m.get("flux_dn_px2"))
            if metric == "Integrated signal"
            else _number(m.get("cd_px"), pixel_nm)
        )
        x, y = _number(m.get("x_px")), _number(m.get("y_px"))
        if not m["valid"] or x is None or y is None:
            value = None
        if value is None:
            x, y = float(m["ref_x_px"] + shift[0]), float(m["ref_y_px"] + shift[1])
        rows.append(
            [
                site_index[key],
                x,
                y,
                value,
                str(m["flag"]),
                _number(m.get("width_px"), pixel_nm),
                _number(m.get("height_px"), pixel_nm),
                _number(m.get("residual_x_nm")),
                _number(m.get("residual_y_nm")),
            ]
        )
    outlines, scale = _outlines(measurements, rows)
    return {"rows": rows, "outlines": outlines, "outlineScale": scale}


def _tiles(measurements):
    return {
        "rows": [
            [
                int(t["region"]),
                int(t["i"]),
                int(t["j"]),
                float(t["x_px"]),
                float(t["y_px"]),
                _number(t["value"]) if t["status"] == "ok" else None,
                str(t["status"]),
            ]
            for t in measurements
        ]
    }


def _limits(values):
    if not len(values):
        return [0.0, 1.0]
    lo, hi = np.percentile(values, [2, 98])
    if lo == hi:
        pad = max(0.5, abs(lo) * 1e-4)
        lo, hi = lo - pad, hi + pad
    return [float(lo), float(hi)]


def _thumbnail_window(study, shape):
    h, w = shape
    center = np.array([(w - 1) / 2, (h - 1) / 2])
    width = min(w, 4 * h / 3)
    if study.regions:
        region = max(study.regions, key=lambda r: len(r.centers))
        if len(region.centers):
            centers = np.asarray(region.centers)
            target = np.median(centers, axis=0)
            center = centers[np.argmin(np.sum((centers - target) ** 2, axis=1))]
            width = min(width, 10 * region.pitch)
    return center, max(1, round(width)), max(1, round(width * 3 / 4))


def _image_levels(out, display, repeat):
    """Low-pass filter fine arrays before displaying them below native resolution."""
    levels = []
    factor = 1
    while max(display.size) / factor > 160:
        factor *= 2
        size = tuple(max(1, round(dim / factor)) for dim in display.size)
        path = f"images/level-{repeat}-{factor}.png"
        display.resize(size, Image.Resampling.LANCZOS).save(out / path)
        levels.append({"scale": 1 / factor, "image": path})
    return levels


def write_viewer_data(out, images, info, study):
    """Return the manifest after writing images and file://-compatible JS layers.

    Site indices refer to the shared [region, site, refX, refY] list. Hole rows
    hold native coordinates. Tile rows and registration bounds use reference
    coordinates. Exact native-pixel measurements remain in the analysis CSVs.
    """
    image_dir = out / "images"
    layer_dir = out / "data"
    image_dir.mkdir(exist_ok=True)
    layer_dir.mkdir(exist_ok=True)
    lut, display_range, display_knots = _display_transfer(images)
    site_index, sites = {}, []
    grouped = defaultdict(list)
    calibrated_flux = False
    for m in study.holes:
        key = (m["region"], m["site"])
        if key not in site_index:
            site_index[key] = len(sites)
            sites.append(
                [int(key[0]), int(key[1]), float(m["ref_x_px"]), float(m["ref_y_px"])]
            )
        grouped[m["repeat"], m["method"]].append(m)
        if m["method"] == "flux" and not calibrated_flux and m["valid"]:
            calibrated_flux = _number(m.get("cd_px")) is not None
    for t in study.tiles:
        grouped[t["repeat"], t["method"]].append(t)

    methods = {}
    for method, name in NAMES.items():
        unit, metric = "nm", "CD"
        if method == "raw":
            unit, metric = "DN", "Brightness"
        elif method == "flux" and not calibrated_flux:
            unit, metric = "DN·px²", "Integrated signal"
        methods[method] = {
            "name": name,
            "support": "hole" if method in METHODS else "tile",
            "unit": unit,
            "metric": metric,
        }

    values = {method: array("d") for method in methods}
    frames = []
    center, crop_w, crop_h = _thumbnail_window(study, images[0].shape)
    for repeat, (image, meta, reg, metrics) in enumerate(
        zip(images, info, study.registration, study.frames)
    ):
        preview = f"images/native-{repeat}.png"
        source = f"images/source-{repeat}.png"
        thumb = f"images/thumb-{repeat}.png"
        display = Image.fromarray(lut[image])
        display.save(out / preview)
        levels = _image_levels(out, display, repeat)
        pnginfo = PngImagePlugin.PngInfo()
        if meta.get("synthetic"):
            pnginfo.add_text("actis-data-origin", "synthetic")
        Image.fromarray(image).save(out / source, pnginfo=pnginfo)
        shift = list(map(float, reg["shift_xy"]))
        cx, cy = center + shift
        left = int(np.clip(round(cx - crop_w / 2), 0, display.width - crop_w))
        top = int(np.clip(round(cy - crop_h / 2), 0, display.height - crop_h))
        display.crop((left, top, left + crop_w, top + crop_h)).resize(
            (96, 72), Image.Resampling.LANCZOS
        ).save(out / thumb)
        layers = {}
        qualified = {}
        for method in methods:
            measurements = grouped.pop((repeat, method), [])
            if method in METHODS:
                layer = _holes(
                    measurements,
                    site_index,
                    methods[method]["metric"],
                    study.pixel_nm,
                    shift,
                )
                value_index = 3
            else:
                layer = _tiles(measurements)
                value_index = 5
            before = len(values[method])
            values[method].extend(
                row[value_index]
                for row in layer["rows"]
                if row[value_index] is not None
            )
            layer["valid"] = len(values[method]) - before
            layer["total"] = len(layer["rows"])
            qualified[method] = layer["valid"]
            layers[method] = f"data/{repeat}-{method}.js"
            with (out / layers[method]).open("w", encoding="utf-8") as f:
                f.write(f"window.ACTIS_LAYER({repeat},{json.dumps(method)},")
                json.dump(
                    layer, f, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                f.write(");\n")
        frames.append(
            {
                "file": meta["file"],
                "image": preview,
                "levels": levels,
                "source": source,
                "thumb": thumb,
                "shift": shift,
                "mean": float(metrics["mean_dn"]),
                "min": int(metrics["min_dn"]),
                "max": int(metrics["max_dn"]),
                "registration": reg["reason"],
                "ambiguous": bool(reg["ambiguous"]),
                "periodicAmbiguity": bool(reg.get("periodic_ambiguity", False)),
                "layers": layers,
                "qualified": qualified,
            }
        )
    cd_values = np.concatenate(
        [np.asarray(values[k]) for k in methods if methods[k]["metric"] == "CD"]
    )
    shared_cd = _limits(cd_values)
    for method, spec in methods.items():
        spec["ownLimits"] = _limits(values[method])
        spec["limits"] = shared_cd if spec["metric"] == "CD" else spec["ownLimits"]

    h, w = images[0].shape
    reference = frames[0]
    overview = (
        reference["levels"][-1]["image"] if reference["levels"] else reference["image"]
    )
    return {
        "width": w,
        "height": h,
        "pixelNm": float(study.pixel_nm),
        "tilePx": float(study.tile_um * 1000 / study.pixel_nm),
        "bounds": list(map(float, study.bounds)),
        "sites": sites,
        "regions": [
            {"id": int(r.region_id), "pitch": float(r.pitch), "shape": r.shape}
            for r in study.regions
        ],
        "methods": methods,
        "frames": frames,
        "overview": overview,
        "displayRange": display_range,
        "displayKnots": display_knots,
    }
