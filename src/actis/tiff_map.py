"""Write the per-tile map of a finished TIFF benchmark as one self-contained page."""

import base64
import json
from pathlib import Path

import numpy as np

# Plain names for the page: (title, what the number is).
ABOUT = {
    "raw": (
        "Raw brightness",
        "Mean brightness of the tile. Measures no size.",
    ),
    "contour": (
        "Contour",
        "Each opening outlined half way up from its dark surround. Size from the outline's area.",
    ),
    "model": (
        "Aperture model",
        "A blurred opening fitted to each hole's pixels. Uses the blur found on the page.",
    ),
    "differential": (
        "Differential fit",
        "One correction step of the aperture model, starting from the contour.",
    ),
    "flux": (
        "Integrated intensity",
        "Light gathered by each opening above its surround. A brightness, not a length.",
    ),
    "harmonic": (
        "Lattice harmonics",
        "One size per tile from the strengths of the array's Fourier orders.",
    ),
    "pooled": (
        "Pooled moments",
        "One size per tile from the summed soft area of its openings.",
    ),
}


def _packed(array, dtype):
    data = np.ascontiguousarray(array, dtype=dtype).tobytes()
    return base64.b64encode(data).decode("ascii")


def write_map(path, result, arrays):
    """Embed the run summary and every tile's value in the map template."""
    pages, rows, columns = arrays["reference"].shape
    methods = []
    for name, row in result["methods"].items():
        title, about = ABOUT[name]
        openings = arrays.get(f"{name}_openings")
        methods.append(
            {
                "name": name,
                "title": title,
                "about": about,
                "unit": row["unit"].replace("^2", "²"),
                "seconds": row["seconds"],
                "per_opening": openings is not None,
                "values": _packed(arrays[name], "<f4"),
                "openings": None if openings is None else _packed(openings, "<i4"),
            }
        )
    blur = result["blur"]
    if blur:
        # One curve per axis: the mean over the references fitted on the page.
        sigma = arrays["blur_sigma_px"]
        fitted = np.isfinite(sigma[..., 0])
        mean = (
            np.where(fitted[..., None], sigma, 0).sum(axis=1)
            / np.maximum(fitted.sum(axis=1), 1)[:, None]
        )
        mean[~fitted.any(axis=1)] = np.nan
        blur = {
            **blur,
            "sigma": _packed(mean, "<f4"),
            "unusable": (fitted & ~arrays["blur_usable"]).any(axis=1).tolist(),
        }
    stages = result["stage_seconds"]
    lattices = [
        {"shape": shape, "pitch": pitch, "tiles": int((arrays["reference"] == k).sum())}
        for k, (shape, pitch) in enumerate(
            zip(result["reference_shapes"], result["reference_pitch_px"])
        )
    ]
    payload = {
        "file": Path(result["input"]).name,
        "gpu": result["gpu"],
        "pages": pages,
        "rows": rows,
        "columns": columns,
        "page_numbers": (arrays["page_index"] + 1).tolist(),
        "tile_size": result["tile_size"],
        "tested_tiles": result["tested_tiles"],
        "accepted_tiles": result["accepted_tiles"],
        "measured_tiles": result["measured_tiles"],
        "total_seconds": result["total_seconds"],
        "shared_fit_seconds": stages.get("contour", 0) + stages.get("fit_setup", 0),
        "lattices": lattices,
        "reference": _packed(arrays["reference"], "<i1"),
        "methods": methods,
        "blur": blur,
    }
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    Path(path).write_text(template.replace("/*DATA*/null", data), encoding="utf-8")
