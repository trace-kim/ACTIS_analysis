"""Lossless reference input and portable result serialization."""

import csv
import gzip
import hashlib
import json
import re
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def read_reference(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Reference folder does not exist: {folder}")
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: [
            int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)
        ],
    )
    if len(paths) != 8:
        raise ValueError(
            f"Expected exactly eight reference PNGs in {folder}, found {len(paths)}."
        )
    images, info = [], []
    for path in paths:
        data = path.read_bytes()
        header = data[:29]
        if (
            len(header) < 29
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
        ):
            raise ValueError(f"{path.name}: invalid PNG header")
        w, h, depth, color = struct.unpack(">IIBB", header[16:26])
        if (depth, color) != (16, 0):
            raise ValueError(
                f"{path.name}: expected native 16-bit grayscale, found depth {depth}, color type {color}"
            )
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(BytesIO(data)) as im:
            pixels = np.asarray(im).astype(np.uint16, copy=False)
            synthetic = im.info.get("actis-data-origin") == "synthetic"
            gamma, srgb = im.info.get("gamma"), im.info.get("srgb")
        if pixels.shape != (h, w) or (images and pixels.shape != images[0].shape):
            raise ValueError(f"{path.name}: image dimensions differ across repeats")
        images.append(pixels)
        info.append(
            {
                "file": path.name,
                "sha256": digest,
                "width": w,
                "height": h,
                "bit_depth": depth,
                "synthetic": synthetic,
                "png_gamma": gamma,
                "png_srgb_intent": srgb,
            }
        )
    return images, info


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return native(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(native(obj), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_table(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            data = {}
            for k in fields:
                v = native(row.get(k))
                data[k] = (
                    json.dumps(v, separators=(",", ":"))
                    if isinstance(v, (list, dict))
                    else v
                )
            writer.writerow(data)
