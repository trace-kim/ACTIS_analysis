"""Generate an offline, native-pixel walkthrough of one lattice harmonic measurement.

From the checkout, run python scripts/inspect_harmonics.py IMAGE [-o REPORT.html].
The default 128-pixel tiles share the native image's top-left pixel boundary.
Supply --pixel-nm only when the image's pixel pitch is known.
"""

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from actis.discovery import discover
from actis.files import native
from actis.harmonics import _tile_orders, iter_harmonic_tiles

CHUNK = 256
HERE = Path(__file__).resolve().parent


def json_text(value):
    # A file name can contain a closing script tag.
    return json.dumps(native(value), separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )


def read_image(path, page):
    with path.open("rb") as stream:
        header = stream.read(26)
    if header.startswith((b"II", b"MM")):
        try:
            import tifffile
        except ImportError:
            pass  # Standard BlackIsZero TIFFs also work with the base install.
        else:
            with tifffile.TiffFile(path) as stack:
                try:
                    frame = stack.pages[page]
                except IndexError:
                    raise ValueError(f"Image page {page} does not exist.") from None
                if (
                    frame.photometric not in (0, 1)
                    or len(frame.shape) != 2
                    or frame.dtype.kind != "u"
                    or frame.bitspersample not in (8, 16)
                ):
                    raise ValueError(
                        "Expected an 8- or 16-bit unsigned grayscale TIFF page."
                    )
                data = frame.asarray().astype(f"<u{frame.dtype.itemsize}", copy=False)
                inverted = frame.photometric == 0
                if inverted:
                    data = np.iinfo(data.dtype).max - data
                return data, None, inverted
    with Image.open(path) as source:
        if source.format not in ("TIFF", "PNG"):
            raise ValueError("Choose an 8- or 16-bit grayscale PNG or TIFF.")
        try:
            source.seek(page)
        except EOFError:
            raise ValueError(f"Image page {page} does not exist.") from None
        if source.format == "TIFF":
            if source.tag_v2.get(262) == 0:
                raise ValueError(
                    "Install tifffile to decode native WhiteIsZero TIFF samples."
                )
            if source.tag_v2.get(258) not in ((8,), (16,)):
                raise ValueError("Expected native 8- or 16-bit grayscale samples.")
        elif len(header) < 26 or header[24] not in (8, 16) or header[25] != 0:
            raise ValueError("Expected a native 8- or 16-bit grayscale PNG.")
        data = np.asarray(source)
        if source.format == "PNG" and source.mode == "I" and header[24] == 16:
            # Older supported Pillow versions unpack 16-bit grayscale PNG into int32.
            data = data.astype("<u2")
            return data, source.info.get("actis-data-origin"), False
        if (
            data.ndim != 2
            or data.dtype.kind != "u"
            or data.dtype.itemsize not in (1, 2)
        ):
            raise ValueError(
                "The image must contain single-channel 8- or 16-bit unsigned detector samples."
            )
        if source.mode not in ("L", "I;16", "I;16L", "I;16B", "I;16N"):
            raise ValueError(
                f"Expected grayscale detector samples, found image mode {source.mode}."
            )
        return (
            data.astype(f"<u{data.itemsize}", copy=False),
            source.info.get("actis-data-origin"),
            False,
        )


def pyramid(out, name, pixels):
    """Embed independently decodable chunks, keeping the native level lossless."""
    levels = []
    while True:
        h, w = pixels.shape
        level = len(levels)
        levels.append([w, h])
        for y in range(0, h, CHUNK):
            for x in range(0, w, CHUNK):
                chunk = pixels[y : y + CHUNK, x : x + CHUNK]
                data = base64.b64encode(zlib.compress(chunk.tobytes(), 4)).decode(
                    "ascii"
                )
                key = f"{name}-{level}-{x // CHUNK}-{y // CHUNK}"
                out.write(
                    f'<script type="application/octet-stream" id="{key}">{data}</script>\n'
                )
        if max(w, h) <= CHUNK:
            return levels
        # Average available samples at the odd right/bottom edges as well.
        smaller = Image.fromarray(pixels.astype(np.int32)).reduce(2)
        pixels = np.asarray(smaller).astype(pixels.dtype)


def observed_orders(patch, freq, anchor_local, *, trace):
    amplitude, noise = _tile_orders(patch, freq, anchor_local, trace=trace)
    weighted = trace.pop("windowed")
    # A native-bin magnitude map supplies visual context for the exact off-bin
    # coefficients recorded by the sampler. The fit never reads this picture.
    magnitude = np.abs(fft.fftshift(fft.fft2(weighted)))
    log_magnitude = np.log1p(magnitude / noise)
    peak = float(log_magnitude.max())
    grayscale = np.rint(log_magnitude * (255 / max(peak, 1e-15))).astype("u1")
    buf = io.BytesIO()
    Image.fromarray(grayscale).save(buf, format="PNG")
    trace.update(
        spectrum=base64.b64encode(buf.getvalue()).decode("ascii"),
        spectrum_peak=float(magnitude.max()),
        weighted_range=[float(weighted.min()), float(weighted.max())],
    )
    return amplitude, noise


def write_report(path, output, *, tile_px=128, pixel_nm=None, shape="auto", page=0):
    path, output = Path(path), Path(output)
    if path.resolve() == output.resolve() or output.suffix.lower() not in (
        ".html",
        ".htm",
    ):
        raise ValueError("Choose a separate .html output file.")
    if (
        tile_px < 4
        or page < 0
        or (pixel_nm is not None and (not np.isfinite(pixel_nm) or pixel_nm <= 0))
    ):
        raise ValueError(
            "Use a tile side of at least 4 px, a nonnegative page and a positive pixel pitch."
        )
    pixels, origin, inverted = read_image(path, page)
    h, w = pixels.shape
    if min(w, h) < 12:
        raise ValueError("Lattice discovery requires at least 12 × 12 pixels.")
    output.parent.mkdir(parents=True, exist_ok=True)
    discovery = {}
    meta = {
        "file": path.name,
        "page": page,
        "origin": origin,
        "white_is_zero": inverted,
        "width": w,
        "height": h,
        "bits": pixels.itemsize * 8,
        "tile_px": tile_px,
        "pixel_nm": pixel_nm,
        "shape": shape,
        "chunk": CHUNK,
        "layers": {},
        "discovery": discovery,
        "regions": [],
        "tiles": [],
    }
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        temporary = Path(staging) / "report.html"
        with temporary.open("w", encoding="utf-8", newline="\n") as out:
            out.write((HERE / "harmonic_view.html").read_text(encoding="utf-8"))
            stride = max(1, int(np.sqrt(pixels.size / 1_000_000)))
            black, white = np.percentile(pixels[::stride, ::stride], [0.5, 99.5])
            meta["layers"]["raw"] = {
                "levels": pyramid(out, "raw", pixels),
                "black": black,
                "white": white,
            }

            def observe(
                bounds, smooth, low, high, threshold, mask, core, noise, labels
            ):
                discovery.update(noise=noise, components=labels)
                for name, layer in (
                    ("smooth", smooth),
                    ("low", low),
                    ("high", high),
                    ("threshold", threshold),
                    ("mask", mask),
                    ("core", core),
                ):
                    floor, ceil = (
                        (0.0, 1.0)
                        if layer.dtype == bool
                        else (float(layer.min()), float(layer.max()))
                    )
                    scaled = np.rint(
                        (layer.astype("f4") - floor) * (255 / max(ceil - floor, 1e-12))
                    )
                    meta["layers"][name] = {
                        "levels": pyramid(out, name, scaled.astype("u1")),
                        "black": floor,
                        "white": ceil,
                    }

            regions = discover(
                pixels, shape=shape, diagnostics=discovery, observe=observe
            )
            for reg in regions:
                info = {
                    "id": reg.region_id,
                    "bounds": reg.bounds,
                    "pitch": reg.pitch,
                    "basis": reg.basis,
                    "angle": reg.angle,
                    "shape": reg.shape,
                    "best": reg.best_shape,
                    "lattice": reg.lattice,
                    "diagnostics": reg.diagnostics,
                    "n_sites": len(reg.centers),
                }
                # Spatial buckets let a zoomed viewer inspect only nearby sites.
                buckets = {}
                for x, y in reg.centers:
                    key = f"{int(x) // CHUNK},{int(y) // CHUNK}"
                    buckets.setdefault(key, []).append([float(x), float(y)])
                info["sites"] = buckets
                meta["regions"].append(info)
                rows = iter_harmonic_tiles(
                    pixels,
                    reg,
                    tile_px=tile_px,
                    origin_xy=(-0.5, -0.5),
                    sample_orders=observed_orders,
                    capture=True,
                )
                for row in rows:
                    index = len(meta["tiles"])
                    trace = row.pop("trace")
                    out.write(
                        f'<script type="application/json" id="tile-{index}">{json_text(trace)}</script>\n'
                    )
                    meta["tiles"].append(row)
            out.write(
                f'<script type="application/json" id="study">{json_text(meta)}</script>\n<script>\n'
            )
            out.write((HERE / "harmonic_view.js").read_text(encoding="utf-8"))
            out.write("\n</script></body></html>\n")
        os.replace(temporary, output)
    return len(meta["tiles"]), sum(row["status"] == "ok" for row in meta["tiles"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "image", type=Path, help="Single bright-hole grayscale PNG or TIFF"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML (default: IMAGE-stem.harmonics.html)",
    )
    parser.add_argument(
        "--tile-px",
        type=int,
        default=128,
        help="Analysis tile side in native pixels (default 128)",
    )
    parser.add_argument(
        "--pixel-nm", type=float, help="Calibrated nanometers per image pixel"
    )
    parser.add_argument(
        "--shape",
        choices=("auto", "circle", "ellipse", "square"),
        default="auto",
        help="Use a known opening shape or keep the default discovery decision",
    )
    parser.add_argument(
        "--page", type=int, default=0, help="Zero-based page index for a TIFF stack"
    )
    args = parser.parse_args()
    output = args.output or args.image.with_suffix(".harmonics.html")
    try:
        total, ok = write_report(
            args.image,
            output,
            tile_px=args.tile_px,
            pixel_nm=args.pixel_nm,
            shape=args.shape,
            page=args.page,
        )
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{ok}/{total} tiles produced a CD. Open {output.resolve()}")


if __name__ == "__main__":
    main()
