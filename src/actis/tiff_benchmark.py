"""Time the seven ACTIS methods on hole-array tiles in one multi-page TIFF."""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .estimators import METHOD_NAMES, _contours, _flux, _model_methods
from .fit import array_backend
from .harmonics import harmonic_tiles
from .pooled import pooled_tiles
from .tiff_tiles import prepare_reference, select_tiles

METHODS = ("raw", *METHOD_NAMES, "harmonic", "pooled")
FITS = ("model", "differential")


class Run:
    def __init__(self, cp):
        self.stream = cp.cuda.get_current_stream()
        self.seconds = Counter()
        self.valid = Counter()
        self.attempted = Counter()
        self.sums = Counter()

    @contextmanager
    def time(self, stage):
        start = perf_counter()
        yield
        self.stream.synchronize()
        self.seconds[stage] += perf_counter() - start

    def record(self, method, values, attempted):
        self.attempted[method] += attempted
        self.valid[method] += len(values)
        self.sums[method] += sum(values)

    def methods(self):
        rows = {}
        for name in METHODS:
            shared = (
                self.seconds["contour"] + self.seconds["fit_setup"]
                if name in FITS
                else 0
            )
            rows[name] = {
                "seconds": self.seconds[name] + shared,
                "exclusive_seconds": self.seconds[name],
                "attempted": self.attempted[name],
                "valid": self.valid[name],
                "unit": {"raw": "DN", "flux": "DN px^2"}.get(name, "px"),
                "mean": self.sums[name] / self.valid[name]
                if self.valid[name]
                else None,
            }
        return rows


def _read_page(page):
    if (
        len(page.shape) != 2
        or page.dtype.kind != "u"
        or page.dtype.itemsize not in (1, 2)
        or page.photometric not in (0, 1)
        or page.bitspersample not in (8, 16)
    ):
        raise ValueError(
            f"TIFF page {page.index} must be single-channel 8- or 16-bit grayscale"
        )
    data = page.asarray()
    # Convert MINISWHITE so bright holes have high DN throughout the pipeline.
    if page.photometric == 0:
        data = np.iinfo(data.dtype).max - data
    return data


def _raw_means(tiles, matches, cp):
    ids = cp.asarray(matches[:, 0], dtype=cp.int32)
    return tiles[ids].mean(axis=(1, 2), dtype=cp.float64).get()


def _decoded_pages(stack):
    pages = iter(stack.pages)

    def read_next():
        start = perf_counter()
        page = next(pages, None)
        if page is None:
            return None
        data = _read_page(page)
        return data, perf_counter() - start

    # Only the reader thread advances the TIFF handle. At most one subsequent
    # decoded page waits in memory while CUDA processes the current one.
    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read_next)
        while (item := pending.result()) is not None:
            pending = reader.submit(read_next)
            yield item


def _reference(path, index, tifffile):
    if path.suffix.lower() == ".png":
        if index != 0:
            raise ValueError("A reference PNG has only page zero")
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode not in (
                "L",
                "I;16",
                "I;16B",
                "I;16L",
                "I",
            ):
                raise ValueError("Reference PNG must be 8- or 16-bit grayscale")
            return np.asarray(im, dtype=np.uint8 if im.mode == "L" else np.uint16)
    with tifffile.TiffFile(path) as ref:
        try:
            return _read_page(ref.pages[index])
        except IndexError:
            raise ValueError(
                f"Reference page {index} does not exist in {path}"
            ) from None


def _measure(tile, region, template, psf, run, cp):
    side = tile.shape[0]
    args = (region.centers, region.pitch, region.shape)
    with run.time("contour"):
        image = tile.astype(cp.float32, copy=False)
        contours = _contours(
            image, *args, region.angle, threshold_diagnostics=True, xp=cp
        )
    run.record("contour", [r["cd_px"] for r in contours if r["valid"]], len(contours))
    with run.time("flux"):
        flux = _flux(tile, *args, contours=None, bright=None, dark=None, xp=cp)
    run.record("flux", [r["flux_dn_px2"] for r in flux if r["valid"]], len(flux))
    # Count contour seeds and blur profiling once in wall time. Each fit also
    # gets its own inclusive time so a one-method cost can be read from the table.
    fit_seconds = Counter()
    with run.time("fit_setup"):
        fits = _model_methods(
            image,
            *args,
            region.angle,
            contours,
            FITS,
            psf,
            side,
            "cuda",
            fit_seconds=fit_seconds,
        )
    run.seconds["fit_setup"] -= sum(fit_seconds.values())
    run.seconds.update(fit_seconds)
    for name, rows in fits.items():
        run.record(name, [r["cd_px"] for r in rows if r["valid"]], len(rows))
    options = {"tile_px": side, "origin_xy": (-0.5, -0.5)}
    with run.time("harmonic"):
        harmonic = harmonic_tiles(
            tile, region, **options, sample_orders=template.sample_orders
        )
    with run.time("pooled"):
        pooled = pooled_tiles(tile, region, **options, xp=cp)
    for name, rows in (("harmonic", harmonic), ("pooled", pooled)):
        run.record(name, [r["cd_px"] for r in rows if r["status"] == "ok"], len(rows))


def benchmark(
    path,
    *,
    reference=None,
    reference_page=0,
    tile_size=128,
    min_corr=0.80,
    shape="auto",
    psf=None,
    device=0,
    progress=None,
):
    try:
        import tifffile
    except ImportError:
        raise RuntimeError(
            'Install the TIFF/CUDA dependencies with pip install -e ".[gpu]"'
        ) from None
    if tile_size < 16 or tile_size % 2 or not 0 < min_corr <= 1:
        raise ValueError(
            "Use an even tile size of at least 16 and a minimum correlation in (0,1]"
        )
    if reference_page < 0 or device < 0:
        raise ValueError("Device and reference-page indices must be nonnegative")
    if psf is not None:
        if len(psf) != 2 or not all(np.isfinite(s) and s > 0 for s in psf):
            raise ValueError(
                "Both calibrated PSF sigma values must be finite and positive"
            )
        psf = tuple(psf)
    total_start = perf_counter()
    cp = array_backend("cuda")
    if device >= cp.cuda.runtime.getDeviceCount():
        raise ValueError(f"CUDA device {device} is unavailable")
    ref_path = Path(reference or path)
    with cp.cuda.Device(device), tifffile.TiffFile(path) as stack:
        run = Run(cp)
        batch_size = max(1, min(64, 2**20 // tile_size**2))
        with run.time("reference_and_warmup"):
            ref = _reference(ref_path, reference_page, tifffile)
            templates = prepare_reference(ref, tile_size, shape, cp)
            first = stack.pages[0]
            if len(first.shape) != 2 or min(first.shape) < tile_size:
                raise ValueError(
                    "First target page must be grayscale and fit at least one complete tile"
                )
            warm_count = min(
                batch_size,
                (first.shape[0] // tile_size) * (first.shape[1] // tile_size),
            )
            for template in templates:
                tile = template.pixels.astype(ref.dtype)
                batch = cp.repeat(tile[None], warm_count, axis=0)
                matches = select_tiles(batch, [template], min_corr, cp)
                if not len(matches):
                    raise ValueError("Reference tile failed its own correlation check")
                if first.dtype != ref.dtype and first.dtype in (np.uint8, np.uint16):
                    # Exercise the target's dtype-specific kernels before timing.
                    pixels = template.pixels
                    scale = (0.8 * np.iinfo(first.dtype).max) / (
                        pixels.max() - pixels.min()
                    )
                    tile = cp.rint((pixels - pixels.min()) * scale).astype(first.dtype)
                    batch = cp.repeat(tile[None], warm_count, axis=0)
                warm = Run(cp)
                _raw_means(batch, matches, cp)
                _measure(tile, template.geometry(np.zeros(2)), template, psf, warm, cp)
        if progress:
            progress(
                f"Prepared {len(templates)} reference hole pattern(s) on CUDA device {device}"
            )
        axis = cp.arange(tile_size)
        pages = tested = accepted = edge_pixels = 0
        scan_start = perf_counter()
        for data, decode_seconds in _decoded_pages(stack):
            run.seconds["decode"] += decode_seconds
            h, w = data.shape
            ny, nx = h // tile_size, w // tile_size
            if not nx * ny:
                raise ValueError(
                    f"Tile size {tile_size} exceeds the dimensions of page {pages}: {w} x {h}"
                )
            edge_pixels += h * w - nx * ny * tile_size**2
            with run.time("selection"):
                frame = cp.asarray(data)
            for start in range(0, ny * nx, batch_size):
                with run.time("selection"):
                    indices = cp.arange(start, min(start + batch_size, ny * nx))
                    y = (indices // nx)[:, None] * tile_size + axis
                    x = (indices % nx)[:, None] * tile_size + axis
                    tiles = frame[y[:, :, None], x[:, None, :]]
                    matches = select_tiles(tiles, templates, min_corr, cp)
                tested += len(tiles)
                accepted += len(matches)
                if len(matches):
                    with run.time("raw"):
                        means = _raw_means(tiles, matches, cp)
                    run.record("raw", means, len(means))
                for idx, ref_id, dx, dy in matches:
                    template = templates[int(ref_id)]
                    region = template.geometry(np.array([dx, dy]))
                    _measure(tiles[int(idx)], region, template, psf, run, cp)
            pages += 1
            if progress and (pages == 1 or pages % 10 == 0):
                progress(f"Read {pages} pages, selected {accepted}/{tested} tiles")
        scan_seconds = perf_counter() - scan_start
        gpu = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
    methods = run.methods()
    ranked = sorted(
        (m for m in METHODS if methods[m]["valid"]), key=lambda m: methods[m]["seconds"]
    )
    return {
        "input": str(path),
        "reference": str(ref_path),
        "reference_page": reference_page,
        "gpu": gpu,
        "device": device,
        "tile_size": tile_size,
        "min_corr": min_corr,
        "reference_shapes": [t.region.shape for t in templates],
        "psf_sigma_px": psf,
        "pages": pages,
        "tested_tiles": tested,
        "accepted_tiles": accepted,
        "excluded_edge_pixels": edge_pixels,
        "stage_seconds": dict(run.seconds),
        "methods": methods,
        "fastest_with_valid_results": ranked[0] if ranked else None,
        "fastest_valid_cd": next(
            (m for m in ranked if methods[m]["unit"] == "px"), None
        ),
        "scan_seconds": scan_seconds,
        "pages_per_second": pages / scan_seconds,
        "total_seconds": perf_counter() - total_start,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tiff", type=Path, help="Multi-page grayscale TIFF to scan")
    parser.add_argument(
        "--reference", type=Path, help="Holes-only TIFF or PNG (default: input TIFF)"
    )
    parser.add_argument(
        "--reference-page",
        type=int,
        default=0,
        help="Zero-based holes-only reference page",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Even tile side in pixels (default: 128)",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.80,
        help="Minimum correlation in every quadrant",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--shape",
        choices=("auto", "square", "circle", "ellipse"),
        default="auto",
        help="Specify only if the reference hole shape is known",
    )
    parser.add_argument(
        "--psf-sigma-px",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Known detector blur in pixels; skips per-tile blur profiling",
    )
    parser.add_argument("--json", type=Path, help="Also write a new timing JSON file")
    args = parser.parse_args(argv)
    try:
        if args.json and args.json.exists():
            raise ValueError(f"Choose a new JSON output path: {args.json}")
        result = benchmark(
            args.tiff,
            reference=args.reference,
            reference_page=args.reference_page,
            tile_size=args.tile_size,
            min_corr=args.min_corr,
            device=args.device,
            shape=args.shape,
            psf=args.psf_sigma_px,
            progress=lambda s: print(s, flush=True),
        )
        if args.json:
            with args.json.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"TIFF benchmark: {exc}\n")
    print(
        f"{result['gpu']} | {result['pages']} pages | {result['accepted_tiles']}/{result['tested_tiles']} "
        f"tiles selected | {result['excluded_edge_pixels']} edge pixels skipped"
    )
    stages = result["stage_seconds"]
    print(
        f"Reference + warm-up {stages['reference_and_warmup']:.3f} s | "
        f"decode {stages.get('decode', 0):.3f} s | selection {stages.get('selection', 0):.3f} s"
    )
    print("Method          Seconds     Valid / attempted   Mean")
    for method, row in result["methods"].items():
        value = "n/a" if row["mean"] is None else f"{row['mean']:.5g} {row['unit']}"
        print(
            f"{method:13} {row['seconds']:9.4f} {row['valid']:9} / {row['attempted']:<9}   {value}"
        )
    print("Both fit times include shared contour and blur setup.")
    print(
        f"Scan {result['scan_seconds']:.3f} s ({result['pages_per_second']:.2f} pages/s) | "
        f"total {result['total_seconds']:.3f} s"
    )
    if result["fastest_with_valid_results"]:
        print(f"Fastest with valid results: {result['fastest_with_valid_results']}")
        if result["fastest_valid_cd"]:
            print(f"Fastest with valid hole dimensions: {result['fastest_valid_cd']}")
    else:
        print("No valid measurements on the target stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
