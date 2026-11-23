"""Screen every TIFF page and time ACTIS methods on accepted tiles."""

import argparse
import json
import os
import sys
import webbrowser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from itertools import groupby, islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Thread
from time import perf_counter

import numpy as np
from PIL import Image

from . import tiff_sites
from .estimators import METHOD_NAMES, _contours, _flux, _model_methods
from .fit import array_backend
from .pooled import pooled_matched_cds
from .tiff_map import write_map
from .tiff_tiles import prepare_reference, select_tiles

METHODS = ("raw", *METHOD_NAMES, "harmonic", "pooled")
FITS = ("model", "differential")
UNITS = {"raw": "DN", "flux": "DN px^2"}


class Progress:
    """Emit a heartbeat even while the main thread waits for I/O or CUDA."""

    def __init__(self, callback):
        self.callback = callback
        self.location = "Starting TIFF benchmark"
        self.stage = "initializing CUDA"
        self.start = perf_counter()
        self.stop = Event()

    def tell(self):
        if self.callback:
            self.callback(
                f"{perf_counter() - self.start:.0f}s | {self.location} | {self.stage}"
            )

    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(15):
                self.tell()

        self.tell()
        if self.callback:
            self.thread = Thread(target=heartbeat, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.callback:
            self.thread.join()


class TileSample:
    """Uniform reservoir of accepted native tiles across the entire stack."""

    def __init__(self, limit):
        self.limit = limit
        self.rng = np.random.default_rng(0)
        self.seen = 0
        self.entries = []

    def consider(self, page, page_index, start, matches, side):
        sequence = np.arange(self.seen, self.seen + len(matches))
        self.seen += len(matches)
        slots = sequence.copy()
        later = sequence >= self.limit
        slots[later] = self.rng.integers(sequence[later] + 1)
        keep = slots < self.limit
        for slot, (idx, ref_id, dx, dy) in zip(slots[keep], matches[keep]):
            tile_index = start + int(idx)
            row, col = divmod(tile_index, page.shape[1] // side)
            y, x = row * side, col * side
            pixels = page[y : y + side, x : x + side].copy()
            entry = (page_index, tile_index, pixels, int(ref_id), dx, dy)
            if slot == len(self.entries):
                self.entries.append(entry)
            else:
                self.entries[slot] = entry


class Run:
    def __init__(self, cp, progress, methods):
        self.stream = cp.cuda.get_current_stream()
        self.progress = progress
        self.selected = methods
        self.seconds = Counter()
        self.valid = Counter()
        self.attempted = Counter()
        self.sums = Counter()

    @contextmanager
    def time(self, stage):
        self.progress.stage = stage
        start = perf_counter()
        yield
        self.stream.synchronize()
        self.seconds[stage] += perf_counter() - start

    def record(self, method, values, attempted):
        self.attempted[method] += attempted
        self.valid[method] += len(values)
        self.sums[method] += float(np.sum(values, dtype=np.float64))

    def methods(self):
        rows = {}
        for name in self.selected:
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
                "unit": UNITS.get(name, "px"),
                "mean": self.sums[name] / self.valid[name]
                if self.valid[name]
                else None,
            }
        return rows


class TileMaps:
    """Every method's value for every tile of the file, kept in page order."""

    def __init__(self, methods=METHODS):
        self.methods = tuple(methods)
        self.page_index = []
        self.grids = []
        self.values = []
        self.openings = []
        self.reference = []
        self.blur = []

    def add_page(self, page_index, rows, columns):
        """Start an empty page and return its position in the scan."""
        size = rows * columns
        self.page_index.append(page_index)
        self.grids.append((rows, columns))
        self.values.append(
            {m: np.full(size, np.nan, dtype=np.float32) for m in self.methods}
        )
        self.openings.append(
            {
                m: np.zeros(size, dtype=np.int32)
                for m in self.methods
                if m in METHOD_NAMES
            }
        )
        self.reference.append(np.full(size, -1, dtype=np.int8))
        return len(self.grids) - 1

    def arrays(self):
        """Stack the pages as (page, tile row, tile column). Smaller pages are padded."""
        rows = max((g[0] for g in self.grids), default=0)
        columns = max((g[1] for g in self.grids), default=0)

        def stack(parts, fill, dtype):
            out = np.full((len(parts), rows, columns), fill, dtype=dtype)
            for page, (part, (r, c)) in enumerate(zip(parts, self.grids)):
                out[page, :r, :c] = part.reshape(r, c)
            return out

        data = {
            m: stack([page[m] for page in self.values], np.nan, np.float32)
            for m in self.methods
        }
        for m in self.methods:
            if m in METHOD_NAMES:
                data[f"{m}_openings"] = stack(
                    [page[m] for page in self.openings], 0, np.int32
                )
        data["reference"] = stack(self.reference, -1, np.int8)
        data["page_index"] = np.asarray(self.page_index, dtype=np.int64)
        references = 1 + max((entry[1] for entry in self.blur), default=0)
        sigma = np.full((len(self.grids), references, 2), np.nan, dtype=np.float32)
        usable = np.zeros((len(self.grids), references), dtype=bool)
        for position, reference, sigma_x, sigma_y, ok in self.blur:
            sigma[position, reference] = sigma_x, sigma_y
            usable[position, reference] = ok
        data["blur_sigma_px"], data["blur_usable"] = sigma, usable
        return data


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


def _raw_means(tiles, cp):
    return tiles.mean(axis=(1, 2), dtype=cp.float64).get()


def _decoded_pages(stack, max_pages=None, page_indices=None):
    pages = (
        (stack.pages[i] for i in page_indices)
        if page_indices is not None
        else islice(stack.pages, max_pages)
    )

    def read_next():
        start = perf_counter()
        page = next(pages, None)
        if page is None:
            return None
        data = _read_page(page)
        return page.index, data, perf_counter() - start

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


def _measure_native_sites(tile, region, psf, run, cp):
    """The general per-tile estimators. Returns {method: (tile mean, valid openings)}."""
    side = tile.shape[0]
    args = (region.centers, region.pitch, region.shape)
    fit_names = tuple(name for name in FITS if name in run.selected)
    per_tile = {}

    def record(name, rows, field="cd_px"):
        values = [r[field] for r in rows if r["valid"]]
        per_tile[name] = (np.mean(values) if values else np.nan, len(values))
        run.record(name, values, len(rows))

    if "contour" in run.selected or fit_names:
        with run.time("contour"):
            image = tile.astype(cp.float32, copy=False)
            contours = _contours(
                image, *args, region.angle, threshold_diagnostics=True, xp=cp
            )
        if "contour" in run.selected:
            record("contour", contours)
    if "flux" in run.selected:
        with run.time("flux"):
            flux = _flux(tile, *args, contours=None, bright=None, dark=None, xp=cp)
        record("flux", flux, "flux_dn_px2")
    if fit_names:
        # Both fits share the contour seeds and blur profile.
        fit_seconds = Counter()
        with run.time("fit_setup"):
            fits = _model_methods(
                image,
                *args,
                region.angle,
                contours,
                fit_names,
                psf,
                side,
                "cuda",
                fit_seconds=fit_seconds,
            )
        run.seconds["fit_setup"] -= sum(fit_seconds.values())
        run.seconds.update(fit_seconds)
        for name, rows in fits.items():
            record(name, rows)
    return per_tile


def _measure_page(tiles, matches, templates, psf, run, cp):
    """Run the selected methods on the accepted tiles of one page.

    Returns one value per tile for each method, the valid openings per tile for
    the per-opening methods, and the blur the aperture fits used per reference.
    """
    count = len(matches)
    references = matches[:, 1].astype(int)
    values, openings, blurs = {}, {}, []

    def column(name, store=values, fill=np.nan):
        if name not in store:
            store[name] = np.full(count, fill)
        return store[name]

    if "raw" in run.selected:
        with run.time("raw"):
            values["raw"] = _raw_means(tiles, cp)
    per_opening = any(name in run.selected for name in METHOD_NAMES)
    for reference in np.unique(references):
        group = np.flatnonzero(references == reference)
        template = templates[reference]
        subset = tiles if len(group) == count else tiles[cp.asarray(group)]
        shifts = matches[group, 2:]
        if "pooled" in run.selected:
            with run.time("pooled"):
                column("pooled")[group] = pooled_matched_cds(
                    subset, template.region, shifts, cp
                )
        if "harmonic" in run.selected:
            with run.time("harmonic"):
                column("harmonic")[group] = template.harmonic_cds(subset, shifts, cp)
        if not per_opening:
            continue
        if tiff_sites.supported(template.region):
            per_tile, blur = tiff_sites.measure_group(
                subset, template, shifts, psf, run, cp
            )
            if blur is not None:
                blurs.append((int(reference), *blur))
        else:
            rows = [
                _measure_native_sites(tile, template.geometry(shift), psf, run, cp)
                for tile, shift in zip(subset, shifts)
            ]
            per_tile = {
                name: tuple(np.array([row[name][i] for row in rows]) for i in (0, 1))
                for name in rows[0]
            }
        for name, (mean, valid) in per_tile.items():
            column(name)[group] = mean
            column(name, openings, 0)[group] = valid
    for name in ("raw", "pooled", "harmonic"):
        if name in values:
            run.record(name, values[name][np.isfinite(values[name])], count)
    return values, openings, blurs


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
    sample_tiles=None,
    max_pages=None,
    checkpoint=None,
    method=None,
    sample_pages=None,
    maps=None,
):
    """Screen the pages of one TIFF and time the methods on its accepted tiles.

    Pass a TileMaps as `maps` to receive every tile's value for every method.
    """
    try:
        import tifffile
    except ImportError:
        raise RuntimeError(
            'Install the TIFF/CUDA dependencies with pip install -e ".[gpu]"'
        ) from None
    if method is not None and method not in METHODS:
        raise ValueError(f"Unknown benchmark method: {method}")
    selected = (method,) if method else METHODS
    if tile_size < 16 or tile_size % 2 or not 0 < min_corr <= 1:
        raise ValueError(
            "Use an even tile size of at least 16 and a minimum correlation in (0,1]"
        )
    if reference_page < 0 or device < 0:
        raise ValueError("Device and reference-page indices must be nonnegative")
    if sample_tiles is not None and sample_tiles < 1:
        raise ValueError(
            "sample_tiles must be positive, or None for exhaustive measurement"
        )
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    if sample_pages is not None and (sample_pages < 1 or max_pages is not None):
        raise ValueError("Use a positive sample_pages or max_pages, one at a time")
    if psf is not None:
        if len(psf) != 2 or not all(np.isfinite(s) and s > 0 for s in psf):
            raise ValueError(
                "Both calibrated PSF sigma values must be finite and positive"
            )
        psf = tuple(psf)
    total_start = perf_counter()
    ref_path = Path(reference or path)
    maps = TileMaps() if maps is None else maps
    maps.methods = selected
    with ExitStack() as resources:
        live = resources.enter_context(Progress(progress))
        cp = array_backend("cuda")
        if device >= cp.cuda.runtime.getDeviceCount():
            raise ValueError(f"CUDA device {device} is unavailable")
        resources.enter_context(cp.cuda.Device(device))
        gpu = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
        live.location = f"CUDA {device}: {gpu}"
        live.stage = "opening TIFF and reading reference"
        live.tell()
        stack = resources.enter_context(tifffile.TiffFile(path))
        page_indices = None
        total_pages = None
        if sample_pages is not None:
            total_pages = len(stack.pages)
            # One random page per contiguous stratum avoids repeatedly hitting
            # the same phase of periodic scan content. Only decode those pages.
            edges = np.linspace(
                0, total_pages, min(sample_pages, total_pages) + 1, dtype=int
            )
            page_indices = (
                np.random.default_rng(0).integers(edges[:-1], edges[1:]).tolist()
            )
        run = Run(cp, live, selected)
        # A page's tiles are screened in bounded chunks. A board page fits in one.
        chunk = max(1, min(1024, 2**23 // tile_size**2))
        with run.time("reference_and_warmup"):
            ref = _reference(ref_path, reference_page, tifffile)
            live.stage = f"discovering reference lattices on CPU ({ref.shape[1]} x {ref.shape[0]})"
            live.tell()
            templates = prepare_reference(ref, tile_size, shape, cp)
            first = stack.pages[0]
            if len(first.shape) != 2 or min(first.shape) < tile_size:
                raise ValueError(
                    "First target page must be grayscale and fit at least one complete tile"
                )
            warm_count = min(
                chunk,
                (first.shape[0] // tile_size) * (first.shape[1] // tile_size),
            )
            for number, template in enumerate(templates, 1):
                live.location = f"Warming reference {number}/{len(templates)}"
                live.stage = "CUDA selection"
                live.tell()
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
                warm = Run(cp, live, selected)
                _measure_page(batch[:1], matches[:1], [template], psf, warm, cp)
        sample = TileSample(sample_tiles) if sample_tiles is not None else None
        mode = (
            "all accepted tiles"
            if sample is None
            else f"uniform sample of up to {sample_tiles} accepted tiles"
        )
        live.location = f"Starting scan, measuring {mode} with {', '.join(selected)}"
        live.stage = (
            f"reading {len(page_indices)} of {total_pages} TIFF pages"
            if page_indices is not None
            else "reading TIFF pages"
        )
        live.tell()
        pages = tested = accepted = edge_pixels = measured = 0
        position_of = {}
        scan_start = perf_counter()
        last_save = -np.inf

        def save_progress():
            nonlocal last_save
            now = perf_counter()
            if checkpoint and now - last_save >= 15:
                checkpoint(
                    {
                        "status": "running",
                        "input": str(path),
                        "gpu": gpu,
                        "sample_tiles": sample_tiles,
                        "sample_page_indices_zero_based": page_indices,
                        "total_pages": total_pages,
                        "pages_completed": pages,
                        "tested_tiles": tested,
                        "accepted_tiles": accepted,
                        "measured_tiles": measured,
                        "methods": run.methods(),
                        "stage_seconds": dict(run.seconds),
                        "elapsed_seconds": now - total_start,
                    }
                )
                last_save = now

        def measure(tiles, matches, position):
            """Measure one page's accepted tiles and file each tile's value."""
            nonlocal measured
            values, openings, blurs = _measure_page(
                tiles, matches, templates, psf, run, cp
            )
            where = matches[:, 0].astype(int)
            maps.reference[position][where] = matches[:, 1]
            for name, column in values.items():
                maps.values[position][name][where] = column
            for name, column in openings.items():
                maps.openings[position][name][where] = column
            maps.blur.extend((position, *blur) for blur in blurs)
            measured += len(matches)
            save_progress()

        for page_index, data, decode_seconds in _decoded_pages(
            stack, max_pages, page_indices
        ):
            run.seconds["decode"] += decode_seconds
            h, w = data.shape
            ny, nx = h // tile_size, w // tile_size
            if not nx * ny:
                raise ValueError(
                    f"Tile size {tile_size} exceeds the dimensions of page {page_index}: {w} x {h}"
                )
            edge_pixels += h * w - nx * ny * tile_size**2
            position = position_of[page_index] = maps.add_page(page_index, ny, nx)
            # This is a strided CPU view. Only complete tiles are packed and uploaded.
            tile_grid = (
                data[: ny * tile_size, : nx * tile_size]
                .reshape(ny, tile_size, nx, tile_size)
                .transpose(0, 2, 1, 3)
            )
            kept, found = [], []
            for start in range(0, ny * nx, chunk):
                live.location = (
                    f"Selecting on page {page_index + 1}, tile {start + 1}/{ny * nx}"
                )
                with run.time("selection"):
                    indices = np.arange(start, min(start + chunk, ny * nx))
                    tiles = cp.asarray(tile_grid[indices // nx, indices % nx])
                    matches = select_tiles(tiles, templates, min_corr, cp)
                    if sample is None and 0 < len(matches) < len(tiles):
                        tiles = tiles[cp.asarray(matches[:, 0], dtype=cp.int32)]
                tested += len(indices)
                accepted += len(matches)
                if sample is not None:
                    sample.consider(data, page_index, start, matches, tile_size)
                elif len(matches):
                    # From here on a tile is known by its index on the page.
                    matches[:, 0] += start
                    kept.append(tiles)
                    found.append(matches)
            if found:
                live.location = f"Measuring page {page_index + 1}"
                measure(
                    kept[0] if len(kept) == 1 else cp.concatenate(kept),
                    np.concatenate(found),
                    position,
                )
            pages += 1
            live.location = f"Read {pages} pages, accepted {accepted}/{tested} tiles, measured {measured}"
            live.stage = "waiting for next TIFF page"
            if pages == 1:
                live.tell()
            save_progress()
        if sample is not None:
            live.location = f"Scanned {pages} pages, measuring {len(sample.entries)} of {accepted} accepted tiles"
            live.stage = "uploading sample"
            live.tell()
            # Each page's sampled tiles are measured together, like a full page.
            entries = sorted(sample.entries, key=lambda e: (position_of[e[0]], e[1]))
            for page_index, group in groupby(entries, key=lambda e: e[0]):
                group = list(group)
                with run.time("sample_upload"):
                    tiles = cp.asarray(np.stack([e[2] for e in group]))
                matches = np.array([(e[1], *e[3:]) for e in group], dtype=float)
                measure(tiles, matches, position_of[page_index])
        scan_seconds = perf_counter() - scan_start
    methods = run.methods()
    # Inclusive model and differential times both contain the same contour
    # and blur setup. Count each timed stage once when extrapolating.
    method_seconds = run.seconds["fit_setup"] + sum(run.seconds[m] for m in METHODS)
    scale = accepted / measured if 0 < measured < accepted else None
    exhaustive_estimate = method_seconds * scale if scale is not None else None
    for row in methods.values():
        row["estimated_all_tiles_seconds"] = (
            row["seconds"] * scale if scale is not None else None
        )
    ranked = sorted(
        (m for m in methods if methods[m]["valid"]), key=lambda m: methods[m]["seconds"]
    )
    blur = None
    if maps.blur:
        sigma = np.array([entry[2:4] for entry in maps.blur], dtype=float)
        blur = {
            "source": "supplied" if psf is not None else "estimated on each page",
            "page_fits": len(maps.blur),
            "unusable_page_fits": sum(not entry[4] for entry in maps.blur),
            "median_sigma_px": np.median(sigma, axis=0).tolist(),
            "lowest_sigma_px": sigma.min(axis=0).tolist(),
            "highest_sigma_px": sigma.max(axis=0).tolist(),
        }
    return {
        "status": "complete",
        "input": str(path),
        "reference": str(ref_path),
        "reference_page": reference_page,
        "gpu": gpu,
        "device": device,
        "tile_size": tile_size,
        "min_corr": min_corr,
        "reference_shapes": [t.region.shape for t in templates],
        "reference_pitch_px": [float(t.region.pitch) for t in templates],
        "psf_sigma_px": psf,
        "blur": blur,
        "pages": pages,
        "total_pages": pages
        if max_pages is None and page_indices is None
        else total_pages,
        "tested_tiles": tested,
        "accepted_tiles": accepted,
        "measured_tiles": measured,
        "estimated_all_tiles_method_seconds": exhaustive_estimate,
        "sample_tiles": sample_tiles,
        "max_pages": max_pages,
        "sample_page_indices_zero_based": page_indices,
        "sample_locations_page_tile_zero_based": [e[:2] for e in sample.entries]
        if sample is not None
        else None,
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


class PartialReport:
    """Publish readable snapshots next to the requested final report."""

    def __init__(self, target):
        self.path = target.with_suffix(".partial.json")
        self.created = False

    def write(self, payload):
        if not self.created:
            with self.path.open("x", encoding="utf-8") as output:
                json.dump(payload, output, indent=2, allow_nan=False)
            self.created = True
            return
        tmp = None
        try:
            with NamedTemporaryFile(
                mode="w", dir=self.path.parent, encoding="utf-8", delete=False
            ) as temp:
                tmp = Path(temp.name)
                json.dump(payload, temp, indent=2, allow_nan=False)
            os.replace(tmp, self.path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


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
        "--method",
        choices=METHODS,
        help="Time one method on the selected tiles (default: all seven methods)",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--sample-tiles",
        type=int,
        help="Limit measurements to a uniform sample of N accepted tiles",
        metavar="N",
    )
    sampling.add_argument(
        "--all-tiles",
        dest="sample_tiles",
        action="store_const",
        const=None,
        help="Measure every accepted tile with the requested methods (default)",
    )
    pages = parser.add_mutually_exclusive_group()
    pages.add_argument(
        "--max-pages",
        type=int,
        help="Read at most this many target pages for a short trial",
    )
    pages.add_argument(
        "--sample-pages",
        type=int,
        help="Limit screening to N pages distributed across the TIFF. Counts and projections cover these pages.",
        metavar="N",
    )
    pages.add_argument(
        "--all-pages",
        action="store_true",
        help="Screen every TIFF page (default)",
    )
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
    parser.add_argument(
        "--map",
        type=Path,
        help="Per-tile map to write. Default: beside --json, otherwise "
        "<TIFF name>.cd-map.html in the current folder. The tile values are "
        "saved beside it as .npz",
        metavar="HTML",
    )
    parser.add_argument(
        "--no-map", action="store_true", help="Do not write the map and tile values"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the map in the browser after the run",
    )
    args = parser.parse_args(argv)
    partial = PartialReport(args.json) if args.json else None
    map_path = None
    if not args.no_map:
        map_path = args.map or (
            args.json.with_suffix(".cd-map.html")
            if args.json
            else Path.cwd() / f"{args.tiff.stem}.cd-map.html"
        )
    maps = TileMaps()
    try:
        if args.json and (args.json.exists() or partial.path.exists()):
            raise ValueError(f"Choose a new JSON output path: {args.json}")
        result = benchmark(
            args.tiff,
            reference=args.reference,
            reference_page=args.reference_page,
            tile_size=args.tile_size,
            min_corr=args.min_corr,
            device=args.device,
            method=args.method,
            sample_tiles=args.sample_tiles,
            max_pages=args.max_pages,
            sample_pages=args.sample_pages,
            maps=maps,
            checkpoint=partial.write if partial else None,
            shape=args.shape,
            psf=args.psf_sigma_px,
            progress=lambda s: print(s, flush=True),
        )
        if args.json:
            with args.json.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
            if partial.created:
                partial.path.unlink()
        if map_path:
            arrays = maps.arrays()
            map_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                map_path.with_suffix(".npz"), tile_size=args.tile_size, **arrays
            )
            write_map(map_path, result, arrays)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"TIFF benchmark: {exc}\n")
    coverage = f"{result['pages']} pages"
    if result["total_pages"] is not None:
        coverage = f"{result['pages']}/{result['total_pages']} pages"
    print(
        f"{result['gpu']} | {coverage} screened | {result['accepted_tiles']}/{result['tested_tiles']} "
        f"tiles selected | {result['measured_tiles']} measured | {result['excluded_edge_pixels']} edge pixels skipped"
    )
    stages = result["stage_seconds"]
    print(
        f"Reference + warm-up {stages['reference_and_warmup']:.3f} s | "
        f"decode {stages.get('decode', 0):.3f} s | selection {stages.get('selection', 0):.3f} s"
    )
    estimate = result["estimated_all_tiles_method_seconds"]
    projection_label = (
        "   Est. all accepted tiles on screened pages (s)"
        if estimate is not None
        else ""
    )
    print(f"Method          Seconds     Valid / attempted   Mean{projection_label}")
    for method, row in result["methods"].items():
        value = "n/a" if row["mean"] is None else f"{row['mean']:.5g} {row['unit']}"
        projection = (
            f"   {row['estimated_all_tiles_seconds']:,.1f}"
            if estimate is not None
            else ""
        )
        print(
            f"{method:13} {row['seconds']:9.4f} {row['valid']:9} / {row['attempted']:<9}   {value}{projection}"
        )
    if any(name in result["methods"] for name in FITS):
        print(
            "Aperture fit times include contour work and "
            f"{stages.get('fit_setup', 0):.3f} s of shared setup."
        )
    blur = result["blur"]
    if blur and blur["source"] != "supplied":
        low, mid, high = (
            blur[f"{key}_sigma_px"] for key in ("lowest", "median", "highest")
        )
        print(
            f"Blur found on each page from its own openings, {blur['page_fits']} times: "
            f"median sigma {mid[0]:.2f} x {mid[1]:.2f} px "
            f"(x {low[0]:.2f} to {high[0]:.2f}, y {low[1]:.2f} to {high[1]:.2f}). "
            f"{blur['unusable_page_fits']} left out because the blur could not be pinned down."
        )
    if estimate is not None:
        print(
            f"At the measured rate of {estimate / result['accepted_tiles']:.3f} s/tile, "
            f"{args.method or 'all seven methods'} on all {result['accepted_tiles']:,} accepted tiles "
            f"would take about {estimate:,.1f} s ({estimate / 3600:.2f} h), "
            "plus reference preparation, input and screening on those pages."
        )
    print(
        f"Scan {result['scan_seconds']:.3f} s ({result['pages_per_second']:.2f} pages/s) | "
        f"total {result['total_seconds']:.3f} s"
    )
    if result["fastest_with_valid_results"]:
        print(f"Fastest with valid results: {result['fastest_with_valid_results']}")
        if result["fastest_valid_cd"]:
            print(f"Fastest with valid hole dimensions: {result['fastest_valid_cd']}")
    else:
        print("No valid measurements on the screened pages.")
        if result["total_pages"] and result["pages"] < result["total_pages"]:
            print("Use --all-pages to screen the entire stack.")
    if map_path:
        print(f"Tile values: {map_path.with_suffix('.npz')}")
        print(f"Tile map: {map_path}")
        # Open the map for a person at a terminal, not for a redirected batch run.
        if not args.no_open and sys.stdout.isatty():
            webbrowser.open(map_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
