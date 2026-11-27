"""Measure every stripe of a scan into tiles and bins for a map.

A scan is a folder that holds a header and, at any depth below it, the board
stacks of every stripe. Several boards record a stripe side by side, each into
its own multi-page TIFF named <stripe>_<vertical segment>_<board>.tiff, and
that name is all a file is found by. Neighbouring boards image the same few
columns. Page by page, the boards are merged with those columns counted once,
and the merged pages of a segment are stacked into one strip.

The strip is cut into square tiles of about 10 µm at the pixel size the header
gives. Across the stripe the tiles are centred, and the columns left over at
its two edges are not used. Along it, tiles run on from one page into the
next. Every tile is screened against the reference and the accepted tiles are
measured. Tiles are then grouped into bins, by default 100 µm along the stripe
and the whole stripe across it, and a bin's value is the mean of its tiles
after those far from the bin's mean are left out.

Every segment gets one .npz with the values of its tiles and of its bins.
stripes.json lists them with the geometry a map needs. On request a second
.npz keeps the outer columns of every merged page, which neighbouring stripes
share, so the shift between stripes can be found without reading the scan again.
"""

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np

from .files import native
from .fit import array_backend, to_host
from .tiff_benchmark import (
    FITS,
    METHODS,
    UNITS,
    Progress,
    Run,
    TileMaps,
    _measure_page,
    _read_page,
    _reference,
)
from .tiff_tiles import prepare_reference, select_tiles

FORMAT = "actis-stripes-2"
MANIFEST = "stripes.json"
NAME = re.compile(r"(\d+)_(\d+)_(\d+)\.tiff?", re.IGNORECASE)
# Read this much of one board before moving to the next, so a hard disk reads
# long runs instead of seeking between the board files at every page.
READ_BLOCK_BYTES = 2**26
# Most pixels screened in one batch. A merged page of four boards fits.
SCREEN_PIXELS = 2**24
# Columns that neighbouring stripes share: the usual width of an edge strip.
EDGE_COLUMNS = 32
# Default tile side and bin length along the stripe, in µm. Across the stripe
# a bin is as wide as the stripe unless asked otherwise.
TILE_UM = 10.0
BIN_Y_UM = 100.0
# Within a bin, tiles farther than this many standard deviations from the
# bin's mean are left out of its value.
REJECT_SIGMA = 1.5
# A pixel size outside this range, in µm, comes from a misread header.
PIXEL_UM_RANGE = (0.005, 0.1)


@dataclass(frozen=True)
class Segment:
    """One vertical segment of a stripe: the board files that record it."""

    stripe: int
    vertical: int
    files: tuple  # one path per board, in board order

    @property
    def name(self):
        return f"{self.stripe}_{self.vertical}"


def find_segments(root, boards=4):
    """Group the board TIFFs under `root`, at any depth, into segments.

    Returns the segments that have boards 1 to `boards`, in stripe order, and
    a list saying which board sets are incomplete.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Scan folder does not exist: {root}")
    found = {}
    for path in sorted(root.rglob("*")):
        match = NAME.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        stripe, vertical, board = map(int, match.groups())
        slot = found.setdefault((stripe, vertical), {})
        if board in slot:
            raise ValueError(
                f"Board {board} of {stripe}_{vertical} is stored twice: "
                f"{slot[board]} and {path}"
            )
        slot[board] = path
    if not found:
        raise ValueError(
            f"No <stripe>_<vertical>_<board>.tiff files in {root} or its subfolders"
        )
    wanted = list(range(1, boards + 1))
    segments, incomplete = [], []
    for (stripe, vertical), slot in sorted(found.items()):
        if sorted(slot) == wanted:
            files = tuple(slot[board] for board in wanted)
            segments.append(Segment(stripe, vertical, files))
        else:
            incomplete.append(
                {"stripe": stripe, "vertical": vertical, "boards_found": sorted(slot)}
            )
    return segments, incomplete


def scan_pixel_size(root, pixel_nm=None):
    """The pixel size of the scan in `root`, in µm across and along the stripe.

    Returns ((x, y), path of the header or None). The header is found and read
    by actis.header. A `pixel_nm` that is given is used for both directions
    instead, and then a header that cannot be found is no error.
    """
    given = None if pixel_nm is None else (pixel_nm / 1000,) * 2
    try:
        from .header import HeaderError, find_header, parse_header
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}.header":
            raise
        if given:
            return given, None
        raise ValueError(
            "Reading the pixel size from the scan header needs actis.header, "
            "which this installation does not have. Give --pixel-nm instead"
        ) from None
    try:
        path = find_header(root)
        problem = None if path else "no header found"
    except HeaderError as error:
        path, problem = None, error
    if given:
        return given, path
    if problem:
        raise ValueError(
            f"Cannot find the header of {root}: {problem}. "
            "Give --pixel-nm to measure without it"
        )
    try:
        header = parse_header(path)
    except HeaderError as error:
        raise ValueError(
            f"Cannot read the pixel size from {path}: {error}. "
            "Give --pixel-nm to measure without it"
        ) from None
    size = float(header.pixel_x), float(header.pixel_y)
    low, high = PIXEL_UM_RANGE
    if not all(low <= s <= high for s in size):
        raise ValueError(
            f"{path} gives pixels of {size[0]:g} x {size[1]:g} um, outside "
            f"{low} to {high} um. Check what actis.header reads, or give --pixel-nm"
        )
    return size, path


def tile_pixels(tile_um, pixel_um):
    """The even tile side, in pixels, nearest to `tile_um` at the mean pixel size."""
    return 2 * int(tile_um / (sum(pixel_um) / 2) / 2 + 0.5)


class Across(NamedTuple):
    """Where tiles and bins lie across one stripe."""

    first_column: int  # merged column at which tile column 0 begins
    tiles: int  # tile columns
    first_tile: int  # tile column at which bin column 0 begins
    bins: int  # bin columns
    bin_tiles: int  # tile columns per bin


@dataclass(frozen=True)
class Tiling:
    """How a stripe is cut into square tiles and the tiles grouped into bins."""

    tile: int  # tile side in pixels
    pixel_um: tuple  # pixel size across and along the stripe
    bin_rows: int  # tiles per bin along the stripe
    bin_columns: int = None  # tiles per bin across it; None for all that fit

    @classmethod
    def from_um(cls, tile_um, bin_y_um, bin_x_um, pixel_um):
        """Sizes in um, each bin side rounded to a whole number of tiles."""
        tile = tile_pixels(tile_um, pixel_um)
        across, along = (tile * size for size in pixel_um)
        rows = max(1, int(bin_y_um / along + 0.5))
        columns = None if bin_x_um is None else max(1, int(bin_x_um / across + 0.5))
        return cls(tile, tuple(pixel_um), rows, columns)

    def across(self, width):
        """Place tiles and bins across a merged stripe `width` columns wide.

        Both are centred: the columns no whole tile fills, and then the tiles
        no whole bin fills, are split between the two edges and not binned.
        """
        tiles, um = width // self.tile, self.pixel_um[0]
        if not tiles:
            raise ValueError(
                f"Tiles of {self.tile} px ({self.tile * um:.2f} um) are wider than "
                f"the stripe, {width} px ({width * um:.2f} um)"
            )
        wide = self.bin_columns or tiles
        if wide > tiles:
            raise ValueError(
                f"Bins {wide} tiles wide ({wide * self.tile * um:.1f} um) do not fit "
                f"across the stripe: its {width} columns ({width * um:.1f} um) hold "
                f"{tiles} tiles of {self.tile} px ({tiles * self.tile * um:.1f} um)"
            )
        bins = tiles // wide
        return Across(
            (width - tiles * self.tile) // 2,
            tiles,
            (tiles - bins * wide) // 2,
            bins,
            wide,
        )

    def describe(self, wide=None):
        x, y = (1000 * s for s in self.pixel_um)
        t, wide = self.tile, wide or self.bin_columns
        tall = self.bin_rows * t * y / 1000
        return (
            f"Pixels of {x:.4g} x {y:.4g} nm. Tiles of {t} px: "
            f"{t * x / 1000:.3f} um across the stripe, {t * y / 1000:.3f} um along it. "
            f"Bins of {self.bin_rows} tiles ({tall:.1f} um) along the stripe by "
            + (
                f"{wide} tiles ({wide * t * x / 1000:.1f} um) across it"
                if wide
                else "the whole stripe across it"
            )
        )


def bin_tiles(values, size, first_column=0, reject_sigma=None):
    """Average the tiles of every bin.

    `values` is shaped (tile row, tile column), NaN where a tile has no value.
    A bin is `size` (tile rows, tile columns). Bins begin at tile row 0 and at
    tile column `first_column`, as many whole ones as fit across, and run to
    the last tile row, so the last bin row may hold fewer tile rows.

    With `reject_sigma`, a tile farther from its bin's mean than that many
    sample standard deviations of the bin's tiles is left out. A bin with one
    tile keeps it.

    Returns the bin values (bin row, bin column), the tiles averaged in each
    bin, and which tiles were averaged, shaped like `values`.
    """
    values = np.asarray(values, dtype=np.float64)
    rows, columns = values.shape
    tall, wide = size
    down, across = -(-rows // tall), max(0, (columns - first_column) // wide)
    end = first_column + across * wide
    block = np.full((down * tall, across * wide), np.nan)
    block[:rows] = values[:, first_column:end]
    block = block.reshape(down, tall, across, wide).transpose(0, 2, 1, 3)
    block = block.reshape(down, across, tall * wide)
    used = np.isfinite(block)
    with np.errstate(invalid="ignore", divide="ignore"):
        if reject_sigma is not None:
            count = used.sum(axis=-1, keepdims=True)
            mean = np.where(used, block, 0).sum(axis=-1, keepdims=True) / count
            squares = np.where(used, (block - mean) ** 2, 0).sum(axis=-1, keepdims=True)
            spread = np.sqrt(squares / (count - 1))  # NaN for a single tile: kept
            used &= ~(np.abs(block - mean) > reject_sigma * spread)
        tiles = used.sum(axis=-1)
        means = np.where(used, block, 0).sum(axis=-1) / tiles
    averaged = np.zeros(values.shape, dtype=bool)
    averaged[:, first_column:end] = (
        used.reshape(down, across, tall, wide)
        .transpose(0, 2, 1, 3)
        .reshape(down * tall, across * wide)[:rows]
    )
    return means.astype(np.float32), tiles.astype(np.int32), averaged


def bin_ranges(rows, size, across):
    """The tile rows and tile columns of every bin, as [first, end) pairs."""
    tall, wide = size
    down = -(-rows // tall)
    row_ranges = [[k * tall, min((k + 1) * tall, rows)] for k in range(down)]
    columns = [
        [across.first_tile + k * wide, across.first_tile + (k + 1) * wide]
        for k in range(across.bins)
    ]
    return (
        np.asarray(row_ranges, dtype=np.int64).reshape(-1, 2),
        np.asarray(columns, dtype=np.int64).reshape(-1, 2),
    )


def _given_up(count, overlap):
    """The columns each of `count` boards gives up at its left and right edge."""
    right = overlap // 2
    left = overlap - right
    return [(left if k else 0, right if k < count - 1 else 0) for k in range(count)]


def merge_boards(boards, overlap, xp=np):
    """Merge one page of every board, left to right, with shared columns counted once.

    Neighbouring boards image the same `overlap` columns. Each gives up half
    of them, so a column comes from the board it lies deeper inside.
    """
    return xp.concatenate(
        [
            board[:, left : board.shape[1] - right]
            for board, (left, right) in zip(boards, _given_up(len(boards), overlap))
        ],
        axis=1,
    )


def seam_columns(count, columns, overlap):
    """The column of the merged page at which each next board begins."""
    widths = [columns - left - right for left, right in _given_up(count, overlap)]
    return np.cumsum(widths)[:-1].tolist()


def seam_agreement(boards, overlap, xp=np):
    """How alike neighbouring boards are in the columns they share.

    One correlation per seam: near 1 where both boards show the same pattern,
    near 0 on an empty page or where the boards do not overlap as assumed.
    """
    if overlap < 1 or len(boards) < 2:
        return np.empty(0, dtype=np.float32)
    left = xp.stack([b[:, -overlap:] for b in boards[:-1]]).astype(xp.float64)
    right = xp.stack([b[:, :overlap] for b in boards[1:]]).astype(xp.float64)
    left -= left.mean(axis=(1, 2), keepdims=True)
    right -= right.mean(axis=(1, 2), keepdims=True)
    energy = xp.sqrt((left * left).sum(axis=(1, 2)) * (right * right).sum(axis=(1, 2)))
    agreement = (left * right).sum(axis=(1, 2)) / xp.where(energy > 0, energy, xp.nan)
    return to_host(xp, agreement).astype(np.float32)


def cut_tiles(page, tile, xp=np):
    """Cut a page into whole tiles from its top-left corner, row by row.

    Returns the tiles shaped (tile row x tile column, tile, tile) and the grid
    size (rows, columns). Rows and columns left over at the right and bottom
    belong to no tile.
    """
    ny, nx = page.shape[0] // tile, page.shape[1] // tile
    if not ny * nx:
        raise ValueError(
            f"Tile size {tile} exceeds the merged page: "
            f"{page.shape[1]} x {page.shape[0]} px"
        )
    tiles = xp.ascontiguousarray(
        page[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3)
    ).reshape(ny * nx, tile, tile)
    return tiles, (ny, nx)


def _board_pages(files, max_pages, seconds, notes, tifffile):
    """Yield (page index, one page per board) for the pages every board has.

    A background thread reads one block ahead. Within a block it takes one
    board at a time. `seconds` gets the time spent reading and the time the
    caller waited for it. `notes` gets a line if the boards differ in length.
    """
    with ExitStack() as opened:
        tiffs = [opened.enter_context(tifffile.TiffFile(path)) for path in files]
        first = tiffs[0].pages[0]
        page_bytes = int(np.prod(first.shape)) * max(1, first.bitspersample // 8)
        block_pages = max(1, READ_BLOCK_BYTES // max(1, page_bytes))
        pages = [iter(tiff.pages) for tiff in tiffs]

        def read(count):
            start = perf_counter()
            block = [[_read_page(p) for p in islice(board, count)] for board in pages]
            return block, perf_counter() - start

        reader = opened.enter_context(ThreadPoolExecutor(max_workers=1))

        def ahead(start):
            count = block_pages
            if max_pages is not None:
                count = min(count, max_pages - start)
            return (reader.submit(read, count), count) if count > 0 else None

        done, pending = 0, ahead(0)
        while pending:
            future, asked = pending
            waiting = perf_counter()
            block, spent = future.result()
            seconds["read_wait"] += perf_counter() - waiting
            seconds["read"] += spent
            have = [len(board) for board in block]
            shared = min(have)
            pending = ahead(done + asked) if shared == asked else None
            for k in range(shared):
                yield done + k, [board[k] for board in block]
            done += shared
            if max(have) > shared:
                short = [path.name for path, n in zip(files, have) if n == shared]
                notes.append(
                    f"{', '.join(short)} ended after {done} pages while other "
                    f"boards go on. Measured the {done} pages all boards have."
                )


def gpu_works(cp):
    """Whether the current GPU still runs work. Some CUDA errors end its context."""
    try:
        cp.cuda.runtime.deviceSynchronize()
        return float(cp.arange(3, dtype=cp.float32).sum()) == 3.0
    except Exception:
        return False


def _error_text(error):
    return f"{type(error).__name__}: {error}"


def _note_problem(problems, name, error):
    entry = problems.setdefault(name, {"failed_batches": 0})
    entry["failed_batches"] += 1
    if "error" not in entry:
        entry["error"] = _error_text(error)
        entry["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )


def _measure_kept(tiles, matches, templates, run, cp, problems):
    """Measure a batch like _measure_page, but a method that fails loses only its own values.

    After an error the batch is measured again one method at a time. A method
    that fails on its own gets no values in this batch, and its first error is
    kept in `problems`. An error after which the GPU no longer works is raised.
    """

    def counts():
        return run.valid.copy(), run.attempted.copy(), run.sums.copy()

    def restore(saved):
        run.valid, run.attempted, run.sums = saved

    saved = counts()
    try:
        return _measure_page(tiles, matches, templates, None, run, cp)
    except Exception:
        if not gpu_works(cp):
            raise
        restore(saved)
    selected, values, openings, blurs = run.selected, {}, {}, []
    try:
        for name in selected:
            run.selected = (name,)
            saved = counts()
            try:
                part = _measure_page(tiles, matches, templates, None, run, cp)
            except Exception as error:
                if not gpu_works(cp):
                    raise
                restore(saved)
                _note_problem(problems, name, error)
                continue
            values.update(part[0])
            openings.update(part[1])
            blurs.extend(part[2])
    finally:
        run.selected = selected
    return values, openings, blurs


def _measure_rows(strip, first_row, templates, settings, tile, run, maps, cp, problems):
    """Cut the whole rows of tiles at the top of `strip`, screen them and
    measure the accepted ones. Tile row 0 of the strip is tile row `first_row`
    of the segment. A method that fails is noted in `problems`.

    Returns how many rows of tiles were cut, screened and accepted.
    """
    with run.time("merge"):
        tiles, (ny, nx) = cut_tiles(strip, tile, cp)
    position = maps.add_page(first_row, ny, nx)
    batch = max(1, SCREEN_PIXELS // tile**2)
    with run.time("selection"):
        matches = []
        for first in range(0, ny * nx, batch):
            found = select_tiles(
                tiles[first : first + batch], templates, settings["min_corr"], cp
            )
            # From here on a tile is known by its index among these rows.
            found[:, 0] += first
            matches.append(found)
        matches = np.concatenate(matches)
        if 0 < len(matches) < len(tiles):
            tiles = tiles[cp.asarray(matches[:, 0], dtype=cp.int32)]
    if len(matches):
        maps.file(
            position,
            matches,
            *_measure_kept(tiles, matches, templates, run, cp, problems),
        )
    return ny, ny * nx, len(matches)


def _tile_arrays(maps, columns):
    """Join the rows of tiles that TileMaps kept batch by batch into one grid.

    Returns the per-tile arrays shaped (tile row, tile column), and the blur
    arrays, which stay per batch, with the first tile row of every batch.
    """
    data = maps.arrays()
    heights = [rows for rows, _ in maps.grids]
    blur = {name: data.pop(name) for name in ("blur_sigma_px", "blur_usable")}
    blur["blur_tile_row"] = data.pop("page_index")
    tiles = {
        name: np.concatenate([part[:h] for part, h in zip(array, heights)])
        if heights
        else array.reshape(0, columns)
        for name, array in data.items()
    }
    return tiles, blur


def _measure_segment(segment, templates, settings, tiling, cp, live, tifffile):
    """Measure every tile of one segment and average the tiles into bins.

    Returns the arrays to save, the edge strips if they are kept, and a
    summary for the manifest.
    """
    tile, overlap = tiling.tile, settings["overlap_px"]
    edge = settings["edge_columns"]
    methods = tuple(settings["methods"])
    run, maps = Run(cp, live, methods), TileMaps(methods)
    notes, seams, left, right, page_index, problems = [], [], [], [], [], {}
    layout = across = strip = None
    tested = accepted = rows = 0
    start = perf_counter()
    pages = _board_pages(
        segment.files, settings["max_pages"], run.seconds, notes, tifffile
    )
    with closing(pages):  # closes the board files, also when a page cannot be read
        for index, boards in pages:
            live.location = f"Segment {segment.name}, page {index + 1}"
            if layout is None:
                layout = boards[0].shape, boards[0].dtype
                if layout[0][1] <= overlap:
                    raise ValueError(
                        f"The boards are {layout[0][1]} columns wide, no wider "
                        f"than the overlap of {overlap}"
                    )
                if edge > layout[0][1] - overlap:
                    raise ValueError(
                        f"Edge strips of {edge} columns do not fit boards of "
                        f"{layout[0][1]} columns that share {overlap}"
                    )
                count = len(boards)
                across = tiling.across(count * layout[0][1] - (count - 1) * overlap)
                kept_columns = slice(
                    across.first_column, across.first_column + across.tiles * tile
                )
            for path, board in zip(segment.files, boards):
                if (board.shape, board.dtype) != layout:
                    raise ValueError(
                        f"Page {index + 1} of {path.name} is {board.shape[1]} x "
                        f"{board.shape[0]} px {board.dtype}, not {layout[0][1]} x "
                        f"{layout[0][0]} px {layout[1]} like the segment's first page"
                    )
            with run.time("merge"):
                parts = [cp.asarray(board) for board in boards]
                seams.append(seam_agreement(parts, overlap, cp))
                page = merge_boards(parts, overlap, cp)[:, kept_columns]
                # Rows that did not fill a tile row continue into this page.
                strip = page if strip is None else cp.concatenate((strip, page))
            page_index.append(index)
            if len(strip) >= tile:
                cut, screened, kept = _measure_rows(
                    strip, rows, templates, settings, tile, run, maps, cp, problems
                )
                strip = strip[cut * tile :]
                rows += cut
                tested += screened
                accepted += kept
            if edge:
                # Copies, because a view would keep its whole page in memory.
                left.append(boards[0][:, :edge].copy())
                right.append(boards[-1][:, -edge:].copy())
            live.stage = "reading the board files"
    if layout is None:
        raise ValueError(f"{segment.files[0].name} has no pages")
    (page_rows, columns), count = layout[0], len(segment.files)
    tiles, blur = _tile_arrays(maps, across.tiles)
    size = tiling.bin_rows, across.bin_tiles
    sigma = settings["reject_sigma"]
    arrays = {f"tile_{name}": array for name, array in tiles.items()}
    rows_of_bins, columns_of_bins = bin_ranges(rows, size, across)
    summary_methods = run.methods()
    for name in methods:
        values, averaged, used = bin_tiles(tiles[name], size, across.first_tile, sigma)
        arrays[f"bin_{name}"] = values
        arrays[f"bin_{name}_tiles"] = averaged
        arrays[f"tile_{name}_used"] = used
        found = np.isfinite(values)
        summary_methods[name].update(
            bins=int(found.sum()),
            used_tiles=int(used.sum()),
            bin_mean=float(values[found].mean(dtype=np.float64))
            if found.any()
            else None,
            # Batches of tile rows in which the method raised an error and so
            # has no values, with the first error and where it was raised.
            **{"failed_batches": 0, **problems.get(name, {})},
        )
    # Tiles that run from one page into the next pass the screen as often as
    # the others only if the pages continue each other without a gap or repeat.
    tile_rows = np.arange(rows)
    across_pages = (tile_rows * tile) // page_rows != (
        (tile_rows + 1) * tile - 1
    ) // page_rows
    matched = tiles["reference"] >= 0
    shares = {
        where: [int(matched[part].sum()), int(matched[part].size)]
        for where, part in (
            ("within_pages", ~across_pages),
            ("across_pages", across_pages),
        )
    }
    if any(name in FITS for name in methods):
        arrays.update(blur)  # no other method needs a blur
    page_index = np.asarray(page_index, dtype=np.int64)
    seams = np.asarray(seams, dtype=np.float32).reshape(len(page_index), -1)
    arrays.update(
        bin_rows=rows_of_bins,
        bin_columns=columns_of_bins,
        page_index=page_index,
        seam_correlation=seams,
        stripe=segment.stripe,
        vertical=segment.vertical,
        tile_size=tile,
        tile_x0=across.first_column,
        page_rows=page_rows,
        pixel_um=np.asarray(tiling.pixel_um, dtype=np.float64),
        reject_sigma=np.nan if sigma is None else sigma,
        overlap_px=overlap,
        boards=count,
        board_columns=columns,
    )
    summary = {
        "pages": len(page_index),
        "board_shape": [page_rows, columns],
        "page_shape": [page_rows, count * columns - (count - 1) * overlap],
        "tile_grid": [rows, across.tiles],
        "tile_x0": across.first_column,
        "rows_left": len(strip),
        "bin_tiles": list(size),
        "bin_grid": [len(rows_of_bins), across.bins],
        "seam_columns": seam_columns(count, columns, overlap),
        # On the best tenth of the pages: empty pages do not pull it down.
        "seam_correlation_p90": [
            float(np.percentile(seam[np.isfinite(seam)], 90))
            if np.isfinite(seam).any()
            else None
            for seam in seams.T
        ],
        "tested_tiles": tested,
        "accepted_tiles": accepted,
        # [accepted, tested] for tile rows inside one page and across two
        "accepted_by_rows": shares,
        "methods": summary_methods,
        "seconds": {"total": perf_counter() - start, **run.seconds},
        "notes": notes,
    }
    edges = {}
    if edge:
        edges = {
            "left": np.stack(left),
            "right": np.stack(right),
            "page_index": arrays["page_index"],
            "stripe": segment.stripe,
            "vertical": segment.vertical,
        }
    return arrays, edges, summary


def _earlier_segments(output, settings):
    """What an earlier run with the same settings left in `output`, by segment."""
    manifest = output / MANIFEST
    if manifest.is_file():
        earlier = json.loads(manifest.read_text(encoding="utf-8"))
        if earlier.get("format") != FORMAT:
            raise ValueError(
                f"{output} holds results in the format {earlier.get('format')}, "
                f"and this version writes {FORMAT}. Choose a new output folder"
            )
        if earlier.get("settings") != settings:
            raise ValueError(
                f"{output} holds a run with other settings. Choose a new output folder"
            )
        return {(s["stripe"], s["vertical"]): s for s in earlier["segments"]}
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            f"Output must be a new or empty folder, or that of an earlier run: {output}"
        )
    return {}


@contextmanager
def _replacing(path):
    """Write a file under another name first, so a stopped run leaves no half of one."""
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as out:
        yield out
    replace(partial, path)


def replace(source, target, attempts=50):
    """os.replace, waiting while another process has `target` open.

    Windows refuses to replace a file that is open for reading, as it is for a
    moment whenever someone checks on a run.
    """
    for attempt in range(attempts):
        try:
            return os.replace(source, target)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1)


def _line(entry):
    name = f"{entry['stripe']}_{entry['vertical']}"
    if entry["status"] != "complete":
        return f"{name}: failed. {entry['error']}"
    rows, columns = entry["page_shape"]
    values = ", ".join(
        f"{method} "
        + (
            "n/a"
            if row["bin_mean"] is None
            else f"{row['bin_mean']:.5g} {row['unit']} from {row['used_tiles']:,} tiles"
        )
        for method, row in entry["methods"].items()
    )
    seams = " ".join(
        "n/a" if seam is None else f"{seam:.2f}"
        for seam in entry["seam_correlation_p90"]
    )
    failures = [
        f"{method} failed in {row['failed_batches']} batches of tile rows, "
        f"first with {row['error']}"
        for method, row in entry["methods"].items()
        if row.get("failed_batches")
    ]
    seconds = entry["seconds"]
    shares = " and ".join(
        f"{accepted / tested:.0%} {where.replace('_', ' ')}"
        for where, (accepted, tested) in entry["accepted_by_rows"].items()
        if tested
    )
    return " ".join(
        [
            f"{name}: {entry['pages']:,} pages of {columns} x {rows} px,",
            f"{entry['accepted_tiles']:,} of {entry['tested_tiles']:,} tiles accepted"
            + (f" ({shares})," if shares else ","),
            "{:,} x {:,} bins,".format(*entry["bin_grid"]),
            f"{values}, seams {seams or 'n/a'},",
            f"{seconds['total']:.0f} s ({seconds['read_wait']:.0f} s waiting for the disk)",
            *entry["notes"],
            *(f"{failure}." for failure in failures),
        ]
    )


def _reference_line(checks, refused):
    """One line on what every method measured on the reference tiles."""
    parts, empty = [], []
    for name, values in checks.items():
        found = [v for v in values if np.isfinite(v)]
        if name in refused:
            parts.append(f"{name} failed ({refused[name]['error']})")
        elif found:
            unit = UNITS.get(name, "px")
            parts.append(f"{name} " + " / ".join(f"{v:.5g}" for v in found) + f" {unit}")
        else:
            parts.append(f"{name} no value")
        if not found:
            empty.append(name)
    line = "On the reference tile itself: " + ", ".join(parts) + "."
    if empty:
        line += (
            f" Expect no values from {', '.join(empty)} on this scan, since only "
            "tiles like the reference are measured."
        )
    return line


def measure_stripes(
    root,
    output,
    *,
    reference,
    reference_page=0,
    methods=("raw",),
    tile_um=TILE_UM,
    bin_y_um=BIN_Y_UM,
    bin_x_um=None,
    reject_sigma=REJECT_SIGMA,
    pixel_nm=None,
    min_corr=0.80,
    overlap=32,
    boards=4,
    shape="auto",
    device=0,
    stripes=None,
    segments=None,
    max_pages=None,
    edge_columns=0,
    progress=None,
):
    """Measure the segments under `root` into `output` and return the manifest.

    `methods` may be "all" for every method. Tiles are `tile_um` square. Bins
    are `bin_y_um` along the stripe and `bin_x_um` across it, the whole stripe
    when None, both rounded to whole tiles. `reject_sigma` None averages every
    accepted tile of a bin. The pixel size comes from the scan's header unless
    `pixel_nm` is given. `device` is the GPU, numbered as CUDA numbers them in
    this process. `stripes` (numbers) and `segments` (names such as "3_1")
    choose what to measure.

    A segment that `output` already holds from an earlier run with the same
    settings is left as it is, so a stopped run continues where it ended.
    `edge_columns` above 0 also saves that many columns from the left and
    from the right edge of every merged page.

    A method that fails on some tiles loses only its own values there, and
    the error is kept in the manifest. A segment that fails is recorded with
    the reason and the run goes on, unless the GPU has stopped working.
    """
    try:
        import tifffile
    except ImportError:
        raise RuntimeError(
            'Install the TIFF/CUDA dependencies with pip install -e ".[gpu]"'
        ) from None
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    methods = METHODS if "all" in methods else tuple(dict.fromkeys(methods))
    if not methods or any(name not in METHODS for name in methods):
        raise ValueError(f"Choose methods from: {', '.join(METHODS)}, or all")
    if not tile_um > 0 or not 0 < min_corr <= 1:
        raise ValueError("Use a positive tile size and a minimum correlation in (0,1]")
    if not bin_y_um >= tile_um or not (bin_x_um is None or bin_x_um >= tile_um):
        raise ValueError(
            f"A bin must be at least one tile, {tile_um:g} um, in both directions"
        )
    if reject_sigma is not None and not reject_sigma > 0:
        raise ValueError(
            "The rejection threshold in standard deviations must be positive"
        )
    if pixel_nm is not None and not pixel_nm > 0:
        raise ValueError("The pixel size must be positive")
    if min(overlap, edge_columns, device, reference_page) < 0 or boards < 1:
        raise ValueError(
            "Overlap, edge columns, device and reference page must be nonnegative, "
            "with at least one board"
        )
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    root, output = Path(root).resolve(), Path(output).resolve()
    reference = Path(reference).resolve()
    wanted = segments
    segments, incomplete = find_segments(root, boards)
    if stripes is not None:
        segments = [s for s in segments if s.stripe in set(stripes)]
    if wanted is not None:
        wanted = {str(name) for name in wanted}
        missing = sorted(wanted - {s.name for s in segments})
        if missing:
            raise ValueError(
                f"No segment {', '.join(missing)} with boards 1 to {boards} under "
                f"{root}" + ("" if stripes is None else " in the chosen stripes")
            )
        segments = [s for s in segments if s.name in wanted]
    if not segments:
        raise ValueError(
            f"No segment with boards 1 to {boards} to measure under {root}"
        )
    pixel_um, header = scan_pixel_size(root, pixel_nm)
    tiling = Tiling.from_um(tile_um, bin_y_um, bin_x_um, pixel_um)
    if tiling.tile < 16:
        raise ValueError(
            f"Tiles of {tile_um:g} um are {tiling.tile} px at this scan's pixel "
            "size. Use tiles of at least 16 px"
        )
    # Stop before any work if the bins cannot fit across the stripe. A first
    # file that cannot be read fails its own segment later, with the reason.
    try:
        with tifffile.TiffFile(segments[0].files[0]) as first:
            columns = first.pages[0].shape[-1]
    except (OSError, IndexError, tifffile.TiffFileError):
        columns = 0
    across = (
        tiling.across(boards * columns - (boards - 1) * overlap)
        if columns > overlap
        else None
    )
    settings = native(
        {
            "reference": str(reference),
            "reference_page": reference_page,
            "methods": methods,
            "pixel_um": pixel_um,
            "tile_um": tile_um,
            "bin_y_um": bin_y_um,
            "bin_x_um": bin_x_um,
            "reject_sigma": reject_sigma,
            "min_corr": min_corr,
            "overlap_px": overlap,
            "boards": boards,
            "shape": shape,
            "max_pages": max_pages,
            "edge_columns": edge_columns,
        }
    )
    entries = _earlier_segments(output, settings)
    say = progress or (lambda line: None)
    with ExitStack() as resources:
        live = Progress(progress)
        live.location = f"Measuring {len(segments)} segments of {root}"
        resources.enter_context(live)
        cp = array_backend("cuda")
        gpus = [
            cp.cuda.runtime.getDeviceProperties(k)["name"].decode()
            for k in range(cp.cuda.runtime.getDeviceCount())
        ]
        if device >= len(gpus):
            listed = "; ".join(f"{k}: {name}" for k, name in enumerate(gpus))
            raise ValueError(f"There is no GPU {device}. CUDA sees {listed}")
        resources.enter_context(cp.cuda.Device(device))
        pci = cp.cuda.runtime.deviceGetPCIBusId(device)
        say(f"GPU {device}: {gpus[device]}, PCI bus {pci}")
        say(
            f"{tiling.describe(across and across.bin_tiles)}. Pixel size from "
            + (f"{header}" if pixel_nm is None else "--pixel-nm")
        )
        live.location = f"GPU {device}: {gpus[device]}"
        live.stage = "preparing the reference"
        live.tell()
        pixels = _reference(reference, reference_page, tifffile)
        templates = prepare_reference(pixels, tiling.tile, shape, cp)
        # Measure the reference tile itself, which compiles the kernels and
        # shows before any scan is read whether a method can measure this
        # pattern at all.
        checks, refused = {name: [] for name in methods}, {}
        for template in templates:
            tile = template.pixels.astype(pixels.dtype)[None]
            matches = select_tiles(tile, [template], min_corr, cp)
            if not len(matches):
                raise ValueError("Reference tile failed its own correlation check")
            values, _, _ = _measure_kept(
                tile, matches, [template], Run(cp, live, methods), cp, refused
            )
            for name in methods:
                checks[name].append(float(values.get(name, [np.nan])[0]))
        say(_reference_line(checks, refused))
        manifest = {
            "format": FORMAT,
            "root": str(root),
            "header": None if header is None else str(header),
            "pixel_source": "header" if pixel_nm is None else "--pixel-nm",
            "settings": settings,
            "tile_size": tiling.tile,
            "bin_rows": tiling.bin_rows,
            "gpu": gpus[device],
            "gpu_index": device,
            "gpu_pci_bus": pci,
            "reference_shapes": [t.region.shape for t in templates],
            "reference_pitch_px": [float(t.region.pitch) for t in templates],
            # What each method measured on each reference tile itself, None
            # for nothing, and the error of a method that failed there.
            "reference_values": checks,
            "reference_errors": {k: v["error"] for k, v in refused.items()},
            "incomplete": incomplete,
        }
        output.mkdir(parents=True, exist_ok=True)
        for number, segment in enumerate(segments, 1):
            key = segment.stripe, segment.vertical
            entry = entries.get(key)
            if (
                entry
                and entry["status"] == "complete"
                and all(
                    (output / entry[kept]).is_file()
                    for kept in ("result", "edges")
                    if kept in entry
                )
            ):
                say(f"{segment.name}: measured by an earlier run")
                continue
            live.location = f"Segment {segment.name} ({number} of {len(segments)})"
            live.stage = "opening the board files"
            entry = {
                "stripe": segment.stripe,
                "vertical": segment.vertical,
                "files": [path.relative_to(root).as_posix() for path in segment.files],
            }
            lost = None
            try:
                arrays, edges, summary = _measure_segment(
                    segment, templates, settings, tiling, cp, live, tifffile
                )
            except (OSError, ValueError, tifffile.TiffFileError) as error:
                entry.update(status="failed", error=str(error))
            except Exception as error:
                # Not a problem of the files. Keep where it happened, and go
                # on with the next segment if the GPU still works.
                entry.update(
                    status="failed",
                    error=_error_text(error),
                    traceback="".join(
                        traceback.format_exception(
                            type(error), error, error.__traceback__
                        )
                    ),
                )
                if not gpu_works(cp):
                    lost = error
                else:
                    cp.get_default_memory_pool().free_all_blocks()
            else:
                entry.update(status="complete", result=f"{segment.name}.npz", **summary)
                live.stage = "saving the results"
                saving = perf_counter()
                if edges:
                    entry["edges"] = f"{segment.name}_edges.npz"
                    # Not compressed. Noisy synthetic strips of a 1,500-page segment
                    # took 6 s to compress and shrank by a third at most.
                    with _replacing(output / entry["edges"]) as out:
                        np.savez(out, **edges)
                with _replacing(output / entry["result"]) as out:
                    np.savez_compressed(out, **arrays)
                saved = perf_counter() - saving
                entry["seconds"]["save"] = saved
                entry["seconds"]["total"] += saved
            entries[key] = native(entry)
            manifest["segments"] = [entries[k] for k in sorted(entries)]
            text = json.dumps(native(manifest), indent=2, allow_nan=False)
            with _replacing(output / MANIFEST) as out:
                out.write(text.encode("utf-8"))
            say(_line(entries[key]))
            if lost is not None:
                raise RuntimeError(
                    f"GPU {device} stopped working while measuring {segment.name} "
                    f"({_error_text(lost)}). The segments measured so far are kept. "
                    "Run the same command again to continue"
                )
    manifest.setdefault("segments", [entries[k] for k in sorted(entries)])
    return native(manifest)


def build_parser():
    """The command-line options of a stripe run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "root",
        type=Path,
        help="Scan folder: holds header.xml and, at any depth, the board stacks",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Holes-only TIFF or PNG at the scan's pixel scale and orientation",
    )
    parser.add_argument(
        "--reference-page",
        type=int,
        default=0,
        help="Zero-based page of a reference TIFF (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New folder for the results, or that of a stopped run to continue",
    )
    parser.add_argument(
        "--method",
        nargs="+",
        choices=(*METHODS, "all"),
        default=["raw"],
        help="Methods to run on every accepted tile, or all for all seven "
        "(default: raw)",
    )
    parser.add_argument(
        "--tile-um",
        type=float,
        default=TILE_UM,
        help="Tile side in um, made the nearest even number of pixels at the scan's "
        f"pixel size (default: {TILE_UM:g})",
    )
    parser.add_argument(
        "--bin-y-um",
        type=float,
        default=BIN_Y_UM,
        help="Bin length along the stripe in um, made a whole number of tiles "
        f"(default: {BIN_Y_UM:g})",
    )
    parser.add_argument(
        "--bin-x-um",
        type=float,
        help="Bin width across the stripe in um, made a whole number of tiles and "
        "no wider than the stripe (default: every tile that fits across, which is "
        "11 tiles, 110 um, at 21 nm per pixel)",
    )
    reject = parser.add_mutually_exclusive_group()
    reject.add_argument(
        "--reject-sigma",
        type=float,
        default=REJECT_SIGMA,
        metavar="K",
        help="Leave a tile out of its bin's value if it lies more than K standard "
        f"deviations from the mean of the bin's tiles (default: {REJECT_SIGMA:g})",
    )
    reject.add_argument(
        "--no-reject",
        action="store_true",
        help="Average every accepted tile of a bin",
    )
    parser.add_argument(
        "--pixel-nm",
        type=float,
        help="Pixel size in nm in both directions, instead of the one in the "
        "scan's header",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.80,
        help="Minimum correlation with the reference in every quadrant of a tile",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Columns that neighbouring boards share (default: 32)",
    )
    parser.add_argument(
        "--boards", type=int, default=4, help="Boards per stripe (default: 4)"
    )
    parser.add_argument(
        "--shape",
        choices=("auto", "square", "circle", "ellipse"),
        default="auto",
        help="Specify only if the reference hole shape is known",
    )
    parser.add_argument(
        "--gpu",
        "--device",
        dest="gpu",
        type=int,
        default=0,
        metavar="N",
        help="GPU to measure on, numbered as nvidia-smi numbers them (default: 0)",
    )
    parser.add_argument(
        "--stripe",
        nargs="+",
        type=int,
        help="Measure only these stripe numbers (default: every stripe)",
        metavar="N",
    )
    parser.add_argument(
        "--segment",
        nargs="+",
        help="Measure only these segments, named <stripe>_<vertical segment> "
        "like 3_1 (default: every segment)",
        metavar="NAME",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Read at most this many pages of each segment, for a short trial",
    )
    parser.add_argument(
        "--edge-strips",
        nargs="?",
        type=int,
        const=EDGE_COLUMNS,
        default=0,
        metavar="COLUMNS",
        help="Also save the outer columns of every merged page, which neighbouring "
        "stripes share, to find the shift between stripes later "
        f"(width when no number is given: {EDGE_COLUMNS} columns per edge)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # By default CUDA puts the fastest GPU first, so its numbers need not be
    # those nvidia-smi shows. Number them by PCI bus, as nvidia-smi does. This
    # holds only if it is set before the first CUDA call of the process.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    start = perf_counter()
    try:
        manifest = measure_stripes(
            args.root,
            args.output,
            reference=args.reference,
            reference_page=args.reference_page,
            methods=args.method,
            tile_um=args.tile_um,
            bin_y_um=args.bin_y_um,
            bin_x_um=args.bin_x_um,
            reject_sigma=None if args.no_reject else args.reject_sigma,
            pixel_nm=args.pixel_nm,
            min_corr=args.min_corr,
            overlap=args.overlap,
            boards=args.boards,
            shape=args.shape,
            device=args.gpu,
            stripes=args.stripe,
            segments=args.segment,
            max_pages=args.max_pages,
            edge_columns=args.edge_strips,
            progress=lambda line: print(line, flush=True),
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"ACTIS stripes: {error}\n")
    for left_out in manifest["incomplete"]:
        print(
            f"{left_out['stripe']}_{left_out['vertical']}: left out, it has boards "
            f"{left_out['boards_found']} and needs 1 to {args.boards}"
        )
    segments = manifest["segments"]
    complete = [s for s in segments if s["status"] == "complete"]
    print(
        f"{len(complete)} of {len(segments)} segments in {args.output.resolve()} "
        f"are complete | this run took {perf_counter() - start:.0f} s"
    )
    return 0 if len(complete) == len(segments) else 1


if __name__ == "__main__":
    raise SystemExit(main())
