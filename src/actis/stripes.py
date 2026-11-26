"""Measure every stripe of a scan and keep each tile's value for a map.

A scan is a folder that holds the board stacks of every stripe, at any depth
below it. Several boards record a stripe side by side, each into its own
multi-page TIFF named <stripe>_<vertical segment>_<board>.tiff, and that name
is all a file is found by. Neighbouring boards image the same few columns. Page by page, the boards are merged with those columns counted
once, the merged page is cut into tiles, the tiles are screened against the
reference, and the accepted tiles are measured.

Every segment gets one .npz of values shaped (page, tile row, tile column).
stripes.json lists them with the geometry a map needs. On request a second
.npz keeps the outer columns of every merged page, which neighbouring stripes
share, so the shift between stripes can be found without reading the scan again.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np

from .files import native
from .fit import array_backend, to_host
from .tiff_benchmark import (
    FITS,
    METHODS,
    Progress,
    Run,
    TileMaps,
    _measure_page,
    _read_page,
    _reference,
)
from .tiff_tiles import prepare_reference, select_tiles

FORMAT = "actis-stripes-1"
MANIFEST = "stripes.json"
NAME = re.compile(r"(\d+)_(\d+)_(\d+)\.tiff?", re.IGNORECASE)
# Read this much of one board before moving to the next, so a hard disk reads
# long runs instead of seeking between the board files at every page.
READ_BLOCK_BYTES = 2**26
# Most pixels screened in one batch. A merged page of four boards fits.
SCREEN_PIXELS = 2**24
# Columns that neighbouring stripes share: the usual width of an edge strip.
EDGE_COLUMNS = 32


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


def _measure_merged_page(boards, index, templates, settings, run, maps, cp):
    """Merge one page of the boards, screen its tiles and measure the accepted ones.

    Returns the agreement at each seam, and how many tiles were screened and accepted.
    """
    tile, overlap = settings["tile_size"], settings["overlap_px"]
    with run.time("merge"):
        parts = [cp.asarray(board) for board in boards]
        seams = seam_agreement(parts, overlap, cp)
        page = merge_boards(parts, overlap, cp)
        tiles, (ny, nx) = cut_tiles(page, tile, cp)
    position = maps.add_page(index, ny, nx)
    batch = max(1, SCREEN_PIXELS // tile**2)
    with run.time("selection"):
        matches = []
        for first in range(0, ny * nx, batch):
            found = select_tiles(
                tiles[first : first + batch], templates, settings["min_corr"], cp
            )
            # From here on a tile is known by its index on the merged page.
            found[:, 0] += first
            matches.append(found)
        matches = np.concatenate(matches)
        if 0 < len(matches) < len(tiles):
            tiles = tiles[cp.asarray(matches[:, 0], dtype=cp.int32)]
    if len(matches):
        maps.file(
            position, matches, *_measure_page(tiles, matches, templates, None, run, cp)
        )
    return seams, ny * nx, len(matches)


def _measure_segment(segment, templates, settings, cp, live, tifffile):
    """Measure every merged page of one segment.

    Returns the arrays to save, the edge strips if they are kept, and a
    summary for the manifest.
    """
    tile, overlap = settings["tile_size"], settings["overlap_px"]
    edge = settings["edge_columns"]
    methods = tuple(settings["methods"])
    run, maps = Run(cp, live, methods), TileMaps(methods)
    notes, seams, left, right, layout = [], [], [], [], None
    tested = accepted = 0
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
            for path, board in zip(segment.files, boards):
                if (board.shape, board.dtype) != layout:
                    raise ValueError(
                        f"Page {index + 1} of {path.name} is {board.shape[1]} x "
                        f"{board.shape[0]} px {board.dtype}, not {layout[0][1]} x "
                        f"{layout[0][0]} px {layout[1]} like the segment's first page"
                    )
            seam, screened, kept = _measure_merged_page(
                boards, index, templates, settings, run, maps, cp
            )
            seams.append(seam)
            tested += screened
            accepted += kept
            if edge:
                # Copies, because a view would keep its whole page in memory.
                left.append(boards[0][:, :edge].copy())
                right.append(boards[-1][:, -edge:].copy())
            live.stage = "reading the board files"
    if layout is None:
        raise ValueError(f"{segment.files[0].name} has no pages")
    (rows, columns), count = layout[0], len(segment.files)
    arrays = maps.arrays()
    if not any(name in FITS for name in methods):
        # No method needed a blur, so there is none to report.
        del arrays["blur_sigma_px"], arrays["blur_usable"]
    seams = np.asarray(seams, dtype=np.float32).reshape(len(maps.grids), -1)
    arrays.update(
        seam_correlation=seams,
        stripe=segment.stripe,
        vertical=segment.vertical,
        tile_size=tile,
        overlap_px=overlap,
        boards=count,
        board_columns=columns,
    )
    summary = {
        "pages": len(maps.grids),
        "board_shape": [rows, columns],
        "page_shape": [rows, count * columns - (count - 1) * overlap],
        "tile_grid": list(maps.grids[0]),
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
        "methods": run.methods(),
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
        # A run from before edge strips existed kept none.
        known = {"edge_columns": 0, **earlier.get("settings", {})}
        if earlier.get("format") != FORMAT or known != settings:
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
    os.replace(partial, path)


def _line(entry):
    name = f"{entry['stripe']}_{entry['vertical']}"
    if entry["status"] != "complete":
        return f"{name}: failed. {entry['error']}"
    rows, columns = entry["page_shape"]
    values = ", ".join(
        f"{method} "
        + ("n/a" if row["mean"] is None else f"{row['mean']:.5g} {row['unit']}")
        for method, row in entry["methods"].items()
    )
    seams = " ".join(
        "n/a" if seam is None else f"{seam:.2f}"
        for seam in entry["seam_correlation_p90"]
    )
    seconds = entry["seconds"]
    return " ".join(
        [
            f"{name}: {entry['pages']:,} pages of {columns} x {rows} px,",
            f"{entry['accepted_tiles']:,} of {entry['tested_tiles']:,} tiles accepted,",
            f"{values}, seams {seams or 'n/a'},",
            f"{seconds['total']:.0f} s ({seconds['read_wait']:.0f} s waiting for the disk)",
            *entry["notes"],
        ]
    )


def measure_stripes(
    root,
    output,
    *,
    reference,
    reference_page=0,
    methods=("raw",),
    tile_size=128,
    min_corr=0.80,
    overlap=32,
    boards=4,
    shape="auto",
    device=0,
    stripes=None,
    max_pages=None,
    edge_columns=0,
    progress=None,
):
    """Measure the segments under `root` into `output` and return the manifest.

    A segment that `output` already holds from an earlier run with the same
    settings is left as it is, so a stopped run continues where it ended.
    `edge_columns` above 0 also saves that many columns from the left and
    from the right edge of every merged page.
    """
    try:
        import tifffile
    except ImportError:
        raise RuntimeError(
            'Install the TIFF/CUDA dependencies with pip install -e ".[gpu]"'
        ) from None
    methods = tuple(dict.fromkeys(methods))
    if not methods or any(name not in METHODS for name in methods):
        raise ValueError(f"Choose methods from: {', '.join(METHODS)}")
    if tile_size < 16 or tile_size % 2 or not 0 < min_corr <= 1:
        raise ValueError(
            "Use an even tile size of at least 16 and a minimum correlation in (0,1]"
        )
    if min(overlap, edge_columns, device, reference_page) < 0 or boards < 1:
        raise ValueError(
            "Overlap, edge columns, device and reference page must be nonnegative, "
            "with at least one board"
        )
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    root, output = Path(root).resolve(), Path(output).resolve()
    reference = Path(reference).resolve()
    segments, incomplete = find_segments(root, boards)
    if stripes is not None:
        segments = [s for s in segments if s.stripe in set(stripes)]
    if not segments:
        raise ValueError(
            f"No segment with boards 1 to {boards} to measure under {root}"
        )
    settings = native(
        {
            "reference": str(reference),
            "reference_page": reference_page,
            "methods": methods,
            "tile_size": tile_size,
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
        if device >= cp.cuda.runtime.getDeviceCount():
            raise ValueError(f"CUDA device {device} is unavailable")
        resources.enter_context(cp.cuda.Device(device))
        gpu = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
        live.location, live.stage = f"CUDA {device}: {gpu}", "preparing the reference"
        live.tell()
        pixels = _reference(reference, reference_page, tifffile)
        templates = prepare_reference(pixels, tile_size, shape, cp)
        for template in templates:
            # Compile the kernels, and stop now if the reference cannot match itself.
            tile = template.pixels.astype(pixels.dtype)[None]
            matches = select_tiles(tile, [template], min_corr, cp)
            if not len(matches):
                raise ValueError("Reference tile failed its own correlation check")
            _measure_page(tile, matches, [template], None, Run(cp, live, methods), cp)
        header = root / "header.xml"
        manifest = {
            "format": FORMAT,
            "root": str(root),
            "header": str(header) if header.is_file() else None,
            "settings": settings,
            "gpu": gpu,
            "reference_shapes": [t.region.shape for t in templates],
            "reference_pitch_px": [float(t.region.pitch) for t in templates],
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
            try:
                arrays, edges, summary = _measure_segment(
                    segment, templates, settings, cp, live, tifffile
                )
            except (OSError, ValueError, tifffile.TiffFileError) as error:
                entry.update(status="failed", error=str(error))
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
    manifest.setdefault("segments", [entries[k] for k in sorted(entries)])
    return native(manifest)


def main(argv=None):
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
        choices=METHODS,
        default=["raw"],
        help="Methods to run on every accepted tile (default: raw)",
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
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--stripe",
        nargs="+",
        type=int,
        help="Measure only these stripe numbers (default: every stripe)",
        metavar="N",
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
    args = parser.parse_args(argv)
    start = perf_counter()
    try:
        manifest = measure_stripes(
            args.root,
            args.output,
            reference=args.reference,
            reference_page=args.reference_page,
            methods=args.method,
            tile_size=args.tile_size,
            min_corr=args.min_corr,
            overlap=args.overlap,
            boards=args.boards,
            shape=args.shape,
            device=args.device,
            stripes=args.stripe,
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
