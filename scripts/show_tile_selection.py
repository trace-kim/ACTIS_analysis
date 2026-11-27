"""Show which tiles of one page pass the reference screen, and each tile's score.

The page is cut into the same tiles as measure_stripes.py cuts it and screened
the same way. Give the four board files of a segment to see the merged page of
the stripe run, or one file to see that file alone. Pages count from 0, like
page_index in the stripe results. Hover over a tile for its score, raw mean
and shift.

The stripe run stacks the pages of a segment and cuts its tiles from that
strip, so on most pages the first tile row begins below the top. A tile that
runs on into the page before or after is not shown. The pixel size comes from
the header of the scan the files lie in, as in the stripe run, unless
--pixel-nm gives it.

All four boards of segment 1_1, page 120.
    python scripts/show_tile_selection.py D:\\ACTIS\\scan\\RAW\\S1\\1_1_?.tiff --reference holes.tiff --page 120

One file, written to a PNG instead of a window (for a machine without a display).
    python scripts/show_tile_selection.py 1_1_2.tiff --reference holes.tiff --page 120 --save page120.png
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.fit import array_backend
from actis.stripes import (
    SCREEN_PIXELS,
    TILE_UM,
    Tiling,
    cut_tiles,
    merge_boards,
    scan_pixel_size,
    seam_columns,
)
from actis.tiff_benchmark import _read_page, _reference
from actis.tiff_tiles import prepare_reference, score_tiles

ACCEPTED = "#4cc3ff"  # sky blue and orange stay apart for red-green colour blindness
REJECTED = "#ff8a1f"
SEAM = "#ffe066"


def read_page(path, page, tifffile):
    """One page of a TIFF, or a grayscale PNG, with the bright holes high."""
    if path.suffix.lower() == ".png":
        if page:
            raise ValueError(f"{path.name} is a PNG, which has only page 0")
        return _reference(path, 0, tifffile)
    with tifffile.TiffFile(path) as stack:
        try:
            return _read_page(stack.pages[page])
        except IndexError:
            raise ValueError(f"{path.name} has no page {page}") from None


def pixel_size(paths, pixel_nm):
    """The pixel size in µm that the stripe run would use for these files.

    A board stack lies a few folders below its scan, as in scan\\RAW\\S1, so
    the nearest folders above the first file are asked for a header.
    """
    if pixel_nm:
        return scan_pixel_size(paths[0].parent, pixel_nm)[0]
    problems = []
    for folder in paths[0].resolve().parents[:4]:
        try:
            return scan_pixel_size(folder)[0]
        except ValueError as error:
            problems.append(error)
    raise ValueError(
        f"No folder above {paths[0].name} gives a pixel size ({problems[-1]}). "
        "Give --pixel-nm"
    )


def screen(paths, args, tifffile):
    """Merge the page of every file, cut it into tiles and score every tile.

    Returns the page, the seams, the references, the tile values and where
    the tiles lie on the page: (first row, first column, tile side).
    """
    boards = [read_page(path, args.page, tifffile) for path in paths]
    for path, board in zip(paths, boards):
        if (board.shape, board.dtype) != (boards[0].shape, boards[0].dtype):
            raise ValueError(
                f"{path.name} is {board.shape[1]} x {board.shape[0]} px {board.dtype}, "
                f"not {boards[0].shape[1]} x {boards[0].shape[0]} px {boards[0].dtype} "
                f"like {paths[0].name}"
            )
    if len(boards) > 1 and boards[0].shape[1] <= args.overlap:
        raise ValueError(f"The boards are no wider than the overlap of {args.overlap}")
    page = merge_boards(boards, args.overlap)
    seams = seam_columns(len(boards), boards[0].shape[1], args.overlap)
    # The bins do not matter here, only where the tiles lie.
    tiling = Tiling.from_um(args.tile_um, args.tile_um, None, pixel_size(paths, args.pixel_nm))
    side = tiling.tile
    if side < 16:
        raise ValueError(f"Tiles of {args.tile_um:g} um are {side} px. Use at least 16 px")
    left = tiling.across(page.shape[1]).first_column
    # Tile rows begin at multiples of the tile side in the stacked pages.
    top = -args.page * page.shape[0] % side
    window = page[top:, left:]
    cp = array_backend("cuda")
    if args.gpu >= cp.cuda.runtime.getDeviceCount():
        raise ValueError(f"There is no GPU {args.gpu}")
    with cp.cuda.Device(args.gpu):
        pixels = _reference(args.reference.resolve(), args.reference_page, tifffile)
        templates = prepare_reference(pixels, side, args.shape, cp)
        tiles, grid = cut_tiles(cp.asarray(window), side, cp)
        # The batch size of the stripe run, so a whole page fits on the GPU.
        batch = max(1, SCREEN_PIXELS // side**2)
        parts = [
            score_tiles(tiles[first : first + batch], templates, cp)
            for first in range(0, len(tiles), batch)
        ]
        score, reference, dx, dy = (
            cp.concatenate(column).get().reshape(grid) for column in zip(*parts)
        )
        # What the stripe run stores as raw for an accepted tile.
        raw = tiles.mean(axis=(1, 2), dtype=cp.float64).get().reshape(grid)
    tiles_of = dict(score=score, reference=reference, dx=dx, dy=dy, raw=raw)
    return page, seams, templates, tiles_of, (top, left, side)


def draw(page, seams, tiles_of, templates, title, placed, args):
    """The page in grey, each tile outlined or shaded by its verdict, with its score."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    top, left, side = placed
    rows, columns = page.shape
    score = tiles_of["score"]
    ny, nx = score.shape
    accepted = score >= args.min_corr
    inches = min(16 / columns, 9 / rows)  # per pixel, so the page fits a screen
    fig, ax = plt.subplots(
        figsize=(max(7, columns * inches), rows * inches + 1.3), layout="constrained"
    )
    low, high = np.percentile(page, (0.5, 99.5))
    ax.imshow(
        page, cmap="gray", vmin=low, vmax=max(high, low + 1), interpolation="nearest"
    )

    def boxes(chosen):
        return [
            Rectangle((left + c * side - 0.5, top + r * side - 0.5), side, side)
            for r, c in zip(*np.nonzero(chosen))
        ]

    shade = to_rgba(REJECTED, 0.2)
    ax.add_collection(
        PatchCollection(boxes(~accepted), facecolor=shade, edgecolor=REJECTED, lw=0.8)
    )
    ax.add_collection(
        PatchCollection(boxes(accepted), facecolor="none", edgecolor=ACCEPTED, lw=1.1)
    )
    # The columns at both edges that no tile holds, and the rows of tiles that
    # run on into the page before or after.
    left_over = dict(fill=False, hatch="///", edgecolor="0.65", lw=0)
    right, bottom = left + nx * side, top + ny * side
    for x, y, w, h in (
        (0, 0, left, rows),
        (right, 0, columns - right, rows),
        (left, 0, nx * side, top),
        (left, bottom, nx * side, rows - bottom),
    ):
        if w > 0 and h > 0:
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), w, h, **left_over))
    for seam in seams:
        ax.axvline(seam - 0.5, color=SEAM, lw=0.9, ls=(0, (6, 4)))

    points = side * inches * 72  # a tile's width on the figure
    size = float(np.clip(points / 3.2, 3, 10))
    outline = [patheffects.withStroke(linewidth=max(1.5, size / 3.5), foreground="black")]
    for (r, c), value in np.ndenumerate(score):
        ax.text(
            left + (c + 0.5) * side - 0.5,
            top + (r + 0.5) * side - 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=size,
            color="white",
            path_effects=outline,
        )

    def where(x, y):
        column, row = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        text = f"x={column} y={row}"
        r, c = (row - top) // side, (column - left) // side
        if not (0 <= r < ny and 0 <= c < nx):
            return text + "  (in no tile of this page)"
        verdict = "accepted" if accepted[r, c] else "rejected"
        text += (
            f"  tile ({r}, {c}): score {score[r, c]:.3f}, {verdict}, "
            f"raw mean {tiles_of['raw'][r, c]:.1f} DN, "
            f"shift ({tiles_of['dx'][r, c]:+.2f}, {tiles_of['dy'][r, c]:+.2f}) px"
        )
        if len(templates) > 1:
            text += f", reference {tiles_of['reference'][r, c]}"
        return text

    ax.format_coord = where
    ax.set_title(
        f"{title}, page {args.page}\n"
        f"{accepted.sum()} of {accepted.size} tiles accepted.  Score = lowest of the "
        f"four quadrant correlations with the reference, accepted from {args.min_corr:.2f}",
        fontsize=10,
    )
    handles = [
        Patch(facecolor="none", edgecolor=ACCEPTED, label="accepted"),
        Patch(facecolor=shade, edgecolor=REJECTED, label="rejected"),
        Patch(fill=False, hatch="///", edgecolor="0.65", label="in no tile of this page"),
    ]
    if seams:
        handles.append(Line2D([], [], color=SEAM, ls=(0, (6, 4)), label="board seam"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    return fig, inches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "image",
        nargs="+",
        help="TIFF or PNG. Several files, or a pattern such as 1_1_?.tiff, are the "
        "boards of one segment, left to right, merged as the stripe run merges them",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Holes-only TIFF or PNG, the one given to measure_stripes.py",
    )
    parser.add_argument(
        "--page", type=int, default=0, help="Zero-based page of the TIFFs (default: 0)"
    )
    parser.add_argument(
        "--reference-page",
        type=int,
        default=0,
        help="Zero-based page of a reference TIFF (default: 0)",
    )
    parser.add_argument(
        "--tile-um",
        type=float,
        default=TILE_UM,
        help=f"Tile side in um, as given to measure_stripes.py (default: {TILE_UM:g})",
    )
    parser.add_argument(
        "--pixel-nm",
        type=float,
        help="Pixel size in nm, instead of the one in the scan's header",
    )
    parser.add_argument(
        "--min-corr",
        type=float,
        default=0.80,
        help="Score from which a tile is accepted (default: 0.80)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Columns that neighbouring boards share (default: 32)",
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
        help="GPU to use, numbered as nvidia-smi numbers them (default: 0)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Write the figure to this PNG instead of opening a window",
    )
    args = parser.parse_args(argv)
    if not args.tile_um > 0 or not 0 < args.min_corr <= 1:
        parser.error("Use a positive --tile-um and a --min-corr in (0, 1]")
    if args.pixel_nm is not None and not args.pixel_nm > 0:
        parser.error("--pixel-nm must be positive")
    if min(args.page, args.reference_page, args.overlap, args.gpu) < 0:
        parser.error("Pages, overlap and GPU must be nonnegative")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # as in measure_stripes.py
    # PowerShell and cmd pass a pattern through unexpanded.
    paths = [Path(p) for arg in args.image for p in (sorted(glob.glob(arg)) or [arg])]

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        parser.exit(1, "No display to open a window on. Pass --save FILE.png\n")
    try:
        import tifffile
    except ImportError:
        parser.exit(1, 'Install the TIFF/CUDA dependencies with pip install -e ".[gpu]"\n')
    try:
        page, seams, templates, tiles_of, placed = screen(paths, args, tifffile)
    except (OSError, ValueError, RuntimeError, tifffile.TiffFileError) as error:
        parser.exit(1, f"Tile selection: {error}\n")

    score = tiles_of["score"]
    accepted = score >= args.min_corr
    close = accepted & (score < args.min_corr + 0.05)
    for t in templates:
        print(
            f"Reference: {t.region.shape} holes, pitch {t.region.pitch:.2f} px, "
            f"shifts searched up to {len(t.search) // 2} px"
        )
    top, left, side = placed
    print(
        f"{page.shape[1]} x {page.shape[0]} px page, tiles of {side} px from row "
        f"{top} and column {left}: {accepted.sum()} of {accepted.size} tiles "
        f"accepted, {close.sum()} of them within 0.05 of the {args.min_corr:.2f} "
        "threshold"
    )
    title = ", ".join(path.name for path in paths)
    fig, inches = draw(page, seams, tiles_of, templates, title, placed, args)
    if args.save:
        # About one image pixel per figure pixel, within reason.
        fig.savefig(args.save, dpi=float(np.clip(1 / inches, 100, 400)))
        print(f"Saved {args.save.resolve()}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
