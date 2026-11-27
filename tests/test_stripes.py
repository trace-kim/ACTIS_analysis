import json
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from actis.fit import array_backend
from actis.stripes import (
    MANIFEST,
    Tiling,
    _board_pages,
    bin_tiles,
    find_segments,
    main,
    measure_stripes,
    merge_boards,
    scan_pixel_size,
    seam_agreement,
    seam_columns,
    tile_pixels,
)
from actis.synthetic import render_repeats
from actis.tiff_benchmark import METHODS

SIDE = 96
PIXEL_NM = 21.0
TILE_UM = SIDE * PIXEL_NM / 1000  # 96 px at 21 nm
# Not a multiple of the tile, so rows of tiles run on from one page into the next.
PAGE_ROWS = 150
# Four boards of 128 columns sharing 32: 416 columns hold 4 tiles, from column 16.
WIDTH, X0 = 4 * SIDE + 32, 16


@pytest.fixture(scope="module")
def cuda():
    try:
        return array_backend("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))


class HeaderError(Exception):
    pass


@pytest.fixture
def header(monkeypatch):
    """Stand in for actis.header, which only the workstation has.

    It finds Header.xml in the folder it is given and reads `size` from it.
    """
    module = types.ModuleType("actis.header")
    module.HeaderError = HeaderError
    module.size = [PIXEL_NM / 1000] * 2

    def find_header(folder):
        path = Path(folder) / "Header.xml"
        if not path.is_file():
            raise HeaderError(f"no Header.xml in {folder}")
        return path

    def parse_header(path):
        if path.read_text() == "broken":
            raise HeaderError("unreadable")
        return SimpleNamespace(pixel_x=module.size[0], pixel_y=module.size[1])

    module.find_header, module.parse_header = find_header, parse_header
    monkeypatch.setitem(sys.modules, "actis.header", module)
    return module


@pytest.fixture(scope="module")
def holes():
    frames, _ = render_repeats(
        shape="square",
        image_shape=(SIDE, SIDE),
        pitch_px=8,
        cd_nm=86,
        n_repeats=1,
        nuisance=False,
        supersample=4,
    )
    return frames[0]


def split(page, boards=4, overlap=32):
    """Cut a merged page into boards that share `overlap` columns with their neighbours."""
    columns = (page.shape[-1] + (boards - 1) * overlap) // boards
    step = columns - overlap
    return [page[..., k * step : k * step + columns] for k in range(boards)]


def merged_strip(tile, where, pages, dim=()):
    """`pages` merged pages, stacked, with the hole tile at the (tile row, tile
    column) places in `where`, 0.4 as bright at the places in `dim`.

    Tile columns begin 16 columns in. Brightness rises from left to right, so
    a tile's mean tells exactly which columns it holds.
    """
    strip = np.full((pages * PAGE_ROWS, WIDTH), 10.0)
    for row, column in where:
        rows = slice(row * SIDE, (row + 1) * SIDE)
        columns = slice(X0 + column * SIDE, X0 + (column + 1) * SIDE)
        strip[rows, columns] = tile * (0.4 if (row, column) in dim else 1.0)
    return np.rint(strip * np.linspace(0.6, 1.0, WIDTH)).astype(np.uint8)


def tile_means(strip):
    rows = len(strip) // SIDE
    tiles = strip[: rows * SIDE, X0 : X0 + 4 * SIDE]
    return tiles.reshape(rows, SIDE, 4, SIDE).mean(axis=(1, 3))


def held(where, rows):
    """Which tiles of a grid `rows` tall hold the hole tile."""
    grid = np.zeros((rows, 4), dtype=bool)
    for row, column in where:
        if row < rows:
            grid[row, column] = True
    return grid


def pages_of(strip):
    return strip.reshape(-1, PAGE_ROWS, strip.shape[1])


def write_boards(folder, stem, strip, tifffile):
    folder.mkdir(parents=True, exist_ok=True)
    for number, board in enumerate(split(pages_of(strip)), 1):
        path = folder / f"{stem}_{number}.tiff"
        tifffile.imwrite(path, board, photometric="minisblack")


def expected_bins(values, used, size, first=0):
    """Bin means written out plainly, to check bin_tiles against."""
    tall, wide = size
    down, across = -(-len(values) // tall), (values.shape[1] - first) // wide
    out = np.full((down, across), np.nan)
    for r in range(down):
        for c in range(across):
            rows = slice(r * tall, (r + 1) * tall)
            columns = slice(first + c * wide, first + (c + 1) * wide)
            if used[rows, columns].any():
                out[r, c] = values[rows, columns][used[rows, columns]].mean()
    return out


def test_boards_merge_with_shared_columns_counted_once():
    rng = np.random.default_rng(5)
    page = rng.integers(0, 255, (40, 416), dtype=np.uint8)
    boards = split(page)
    assert [b.shape for b in boards] == [(40, 128)] * 4
    np.testing.assert_array_equal(merge_boards(boards, 32), page)
    # Each board begins 16 columns into the 32 it shares with the one before.
    assert seam_columns(4, 128, 32) == [112, 208, 304]
    np.testing.assert_allclose(seam_agreement(boards, 32), 1, atol=1e-6)
    assert abs(seam_agreement(boards[::-1], 32)).max() < 0.2

    odd = rng.integers(0, 255, (8, 50), dtype=np.uint16)
    np.testing.assert_array_equal(merge_boards(split(odd, 3, 5), 5), odd)
    assert seam_columns(3, 20, 5) == [18, 33]
    np.testing.assert_array_equal(merge_boards(split(odd, 2, 0), 0), odd)
    assert len(seam_agreement(split(odd, 2, 0), 0)) == 0
    np.testing.assert_array_equal(merge_boards([odd], 32), odd)
    assert seam_columns(1, 50, 32) == []


def test_segments_are_found_by_file_name(tmp_path):
    # Stripe folders inside a folder, beside the scan's own files, and no folder at all.
    names = ["RAW/S1/1_1_%d.tiff", "RAW/S1/1_2_%d.tiff", "RAW/S10/10_1_%d.TIF"]
    names += ["S2/2_1_%d.tiff", "3_1_%d.tiff"]
    for name in names:
        for board in range(1, 5):
            path = tmp_path / (name % board)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    (tmp_path / "S2" / "2_1_3.tiff").unlink()
    for other in ("header.xml", "RAW/S1/notes.txt", "RAW/S1/1_1_1_copy.tiff"):
        (tmp_path / other).touch()
    segments, incomplete = find_segments(tmp_path)
    # In number order: stripe 10 follows stripe 3, not stripe 1.
    assert [(s.stripe, s.vertical, s.name) for s in segments] == [
        (1, 1, "1_1"),
        (1, 2, "1_2"),
        (3, 1, "3_1"),
        (10, 1, "10_1"),
    ]
    assert [p.name for p in segments[3].files] == [f"10_1_{b}.TIF" for b in range(1, 5)]
    assert incomplete == [{"stripe": 2, "vertical": 1, "boards_found": [1, 2, 4]}]
    assert len(find_segments(tmp_path, boards=3)[0]) == 0
    assert [s.name for s in find_segments(tmp_path / "RAW")[0]] == [
        "1_1",
        "1_2",
        "10_1",
    ]
    assert [s.name for s in find_segments(tmp_path / "RAW" / "S10")[0]] == ["10_1"]

    (tmp_path / "S2" / "1_1_4.tif").touch()
    with pytest.raises(ValueError, match="stored twice"):
        find_segments(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No <stripe>_<vertical>_<board>"):
        find_segments(tmp_path / "empty")
    with pytest.raises(ValueError, match="does not exist"):
        find_segments(tmp_path / "missing")


@pytest.mark.parametrize("block_bytes", [1, 5 * 6 * 2, 2**26])
def test_boards_are_read_together_up_to_the_shortest(
    tmp_path, monkeypatch, block_bytes
):
    tifffile = pytest.importorskip("tifffile")
    monkeypatch.setattr("actis.stripes.READ_BLOCK_BYTES", block_bytes)
    files = []
    for board, count in enumerate((7, 4, 7)):
        pages = 100 * board + np.arange(count, dtype=np.uint8)[:, None, None]
        files.append(tmp_path / f"1_1_{board + 1}.tiff")
        tifffile.imwrite(
            files[-1], np.broadcast_to(pages, (count, 5, 6)), photometric="minisblack"
        )

    def read(chosen, max_pages=None):
        seconds, notes = Counter(), []
        rows = [
            (index, [int(page[0, 0]) for page in boards])
            for index, boards in _board_pages(
                chosen, max_pages, seconds, notes, tifffile
            )
        ]
        assert seconds["read"] > 0 and seconds["read_wait"] > 0
        return rows, notes

    rows, notes = read(files)
    assert rows == [(k, [k, 100 + k, 200 + k]) for k in range(4)]
    assert len(notes) == 1 and notes[0].startswith("1_1_2.tiff ended after 4 pages")
    assert read(files[::2]) == ([(k, [k, 200 + k]) for k in range(7)], [])
    assert read(files, max_pages=3) == (rows[:3], [])


def test_pixel_size_comes_from_the_header(tmp_path, header, monkeypatch):
    path = tmp_path / "Header.xml"
    path.write_text("<header/>")
    assert scan_pixel_size(tmp_path) == ((0.021, 0.021), path)
    header.size[:] = [0.0211, 0.0209]
    assert scan_pixel_size(tmp_path) == ((0.0211, 0.0209), path)
    # A size that is given wins, for both directions.
    assert scan_pixel_size(tmp_path, pixel_nm=20) == ((0.02, 0.02), path)
    header.size[:] = [21, 21]  # nm taken for µm
    with pytest.raises(ValueError, match="outside 0.005 to 0.1 um"):
        scan_pixel_size(tmp_path)
    path.write_text("broken")
    with pytest.raises(ValueError, match="Cannot read the pixel size .*unreadable"):
        scan_pixel_size(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Cannot find the header .*no Header.xml"):
        scan_pixel_size(empty)
    assert scan_pixel_size(empty, pixel_nm=21) == ((0.021, 0.021), None)
    # As on a PC without the workstation's module.
    monkeypatch.setitem(sys.modules, "actis.header", None)
    with pytest.raises(ValueError, match="needs actis.header"):
        scan_pixel_size(tmp_path)
    assert scan_pixel_size(tmp_path, pixel_nm=21) == ((0.021, 0.021), None)


def test_tiles_and_bins_are_centred_across_the_stripe():
    pixel = (0.021, 0.021)
    assert tile_pixels(10, pixel) == 476 and tile_pixels(10, (0.0215,) * 2) == 466
    tiling = Tiling.from_um(10, 100, None, pixel)
    assert (tiling.tile, tiling.bin_rows, tiling.bin_columns) == (476, 10, None)
    # 5,408 merged columns hold 11 tiles and leave 172, 86 at each edge.
    assert tiling.across(5408) == (86, 11, 0, 1, 11)
    assert Tiling.from_um(10, 100, 110, pixel).across(5408) == (86, 11, 0, 1, 11)
    # Two bins of 5 tiles leave one tile, at the right. Three of 3 leave one at each edge.
    assert Tiling.from_um(10, 100, 50, pixel).across(5408) == (86, 11, 0, 2, 5)
    assert Tiling.from_um(10, 100, 30, pixel).across(5408) == (86, 11, 1, 3, 3)
    # At 20 nm a pixel the stripe is 108 µm: too narrow for bins of 110 µm,
    # and bins of the whole stripe take the 10 tiles that fit.
    with pytest.raises(ValueError, match="do not fit across the stripe"):
        Tiling.from_um(10, 100, 110, (0.02, 0.02)).across(5408)
    assert Tiling.from_um(10, 100, None, (0.02, 0.02)).across(5408) == (
        204,
        10,
        0,
        1,
        10,
    )
    with pytest.raises(ValueError, match="wider than the stripe"):
        Tiling(600, pixel, 1).across(500)


def test_bins_average_their_tiles_without_the_far_ones():
    nan = np.nan
    values = np.array(
        [
            [100, 10, 10, 10, 1, 2, 3],
            [100, 10, 10, 30, 4, 5, nan],
            [100, nan, nan, nan, nan, nan, nan],
            [100, nan, nan, nan, nan, 7, nan],
            [100, 2, 4, nan, nan, nan, nan],
        ]
    )
    # Bins of 2 x 3 tiles from tile column 1: column 0 is in none, and the
    # last bin row holds one tile row.
    means, counts, used = bin_tiles(values, (2, 3), 1, reject_sigma=1.5)
    # 30 lies 2.04 standard deviations above its bin's mean of 13.3. A lone tile stays.
    np.testing.assert_allclose(means, [[10, 3], [nan, 7], [3, nan]])
    np.testing.assert_array_equal(counts, [[5, 5], [0, 1], [2, 0]])
    expected = np.isfinite(values)
    expected[:, 0] = expected[1, 3] = False
    np.testing.assert_array_equal(used, expected)
    assert means.dtype == np.float32
    for sigma in (None, 2.1):
        means, counts, used = bin_tiles(values, (2, 3), 1, reject_sigma=sigma)
        assert means[0, 0] == pytest.approx(80 / 6) and counts[0, 0] == 6 and used[1, 3]
    # No tile of n can lie more than (n - 1) / sqrt(n) standard deviations
    # from their mean, which is 1.5 for four tiles.
    one_far = np.array([[1.0, 1.0], [1.0, 9.0]])
    assert bin_tiles(one_far, (2, 2), reject_sigma=1.5)[1][0, 0] == 4
    assert bin_tiles(one_far, (2, 2), reject_sigma=1.49)[1][0, 0] == 3


def test_every_stripe_is_measured_in_tiles_and_bins(
    cuda, holes, header, tmp_path, monkeypatch
):
    tifffile = pytest.importorskip("tifffile")
    tile = holes // 256
    root, out = tmp_path / "scan", tmp_path / "out"
    root.mkdir()
    (root / "Header.xml").write_text("<header/>")
    reference = tmp_path / "reference.tiff"
    tifffile.imwrite(reference, holes, photometric="minisblack")
    # Tile rows 1 and 3 run on from one page into the next, tile columns 1 to
    # 3 each lie across a seam between two boards, and one tile is dim.
    where = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    where += [(2, 1), (2, 3), (3, 0), (3, 2)]
    dim = {(1, 3)}
    first = merged_strip(tile, where, 3, dim)
    later = [(0, 3), (1, 0), (2, 2)]
    second = merged_strip(tile, later, 3)
    write_boards(root / "RAW" / "S1", "1_1", first, tifffile)
    write_boards(root / "RAW" / "S1", "1_2", second, tifffile)
    short = split(pages_of(second)[:2])[2]
    tifffile.imwrite(
        root / "RAW" / "S1" / "1_2_3.tiff", short, photometric="minisblack"
    )
    write_boards(root / "RAW" / "S2", "2_1", first, tifffile)
    (root / "RAW" / "S2" / "2_1_4.tiff").unlink()
    write_boards(root / "RAW" / "S3", "3_1", first, tifffile)
    narrow = split(pages_of(first))[1][..., :120]
    tifffile.imwrite(
        root / "RAW" / "S3" / "3_1_2.tiff", narrow, photometric="minisblack"
    )

    lines = []
    # Bins two tiles long and the whole stripe wide.
    kwargs = {"reference": reference, "tile_um": TILE_UM, "shape": "square"}
    kwargs["bin_y_um"] = 2 * TILE_UM
    manifest = measure_stripes(root, out, progress=lines.append, **kwargs)
    assert manifest == json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["format"] == "actis-stripes-2"
    assert manifest["header"] == str(root / "Header.xml")
    assert manifest["pixel_source"] == "header"
    assert manifest["settings"]["pixel_um"] == [0.021, 0.021]
    assert manifest["settings"]["methods"] == ["raw"]
    assert (manifest["tile_size"], manifest["bin_rows"]) == (SIDE, 2)
    assert manifest["gpu_index"] == 0 and manifest["gpu_pci_bus"]
    assert any(line.startswith("GPU 0: ") and "PCI bus" in line for line in lines)
    assert any(
        line.startswith("Pixels of 21 x 21 nm. Tiles of 96 px")
        and "Bins of 2 tiles" in line
        and "by 4 tiles" in line
        for line in lines
    )
    assert manifest["incomplete"] == [
        {"stripe": 2, "vertical": 1, "boards_found": [1, 2, 3]}
    ]
    one, two, three = manifest["segments"]
    assert [
        (s["stripe"], s["vertical"], s["status"]) for s in manifest["segments"]
    ] == [
        (1, 1, "complete"),
        (1, 2, "complete"),
        (3, 1, "failed"),
    ]
    assert "3_1_2.tiff is 120 x 150 px" in three["error"] and "result" not in three
    assert not (out / "3_1.npz").exists()
    assert one["files"] == [f"RAW/S1/1_1_{b}.tiff" for b in range(1, 5)]
    assert (one["pages"], one["board_shape"], one["page_shape"]) == (
        3,
        [150, 128],
        [150, 416],
    )
    # 450 rows hold 4 rows of tiles and leave 66. 416 columns hold 4 tiles and
    # leave 16 at each edge.
    assert (one["tile_grid"], one["tile_x0"], one["rows_left"]) == ([4, 4], 16, 66)
    assert (one["bin_tiles"], one["bin_grid"]) == ([2, 4], [2, 1])
    assert one["seam_columns"] == [112, 208, 304]
    assert (one["tested_tiles"], one["accepted_tiles"]) == (16, 11)
    # Tile rows 0 and 2 lie inside a page, rows 1 and 3 across two.
    assert one["accepted_by_rows"] == {"within_pages": [5, 8], "across_pages": [6, 8]}
    assert one["notes"] == [] and min(one["seam_correlation_p90"]) > 0.999
    assert one["seconds"]["total"] >= one["seconds"]["read_wait"] > 0
    # Board 3 of 1_2 ends after 2 pages: 300 rows, 3 rows of tiles, and a last
    # bin one tile row long.
    assert (two["pages"], two["tile_grid"], two["rows_left"]) == (2, [3, 4], 12)
    assert two["bin_grid"] == [2, 1] and two["accepted_tiles"] == 3
    assert two["notes"][0].startswith("1_2_3.tiff ended after 2 pages")
    assert any(
        line.startswith(
            "1_1: 3 pages of 416 x 150 px, 11 of 16 tiles accepted (62% within "
            "pages and 75% across pages), 2 x 1 bins, raw "
        )
        for line in lines
    )
    assert any(line.startswith("3_1: failed.") for line in lines)

    for entry, strip, places, faint in (
        (one, first, where, dim),
        (two, second[: 2 * PAGE_ROWS], later, ()),
    ):
        data = np.load(out / entry["result"])
        rows = entry["tile_grid"][0]
        accepted, expected = held(places, rows), tile_means(strip)
        raw = data["tile_raw"]
        assert raw.dtype == np.float32 and raw.shape == (rows, 4)
        np.testing.assert_array_equal(np.isfinite(raw), accepted)
        np.testing.assert_array_equal(data["tile_reference"] >= 0, accepted)
        # Off by one column or row, at a seam or between pages, these would change.
        np.testing.assert_allclose(raw[accepted], expected[accepted], rtol=1e-6)
        used = accepted & ~held(faint, rows)
        np.testing.assert_array_equal(data["tile_raw_used"], used)
        np.testing.assert_allclose(
            data["bin_raw"], expected_bins(expected, used, (2, 4)), rtol=1e-6
        )
        np.testing.assert_array_equal(
            data["bin_raw_tiles"], [[used[r : r + 2].sum()] for r in range(0, rows, 2)]
        )
        np.testing.assert_array_equal(
            data["bin_rows"], [[r, min(r + 2, rows)] for r in range(0, rows, 2)]
        )
        np.testing.assert_array_equal(data["bin_columns"], [[0, 4]])
        np.testing.assert_array_equal(data["page_index"], np.arange(entry["pages"]))
        assert data["seam_correlation"].shape == (entry["pages"], 3)
        assert np.nanmin(data["seam_correlation"]) > 0.999
        assert (int(data["stripe"]), int(data["vertical"])) == (1, entry["vertical"])
        assert (int(data["tile_size"]), int(data["tile_x0"])) == (SIDE, X0)
        assert (int(data["page_rows"]), float(data["reject_sigma"])) == (PAGE_ROWS, 1.5)
        np.testing.assert_array_equal(data["pixel_um"], [0.021, 0.021])
        assert (int(data["overlap_px"]), int(data["boards"])) == (32, 4)
        assert int(data["board_columns"]) == 128
        assert not any(name.startswith("blur") for name in data.files)
        row = entry["methods"]["raw"]
        assert row["mean"] == pytest.approx(expected[accepted].mean(), rel=1e-6)
        assert row["used_tiles"] == used.sum()
        assert row["bins"] == np.isfinite(data["bin_raw"]).sum()
    # The dim tile is left out of its bin, and the six others are averaged.
    assert np.load(out / "1_1.npz")["bin_raw_tiles"].tolist() == [[6], [4]]

    # A second run keeps what is complete and tries the failed segment again.
    measured = []
    original = measure_stripes.__globals__["_measure_segment"]

    def recorded(segment, *args):
        measured.append(segment.name)
        return original(segment, *args)

    monkeypatch.setattr("actis.stripes._measure_segment", recorded)
    again = measure_stripes(root, out, **kwargs)
    assert measured == ["3_1"] and again == manifest
    for changed in (
        {"min_corr": 0.7},
        {"reject_sigma": None},
        {"bin_y_um": 3 * TILE_UM},
        {"pixel_nm": 20},
        {"edge_columns": 32},
    ):
        with pytest.raises(ValueError, match="other settings"):
            measure_stripes(root, out, **{**kwargs, **changed})
    (tmp_path / "old").mkdir()
    earlier = {**manifest, "format": "actis-stripes-1"}
    (tmp_path / "old" / MANIFEST).write_text(json.dumps(earlier), encoding="utf-8")
    with pytest.raises(ValueError, match="format actis-stripes-1"):
        measure_stripes(root, tmp_path / "old", **kwargs)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file.txt").touch()
    with pytest.raises(ValueError, match="new or empty folder"):
        measure_stripes(root, tmp_path / "taken", **kwargs)
    with pytest.raises(ValueError, match="No segment"):
        measure_stripes(root, tmp_path / "none", stripes=[7], **kwargs)

    # Bins two tiles wide, with every accepted tile averaged.
    halves = {**kwargs, "stripes": [1], "bin_x_um": 2 * TILE_UM, "reject_sigma": None}
    assert measure_stripes(root, tmp_path / "x2", **halves)["segments"][0][
        "bin_grid"
    ] == [2, 2]
    data = np.load(tmp_path / "x2" / "1_1.npz")
    accepted = held(where, 4)
    np.testing.assert_array_equal(data["tile_raw_used"], accepted)
    np.testing.assert_array_equal(data["bin_columns"], [[0, 2], [2, 4]])
    np.testing.assert_allclose(
        data["bin_raw"],
        expected_bins(tile_means(first), accepted, (2, 2)),
        rtol=1e-6,
    )
    assert np.isnan(data["reject_sigma"])
    # Three tiles wide leave one tile over, at the right. Five do not fit.
    measure_stripes(root, tmp_path / "x3", **{**halves, "bin_x_um": 3 * TILE_UM})
    assert np.load(tmp_path / "x3" / "1_1.npz")["bin_columns"].tolist() == [[0, 3]]
    with pytest.raises(ValueError, match="do not fit across the stripe"):
        measure_stripes(root, tmp_path / "x5", **{**kwargs, "bin_x_um": 5 * TILE_UM})
    with pytest.raises(ValueError, match="at least one tile"):
        measure_stripes(root, tmp_path / "small", **{**kwargs, "bin_y_um": TILE_UM / 2})
    with pytest.raises(ValueError, match="There is no GPU"):
        gpus = cuda.cuda.runtime.getDeviceCount()
        measure_stripes(root, tmp_path / "gpu", device=gpus, **kwargs)

    # Every method goes through the same tiles, and a trial stops early.
    every = measure_stripes(
        root, tmp_path / "every", methods=METHODS, stripes=[1], max_pages=1, **kwargs
    )
    assert [s["pages"] for s in every["segments"]] == [1, 1]
    data = np.load(tmp_path / "every" / "1_1.npz")
    accepted = held(where, 1)  # 150 rows hold one row of tiles
    for name in METHODS:
        np.testing.assert_array_equal(
            np.isfinite(data[f"tile_{name}"]), accepted, err_msg=name
        )
        assert data[f"bin_{name}"].shape == (1, 1)
    assert data["blur_sigma_px"].shape == (1, 1, 2)
    assert data["blur_tile_row"].tolist() == [0]

    # On request the outer columns of every merged page are kept as stored.
    strips = {**kwargs, "stripes": [1], "edge_columns": 32}
    kept = measure_stripes(root, tmp_path / "edges", **strips)
    assert kept["settings"]["edge_columns"] == 32
    assert "edges" not in one and "save" in one["seconds"]
    for entry, strip in zip(kept["segments"], (first, second[: 2 * PAGE_ROWS])):
        assert entry["edges"] == f"1_{entry['vertical']}_edges.npz"
        pages = pages_of(strip)
        with np.load(tmp_path / "edges" / entry["edges"]) as edges:
            assert edges["left"].dtype == np.uint8
            np.testing.assert_array_equal(edges["left"], pages[:, :, :32])
            np.testing.assert_array_equal(edges["right"], pages[:, :, -32:])
            np.testing.assert_array_equal(edges["page_index"], np.arange(len(pages)))
            assert (int(edges["stripe"]), int(edges["vertical"])) == (
                1,
                entry["vertical"],
            )
        # Keeping the strips does not change the values.
        with np.load(tmp_path / "edges" / entry["result"]) as with_strips:
            for name in ("tile_raw", "bin_raw"):
                np.testing.assert_array_equal(
                    with_strips[name], np.load(out / entry["result"])[name]
                )
    # A segment whose strips are gone is measured again.
    (tmp_path / "edges" / "1_2_edges.npz").unlink()
    del measured[:]
    redone = measure_stripes(root, tmp_path / "edges", **strips)
    assert measured == ["1_2"] and redone["segments"][0] == kept["segments"][0]
    assert (tmp_path / "edges" / "1_2_edges.npz").is_file()
    # The boards are 128 columns wide and share 32, so 96 are a board's own.
    wide = measure_stripes(root, tmp_path / "wide", **{**strips, "edge_columns": 97})
    assert all("Edge strips of 97" in s["error"] for s in wide["segments"])
    with pytest.raises(ValueError, match="nonnegative"):
        measure_stripes(root, tmp_path / "negative", **{**strips, "edge_columns": -1})

    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    arguments = [str(root), "--reference", str(reference), "--tile-um", str(TILE_UM)]
    arguments += ["--bin-y-um", str(2 * TILE_UM), "--shape", "square", "--gpu", "0"]
    arguments += ["--output"]
    assert main([*arguments, str(tmp_path / "cli"), "--stripe", "1"]) == 0
    saved = json.loads((tmp_path / "cli" / MANIFEST).read_text(encoding="utf-8"))
    assert [s["result"] for s in saved["segments"]] == ["1_1.npz", "1_2.npz"]
    np.testing.assert_array_equal(
        np.load(tmp_path / "cli" / "1_1.npz")["bin_raw"],
        np.load(out / "1_1.npz")["bin_raw"],
    )
    assert main([*arguments, str(tmp_path / "cli")]) == 1  # 3_1 cannot be merged
    # Without a header the pixel size must be given.
    (root / "Header.xml").unlink()
    with pytest.raises(SystemExit):
        main([*arguments, str(tmp_path / "cli-none")])
    trial = [*arguments, str(tmp_path / "cli-given"), "--stripe", "1"]
    trial += ["--max-pages", "1", "--pixel-nm", "21", "--no-reject"]
    assert main([*trial, "--edge-strips"]) == 0
    given = json.loads((tmp_path / "cli-given" / MANIFEST).read_text(encoding="utf-8"))
    assert (given["header"], given["pixel_source"]) == (None, "--pixel-nm")
    assert given["settings"]["reject_sigma"] is None
    edges = np.load(tmp_path / "cli-given" / "1_1_edges.npz")
    assert edges["right"].shape == (1, 150, 32)
    with pytest.raises(SystemExit):  # the folder holds strips of 32 columns, not 16
        main([*trial, "--edge-strips", "16"])
    with pytest.raises(SystemExit):  # a threshold, or no rejection, not both
        main([*trial, "--reject-sigma", "2"])


def test_a_failing_method_or_segment_does_not_stop_the_run(
    cuda, holes, tmp_path, monkeypatch
):
    tifffile = pytest.importorskip("tifffile")
    from actis.tiff_tiles import ReferenceTile

    tile = holes // 256
    root = tmp_path / "scan"
    reference = tmp_path / "reference.tiff"
    tifffile.imwrite(reference, holes, photometric="minisblack")
    where = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
    where += [(2, 1), (2, 3), (3, 0), (3, 2)]
    write_boards(root / "RAW" / "S1", "1_1", merged_strip(tile, where, 3), tifffile)
    write_boards(root / "RAW" / "S1", "1_2", merged_strip(tile, where, 3), tifffile)
    kwargs = {"reference": reference, "tile_um": TILE_UM, "shape": "square"}
    kwargs.update(pixel_nm=PIXEL_NM, bin_y_um=2 * TILE_UM, methods="all")

    # The pages are measured in three batches: tile row 0, rows 1 and 2, and
    # row 3. The harmonic fit fails on the six tiles of the second batch.
    harmonic = ReferenceTile.harmonic_cds

    def failing(self, tiles, shifts, cp):
        if len(tiles) == 6:
            raise RuntimeError("boom")
        return harmonic(self, tiles, shifts, cp)

    monkeypatch.setattr(ReferenceTile, "harmonic_cds", failing)
    lines = []
    manifest = measure_stripes(
        root, tmp_path / "out", segments=["1_1"], progress=lines.append, **kwargs
    )
    assert manifest["settings"]["methods"] == list(METHODS)
    assert [s["vertical"] for s in manifest["segments"]] == [1]
    # Every method measured the reference tile, which has one batch of one tile.
    assert set(manifest["reference_values"]) == set(METHODS)
    assert manifest["reference_errors"] == {}
    assert any(line.startswith("On the reference tile itself: raw ") for line in lines)
    entry = manifest["segments"][0]
    assert entry["status"] == "complete", entry.get("traceback") or entry.get("error")
    row = entry["methods"]["harmonic"]
    assert row["failed_batches"] == 1 and row["error"] == "RuntimeError: boom"
    assert "boom" in row["traceback"] and "failing" in row["traceback"]
    assert all(
        entry["methods"][name]["failed_batches"] == 0
        for name in METHODS
        if name != "harmonic"
    )
    assert any("harmonic failed in 1 batches" in line for line in lines)
    data = np.load(tmp_path / "out" / "1_1.npz")
    accepted = held(where, 4)
    np.testing.assert_array_equal(np.isfinite(data["tile_raw"]), accepted)
    # Only the harmonic values of the failed batch are missing.
    lost = accepted.copy()
    lost[[0, 3]] = False
    np.testing.assert_array_equal(np.isfinite(data["tile_harmonic"]), accepted & ~lost)
    np.testing.assert_array_equal(np.isfinite(data["tile_contour"]), accepted)
    assert row["valid"] == (accepted & ~lost).sum()

    # A segment that fails for any reason is recorded, and the run goes on.
    monkeypatch.setattr(ReferenceTile, "harmonic_cds", harmonic)
    original = measure_stripes.__globals__["_measure_segment"]

    def broken(segment, *args):
        if segment.name == "1_1":
            raise TypeError("not a file problem")
        return original(segment, *args)

    monkeypatch.setattr("actis.stripes._measure_segment", broken)
    both = measure_stripes(root, tmp_path / "both", **kwargs)
    first, second = both["segments"]
    assert (first["status"], first["error"]) == ("failed", "TypeError: not a file problem")
    assert "broken" in first["traceback"] and second["status"] == "complete"

    # If the GPU no longer works afterwards, the run stops there.
    monkeypatch.setattr("actis.stripes.gpu_works", lambda cp: False)
    with pytest.raises(RuntimeError, match="GPU 0 stopped working while measuring 1_1"):
        measure_stripes(root, tmp_path / "lost", **kwargs)
    saved = json.loads((tmp_path / "lost" / MANIFEST).read_text(encoding="utf-8"))
    assert [s["status"] for s in saved["segments"]] == ["failed"]

    with pytest.raises(ValueError, match="No segment 9_9 with boards"):
        measure_stripes(root, tmp_path / "none", segments=["1_1", "9_9"], **kwargs)
    with pytest.raises(ValueError, match="or all"):
        measure_stripes(root, tmp_path / "none", **{**kwargs, "methods": ["fast"]})
