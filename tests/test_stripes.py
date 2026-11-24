import json
from collections import Counter

import numpy as np
import pytest

from actis.fit import array_backend
from actis.stripes import (
    MANIFEST,
    _board_pages,
    find_segments,
    main,
    measure_stripes,
    merge_boards,
    seam_agreement,
    seam_columns,
)
from actis.synthetic import render_repeats
from actis.tiff_benchmark import METHODS

SIDE = 96


@pytest.fixture(scope="module")
def cuda():
    try:
        return array_backend("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))


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


def merged_page(tile, where):
    """Two rows of four tiles, with the hole tile at the (row, column) places in `where`.

    Three rows and 32 columns are left over at the edges. Brightness rises
    from left to right, so a tile's mean tells exactly which columns it holds.
    """
    page = np.full((2 * SIDE + 3, 4 * SIDE + 32), 10.0)
    for row, column in where:
        page[row * SIDE :, column * SIDE :][:SIDE, :SIDE] = tile
    return np.rint(page * np.linspace(0.6, 1.0, page.shape[1])).astype(np.uint8)


def tile_means(page):
    return page[: 2 * SIDE, : 4 * SIDE].reshape(2, SIDE, 4, SIDE).mean(axis=(1, 3))


def hole_tiles(layout):
    """Which (page, tile row, tile column) hold the hole tile, from each page's places."""
    held = np.zeros((len(layout), 2, 4), dtype=bool)
    for page, where in enumerate(layout):
        for row, column in where:
            held[page, row, column] = True
    return held


def write_boards(folder, stem, pages, tifffile):
    folder.mkdir(exist_ok=True)
    for number, board in enumerate(split(np.stack(pages)), 1):
        path = folder / f"{stem}_{number}.tiff"
        tifffile.imwrite(path, board, photometric="minisblack")


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
    names = ["S1/1_1_%d.tiff", "S1/1_2_%d.tiff", "S10/10_1_%d.TIF", "S2/2_1_%d.tiff"]
    for name in names:
        for board in range(1, 5):
            path = tmp_path / (name % board)
            path.parent.mkdir(exist_ok=True)
            path.touch()
    (tmp_path / "S2" / "2_1_3.tiff").unlink()
    for other in ("header.xml", "S1/notes.txt", "S1/1_1_1_copy.tiff", "9_9_1.tiff"):
        (tmp_path / other).touch()
    segments, incomplete = find_segments(tmp_path)
    # In number order: stripe 10 follows stripe 1, not the other way round.
    assert [(s.stripe, s.vertical, s.name) for s in segments] == [
        (1, 1, "1_1"),
        (1, 2, "1_2"),
        (10, 1, "10_1"),
    ]
    assert [p.name for p in segments[2].files] == [f"10_1_{b}.TIF" for b in range(1, 5)]
    assert incomplete == [{"stripe": 2, "vertical": 1, "boards_found": [1, 2, 4]}]
    assert len(find_segments(tmp_path, boards=3)[0]) == 0

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


def test_every_stripe_is_measured_on_merged_pages(cuda, holes, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    tile = holes // 256
    root, out = tmp_path / "scan", tmp_path / "out"
    root.mkdir()
    (root / "header.xml").write_text("<header/>")
    reference = tmp_path / "reference.tiff"
    tifffile.imwrite(reference, holes, photometric="minisblack")
    # Tile columns 1 to 3 each lie across a seam between two boards.
    places = [[(0, 0), (0, 1), (1, 3)], [(0, 2), (1, 1)], []], [[(1, 0)], [(0, 3)]]
    first = [merged_page(tile, where) for where in places[0]]
    second = [merged_page(tile, where) for where in places[1]] + first[:1]
    write_boards(root / "S1", "1_1", first, tifffile)
    write_boards(root / "S1", "1_2", second, tifffile)
    short = split(np.stack(second[:2]))[2]
    tifffile.imwrite(root / "S1" / "1_2_3.tiff", short, photometric="minisblack")
    write_boards(root / "S2", "2_1", first, tifffile)
    (root / "S2" / "2_1_4.tiff").unlink()
    write_boards(root / "S3", "3_1", first, tifffile)
    narrow = split(np.stack(first))[1][..., :120]
    tifffile.imwrite(root / "S3" / "3_1_2.tiff", narrow, photometric="minisblack")

    lines = []
    kwargs = {"reference": reference, "tile_size": SIDE, "shape": "square"}
    manifest = measure_stripes(root, out, progress=lines.append, **kwargs)
    assert manifest == json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["header"] == str(root / "header.xml")
    assert manifest["settings"]["methods"] == ["raw"]
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
    assert "3_1_2.tiff is 120 x 195 px" in three["error"] and "result" not in three
    assert not (out / "3_1.npz").exists()
    assert one["files"] == [f"S1/1_1_{b}.tiff" for b in range(1, 5)]
    assert (one["pages"], one["board_shape"], one["page_shape"]) == (
        3,
        [195, 128],
        [195, 416],
    )
    assert (one["tile_grid"], one["seam_columns"]) == ([2, 4], [112, 208, 304])
    assert (one["tested_tiles"], one["accepted_tiles"]) == (24, 5)
    assert one["notes"] == [] and min(one["seam_correlation_p90"]) > 0.999
    assert one["seconds"]["total"] >= one["seconds"]["read_wait"] > 0
    assert two["pages"] == 2 and two["accepted_tiles"] == 2
    assert two["notes"][0].startswith("1_2_3.tiff ended after 2 pages")
    assert any(line.startswith("1_1: 3 pages of 416 x 195 px") for line in lines)
    assert any(line.startswith("3_1: failed.") for line in lines)

    for entry, pages, layout in zip((one, two), (first, second[:2]), places):
        data = np.load(out / entry["result"])
        expected = np.stack([tile_means(page) for page in pages])
        accepted = hole_tiles(layout)
        assert data["raw"].dtype == np.float32 and data["raw"].shape == accepted.shape
        np.testing.assert_array_equal(np.isfinite(data["raw"]), accepted)
        np.testing.assert_array_equal(data["reference"] >= 0, accepted)
        # A merge that is off by one column would change these means.
        np.testing.assert_allclose(data["raw"][accepted], expected[accepted], rtol=1e-6)
        np.testing.assert_array_equal(data["page_index"], np.arange(len(pages)))
        assert data["seam_correlation"].shape == (len(pages), 3)
        assert np.nanmin(data["seam_correlation"]) > 0.999
        assert (int(data["stripe"]), int(data["vertical"])) == (1, entry["vertical"])
        assert (int(data["tile_size"]), int(data["overlap_px"])) == (SIDE, 32)
        assert (int(data["boards"]), int(data["board_columns"])) == (4, 128)
        assert "blur_sigma_px" not in data.files
        mean = entry["methods"]["raw"]["mean"]
        assert mean == pytest.approx(expected[accepted].mean(), rel=1e-6)

    # A second run keeps what is complete and tries the failed segment again.
    measured = []
    original = measure_stripes.__globals__["_measure_segment"]

    def recorded(segment, *args):
        measured.append(segment.name)
        return original(segment, *args)

    monkeypatch.setattr("actis.stripes._measure_segment", recorded)
    again = measure_stripes(root, out, **kwargs)
    assert measured == ["3_1"] and again == manifest
    with pytest.raises(ValueError, match="other settings"):
        measure_stripes(root, out, **{**kwargs, "min_corr": 0.7})
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file.txt").touch()
    with pytest.raises(ValueError, match="new or empty folder"):
        measure_stripes(root, tmp_path / "taken", **kwargs)
    with pytest.raises(ValueError, match="No segment"):
        measure_stripes(root, tmp_path / "none", stripes=[7], **kwargs)

    # Every method goes through the same pages, and a trial stops early.
    every = measure_stripes(
        root, tmp_path / "every", methods=METHODS, stripes=[1], max_pages=1, **kwargs
    )
    assert [s["pages"] for s in every["segments"]] == [1, 1]
    data = np.load(tmp_path / "every" / "1_1.npz")
    accepted = hole_tiles(places[0][:1])
    for name in METHODS:
        np.testing.assert_array_equal(np.isfinite(data[name]), accepted, err_msg=name)
    assert data["blur_sigma_px"].shape == (1, 1, 2)

    arguments = [str(root), "--reference", str(reference), "--tile-size", str(SIDE)]
    arguments += ["--shape", "square", "--output"]
    assert main([*arguments, str(tmp_path / "cli"), "--stripe", "1"]) == 0
    saved = json.loads((tmp_path / "cli" / MANIFEST).read_text(encoding="utf-8"))
    assert [s["result"] for s in saved["segments"]] == ["1_1.npz", "1_2.npz"]
    assert main([*arguments, str(tmp_path / "cli")]) == 1  # 3_1 cannot be merged
