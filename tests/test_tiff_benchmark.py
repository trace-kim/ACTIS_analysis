import json

import numpy as np
import pytest
from scipy.ndimage import shift

from actis.estimators import estimate_holes
from actis.fit import array_backend
from actis.harmonics import harmonic_tiles
from actis.pooled import pooled_matched_cds, pooled_tiles
from actis.synthetic import render_repeats
from actis.tiff_benchmark import METHODS, PartialReport, TileSample, benchmark, main
from actis.tiff_tiles import prepare_reference, select_tiles


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
        image_shape=(96, 96),
        pitch_px=8,
        cd_nm=86,
        n_repeats=1,
        nuisance=False,
        supersample=4,
    )
    return frames[0]


def test_gpu_selection_distinguishes_mixed_tiles(cuda, holes):
    template = prepare_reference(holes, 96, "square", cuda)
    # Exact-edge samples must stay inside the reference interpolation stencil.
    storage = cuda.full(holes.size + holes.shape[1] + 1, cuda.nan, dtype=cuda.float32)
    pixels = storage[: holes.size].reshape(holes.shape)
    pixels[:] = template[0].pixels
    template[0].pixels = pixels
    assert len(select_tiles(pixels[None], template, 1.0, cuda)) == 1
    moved = shift(holes.astype(float), (2.25, -1.33), order=1, mode="grid-wrap")
    stripes = np.broadcast_to(holes[holes.mean(axis=1).argmax()], holes.shape)
    partial = holes.copy()
    partial[:48, :48] = np.mean(holes)
    # A whole-tile correlation cutoff alone admits this partially blank tile.
    assert np.corrcoef(holes.ravel(), partial.ravel())[0, 1] > 0.8
    noise = np.random.default_rng(23).normal(2000, 50, holes.shape)
    tiles = np.stack(
        [holes, moved * 0.45 + 12000, stripes, stripes.T, partial, noise, holes * 0]
    )
    matches = select_tiles(cuda.asarray(tiles), template, 0.8, cuda)
    np.testing.assert_array_equal(matches[:, 0], [0, 1])
    recovered = matches[1, 2:] - [-1.33, 2.25]
    np.testing.assert_allclose(recovered - 8 * np.rint(recovered / 8), 0, atol=0.15)


def test_batched_pooled_matches_independent_shifted_tiles(cuda, holes):
    # A halo supplies sites that enter or leave the tile at different shifts.
    template = prepare_reference(np.tile(holes, (2, 2)), 96, "square", cuda)[0]
    pixels = template.pixels.get().astype("u2")
    shifts = np.array([[0.0, 0.0], [-2.25, 1.5], [2.75, -1.5], [0.0, 0.0]])
    tiles = np.stack(
        [shift(pixels, (dy, dx), order=1, mode="grid-wrap") for dx, dy in shifts]
    )
    tiles[-1].fill(65535)
    expected = [
        pooled_tiles(
            tile, template.geometry(delta), tile_px=96, origin_xy=(-0.5, -0.5)
        )[0]["cd_px"]
        for tile, delta in zip(tiles, shifts)
    ]
    assert np.isfinite(expected[:-1]).all() and np.isnan(expected[-1])
    actual = pooled_matched_cds(cuda.asarray(tiles), template.region, shifts, cuda)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0, equal_nan=True)


def test_streamed_gpu_methods_agree_with_native_estimators(cuda, holes, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    # Nonzero reference index, 16-bit reference and compressed 8-bit target.
    ref = tmp_path / "reference.tiff"
    tifffile.imwrite(ref, np.stack([holes * 0, holes]), photometric="minisblack")
    tile = (holes // 256).astype("u1")
    path = tmp_path / "scan.tiff"
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for left, right in ((tile, tile * 0), (tile * 0, tile), (tile * 0, tile * 0)):
            page = np.pad(np.hstack([left, right]), ((0, 3), (0, 5)))
            tif.write(page, compression="deflate")
    kwargs = {
        "reference": ref,
        "reference_page": 1,
        "shape": "square",
        "psf": (0.78, 0.96),
        "tile_size": 96,
    }
    result = benchmark(path, **kwargs)
    assert (result["pages"], result["tested_tiles"], result["accepted_tiles"]) == (
        3,
        6,
        2,
    )
    assert result["excluded_edge_pixels"] == 3 * (99 * 197 - 96 * 192)
    assert set(result["methods"]) == set(METHODS)
    assert result["measured_tiles"] == 2
    assert result["scan_seconds"] < result["total_seconds"]
    assert result["stage_seconds"]["fit_setup"] > 0

    templates = prepare_reference(holes, 96, "square", cuda)
    _, _, dx, dy = select_tiles(cuda.asarray(tile[None]), templates, 0.8, cuda)[0]
    region = templates[0].geometry(np.array([dx, dy]))
    estimates = estimate_holes(
        tile,
        region.centers,
        pitch_px=region.pitch,
        shape=region.shape,
        angle_rad=region.angle,
        psf_sigma_px=kwargs["psf"],
        tile_size_px=96,
    )
    expected = {"raw": [tile.mean()]}
    for name, rows in estimates.items():
        field = "flux_dn_px2" if name == "flux" else "cd_px"
        expected[name] = [r[field] for r in rows if r["valid"]]
    for name, estimator in (("harmonic", harmonic_tiles), ("pooled", pooled_tiles)):
        rows = estimator(tile, region, tile_px=96, origin_xy=(-0.5, -0.5))
        expected[name] = [r["cd_px"] for r in rows if r["status"] == "ok"]
    for name, values in expected.items():
        row = result["methods"][name]
        assert row["seconds"] >= row["exclusive_seconds"] > 0
        assert row["valid"] == 2 * len(values) > 0
        assert row["mean"] == pytest.approx(np.mean(values), rel=1e-4)

    tifffile.imwrite(tmp_path / "blank.tif", tile * 0)
    empty = benchmark(tmp_path / "blank.tif", **kwargs)
    assert empty["fastest_with_valid_results"] is None
    assert empty["tested_tiles"] == 1 and empty["accepted_tiles"] == 0
    assert all(r["attempted"] == r["seconds"] == 0 for r in empty["methods"].values())
    assert empty["estimated_all_tiles_method_seconds"] is None


def test_tile_sample_is_repeatable_across_batch_boundaries():
    def sample(batch_size):
        sampler = TileSample(4)
        for page_id in range(6):
            page = np.arange(12, dtype=np.uint16)[None] + 12 * page_id
            for start in range(0, 12, batch_size):
                count = min(batch_size, 12 - start)
                matches = [(idx, 0, 0.0, 0.0) for idx in range(count)]
                sampler.consider(page, page_id, start, np.asarray(matches), 1)
            page.fill(0)
        assert sampler.seen == 72
        return [(p, t, im.item()) for p, t, im, *_ in sampler.entries]

    selected = sample(1)
    assert len(selected) == len(set(selected)) == 4
    assert any(page > 0 for page, _, _ in selected)
    assert all(value == 12 * page + tile for page, tile, value in selected)
    assert selected == sample(5)


def test_single_pooled_method_skips_other_work_even_in_warmup(
    cuda, holes, tmp_path, monkeypatch
):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "scan.tif"
    tifffile.imwrite(
        path, np.stack([holes, holes * 0, holes]), photometric="minisblack"
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Ran an unrequested estimator")

    for name in (
        "_model_methods",
        "_contours",
        "harmonic_tiles",
        "_flux",
        "_raw_means",
    ):
        monkeypatch.setattr(f"actis.tiff_benchmark.{name}", unexpected)
    kwargs = {"method": "pooled", "tile_size": 96, "shape": "square"}
    full = benchmark(path, sample_tiles=None, **kwargs)
    sample = benchmark(path, sample_tiles=1, **kwargs)
    assert set(full["methods"]) == {"pooled"}
    assert full["pages"] == 3 and full["tested_tiles"] == 3
    assert (
        full["methods"]["pooled"]["valid"]
        == full["measured_tiles"]
        == full["accepted_tiles"]
        == 2
    )
    row = sample["methods"]["pooled"]
    assert row["valid"] == 1 and row["mean"] == pytest.approx(
        full["methods"]["pooled"]["mean"]
    )
    assert row["estimated_all_tiles_seconds"] == pytest.approx(2 * row["seconds"])
    assert full["methods"]["pooled"]["estimated_all_tiles_seconds"] is None


def test_page_sample_decodes_only_chosen_pages_and_keeps_native_indices(
    cuda, holes, tmp_path, monkeypatch
):
    from actis import tiff_benchmark

    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "sample.tif"
    base = holes // 2
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for i in range(12):
            tif.write(base + i, compression="deflate")
    decoded = []
    read = tiff_benchmark._read_page

    def record_page(page):
        decoded.append(page.index)
        return read(page)

    monkeypatch.setattr(tiff_benchmark, "_read_page", record_page)
    result = benchmark(path, method="raw", tile_size=96, shape="square", sample_pages=3)
    indices = result["sample_page_indices_zero_based"]
    assert decoded == [0, *indices]  # Reference plus three target pages
    assert [i // 4 for i in indices] == [0, 1, 2]
    assert result["pages"] == result["measured_tiles"] == 3
    assert result["sample_locations_page_tile_zero_based"] == [(i, 0) for i in indices]
    assert result["methods"]["raw"]["mean"] == pytest.approx(
        base.mean() + np.mean(indices)
    )


def test_cli_defaults_bound_comparison_and_cover_every_page_for_all_tiles(
    cuda, holes, tmp_path
):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "many.tif"
    tifffile.imwrite(path, np.repeat(holes[None], 40, axis=0), photometric="minisblack")
    for name, flags, pages, measured in (
        ("quick", [], 32, 32),
        ("all_pages", ["--all-pages"], 40, 32),
        ("full", ["--all-tiles"], 40, 40),
    ):
        output = tmp_path / f"{name}.json"
        assert (
            main(
                [
                    str(path),
                    "--tile-size",
                    "96",
                    "--shape",
                    "square",
                    "--method",
                    "raw",
                    "--json",
                    str(output),
                    *flags,
                ]
            )
            == 0
        )
        result = json.loads(output.read_text())
        assert result["total_pages"] == 40
        assert result["pages"] == result["accepted_tiles"] == pages
        assert result["measured_tiles"] == result["methods"]["raw"]["valid"] == measured


def test_sample_exhaustive_and_interrupted_report(cuda, holes, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "mixed.tif"
    with tifffile.TiffWriter(path) as tif:
        for _ in range(3):
            tif.write(np.hstack((holes, holes)))
        tif.write(np.zeros((96, 96, 3), dtype="u1"), photometric="rgb")
    kwargs = {
        "reference_page": 0,
        "tile_size": 96,
        "shape": "square",
        "psf": (0.78, 0.96),
    }
    sampled = benchmark(path, sample_tiles=2, max_pages=3, **kwargs)
    full = benchmark(path, sample_tiles=None, max_pages=3, **kwargs)
    assert sampled["status"] == full["status"] == "complete"
    assert sampled["accepted_tiles"] == full["measured_tiles"] == 6
    assert sampled["measured_tiles"] == sampled["methods"]["raw"]["valid"] == 2
    assert full["methods"]["raw"]["valid"] == 6
    assert len(sampled["sample_locations_page_tile_zero_based"]) == 2
    sample_cost = sampled["stage_seconds"]["fit_setup"] + sum(
        row["exclusive_seconds"] for row in sampled["methods"].values()
    )
    assert sampled["estimated_all_tiles_method_seconds"] == pytest.approx(
        sample_cost * 3
    )
    assert full["estimated_all_tiles_method_seconds"] is None
    partial = PartialReport(tmp_path / "timing.json")
    with pytest.raises(ValueError, match="single-channel"):
        benchmark(path, checkpoint=partial.write, sample_tiles=None, **kwargs)
    saved = json.loads(partial.path.read_text())
    assert saved["status"] == "running" and saved["measured_tiles"] >= 1
    assert saved["methods"]["raw"]["valid"] == saved["measured_tiles"]
