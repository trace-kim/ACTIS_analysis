import numpy as np
import pytest
from scipy.ndimage import shift

from actis.estimators import estimate_holes
from actis.fit import array_backend
from actis.harmonics import harmonic_tiles
from actis.pooled import pooled_tiles
from actis.synthetic import render_repeats
from actis.tiff_benchmark import METHODS, benchmark
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
