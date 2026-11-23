from types import SimpleNamespace

import numpy as np
import pytest

from actis.fit import array_backend
from actis.tiff_tiles import _quadrant_scores, select_tiles


@pytest.fixture(scope="module")
def cuda():
    try:
        return array_backend("cuda")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_translated_quadrants_agree_with_explicit_interpolation(cuda):
    cp = cuda
    from cupyx.scipy.ndimage import map_coordinates

    # Weak variations on a 16-bit background expose interpolation/centering
    # errors. Use a strided reference and a side not divisible by 32.
    side = 130
    rng = np.random.default_rng(47)
    pixels = rng.integers(64990, 65001, (side, side), dtype=np.uint16)
    reference = cp.asarray(pixels, dtype=cp.float32).T
    shifts = [(0, 0), (1, -2), (-0.375, 2.25), (1e-9, -1e-8), (3.000001, -2.000001)]
    dx, dy = cp.asarray(shifts).T
    axis = cp.arange(side)
    x, y = axis - dx[:, None], axis - dy[:, None]
    coords = cp.stack(cp.broadcast_arrays(y[:, :, None], x[:, None, :]))
    overlap = ((coords >= 0) & (coords <= side - 1)).all(axis=0)
    aligned = map_coordinates(
        reference, coords, order=1, mode="nearest", prefilter=False
    )
    noise = cp.asarray(rng.normal(0, 1.2, aligned.shape), dtype=cp.float32)
    tiles = cp.asfortranarray(aligned + noise)
    tiles[0] = reference
    tiles[1, : side // 2, : side // 2] = 65000

    def quadrants(a):
        return a.reshape(-1, 2, side // 2, 2, side // 2)

    def centered(a):
        mask = quadrants(overlap)
        means = (quadrants(a) * mask).sum(axis=(2, 4), keepdims=True) / mask.sum(
            axis=(2, 4), keepdims=True
        )
        return (quadrants(a) - means) * mask

    a, b = centered(tiles), centered(aligned)
    aa, bb, ab = ((p * q).sum(axis=(2, 4)) for p, q in ((a, a), (b, b), (a, b)))
    scores = (ab / cp.maximum(cp.sqrt(aa * bb), 1e-20)).min(axis=(1, 2))
    actual = _quadrant_scores(tiles, reference, dx, dy, cp).get()
    np.testing.assert_allclose(actual, scores.get(), atol=1e-12, rtol=0)
    assert actual[0] == 1
    assert actual[1] == 0


def test_select_highest_scoring_reference_and_first_tie(cuda):
    cp = cuda
    side = 64
    window = np.outer(np.hanning(side), np.hanning(side))
    weights = cp.asarray(window / window.sum(), dtype=cp.float32)
    rng = np.random.default_rng(304)
    tiles = cp.asarray(rng.integers(0, 256, (3, side, side)), dtype=cp.float32)
    templates = [
        SimpleNamespace(
            window=weights,
            pixels=tile,
            search=cp.arange(-6, 7),
            spectrum=cp.fft.rfft2((tile - (tile * weights).sum()) * weights).conj(),
        )
        for tile in tiles[:2]
    ]
    result = select_tiles(tiles, [*templates, templates[0]], 0.80, cp)
    np.testing.assert_array_equal(result[:, :2], [[0, 0], [1, 1]])
