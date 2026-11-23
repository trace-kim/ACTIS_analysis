"""Image operations of the contour lab: levels, noise, deblurring and edges."""

import numpy as np
import pytest
from scipy import ndimage

from actis.contourlab import contours as geometry
from actis.contourlab import ops
from actis.synthetic import render_repeats

HALF = np.sqrt(2 * np.log(2))  # radius of a Gaussian's half maximum, in sigma


def blobs(
    shape=(96, 120), pitch=12.0, sigma=1.6, height=1000.0, base=100.0, shift=(0.3, -0.2)
):
    """Gaussian blobs on a square lattice, and their centres (x, y).

    The outline of each at half its height is a circle of radius sigma * HALF,
    and its slope is steepest on the circle of radius sigma.
    """
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    xs = np.arange(pitch, cols - pitch / 2, pitch) + shift[0]
    ys = np.arange(pitch, rows - pitch / 2, pitch) + shift[1]
    centres = np.array([(cx, cy) for cy in ys for cx in xs])
    image = np.full(shape, float(base))
    for cx, cy in centres:
        image += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return image.astype(np.float32), centres


def circle(centre, radius, points=96):
    t = np.arange(points) * (2 * np.pi / points)
    return np.column_stack(
        [centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)]
    )


def test_the_noise_estimate_is_not_inflated_by_a_dense_array():
    rng = np.random.default_rng(0)
    dense, _ = blobs((256, 256), pitch=6.77, sigma=1.5, height=20000)
    for true in (20.0, 200.0):
        found = ops.noise_sigma(dense + rng.normal(0, true, dense.shape))
        assert found == pytest.approx(true, rel=0.12)
    flat = rng.normal(1000, 50, (256, 224))
    assert ops.noise_sigma(flat) == pytest.approx(50, rel=0.08)
    eight_bit = np.rint(rng.normal(20, 2.0, (300, 300))).astype(np.uint8)
    assert ops.noise_sigma(eight_bit) == pytest.approx(2.0, rel=0.1)
    assert ops.noise_sigma(rng.normal(0, 3.0, (12, 12))) == pytest.approx(3.0, rel=0.6)
    assert ops.noise_sigma(np.zeros((40, 40))) == 0.0


def test_stripes_do_not_count_as_noise():
    rng = np.random.default_rng(1)
    image = rng.normal(0, 5.0, (256, 256)) + rng.normal(0, 40.0, (256, 1))

    assert ops.noise_sigma(image) == pytest.approx(5.0, rel=0.1)


@pytest.mark.parametrize(
    "lattice, pitch", [("square", 6.77), ("staggered", 12.0), ("square", 17.3)]
)
def test_the_pattern_spacing_is_measured_from_the_image(lattice, pitch):
    frames, _ = render_repeats(
        image_shape=(200, 180), lattice=lattice, pitch_px=pitch, cd_nm=80, n_repeats=1
    )

    assert ops.estimate_pitch(frames[0]) == pytest.approx(pitch, abs=0.15)


def test_an_image_without_a_repeating_pattern_has_no_spacing():
    rng = np.random.default_rng(2)

    assert ops.estimate_pitch(rng.normal(0, 1, (200, 200))) is None
    assert ops.estimate_pitch(np.zeros((200, 200))) is None
    assert ops.estimate_pitch(np.ones((8, 8))) is None


def test_otsu_levels_divide_the_classes_of_grey_level():
    rng = np.random.default_rng(3)
    two = np.concatenate([rng.normal(100, 8, 6000), rng.normal(400, 20, 2000)])
    three = np.concatenate([two, rng.normal(900, 30, 1500)])

    (level,) = ops.otsu_levels(two.reshape(80, 100))
    lower, upper = ops.otsu_levels(three.reshape(95, 100), classes=3)

    assert 150 < level < 330
    assert 150 < lower < 330 and 480 < upper < 800
    assert ops.otsu_levels(np.full((10, 10), 7.0)) == [7.0]
    assert ops.otsu_levels(np.full((10, 10), 7.0), classes=3) == [7.0, 7.0]


def test_tile_levels_follow_brightness_that_varies_across_the_image():
    image, _ = blobs((256, 384), pitch=8, sigma=1.5, height=1000, base=0)
    gain = np.linspace(1.0, 2.0, 384, dtype=np.float32)[None, :]
    image = image * gain + 50

    low, high, ys, xs = ops.tile_levels(image, 64, 5, 99.5)
    bright = ops.expand_tiles(high, ys, xs, image.shape)

    assert low.shape == high.shape == (4, 6) and len(ys) == 4 and len(xs) == 6
    assert low.min() > 49 and low.max() < 75  # the offset, and the feet of the blobs
    centre_gain = np.interp(xs, np.arange(384), gain[0])
    np.testing.assert_allclose(high[0], 50 + 1000 * centre_gain, rtol=0.05)
    assert bright.shape == image.shape and bright.dtype == np.float32
    assert np.all(np.diff(bright[100]) >= -1e-3)  # rising with the gain, no steps
    assert np.ptp(bright[:, :10], axis=1).max() == 0  # flat past the outermost centre


def test_feature_levels_are_each_feature_s_own():
    image, _centres = blobs((60, 96), pitch=12, sigma=1.6, shift=(0, 0))
    image[:, 48:] = (
        image[:, 48:] - 100
    ) * 0.5 + 100  # the right half is half as bright

    low, high = ops.feature_levels(image, 12)

    left, right = high[24, 24], high[24, 72]
    assert left == pytest.approx(1100, rel=0.03) and right == pytest.approx(
        600, rel=0.03
    )
    assert low[24, 24] == pytest.approx(100, abs=2)


def test_upscaling_keeps_the_image_in_place():
    image, centres = blobs((36, 48), pitch=12, sigma=1.6, shift=(0.3, -0.2))

    for method in ops.UPSCALERS:
        fine = ops.upscale(image, 4, method)
        assert fine.shape == (144, 192) and fine.dtype == np.float32
        window = fine[20:76, 20:76] - 100  # the pixels of the first blob
        y, x = np.mgrid[20:76, 20:76]
        cx = (x * window).sum() / window.sum()
        cy = (y * window).sum() / window.sum()
        # pixel (x, y) of the result lies at ((x + 0.5) / 4 - 0.5, ...) of the image
        np.testing.assert_allclose(
            [(cx + 0.5) / 4 - 0.5, (cy + 0.5) / 4 - 0.5], centres[0], atol=0.03
        )
    assert ops.upscale(image, 1) is image


def test_deblurring_steepens_edges_and_keeps_the_light():
    y, x = np.mgrid[0:64, 0:64]
    sharp = 100 + 1000.0 * (np.hypot(x - 31.6, y - 30.2) < 6)
    blurred = ndimage.gaussian_filter(sharp, (1.2, 0.9)).astype(np.float32)

    restored = ops.richardson_lucy(blurred, 0.9, 1.2, 20)

    def steepest(image):
        return np.abs(np.gradient(image.astype(float), axis=1)).max()

    assert steepest(restored) > 1.4 * steepest(blurred)
    assert restored.sum() == pytest.approx(blurred.sum(), rel=1e-4)
    assert restored.min() > 99 and restored.dtype == np.float32  # no undershoot
    assert np.abs(restored - sharp).mean() < 0.8 * np.abs(blurred - sharp).mean()
    assert ops.richardson_lucy(blurred, 0.9, 1.2, 0) is blurred
    assert ops.richardson_lucy(blurred, 0, 0, 5) is blurred


def test_non_local_means_lowers_the_noise_of_a_repeating_pattern():
    rng = np.random.default_rng(4)
    clean, _ = blobs((96, 96), pitch=12, sigma=1.6, shift=(0, 0))
    noisy = clean + rng.normal(0, 40, clean.shape).astype(np.float32)

    smooth = ops.non_local_means(noisy, 40.0, strength=1.0, patch=3, search=25)

    assert np.abs(smooth - clean).std() < 0.6 * np.abs(noisy - clean).std()
    assert smooth.shape == clean.shape and smooth.dtype == np.float32


def test_stripes_are_removed_and_the_pattern_stays():
    rng = np.random.default_rng(5)
    clean = rng.normal(500, 20, (200, 240)).astype(np.float32)
    rows, cols = rng.normal(0, 30, (200, 1)), rng.normal(0, 30, (1, 240))

    both = ops.remove_stripes(clean + rows + cols, True, True, 65)
    rows_only = ops.remove_stripes(clean + rows + cols, True, False, 65)

    assert (rows + cols + 0 * clean).std() > 40
    # what is left: the scatter of the 65 neighbouring medians each line is compared with
    assert (both - clean).std() < 15
    assert (rows_only - clean - cols).std() < 11
    assert (rows_only - clean).std() > 25  # the column stripes are still there


def test_a_blur_lowers_white_noise_by_its_gain():
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 1, (400, 400))

    for sigma in (0.5, 1.0, 2.0):
        measured = ops.blur(noise, sigma).std()
        assert ops.blur_noise_gain(sigma) == pytest.approx(measured, rel=0.03)
    assert ops.blur_noise_gain(0) == 1.0 and ops.blur(noise, 0) is noise


@pytest.mark.parametrize("rule", ops.RULES)
def test_refining_moves_every_point_to_its_edge(rule):
    image, centres = blobs()
    start = geometry.from_list([circle(c + [0.15, -0.1], 2.3) for c in centres])

    # far enough out to reach the background between the blobs
    refined = ops.refine(image, start, rule, search=4.0, blur_sigma=0, level=600.0)

    # The spline through pixels 1.6 sigma apart is a fraction of a percent off the
    # blob, which shows most where levels or slopes are taken from it.
    wanted, within = {
        "half level": (2 * 1.6 * HALF, 0.02),
        "steepest gradient": (2 * 1.6, 0.04),
        "threshold level": (2 * 1.6 * HALF, 0.003),
    }[rule]
    measures = geometry.measure(refined)
    np.testing.assert_allclose(measures["diameter"], wanted, atol=within)
    np.testing.assert_allclose(measures["centre"], centres, atol=0.02)
    assert list(refined.counts) == list(start.counts) and refined.closed.all()


def test_the_half_level_of_a_small_feature_counts_from_its_peak():
    image, centres = blobs()
    start = geometry.from_list([circle(c, 1.9) for c in centres])

    # the search does not reach the peak from the edge: the centroid's level does
    near = ops.refine(image, start, "half level", search=1.0, blur_sigma=0)

    reached = 100 + 1000 * np.exp(-((1.9 + 1.0) ** 2) / (2 * 1.6**2))  # lowest sample
    level = reached + 0.5 * (1100 - reached)
    radius = 1.6 * np.sqrt(2 * np.log(1000 / (level - 100)))
    np.testing.assert_allclose(
        geometry.measure(near)["diameter"], 2 * radius, atol=0.01
    )


def test_refining_dark_features_is_the_same_on_the_negative():
    image, centres = blobs()
    start = geometry.from_list([circle(c, 2.2) for c in centres])

    bright = ops.refine(image, start, "half level", blur_sigma=0)
    dark = ops.refine(1200 - image, start, "half level", blur_sigma=0, bright=False)

    np.testing.assert_allclose(dark.xy, bright.xy, atol=1e-3)
    with pytest.raises(ValueError, match="rule must be one of"):
        ops.refine(image, start, "brightest")
    assert len(ops.refine(image, geometry.empty())) == 0


def test_a_point_without_an_edge_takes_its_neighbours_place():
    image, centres = blobs()
    ring = circle(centres[0], 1.6 * HALF)
    ring[10] += 6.0  # far off, in the flat background: no edge within reach
    start = geometry.from_list([ring])

    refined = ops.refine(
        image, start, "threshold level", search=1.0, blur_sigma=0, level=600.0
    )

    radius = np.hypot(*(refined.xy - centres[0]).T)
    assert abs(np.delete(radius, 10) - 1.6 * HALF).max() < 0.01
    np.testing.assert_allclose(refined.xy[10], ring[10], atol=0.05)  # as its neighbours


def test_edge_contrast_is_the_step_across_each_contour():
    image, centres = blobs()
    rings = geometry.from_list([circle(c, 1.6 * HALF) for c in centres])

    step = ops.edge_contrast(image, rings, distance=1.0)

    profile = 1000 * np.exp(
        -(np.array([1.6 * HALF - 1, 1.6 * HALF + 1]) ** 2) / (2 * 1.6**2)
    )
    np.testing.assert_allclose(step, profile[0] - profile[1], rtol=0.03)
    assert len(ops.edge_contrast(image, geometry.empty())) == 0


def test_gradient_edges_lie_on_the_steepest_slope():
    image, centres = blobs()

    edges, upper = ops.gradient_edges(image, sigma=0.8, oversample=3)

    measures = geometry.measure(edges)
    assert len(edges) == len(centres) and edges.closed.all() and upper > 0
    # the blur of the derivatives adds to the blob's own: sigma 1.6 and 0.8 in quadrature
    np.testing.assert_allclose(measures["diameter"], 2 * np.hypot(1.6, 0.8), atol=0.03)
    order = np.lexsort((measures["centre"][:, 0], np.rint(measures["centre"][:, 1])))
    np.testing.assert_allclose(measures["centre"][order], centres, atol=0.02)
    assert (measures["area"] > 0).all()  # the brighter side is the inside


def test_gradient_edges_leave_out_weak_and_short_edges():
    rng = np.random.default_rng(7)
    image, centres = blobs()
    faint = image.copy()
    faint[:, 66:] = (
        faint[:, 66:] - 100
    ) * 0.1 + 100  # the right part: a tenth as steep
    noisy = faint + rng.normal(0, 2, faint.shape).astype(np.float32)

    edges, _ = ops.gradient_edges(noisy, sigma=0.8, high=0.3, low_ratio=0.5)

    x = geometry.measure(edges)["centre"][:, 0]
    assert len(edges) == (centres[:, 0] < 66).sum() and x.max() < 66
    assert edges.left_out["weak"] > 0
    empty, _ = ops.gradient_edges(np.full((40, 40), 5.0, np.float32))
    assert len(empty) == 0
