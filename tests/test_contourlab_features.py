"""What was made for ACTIS images: the optical low pass, noise by brightness, and
thresholds that the features set for themselves."""

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from actis.contourlab import contours as geometry
from actis.contourlab import ops
from actis.contourlab.modules import WorkflowError, image_contours
from actis.contourlab.workflow import (
    START,
    preset,
    run_workflow,
    step,
    workflow_contours,
)
from actis.synthetic import render_repeats

HALF = np.sqrt(2 * np.log(2))  # radius of a Gaussian's half maximum, in sigma
WHOLE = dict(holes=False, open_at_border=False)


def blob(image, cx, cy, height, sigma=1.6):
    """Add a Gaussian blob to an image."""
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    image += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))


def lattice(x0, y0, x1, y1, pitch):
    """Centres (x, y) on a square lattice inside a box, off the pixel grid."""
    xs = np.arange(x0 + pitch / 2 + 0.3, x1 - pitch / 2, pitch)
    ys = np.arange(y0 + pitch / 2 - 0.2, y1 - pitch / 2, pitch)
    return np.array([(cx, cy) for cy in ys for cx in xs])


def found_at(contours, centres, within=1.5):
    """For each centre the number of the closed contour found there (-1: none)."""
    measures = geometry.measure(contours)
    whole = np.flatnonzero(contours.closed & np.isfinite(measures["diameter"]))
    if not len(whole):
        return np.full(len(centres), -1)
    distance, index = cKDTree(measures["centre"][whole]).query(centres)
    return np.where(distance < within, whole[index], -1)


@pytest.fixture(scope="module")
def sharp():
    """A hole array as optics with a hard cutoff at 0.27 cycles per pixel deliver it."""
    frames, _ = render_repeats(
        shape="square",
        image_shape=(320, 384),
        pitch_px=6.77,
        cd_nm=80.0,
        n_repeats=1,
        nuisance=False,
        sigma_xy_px=(0.4, 0.4),
    )
    return ops.optical_filter(frames[0].astype(np.float32), 0.27, 0.27, edge=0.02)[0]


# --- The pass band of the optics --------------------------------------------


def test_the_pass_band_ends_where_the_optics_cut_off(sharp):
    rng = np.random.default_rng(1)

    for sigma in (30.0, 300.0):
        band = ops.pass_band(
            sharp + rng.normal(0, sigma, sharp.shape).astype(np.float32)
        )
        # (on an image this small one of the two directions may come out higher)
        assert 0.25 < min(band) < 0.31 and max(band) < 0.42, sigma
    # noise alone holds no pattern at any frequency
    assert max(ops.pass_band(rng.normal(40, 3, (384, 384)))) < 0.1
    # too small an image to tell: nothing is taken for noise
    assert ops.pass_band(np.zeros((20, 20))) == (0.5, 0.5)


def test_lines_along_the_pixel_grid_keep_their_harmonics():
    # all of their spectrum lies on one axis, where stripe noise lies too
    rng = np.random.default_rng(3)
    _y, x = np.mgrid[0:256, 0:256]
    lines = 2000.0 * (np.cos(2 * np.pi * x / 8) > 0.2) + rng.normal(0, 40, (256, 256))

    along_x, along_y = ops.pass_band(lines)

    assert along_x > 0.4 and along_y < 0.1


def test_the_low_pass_removes_the_noise_beyond_the_band_and_nothing_else(sharp):
    rng = np.random.default_rng(4)
    noisy = sharp + rng.normal(0, 300.0, sharp.shape).astype(np.float32)

    kept, same = ops.optical_filter(sharp, 0.3, 0.3)
    cleaned, gain = ops.optical_filter(noisy, 0.3, 0.3)

    assert same == gain and 0.5 < gain < 0.62  # the share of the frequency plane kept
    inside = (slice(16, -16), slice(16, -16))  # (the image is mirrored at its borders)
    assert np.abs(kept - sharp)[inside].max() < 1e-3 * np.ptp(
        sharp
    )  # the pattern as it was
    assert np.std(cleaned - sharp) == pytest.approx(300.0 * gain, rel=0.05)
    assert cleaned.shape == sharp.shape and cleaned.dtype == np.float32


def test_a_blur_is_divided_out_inside_the_band():
    image = np.zeros((96, 96), np.float32)
    for cx, cy in lattice(0, 0, 96, 96, 16.0):
        blob(image, cx, cy, 1000.0, sigma=2.0)
    blurred = ndimage.gaussian_filter(image, 0.9, mode="nearest")

    restored, gain = ops.optical_filter(blurred, 0.45, 0.45, 0.9, 0.9, max_gain=20.0)

    inside = (slice(12, -12), slice(12, -12))
    assert (
        np.abs(restored - image)[inside].max()
        < 0.2 * np.abs(blurred - image)[inside].max()
    )
    assert gain > 1  # and noise would have been amplified


# --- Noise by brightness ----------------------------------------------------


def test_bright_areas_are_known_to_be_noisier_than_dark_ones():
    rng = np.random.default_rng(5)
    image = np.full((256, 384), 100.0)
    image[:, 192:] = 4000.0
    image += rng.normal(0, 1, image.shape) * np.where(image > 1000, 12.0, 2.0)
    image = image.astype(np.float32)

    levels, sigmas = ops.noise_by_level(image)
    sigma, overall = ops.noise_map(image)

    assert np.interp(100.0, levels, sigmas) == pytest.approx(2.0, rel=0.25)
    assert np.interp(4000.0, levels, sigmas) == pytest.approx(12.0, rel=0.25)
    assert np.median(sigma[:, :150]) == pytest.approx(2.0, rel=0.25)
    assert np.median(sigma[:, 230:]) == pytest.approx(12.0, rel=0.25)
    assert sigma.shape == image.shape and 2.0 < overall < 12.0
    assert ops.noise_map(image)[0] is sigma  # known for as long as the image is


def test_a_blur_lowers_noise_less_once_a_low_pass_has_confined_it():
    white = ops.blur_noise_gain(0.8)
    confined = ops.blur_noise_gain(0.8, band=(0.15, 0.15))

    assert 0.3 < white < 0.4 and confined > 0.85
    assert ops.blur_noise_gain(0.0, band=(0.2, 0.2)) == 1.0


# --- Thresholds the features set for themselves -----------------------------


@pytest.fixture(scope="module")
def mixed():
    """Faint holes, bright holes, a wide pad with faint holes close beside it,
    and an empty quarter, with shot noise: (image, what is where)."""
    rng = np.random.default_rng(6)
    image = np.full((288, 384), 150.0)
    faint = lattice(8, 8, 180, 130, 10.0)
    bright = lattice(200, 8, 376, 130, 12.0)
    beside = lattice(292, 150, 376, 280, 10.0)
    for cx, cy in np.vstack([faint, beside]):
        blob(image, cx, cy, 500.0)
    for cx, cy in bright:
        blob(image, cx, cy, 3000.0)
    pad = np.zeros_like(image)
    pad[150:270, 130:280] = 3000.0
    image += ndimage.gaussian_filter(pad, 1.2)
    image *= 1 + 0.08 * np.linspace(-1, 1, 384)[None, :]  # uneven illumination
    image += rng.normal(0, 1, image.shape) * np.sqrt(64 + 0.6 * image)
    where = dict(faint=faint, bright=bright, beside=beside, pad=(130, 150, 280, 270))
    return image.astype(np.float32), where


def test_every_kind_of_feature_is_outlined_on_one_page(mixed):
    image, where = mixed

    run = run_workflow(image, preset(START))

    assert run.error is None and "tops set levels" in run.states[1].note
    contours = image_contours(run.final)
    measures = geometry.measure(contours)
    for kind, sigma_px in (("faint", 1.6), ("bright", 1.6), ("beside", 1.6)):
        index = found_at(contours, where[kind])
        assert (index >= 0).all(), kind
        # each at half of its own kind's height, whatever the others' brightness
        diameter = np.median(measures["diameter"][index])
        assert diameter == pytest.approx(2 * sigma_px * HALF, rel=0.08), kind
    # the pad: one outline along its edge, and nothing inside it or in the empty quarter
    x0, y0, x1, y1 = where["pad"]
    low = np.minimum.reduceat(contours.xy, contours.start[:-1], axis=0)
    high = np.maximum.reduceat(contours.xy, contours.start[:-1], axis=0)
    # (to 3 px: faint holes close beside a wide feature pull its outline towards
    # them by a pixel or two where their top is the nearest, a known weakness)
    edge = (np.abs(low - [x0, y0]) < 3).all(axis=1) & (
        np.abs(high - [x1 - 1, y1 - 1]) < 3
    ).all(axis=1)
    assert edge.sum() == 1 and contours.closed[edge].all()
    inside = (low > [x0 + 4, y0 + 4]).all(axis=1) & (high < [x1 - 4, y1 - 4]).all(
        axis=1
    )
    empty = (high[:, 0] < 125) & (low[:, 1] > 140)
    assert not inside.any() and not empty.any()


def test_the_tiles_get_the_same_page_wrong(mixed):
    # what the feature levels were made for: a tile holds two kinds of pattern
    image, where = mixed

    contours = workflow_contours(image, preset("Local tiles"))

    assert (found_at(contours, where["beside"]) < 0).mean() > 0.2


def test_noise_and_uneven_brightness_are_no_features():
    rng = np.random.default_rng(7)
    dark = rng.normal(40, 3, (256, 256))
    bright = 3000 * (1 + 0.15 * np.linspace(-1, 1, 320))[None, :] + rng.normal(
        0, 40, (240, 320)
    )

    for image in (dark, bright, np.full((64, 64), 7.0)):
        run = run_workflow(image, preset(START))
        assert run.error is None and len(run.final.contours) == 0
        assert run.states[1].note.startswith("0 tops set levels")


def test_shared_levels_make_a_fainter_hole_smaller_and_own_levels_do_not():
    image = np.full((120, 120), 100.0)
    centres = lattice(0, 0, 120, 120, 12.0)
    odd = len(centres) // 2
    for k, (cx, cy) in enumerate(centres):
        blob(image, cx, cy, 800.0 if k == odd else 1000.0)
    image = image.astype(np.float32)

    def diameters(peers):
        steps = [
            step("feature_threshold", peers=peers, blur_px=0.0),
            step("trace_contours", **WHOLE),
        ]
        contours = workflow_contours(image, steps)
        return geometry.measure(contours)["diameter"][found_at(contours, centres)]

    shared, own = diameters(64), diameters(1)

    full = 2 * 1.6 * HALF
    others = np.arange(len(centres)) != odd
    np.testing.assert_allclose(shared[others], full, rtol=0.02)
    np.testing.assert_allclose(
        own, full, rtol=0.02
    )  # every hole at half of its own height
    # at the level of its neighbours a hole of 0.8 times their height is this much narrower
    assert shared[odd] / full == pytest.approx(
        np.sqrt(np.log(1.6) / np.log(2)), rel=0.03
    )


def test_what_is_cut_by_the_border_takes_its_neighbours_level():
    image = np.full((60, 96), 100.0)
    centres = lattice(0, 0, 96, 60, 8.0)
    for cx, cy in centres:
        blob(image, cx, cy, 1000.0)
    for cx in centres[:, 0]:  # the next row, whose tops are outside the image
        blob(image, cx, 61.8, 1000.0)
    image = image.astype(np.float32)

    contours = workflow_contours(
        image, [step("feature_threshold"), step("trace_contours")]
    )

    # as features of their own the slivers would flood the gap and join the last row to the border
    index = found_at(contours, centres)
    assert (index >= 0).all()
    np.testing.assert_allclose(
        geometry.measure(contours)["diameter"][index], 2 * 1.6 * HALF, rtol=0.05
    )


# --- The steps as a workflow uses them --------------------------------------


def test_the_optical_low_pass_says_what_it_found(sharp):
    rng = np.random.default_rng(8)
    noisy = sharp + rng.normal(0, 300.0, sharp.shape).astype(np.float32)

    cut = run_workflow(noisy, [step("optical_filter")]).final
    by_hand = run_workflow(noisy, [step("optical_filter", cutoff=0.4)]).final
    frames, _ = render_repeats(
        image_shape=(160, 192), shape="circle", pitch_px=8.0, cd_nm=90, n_repeats=1
    )
    plain = run_workflow(frames[0], [step("optical_filter")])

    # never below 0.3 by itself: the measurement falls short on a noisy image
    assert cut.note.startswith("cut at 0.30 cycles/px") and cut.band is not None
    assert cut.noise == pytest.approx(300.0 * 0.56, rel=0.1)
    assert (
        by_hand.note.startswith("cut at 0.40 cycles/px") and by_hand.noise > cut.noise
    )
    # the renderer's Gaussian blur leaves pattern up to the sampling limit
    assert (
        "nothing to remove" in plain.final.note
        and plain.final.image is plain.states[0].image
    )
    with pytest.raises(WorkflowError, match="before Upscale"):
        workflow_contours(
            noisy, [step("upscale"), step("optical_filter"), step("feature_threshold")]
        )


def test_a_wide_feature_keeps_its_outline_when_holes_become_ellipses(mixed):
    image, where = mixed

    contours = workflow_contours(image, preset("One ellipse per hole"))

    measures = geometry.measure(contours)
    wide = np.nan_to_num(measures["diameter"]) > 16
    assert (
        wide.sum() == 1 and measures["points"][wide][0] > 200
    )  # the pad, point by point
    assert (
        measures["points"][contours.closed & ~wide] == 64
    ).all()  # every hole an ellipse
