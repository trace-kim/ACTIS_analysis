"""Packed contours and the outlines traced between the pixels."""

import numpy as np
import pytest

from actis.contourlab import contours as geometry
from actis.contourlab import ops


def _grid(rows=60, cols=70):
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return x, y


def _ellipse(a=6.0, b=3.0, degrees=25.0, centre=(50.0, 40.0), points=400):
    angle = np.deg2rad(degrees)
    t = np.linspace(0, 2 * np.pi, points, endpoint=False)
    xy = np.column_stack([a * np.cos(t), b * np.sin(t)])
    turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return xy @ turn + centre


def test_a_level_set_is_traced_between_the_pixels():
    x, y = _grid()
    field = (10.0 - np.hypot(x - 31.3, y - 27.6)).astype(np.float32)

    pixels = ops.trace_level(field, spacing=0.25, subpixel=False)
    traced = ops.trace_level(field, spacing=0.25)

    measures = geometry.measure(traced)
    assert len(traced) == 1 and traced.closed.all()
    assert measures["diameter"][0] == pytest.approx(20.0, abs=0.003)
    np.testing.assert_allclose(measures["centre"][0], [31.3, 27.6], atol=1e-4)
    assert measures["length"][0] == pytest.approx(2 * np.pi * 10, abs=0.01)
    radius = np.hypot(*(traced.xy - [31.3, 27.6]).T)
    assert abs(radius - 10).max() < 2e-3
    # marching squares alone cuts the corners of the curve between the pixels
    assert geometry.measure(pixels)["diameter"][0] < 19.99
    step = np.linalg.norm(np.diff(traced.xy, axis=0), axis=1)
    assert step.min() > 0.2 and step.max() < 0.3  # the spacing asked for


def test_contours_run_with_their_region_inside_and_normals_pointing_out():
    x, y = _grid()
    radius = np.hypot(x - 31.3, y - 27.6)
    ring = np.minimum(20 - radius, radius - 8).astype(np.float32)

    traced = ops.trace_level(ring, 0.25)

    area = geometry.measure(traced)["area"]
    assert sorted(np.sign(area)) == [-1, 1]  # the hole's boundary is negative
    np.testing.assert_allclose(sorted(area), [-np.pi * 64, np.pi * 400], rtol=1e-3)
    outer = traced.select(area > 0)
    away = (outer.xy - [31.3, 27.6]) / 20
    assert (geometry.normals(outer) * away).sum(axis=1).min() > 0.999
    inner = traced.select(area < 0)
    towards = ([31.3, 27.6] - inner.xy) / 8  # out of the region is into the hole
    assert (geometry.normals(inner) * towards).sum(axis=1).min() > 0.999


def test_a_region_that_leaves_the_image_gives_an_open_contour():
    x, _ = _grid()

    traced = ops.trace_level((x - 20.4).astype(np.float32), 0.5)

    assert len(traced) == 1 and not traced.closed[0]
    np.testing.assert_allclose(traced.xy[:, 0], 20.4, atol=1e-5)
    assert {traced.xy[0, 1], traced.xy[-1, 1]} == {0.0, 59.0}  # ends at the border
    measures = geometry.measure(traced)
    assert np.isnan(measures["area"][0]) and np.isnan(measures["diameter"][0])
    assert measures["length"][0] == pytest.approx(59.0)
    np.testing.assert_allclose(geometry.normals(traced)[5], [-1, 0], atol=1e-9)


def test_a_field_without_a_zero_level_has_no_contours():
    assert len(ops.trace_level(np.ones((20, 20), np.float32))) == 0
    assert len(ops.level_lines(np.zeros((1, 5), np.float32))) == 0
    assert geometry.measure(geometry.empty())["centre"].shape == (0, 2)


def test_measures_of_an_ellipse():
    ring = _ellipse()
    packed = geometry.from_list(
        [ring, ring[::-1] + 30, ring[:100]], [True, True, False]
    )

    measures = geometry.measure(packed)

    np.testing.assert_allclose(
        measures["area"][:2], [np.pi * 18, -np.pi * 18], rtol=1e-3
    )
    np.testing.assert_allclose(measures["major"][:2], 12, atol=0.01)
    np.testing.assert_allclose(measures["minor"][:2], 6, atol=0.01)
    np.testing.assert_allclose(np.degrees(measures["angle"][:2]), 25, atol=0.01)
    np.testing.assert_allclose(
        measures["centre"], [[50, 40], [80, 70], ring[:100].mean(0)]
    )
    assert np.isnan(measures["major"][2]) and list(measures["points"]) == [
        400,
        400,
        100,
    ]
    assert list(packed.counts) == [400, 400, 100] and len(packed.split()) == 3
    np.testing.assert_array_equal(packed.split()[1], ring[::-1] + 30)


def test_select_keeps_the_chosen_contours_and_counts_the_others():
    ring = _ellipse()
    packed = geometry.from_list([ring, ring + 1, ring[:7]], [True, True, False])

    kept = packed.select([True, False, True], small=1, large=0)
    again = kept.select([False, True], small=1)

    assert list(kept.counts) == [400, 7] and list(kept.closed) == [True, False]
    np.testing.assert_array_equal(kept.split()[1], ring[:7])
    assert kept.left_out == {"small": 1} and again.left_out == {"small": 2}
    assert packed.left_out == {}  # the contours given are not changed


def test_step_index_wraps_a_closed_contour_and_stops_at_the_ends_of_an_open_one():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    packed = geometry.from_list([square, square + 5], [True, False])

    ahead, valid = geometry.step_index(packed, 1)
    behind, known = geometry.step_index(packed, -2)

    assert list(ahead) == [1, 2, 3, 0, 5, 6, 7, 7]
    assert list(valid) == [True] * 4 + [True, True, True, False]
    assert list(behind) == [2, 3, 0, 1, 4, 4, 4, 5]
    assert list(known) == [True] * 4 + [False, False, True, True]


def test_resample_puts_the_points_at_equal_steps():
    ring = _ellipse()
    packed = geometry.from_list([ring, ring[:100]], [True, False])

    spaced = geometry.resample(packed, 0.5)

    closed, open_ = spaced.split()
    assert len(closed) == 58 and len(open_) == 15  # 29.07 and 7.17 px long
    np.testing.assert_allclose(open_[[0, -1]], ring[[0, 99]], atol=1e-9)  # both ends
    np.testing.assert_allclose(closed[0], ring[0], atol=1e-9)
    steps = np.linalg.norm(np.diff(open_, axis=0), axis=1)
    assert steps.std() < 0.01 * steps.mean()
    assert geometry.measure(spaced)["area"][0] == pytest.approx(np.pi * 18, rel=0.01)
    tiny = geometry.from_list([ring[:3] * 0.01])
    assert geometry.resample(tiny, 0.5).counts[0] == 8  # never fewer points


def test_smoothing_removes_scatter_without_shrinking_a_closed_contour():
    ring = _ellipse()
    rng = np.random.default_rng(0)
    noisy = geometry.from_list([ring + rng.normal(0, 0.05, ring.shape), ring[:6]])

    smooth = geometry.smooth(noisy, 15)

    before = np.sqrt(((noisy.split()[0] - ring) ** 2).mean())
    after = np.sqrt(((smooth.split()[0] - ring) ** 2).mean())
    assert after < 0.5 * before
    area = geometry.measure(smooth)["area"][0]
    assert area == pytest.approx(np.pi * 18, rel=2e-3)
    np.testing.assert_array_equal(
        smooth.split()[1], ring[:6]
    )  # shorter than the window


def test_one_harmonic_makes_an_ellipse_of_a_closed_contour():
    ring = _ellipse()
    rng = np.random.default_rng(1)
    noisy = geometry.from_list(
        [ring + rng.normal(0, 0.08, ring.shape), ring[::-1] + 30, ring[:50]],
        [True, True, False],
    )

    simple = geometry.fourier_shape(noisy, 1)

    measures = geometry.measure(simple)
    assert list(simple.counts) == [64, 64, 50]
    assert list(simple.closed) == [True, True, False]
    np.testing.assert_allclose(measures["major"][:2], 12, atol=0.05)
    np.testing.assert_allclose(measures["minor"][:2], 6, atol=0.05)
    np.testing.assert_allclose(np.degrees(measures["angle"][:2]), 25, atol=0.5)
    # the area is the contour's own, and so is the way round it runs
    np.testing.assert_allclose(
        measures["area"][:2], geometry.measure(noisy)["area"][:2], rtol=2e-3
    )
    np.testing.assert_array_equal(simple.split()[2], ring[:50])  # open: as it was
    centred = simple.split()[0] - measures["centre"][0]
    turn = np.deg2rad(25)
    u = centred @ [np.cos(turn), np.sin(turn)]
    v = centred @ [-np.sin(turn), np.cos(turn)]
    assert abs((u / 6) ** 2 + (v / 3) ** 2 - 1).max() < 0.03  # it is an ellipse


def test_more_harmonics_follow_the_outline_and_keep_its_area():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    radius = 5 * (1 + 0.12 * np.cos(4 * t))  # a rounded square
    ring = np.column_stack([20 + radius * np.cos(t), 30 + radius * np.sin(t)])
    rng = np.random.default_rng(2)
    noisy = geometry.from_list([ring + rng.normal(0, 0.05, ring.shape)])
    area = geometry.measure(geometry.from_list([ring]))["area"][0]

    for harmonics in (2, 3, 5, 8):
        simple = geometry.fourier_shape(noisy, harmonics)
        measures = geometry.measure(simple)
        assert measures["area"][0] == pytest.approx(area, rel=3e-3)
        np.testing.assert_allclose(measures["centre"][0], [20, 30], atol=0.02)
    spread = np.ptp(np.hypot(*(simple.xy - [20, 30]).T))
    assert spread == pytest.approx(2 * 5 * 0.12, abs=0.08)  # the corners are back
    round_ = geometry.fourier_shape(noisy, 2)
    assert np.ptp(np.hypot(*(round_.xy - [20, 30]).T)) < 0.5 * spread


def test_runs_cuts_contours_where_points_are_dropped():
    ring = _ellipse()
    packed = geometry.from_list([ring, ring + 30, ring[:100]], [True, True, False])
    keep = np.ones(900, bool)
    keep[10:20] = keep[395:400] = False  # two gaps in the first
    keep[450:460] = False  # one gap in the second: the rest joins across its start

    pieces, origin = ops.runs(packed, keep)

    assert list(pieces.counts) == [375, 10, 390, 100] and list(origin) == [0, 0, 1, 2]
    assert not pieces.closed.any()
    np.testing.assert_array_equal(pieces.split()[2][0], ring[60] + 30)
    np.testing.assert_array_equal(pieces.split()[2][-1], ring[49] + 30)
    whole, _ = ops.runs(packed, np.ones(900, bool))
    assert list(whole.closed) == [True, True, False] and list(whole.counts) == [
        400,
        400,
        100,
    ]
    none, _ = ops.runs(packed, np.zeros(900, bool))
    assert len(none) == 0


def test_group_median_ignores_missing_values():
    values = np.array([3.0, 1.0, 2.0, np.nan, 5.0, 7.0, np.nan])

    medians = geometry.group_median(values, np.array([0, 3, 6, 7]))

    np.testing.assert_array_equal(medians, [2.0, 6.0, np.nan])


def test_an_isolated_shift_takes_the_median_of_its_neighbours():
    ring = _ellipse()
    packed = geometry.from_list([ring, ring[:50]], [True, False])
    shift = np.full(450, 0.2)
    shift[5], shift[120], shift[449] = 2.0, np.nan, np.nan
    shift[200:230] = 0.9  # a real deviation spans many points: kept

    cleaned = geometry.without_isolated(shift, packed)

    assert cleaned[5] == cleaned[120] == cleaned[449] == 0.2
    np.testing.assert_array_equal(cleaned[205:225], 0.9)
    assert geometry.without_isolated(np.full(450, np.nan), packed).max() == 0.0
