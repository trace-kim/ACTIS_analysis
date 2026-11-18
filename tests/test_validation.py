import numpy as np
import pytest

from actis.statistics import precision, response_fit, tile_indices, triplet_scores
from actis.synthetic import render_repeats


def test_sample_precision_is_computed_across_acquisitions_per_bin():
    values = np.array([[9, 1, np.nan, -1], [10, 2, 4, 0], [11, 3, 4, 1]])
    result = precision(values)
    np.testing.assert_array_equal(result["count"], [3, 3, 2, 3])
    np.testing.assert_allclose(result["sigma3"], [3, 3, np.nan, 3], equal_nan=True)
    np.testing.assert_allclose(
        result["percent"], [30, 150, np.nan, np.nan], equal_nan=True
    )
    assert precision(values[:, 0])["percent"] == pytest.approx(30)


def test_all_triplets_use_all_acquisition_common_support():
    values = np.column_stack((np.arange(10.0, 18.0), np.ones(8)))
    values[-1, 1] = np.nan
    rows = triplet_scores(values)
    assert len(rows) == 56
    assert all(row["bins"] == 1 for row in rows)
    assert rows[0]["repeats"] == (0, 1, 2)
    assert rows[-1]["repeats"] == (5, 6, 7)
    assert rows[0]["percent"] == pytest.approx(300 / 11.0)
    assert rows[0]["sigma3"] == pytest.approx(3)
    assert not triplet_scores(values[:2])


def test_tiles_cover_only_complete_areas_and_use_row_major_ids():
    tile = 5000 / 21
    xy = [[0, 0], [tile, tile], [2 * tile, 0], [0, 8 * tile], [np.nan, 50], [-1, 0]]
    ids, shape = tile_indices(xy, (0, 0, 676, 2048))
    assert shape == (8, 2)
    np.testing.assert_array_equal(ids, [0, 3, -1, -1, -1, -1])
    ids, shape = tile_indices(xy, (0, 0, 676, 2048), tile_um=100)
    assert shape == (0, 0) and np.all(ids == -1)


def test_sensitivity_detects_measurement_that_hides_true_cd_changes():
    truth = np.array([80, 90, 100, 110, 120, np.nan])
    inferred = 3 + 0.5 * truth
    fit = response_fit(truth, inferred)
    assert fit["slope"] == pytest.approx(0.5)
    assert fit["intercept"] == pytest.approx(3)
    assert fit["mean_error"] == pytest.approx(-47)
    assert fit["rmse"] > 47 and fit["count"] == 5
    assert response_fit(truth, np.full(6, 100.0))["slope"] == pytest.approx(0)
    assert np.isnan(response_fit([80, 80], [82, 84])["slope"])


@pytest.mark.parametrize("shape", ["circle", "square", "ellipse"])
def test_known_right_half_cd_change_is_present_in_image_and_truth(shape):
    args = dict(
        shape=shape,
        lattice="staggered",
        image_shape=(64, 72),
        pitch_px=13.5,
        n_repeats=1,
        cd_nm=85,
        supersample=5,
        nuisance=False,
        angle_deg=4.0,
    )
    base, truth = render_repeats(**args)
    changed, new_truth = render_repeats(**args, local_delta_nm=12)
    np.testing.assert_array_equal(truth["site_id"], new_truth["site_id"])
    np.testing.assert_allclose(truth["centers_xy_px"], new_truth["centers_xy_px"])
    right = truth["local_region"]
    assert right.any() and (~right).any()
    np.testing.assert_array_equal(
        new_truth["cd_x_nm"] - truth["cd_x_nm"], np.where(right, 12, 0)
    )
    assert changed[:, :, 42:].mean() > base[:, :, 42:].mean() * 1.10
    np.testing.assert_array_equal(base[:, :, :22], changed[:, :, :22])
    ratio = {"circle": 1, "square": 2 / np.sqrt(np.pi), "ellipse": np.sqrt(0.75)}[shape]
    np.testing.assert_allclose(
        new_truth["area_diameter_nm"], ratio * new_truth["cd_x_nm"]
    )


def test_repeats_independently_change_detector_effects_but_preserve_geometry():
    args = dict(image_shape=(64, 72), n_repeats=3, supersample=4, seed=121)
    frames, truth = render_repeats(**args)
    rerun, _ = render_repeats(**args)
    _, shifted = render_repeats(
        **args, local_shift_xy_nm=(10.5, -4.2), global_delta_nm=2
    )
    assert frames.shape == (3, 64, 72) and frames.dtype == np.uint16
    assert np.max(frames) > 255 and not np.array_equal(frames[0], frames[1])
    np.testing.assert_array_equal(frames, rerun)
    for key in ["gain", "offset_dn", "sigma_xy_px", "shift_xy_px"]:
        assert np.ptp(truth[key], axis=0).min() > 0
        np.testing.assert_array_equal(truth[key], shifted[key])
    np.testing.assert_allclose(shifted["nominal_centers_xy_px"], truth["centers_xy_px"])
    local = truth["local_region"]
    np.testing.assert_allclose(
        shifted["centers_xy_px"] - truth["centers_xy_px"], local[:, None] * [0.5, -0.2]
    )
    np.testing.assert_allclose(shifted["cd_nm"] - truth["cd_nm"], 2)


def test_mask_translation_gives_expected_native_image_shift():
    args = dict(
        image_shape=(60, 80),
        pitch_px=14,
        cd_nm=100,
        n_repeats=2,
        supersample=4,
        nuisance=False,
    )
    reference, _ = render_repeats(**args)
    shifted, _ = render_repeats(**args, local_shift_xy_nm=(21.0, 0.0))
    np.testing.assert_array_equal(reference[0], reference[1])
    np.testing.assert_array_equal(shifted[0, 12:48, 54:67], reference[0, 12:48, 53:66])
    np.testing.assert_array_equal(shifted[0, 12:48, :24], reference[0, 12:48, :24])


def test_stress_scoring_uses_blind_analysis_and_independent_truth():
    from actis.stress import run_stress

    rows = run_stress([("circle", "square", 12, 100)] * 2, pixel_nm=21)
    by_key = {(r["method"], r["scenario"]): r for r in rows}
    assert len(rows) == len(by_key)
    assert {r["method"] for r in rows} == {
        "raw",
        "flux",
        "contour",
        "model",
        "differential",
        "harmonic",
        "pooled",
    }
    raw = by_key["raw", "global_cd"]
    assert raw["count"] == 3 and raw["unit"] == "DN" and raw["slope"] > 0
    assert np.isnan(raw["rmse"])  # raw brightness and geometric CD have different units
    assert raw["fractional_slope"] == pytest.approx(raw["slope"] / raw["nominal_value"])
    for scenario in [
        "global_cd",
        "global_cd_tiles",
        "local_cd",
        "placement_x",
        "placement_y",
    ]:
        row = by_key["contour", scenario]
        assert row["status"] == "ok" and row["support"] > 0
        assert 0.65 < row["slope"] < 1.35
    assert by_key["flux", "global_cd"]["unit"] == "DN·px²"
    assert by_key["raw", "nuisance"]["sigma3"] > 0
    assert 0 < by_key["raw", "clipping"]["clipped_fraction"] < 1
