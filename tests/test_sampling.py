import numpy as np

from actis.stress import _patterns, run_stress
from actis.synthetic import render_repeats


def test_oblique_lattice_is_independent_of_aperture_orientation_and_global_shift():
    basis = np.array([[12.2, 2.1], [5.1, 11.8]])
    opts = dict(
        shape="ellipse",
        lattice="oblique",
        image_shape=(56, 66),
        pitch_px=12.0,
        angle_deg=29.0,
        basis_xy=basis,
        n_repeats=1,
        nuisance=False,
    )
    image, truth = render_repeats(**opts)
    shifted, moved = render_repeats(**opts, global_shift_xy_px=(1.0, 0.0))
    # The whole-pixel special case recovers a translated patch. Fractional
    # exposures use the same object renderer before optics and sampling.
    np.testing.assert_array_equal(shifted[0, 15:-15, 15:-15], image[0, 15:-15, 14:-16])
    np.testing.assert_allclose(
        truth["lattice_rc"][:, ::-1] @ basis + [32.73, 27.33],
        truth["nominal_centers_xy_px"],
    )
    np.testing.assert_array_equal(moved["shift_xy_px"], [[1.0, 0.0]])
    np.testing.assert_array_equal(truth["basis_xy"], basis)


def test_distinct_source_regions_keep_orientation_size_and_axis_ratio():
    common = dict(
        shape="ellipse",
        lattice="square",
        pitch_px=12.0,
        cd_nm=130.0,
        angle_deg=11.0,
        ellipse_ratio=0.66,
        basis_xy=[[12.0, 0.0], [0.0, 12.0]],
    )
    regions = list(
        _patterns(
            [dict(common, region=2), dict(common, region=3, cd_nm=76.0, angle_deg=31.0)]
        )
    )
    assert [r["region"] for r in regions] == [2, 3]
    assert [r["cd_nm"] for r in regions] == [130.0, 76.0]
    assert [r["angle_deg"] for r in regions] == [11.0, 31.0]
    assert regions[1]["ellipse_ratio"] == 0.66


def test_subpixel_phase_scores_retain_paired_changes_and_independent_small_size():
    rows = run_stress(
        [
            dict(
                region=9,
                shape="circle",
                lattice="square",
                pitch_px=12.0,
                cd_nm=105.0,
                angle_deg=0.0,
                ellipse_ratio=1.0,
                basis_xy=[[12.0, 0.0], [0.0, 12.0]],
            )
        ],
        21.0,
    )
    contour = {r["scenario"]: r for r in rows if r["method"] == "contour"}
    assert contour["pixel_phase_cd"]["status"] == "ok"
    assert contour["pixel_phase_cd"]["region"] == 9
    assert contour["pixel_phase_cd"]["cd_levels_nm"] == [105.0, 94.5, 115.5]
    assert contour["pixel_phase_x"]["count"] == 5
    np.testing.assert_allclose(
        contour["pixel_phase_y"]["expected_changes"], [0, 0, 10.5, -10.5, -5.25]
    )
    assert contour["pixel_phase_x"]["peak_abs_error"] < 2.0
    assert contour["small_cd70"]["nominal_cd_nm"] == 70.0
    assert contour["small_cd70"]["cd_levels_nm"] == [70.0, 63.0, 77.0]
    assert contour["small_cd70"]["source_cd_nm"] == 105.0
    assert contour["small_cd70_phase"]["count"] == 3
