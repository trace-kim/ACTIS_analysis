from types import SimpleNamespace

import numpy as np
import pytest

from actis import pooled
from actis.synthetic import render_repeats


def _geometry(truth, shape_yx, shape, pitch, theta, staggered):
    unit = np.array([np.cos(theta), np.sin(theta)])
    cross = np.array([-np.sin(theta), np.cos(theta)])
    basis = np.stack(
        (pitch * unit, pitch * (cross + (0.5 if staggered else 0.0) * unit))
    )
    xy = truth["nominal_centers_xy_px"]
    return SimpleNamespace(
        centers=xy,
        basis=basis,
        pitch=pitch,
        shape=shape,
        angle=theta,
        bounds=(0, 0, shape_yx[1], shape_yx[0]),
        region_id=0,
        diagnostics={"anchor_xy": xy[0]},
    )


@pytest.mark.parametrize(
    "shape,lattice",
    [("square", "staggered"), ("ellipse", "staggered"), ("circle", "square")],
)
def test_pooled_recovers_cd_placement_and_gain_on_rotated_native_lattice(
    shape, lattice
):
    params = dict(
        shape=shape,
        lattice=lattice,
        image_shape=(96, 192),
        pitch_px=13,
        cd_nm=120,
        n_repeats=1,
        nuisance=False,
        angle_deg=12,
        supersample=5,
    )
    base, truth = render_repeats(**params)
    region = _geometry(
        truth, base.shape[1:], shape, 13, np.deg2rad(12), lattice == "staggered"
    )
    shifted, _ = render_repeats(**params, local_shift_xy_nm=(8.4, -6.3))
    changed, _ = render_repeats(**params, local_delta_nm=10)

    def score(image):
        return pooled.pooled_tiles(image, region, tile_px=90, origin_xy=(-0.5, -0.5))

    ref, moved, sized = score(base[0]), score(shifted[0]), score(changed[0])
    illuminated = score(base[0].astype(float) * 1.3 + 800)
    assert len(ref) == 2 and all(
        r["status"] == "ok" for r in ref + moved + sized + illuminated
    )
    assert min(r["valid_fraction"] for r in ref) > 0.9
    np.testing.assert_allclose(
        [r["cd_px"] for r in ref], [r["cd_px"] for r in illuminated], atol=1e-7
    )
    right = 1
    cd_change = 21 * (sized[right]["cd_px"] - ref[right]["cd_px"])
    assert 6 <= cd_change <= 13
    assert (
        np.hypot(
            ref[0]["dx_px"] - moved[0]["dx_px"], ref[0]["dy_px"] - moved[0]["dy_px"]
        )
        < 0.02
    )
    np.testing.assert_allclose(
        [
            moved[right]["dx_px"] - ref[right]["dx_px"],
            moved[right]["dy_px"] - ref[right]["dy_px"],
        ],
        [0.4, -0.3],
        atol=0.12,
    )
    if shape == "ellipse":
        assert ref[0]["cd_x_px"] > ref[0]["cd_y_px"] > 0


def test_degenerate_data_rejects_all_apertures_and_ambiguous_shape_uses_area():
    frames, truth = render_repeats(image_shape=(84, 84), n_repeats=1, nuisance=False)
    region = _geometry(truth, (84, 84), "ambiguous", 12, 0, False)
    for fill in (1400, 65535):
        result = pooled.pooled_tiles(np.full_like(frames[0], fill), region, tile_px=80)[
            0
        ]
        assert result["status"] == "insufficient_valid_apertures"
        assert result["valid_sites"] == 0 and np.isnan(result["cd_px"])
    valid = pooled.pooled_tiles(frames[0], region, tile_px=80)[0]
    assert valid["cd_px"] == valid["area_diameter_px"] and np.isnan(valid["cd_x_px"])
    assert pooled.pooled_tiles(frames[0], region, tile_px=100) == []


def test_vectorized_and_native_moment_reductions_match(monkeypatch):
    frames, truth = render_repeats(image_shape=(84, 84), n_repeats=1, nuisance=True)
    region = _geometry(truth, (84, 84), "circle", 12, 0, False)
    forward = pooled.pooled_tiles(frames[0], region, tile_px=80)[0]
    monkeypatch.setattr(pooled, "_moments", pooled._moments_numpy)
    fallback = pooled.pooled_tiles(frames[0], region, tile_px=80)[0]
    assert forward["status"] == fallback["status"] == "ok"
    for key in ("cd_px", "dx_px", "dy_px", "valid_fraction"):
        np.testing.assert_allclose(fallback[key], forward[key], atol=1e-12)


def test_repeat_score_intersects_actual_site_ids_and_isolates_failed_challenge():
    frames, truth = render_repeats(
        image_shape=(120, 120),
        n_repeats=1,
        nuisance=False,
        local_delta_nm=16,
        pitch_px=14,
    )
    region = _geometry(truth, (120, 120), "circle", 14, 0, False)
    left, right = frames[0].copy(), frames[0].copy()
    for target, near in [(left, [30, 30]), (right, [89, 89])]:
        i = np.argmin(np.linalg.norm(region.centers - near, axis=1))
        x, y = np.rint(region.centers[i]).astype(int)
        target[y - 6 : y + 7, x - 6 : x + 7] = 1400
    exposure = [left, right, np.full_like(left, 1400)]
    result = pooled.pooled_group_tiles(
        exposure, region, np.zeros((3, 2)), 116, (-0.5, -0.5), [(0, 1), (0, 2)]
    )
    first, second = (rows[0] for rows in result[0, 1])
    assert first["status"] == second["status"] == "ok"
    assert (
        first["valid_sites"]
        == second["valid_sites"]
        < min(first["native_valid_sites"], second["native_valid_sites"])
    )
    assert first["cd_px"] == pytest.approx(second["cd_px"], abs=1e-12)
    assert all(rows[0]["valid_sites"] == 0 for rows in result[0, 2])
