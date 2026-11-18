import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from actis.estimators import estimate_holes


def _test_image(shape="circle", size=5.0, angle=0.28, sigma=(0.75, 1.0)):
    """Render binary masks on a grid distinct from the fitting library."""
    centers = np.array(
        [[x + 0.23, y - 0.31] for x in (13, 25, 37) for y in (13, 25, 37)]
    )
    s = 12
    axis = (np.arange(51 * s) + 0.5) / s - 0.5
    y, x = np.meshgrid(axis, axis, indexing="ij")
    obj = np.zeros_like(x, dtype=np.float32)
    for cx, cy in centers:
        dx, dy = x - cx, y - cy
        u = np.cos(angle) * dx + np.sin(angle) * dy
        v = -np.sin(angle) * dx + np.cos(angle) * dy
        if shape == "square":
            inside = np.maximum(abs(u), abs(v)) < size / 2
        else:
            ratio = 0.68 if shape == "ellipse" else 1
            inside = np.hypot(u, v / ratio) < size / 2
        obj += inside
    image = (
        gaussian_filter(obj, (s * sigma[1], s * sigma[0]))
        .reshape(51, s, 51, s)
        .mean(axis=(1, 3))
    )
    return 3.5 + 47 * image, centers


@pytest.mark.parametrize("shape", ["square", "circle", "ellipse"])
def test_size_and_position_independent_from_affine_counts(shape):
    image, centers = _test_image(shape=shape)
    opts = dict(pitch_px=12, shape=shape, angle_rad=0.28, psf_sigma_px=(0.75, 1.0))
    seeds = centers + (0.35, -0.2)
    a = estimate_holes(image, seeds, **opts)
    b = estimate_holes(2.2 * image + 300, seeds, **opts)
    for method in ("contour", "model", "differential"):
        assert all(r["valid"] for r in a[method])
        assert np.allclose(
            [r["cd_px"] for r in a[method]], [r["cd_px"] for r in b[method]], atol=2e-3
        )
        assert all(r["contour_xy"].shape[1] == 2 for r in a[method])
    for row in a["contour"]:
        assert (
            row["contour_area_diameter40_px"]
            > row["area_diameter_px"]
            > row["contour_area_diameter60_px"]
        )
    result = a["model"]
    assert abs(np.median([r["cd_px"] for r in result]) - 5.0) < 0.2
    if shape == "ellipse":
        assert abs(np.median([r["height_px"] for r in result]) - 3.4) < 0.2
    assert (
        np.max(
            np.linalg.norm(
                np.array([(r["x_px"], r["y_px"]) for r in result]) - centers, axis=1
            )
        )
        < 0.12
    )
    assert np.isfinite(a["flux"][0]["flux_dn_px2"])
    assert np.isnan(a["flux"][0]["cd_px"])
    assert a["flux"][0]["flag"] == "uncalibrated_flux"


def test_width_changes_survive_tile_calibration_and_contours_reject_border():
    estimates = []
    for size in (3.4, 5.4):
        image, centers = _test_image(size=size)
        out = estimate_holes(
            image,
            np.vstack((centers, [1.0, 1.0])),
            pitch_px=12,
            shape="circle",
            psf_sigma_px=(0.75, 1.0),
            methods=("model", "flux"),
            dark_dn=3.5,
            bright_dn=50.5,
        )
        assert not out["model"][-1]["valid"]
        assert out["model"][-1]["flag"] == "border"
        assert out["flux"][3]["flag"] == "reference_normalized_flux"
        assert abs(out["flux"][3]["cd_px"] - size) < 0.2
        estimates.append(np.median([r["cd_px"] for r in out["model"] if r["valid"]]))
    assert abs(estimates[1] - estimates[0] - 2.0) < 0.25


def test_unresolved_and_ambiguous_geometry():
    image, centers = _test_image()
    result = estimate_holes(image, centers, pitch_px=12, shape="ambiguous")
    assert all(r["valid"] for r in result["contour"])
    assert all(
        not r["valid"] and r["flag"] == "ambiguous_shape" for r in result["model"]
    )
    assert not any(
        r["valid"]
        for r in estimate_holes(np.zeros((32, 32)), [[16, 16]], pitch_px=8)["contour"]
    )


def test_severe_blur_cannot_pass_as_a_larger_opening():
    image, centers = _test_image(size=3.4)
    rows = estimate_holes(
        gaussian_filter(image, 2.5),
        centers,
        pitch_px=12,
        shape="circle",
        methods=("model",),
    )["model"]
    assert all(not r["valid"] for r in rows)
    assert any(r["flag"] == "blur_at_search_limit" for r in rows)


def test_cuda_matches_cpu_when_available():
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except (OSError, RuntimeError):
        pytest.skip("CUDA runtime is unavailable")
    if not devices:
        pytest.skip("CUDA device is unavailable")
    image, centers = _test_image(shape="ellipse")
    opts = dict(
        pitch_px=12,
        shape="ellipse",
        angle_rad=0.28,
        methods=("model", "differential"),
        psf_sigma_px=(0.75, 1.0),
    )
    cpu = estimate_holes(image, centers, **opts)
    gpu = estimate_holes(image, centers, backend="cuda", **opts)
    for name in opts["methods"]:
        assert [r["flag"] for r in cpu[name]] == [r["flag"] for r in gpu[name]]
        for field in ("x_px", "y_px", "cd_px", "height_px"):
            np.testing.assert_allclose(
                [r[field] for r in cpu[name]], [r[field] for r in gpu[name]], atol=1e-4
            )
