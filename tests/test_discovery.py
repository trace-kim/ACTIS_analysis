import numpy as np
from scipy import ndimage

from actis.discovery import detect_centers, discover, match_centers, register
from actis.harmonics import _interpolated_dft, harmonic_tiles


def array_image(
    shape=(192, 192),
    size=7.0,
    rotation=0.0,
    stagger=False,
    shift=(0.0, 0.0),
    opening="circle",
    pitch=16.0,
    aspect=1.0,
    blur=0.85,
):
    # Rasterize subpixels independently from the frequency-domain fitting model.
    y, x = np.indices(shape, dtype=float)
    c, s = np.cos(rotation), np.sin(rotation)
    image = np.zeros(shape)
    for sy in (-0.375, -0.125, 0.125, 0.375):
        for sx in (-0.375, -0.125, 0.125, 0.375):
            xx = x + sx - shift[0] - 8
            yy = y + sy - shift[1] - 8
            u, v = c * xx + s * yy, -s * xx + c * yy
            row = np.rint(v / pitch)
            u -= row * (pitch / 2 if stagger else 0.0)
            u -= pitch * np.rint(u / pitch)
            v -= pitch * row
            if opening == "square":
                image += np.maximum(abs(u), abs(v)) <= size / 2
            else:
                image += (2 * u / size) ** 2 + (2 * v * aspect / size) ** 2 <= 1
    return ndimage.gaussian_filter(image / 16, blur)


def test_rotated_stagger_basis_and_native_correspondence():
    img = array_image(stagger=True, rotation=0.14)
    regions = discover(img)
    assert len(regions) == 1
    r = regions[0]
    assert r.lattice == "staggered"
    assert abs(abs(np.linalg.det(r.basis)) - 16**2) < 1
    assert r.confidence > 0.95 and r.pitch > 15
    shift = np.array([1.3, -0.6])
    shuffled = (r.centers + shift)[::-1]
    pairs = match_centers(r.centers, shuffled, shift, r.pitch)
    assert np.allclose(r.centers[pairs[:, 0]] + shift, shuffled[pairs[:, 1]])


def test_dense_native_pixel_lattice_keeps_phase_across_full_detector_height():
    # A rounded threshold centroid depends on the subpixel sampling phase.
    # On this field the median apparent neighbor spacing formerly lost nearly
    # three rows of phase and all measurements were dropped at discovery.
    h, w = 2048, 676
    pitch = 5.9
    y, x = np.ogrid[:h, :w]
    dx = (x + 0.30 + pitch / 2) % pitch - pitch / 2
    dy = (y + 0.13 + pitch / 2) % pitch - pitch / 2
    exposure = (
        3000 + 11000 * np.exp(-0.5 * ((dx / 1.24) ** 2 + (dy / 1.34) ** 2))
    ).astype("uint16")
    diagnostics = {}
    regions = discover(exposure, diagnostics=diagnostics)
    assert diagnostics["candidate_centers"] > 39000
    assert diagnostics["accepted_regions"] == len(regions) == 1
    assert diagnostics["rejected"] == []
    region = regions[0]
    assert region.confidence > 0.99
    np.testing.assert_allclose(region.basis, pitch * np.eye(2), atol=0.001)


def test_discovery_rejections_distinguish_missing_sites_from_a_line_of_sites():
    image = np.zeros((100, 200))
    diagnostics = {}
    assert not discover(image, diagnostics=diagnostics)
    assert diagnostics["candidate_centers"] == 0
    assert diagnostics["rejected"][0]["reason"] == "too_few_centers"

    for x in range(8, 195, 12):
        image[48:52, x : x + 4] = 1
    assert not discover(image, diagnostics=diagnostics)
    assert diagnostics["candidate_centers"] == 16
    assert diagnostics["rejected"][0]["reason"] == "no_independent_lattice"


def test_subpixel_registration_and_aliases():
    img = array_image(shape=(192, 208))
    delta = (1.45, -2.25)
    mov = array_image(shape=img.shape, shift=delta) * 1.42 + 0.3
    reg = register(img, mov, 16.0)
    np.testing.assert_allclose(reg.shift_xy, delta, atol=0.21)
    assert not reg.ambiguous
    assert reg.periodic_ambiguity
    assert register(img, mov, 16.0, max_shift_px=22).ambiguous
    assert register(img * 0, mov, 16.0).ambiguous


def test_lattice_alias_rejected_when_field_structure_contradicts_small_shift():
    img = array_image(shape=(192, 208))
    weights = np.random.default_rng(10).uniform(0.4, 1.8, (12, 13))
    img *= ndimage.zoom(weights, (16, 16), order=1)
    moving = ndimage.shift(img, (-2.25, 17.45), order=3)
    assert register(img, moving, 16).reason == "stronger_peak_outside_prior"
    anchored = register(img, moving, 16, max_shift_px=24)
    np.testing.assert_allclose(anchored.shift_xy, [17.45, -2.25], atol=0.23)
    assert not anchored.ambiguous
    assert len(detect_centers(img)) > 100


def test_shape_evidence_and_separated_regions():
    a = array_image((160, 128), size=10, opening="square")
    b = array_image((160, 128), size=9, opening="ellipse", aspect=1.7, pitch=13)
    regions = discover(np.hstack([a, np.zeros((160, 36)), b]))
    assert len(regions) >= 2
    assert "square" in {r.best_shape for r in regions}
    assert "ellipse" in {r.best_shape for r in regions}
    tiny = discover(array_image(size=3.6, blur=1.5), pitch_px=16)
    assert tiny and tiny[0].shape == "ambiguous"


def test_adjacent_domains_are_resolved_through_reserved_orders():
    square = array_image((192, 192), size=5.0, opening="square")
    circle = array_image((192, 192), size=5.0, opening="circle")
    groups = sorted(discover(np.hstack([square, circle])), key=lambda r: r.bounds)
    assert [r.shape for r in groups] == ["square", "circle"]
    assert abs(groups[0].bounds[2] - 192) < 2
    assert all(r.diagnostics["tested_orders"] > 5 for r in groups)


def test_independent_spatial_profile_check_preserves_gain_and_shift_invariance():
    a = array_image((192, 352), size=7.0, opening="circle")
    moving = array_image(a.shape, size=7.0, opening="circle", shift=(0.45, -0.3))
    a[:, 176:] = 1.4 * moving[:, 176:] + 0.12
    r = discover(a, regions=[{"bounds": (0, 0, 352, 192)}])[0]
    assert r.shape == "circle"
    assert r.diagnostics["spatial_holdout_status"] == "validated"
    train, test = r.diagnostics["shape_patch_bounds_xyxy"]
    assert train[2] <= test[0] or train[3] <= test[1]
    scores = r.diagnostics["spatial_holdout_rms"]
    assert scores["circle"] < scores["square"]
    small = discover(array_image((144, 144), size=7.0))[0]
    assert (
        small.diagnostics["spatial_holdout_status"]
        == "small_region_reserved_orders_only"
    )


def test_large_tile_fourier_interpolation_against_direct_native_samples():
    rng = np.random.default_rng(101)
    extent = (211, 237)
    patch = (
        rng.normal(size=extent) * np.hanning(extent[0])[:, None] * np.hanning(extent[1])
    )
    q = rng.uniform(-0.45, 0.45, (23, 2))
    tx = np.exp(-2j * np.pi * np.outer(q[:, 0], np.arange(extent[1])))
    ty = np.exp(-2j * np.pi * np.outer(q[:, 1], np.arange(extent[0])))
    direct = np.sum((ty @ patch) * tx, axis=1)
    estimated = _interpolated_dft(patch, q)
    assert np.max(abs(estimated - direct)) < 0.004 * np.max(abs(direct))


def test_signed_harmonics_recover_array_cd_gain_invariance_and_grid_origin():
    base = array_image()
    reg = discover(base, shape="circle")[0]
    tiles = harmonic_tiles(base, reg, tile_px=96, origin_xy=(8.0, 7.0))
    assert len(tiles) == 1 and tiles[0]["status"] == "ok"
    assert tiles[0]["x"] == 56.0 and tiles[0]["tile_i"] == 0
    assert abs(tiles[0]["cd_px"] - 7.0) < 0.5
    assert np.hypot(tiles[0]["dx_px"], tiles[0]["dy_px"]) < 0.1
    changed = harmonic_tiles(410 * base + 5300, reg, tile_px=96, origin_xy=(8.0, 7.0))
    assert abs(tiles[0]["cd_px"] - changed[0]["cd_px"]) < 1e-6
