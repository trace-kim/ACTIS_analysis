"""Independent mask renderer for controlled image-domain validation on Windows.

The intensity model supplies known geometry with finite sampling, an optical
field/intensity mixture, flare and detector effects. Its parameters describe
synthetic stress tests. They have no claimed calibration to an ACTIS recipe.
Native pixel centers have integer coordinates in (x, y), increasing right/down.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def lattice_basis(lattice, pitch, angle_deg, basis_xy=None):
    """Primitive row vectors in detector (x, y). Shape orientation is separate."""
    if basis_xy is None:
        if lattice == "oblique":
            raise ValueError(
                "An oblique lattice requires explicit primitive basis vectors"
            )
        ang = np.deg2rad(angle_deg)
        rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        basis_xy = (
            np.array(
                [[pitch, 0.0], [0.5 * pitch if lattice == "staggered" else 0.0, pitch]]
            )
            @ rot
        )
    basis = np.asarray(basis_xy, dtype=float)
    if (
        basis.shape != (2, 2)
        or not np.isfinite(basis).all()
        or np.linalg.cond(basis) > 100
    ):
        raise ValueError(
            "Primitive lattice vectors must form a finite nonsingular 2x2 matrix"
        )
    return basis


def _lattice_sites(image_shape, basis, margin):
    h, w = image_shape
    origin = np.array([(w - 1) / 2 + 0.23, (h - 1) / 2 - 0.17])
    corners = np.array(
        [
            [-margin, -margin],
            [w + margin, -margin],
            [w + margin, h + margin],
            [-margin, h + margin],
        ]
    )
    fractions = (corners - origin) @ np.linalg.inv(basis)
    lower = np.floor(fractions.min(axis=0)).astype(int)
    upper = np.ceil(fractions.max(axis=0)).astype(int)
    rr, cc = np.mgrid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
    indices = np.column_stack((cc.ravel(), rr.ravel()))
    centers = indices @ basis + origin
    near = ((centers >= -margin) & (centers < [w + margin, h + margin])).all(axis=1)
    return centers[near], indices[near, ::-1]


def _aperture(image_shape, centers, widths, heights, square, cos_a, sin_a, samples):
    """Sample a signed-distance approximation on a fine grid before optical blur."""
    h, w = image_shape
    aperture = np.zeros((h * samples, w * samples), dtype=np.float32)
    axes_x = (np.arange(w * samples) + 0.5) / samples - 0.5
    axes_y = (np.arange(h * samples) + 0.5) / samples - 0.5
    for (cx, cy), width, height in zip(centers, widths, heights):
        # The radius encloses a rotated square or ellipse at any angle.
        radius = np.hypot(width, height) / 2 + 1.0 / samples
        x0, x1 = axes_x.searchsorted([cx - radius, cx + radius])
        y0, y1 = axes_y.searchsorted([cy - radius, cy + radius])
        if x0 == x1 or y0 == y1:
            continue
        dx = axes_x[None, x0:x1] - cx
        dy = axes_y[y0:y1, None] - cy
        u, v = cos_a * dx + sin_a * dy, -sin_a * dx + cos_a * dy
        half_w, half_h = width / 2.0, height / 2.0
        if square:
            inside = np.minimum(half_w - np.abs(u), half_h - np.abs(v))
        else:
            inside = (1.0 - np.hypot(u / half_w, v / half_h)) * min(half_w, half_h)
        patch = aperture[y0:y1, x0:x1]
        np.maximum(patch, np.clip(0.5 + samples * inside, 0, 1), out=patch)
    return aperture


def _capture(aperture, blur_xy, samples, h, w):
    # Two terms with a common illustrative pupil scale. The nonlinear term
    # perturbs the apparent contour of small openings; no estimation code runs here.
    sx, sy = blur_xy
    field = gaussian_filter(aperture, (sy * samples, sx * samples), mode="nearest")
    field *= 0.80 + 0.20 * field
    native = field.reshape(h, samples, w, samples).mean(axis=(1, 3))
    return 0.96 * native + 0.04 * gaussian_filter(native, 2.5, mode="nearest")


def render_repeats(
    *,
    shape="circle",
    lattice="square",
    image_shape=(192, 160),
    pixel_nm=21.0,
    pitch_px=12.0,
    cd_nm=100.0,
    cd_variation_nm=0.0,
    global_delta_nm=0.0,
    local_delta_nm=0.0,
    n_repeats=8,
    seed=47,
    supersample=8,
    nuisance=True,
    angle_deg=0.0,
    local_shift_xy_nm=(0.0, 0.0),
    ellipse_ratio=0.75,
    basis_xy=None,
    global_shift_xy_px=(0.0, 0.0),
    sigma_xy_px=(0.78, 0.96),
):
    """Return a uint16 array [run,y,x] and ground truth in the nominal mask frame.

    The base CD is a square side, a circle diameter, or an ellipse major
    diameter. Ellipse minor/major ratio defaults to 0.75. Global and
    right-half CD steps add in nm before optical formation. Right-half
    membership uses the original center. Local displacement is applied before
    the acquisition translation. Positive angles rotate +x toward +y.

    cd_variation_nm bounds an optional fixed mask variation shared by all
    acquisitions. Two smooth local size changes and a smaller site-to-site
    component provide landmarks for comparing measured dimensions.

    Geometry and per-acquisition nuisance parameters are consistent across CD
    sweeps for a fixed seed and configuration. Setting nuisance=False selects
    one optical condition with unit gain and no noise. The explicit global
    translation moves the mask before optics and native pixel integration.
    Returned site centers exclude the half-pitch border. Partial apertures are
    still rendered there. Widths along the pattern axes and area-equivalent
    circle diameter are supplied separately.
    """
    if shape not in {"circle", "square", "ellipse"} or lattice not in {
        "square",
        "staggered",
        "oblique",
    }:
        raise ValueError("Unsupported aperture shape or lattice family")
    dims = np.asarray(image_shape)
    if dims.shape != (2,) or dims.dtype.kind not in "iu" or np.any(dims < 1):
        raise ValueError("image_shape must be two positive integers (height, width)")
    if not isinstance(supersample, (int, np.integer)) or supersample < 1:
        raise ValueError("supersample must be a positive integer")
    if not isinstance(n_repeats, (int, np.integer)) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    shifts_nm = np.asarray(local_shift_xy_nm, dtype=float)
    if shifts_nm.shape != (2,) or not np.isfinite(shifts_nm).all():
        raise ValueError("local_shift_xy_nm must be a finite (x, y) pair")
    global_shift = np.asarray(global_shift_xy_px, dtype=float)
    if global_shift.shape != (2,) or not np.isfinite(global_shift).all():
        raise ValueError("global_shift_xy_px must be a finite (x, y) pair")
    sigma_xy = np.asarray(sigma_xy_px, dtype=float)
    if (
        sigma_xy.shape != (2,)
        or not np.isfinite(sigma_xy).all()
        or np.any(sigma_xy <= ([0.18, 0.20] if nuisance else 0))
    ):
        raise ValueError("Optical blur widths must stay positive across acquisitions")
    parameters = [
        pixel_nm,
        pitch_px,
        cd_nm,
        cd_variation_nm,
        global_delta_nm,
        local_delta_nm,
        angle_deg,
    ]
    if not np.isfinite(parameters).all() or min(pixel_nm, pitch_px) <= 0.0:
        raise ValueError(
            "geometry requires finite parameters and positive pixel and pitch sizes"
        )
    if not np.isfinite(ellipse_ratio) or not 0 < ellipse_ratio <= 1:
        raise ValueError("ellipse_ratio must be in (0, 1]")
    cd_levels = [cd_nm + global_delta_nm, cd_nm + global_delta_nm + local_delta_nm]
    if cd_variation_nm < 0:
        raise ValueError("CD variation must be nonnegative")
    if (
        min(cd_levels) - cd_variation_nm <= 0
        or max(cd_levels) + cd_variation_nm >= pixel_nm * pitch_px
    ):
        raise ValueError("CD levels must be positive and smaller than the pitch")

    h, w = map(int, dims)
    radians = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    basis = lattice_basis(lattice, pitch_px, angle_deg, basis_xy)
    halo = pitch_px + max(abs(global_shift)) + max(abs(shifts_nm)) / pixel_nm
    centers, lattice_rc = _lattice_sites((h, w), basis, halo)
    local = centers[:, 0] >= (w - 1) / 2
    moved = centers + local[:, None] * (shifts_nm / pixel_nm)
    major = np.where(local, cd_levels[1], cd_levels[0])
    if cd_variation_nm:
        x, y = centers.T / np.maximum([w - 1, h - 1], 1)[:, None]
        broader = np.exp(-0.5 * (((x - 0.36) / 0.15) ** 2 + ((y - 0.45) / 0.20) ** 2))
        narrower = np.exp(-0.5 * (((x - 0.75) / 0.12) ** 2 + ((y - 0.67) / 0.16) ** 2))
        mask_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        variation = 0.75 * (broader - narrower) + 0.25 * mask_rng.uniform(
            -1, 1, len(major)
        )
        major = major + cd_variation_nm * variation
    minor = major * (ellipse_ratio if shape == "ellipse" else 1.0)
    area_cd = (
        2 * major / np.sqrt(np.pi) if shape == "square" else np.sqrt(major * minor)
    )
    margin = pitch_px / 2
    keep = (
        (centers >= margin - 0.5) & (centers <= [w - 0.5 - margin, h - 0.5 - margin])
    ).all(axis=1)
    frames = np.empty((n_repeats, h, w), dtype=np.uint16)
    shifts = np.broadcast_to(global_shift, (n_repeats, 2)).copy()
    blurs = np.empty((n_repeats, 2))
    gains = np.ones(n_repeats)
    offsets = np.full(n_repeats, 1400.0)
    xn = np.linspace(-1, 1, w)[None, :]
    yn = np.linspace(-1, 1, h)[:, None]
    for run in range(n_repeats):
        if run and not nuisance:
            frames[run] = frames[0]
            blurs[run] = blurs[0]
            continue
        optics_rng = np.random.default_rng(np.random.SeedSequence([seed, run, 1]))
        sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, run, 2]))
        blurs[run] = sigma_xy
        illumination = 1.0
        if nuisance:
            if run:
                shifts[run] += optics_rng.uniform(-0.65, 0.65, size=2)
            blurs[run] += optics_rng.uniform([-0.18, -0.20], [0.34, 0.48])
            gains[run] = optics_rng.uniform(0.79, 1.21)
            offsets[run] += optics_rng.uniform(-360.0, 360.0)
            illumination = (
                1
                + optics_rng.uniform(-0.08, 0.08) * xn
                + optics_rng.uniform(-0.05, 0.05) * yn
            )
        aperture = _aperture(
            (h, w),
            moved + shifts[run],
            major / pixel_nm,
            minor / pixel_nm,
            shape == "square",
            cos_a,
            sin_a,
            supersample,
        )
        signal = (
            22000.0
            * gains[run]
            * illumination
            * _capture(aperture, blurs[run], supersample, h, w)
        )
        if nuisance:
            # Detector-column response, TDI line offsets, shot and read noise.
            signal *= sensor_rng.normal(1.0, 0.008, size=(1, w))
            signal = 6.0 * sensor_rng.poisson(np.maximum(signal, 0.0) / 6.0)
            signal += sensor_rng.normal(0.0, 27.0, size=(h, 1))
            signal += sensor_rng.normal(0.0, 36.0, size=(h, w))
        frames[run] = np.clip(np.rint(signal + offsets[run]), 0, 65535).astype(
            np.uint16
        )

    truth = {
        "shape": shape,
        "lattice": lattice,
        "angle_deg": float(angle_deg),
        "cd_variation_nm": float(cd_variation_nm),
        "ellipse_ratio": float(ellipse_ratio),
        "pitch_px": float(pitch_px),
        "pixel_nm": float(pixel_nm),
        "basis_xy": basis,
        "centers_xy_px": moved[keep],
        "nominal_centers_xy_px": centers[keep],
        "site_id": np.arange(np.count_nonzero(keep)),
        "lattice_rc": lattice_rc[keep],
        "local_region": local[keep],
        "local_shift_xy_nm": shifts_nm.copy(),
        "cd_nm": major[keep],
        "cd_x_nm": major[keep],
        "cd_y_nm": minor[keep],
        "area_diameter_nm": area_cd[keep],
        "shift_xy_px": shifts,
        "sigma_xy_px": blurs,
        "gain": gains,
        "offset_dn": offsets,
    }
    return frames, truth
