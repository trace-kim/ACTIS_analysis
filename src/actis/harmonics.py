"""Local array-average CD from signed lattice harmonics and translation phase.

This is an effective, linear intensity-transfer model. A filled opening's
form factor, pixel integration, and an anisotropic Gaussian blur determine the
harmonic envelope. Instrument characterization is needed for physical CD.
"""

from dataclasses import replace

import numpy as np
from scipy import fft
from scipy.optimize import least_squares
from scipy.special import j0, j1


def _form_factor(freq, width, height, angle, shape):
    c, s = np.cos(angle), np.sin(angle)
    u = width * (freq[:, 0] * c + freq[:, 1] * s)
    v = height * (-freq[:, 0] * s + freq[:, 1] * c)
    if shape == "square":
        return np.sinc(u) * np.sinc(v)
    z = np.pi * np.hypot(u, v)
    return np.divide(2 * j1(z), z, out=np.ones_like(z), where=z > 1e-9)


def _fit_orders(amplitude, freq, noise, pitch, region):
    shape = region.shape
    ellipse = shape == "ellipse"
    scale = max(float(np.abs(amplitude).max()), 1e-15)
    target = amplitude / scale
    error = np.maximum(noise / scale, 0.008)
    pixel = np.sinc(freq[:, 0]) * np.sinc(freq[:, 1])

    def unpack(z):
        return (
            z[0] * pitch,
            (z[1] if ellipse else z[0]) * pitch,
            z[-3] * pitch,
            z[-2] * pitch,
            np.exp(z[-1]),
        )

    cos_a, sin_a = np.cos(region.angle), np.sin(region.angle)
    along = freq[:, 0] * cos_a + freq[:, 1] * sin_a
    across = -freq[:, 0] * sin_a + freq[:, 1] * cos_a

    def residual(z):
        wid, ht, sx, sy, gain = unpack(z)
        form = _form_factor(freq, wid, ht, region.angle, shape)
        mtf = np.exp(-2 * np.pi**2 * ((sx * freq[:, 0]) ** 2 + (sy * freq[:, 1]) ** 2))
        return (gain * form * mtf * pixel - target) / error

    def jacobian(z):
        # Gaussian parameters and aperture dimensions have separate analytic
        # derivatives. Avoid finite-differencing each parameter for every tile.
        wid, ht, sx, sy, gain = unpack(z)
        u, v = wid * along, ht * across
        if shape == "square":
            su, sv = np.sinc(u), np.sinc(v)
            form = su * sv
            # d sinc(wf) / d w = [cos(pi*w*f) - sinc(w*f)] / w.
            dw = (np.cos(np.pi * u) - su) * sv / wid
            dh = (np.cos(np.pi * v) - sv) * su / ht
        else:
            radius = np.pi * np.hypot(u, v)
            form = np.divide(
                2 * j1(radius), radius, out=np.ones_like(radius), where=radius > 1e-8
            )
            gradient = (
                np.divide(
                    2 * (j0(radius) - form),
                    radius**2,
                    out=np.full_like(radius, -0.25),
                    where=radius > 1e-4,
                )
                * np.pi**2
            )
            dw, dh = gradient * wid * along**2, gradient * ht * across**2
        prefactor = (
            gain
            * pixel
            * np.exp(-2 * np.pi**2 * ((sx * freq[:, 0]) ** 2 + (sy * freq[:, 1]) ** 2))
            / error
        )
        size_cols = [pitch * dw, pitch * dh] if ellipse else [pitch * (dw + dh)]
        d_blur = -4 * np.pi**2 * pitch * form
        values = size_cols + [
            d_blur * sx * freq[:, 0] ** 2,
            d_blur * sy * freq[:, 1] ** 2,
            form,
        ]
        return prefactor[:, None] * np.column_stack(values)

    low = ([0.075, 0.075] if ellipse else [0.075]) + [0.004, 0.004, -2.0]
    high = ([1.20, 1.20] if ellipse else [1.20]) + [0.35, 0.35, 4.0]
    best = None
    for width in (0.23, 0.46, 0.74):
        init = ([width, width * 0.78] if ellipse else [width]) + [0.065, 0.065, 0.25]
        fit = least_squares(
            residual,
            init,
            jac=jacobian,
            bounds=(low, high),
            max_nfev=100,
            ftol=1e-7,
            gtol=1e-7,
            xtol=1e-7,
        )
        if best is None or fit.cost < best.cost:
            best = fit
    J = best.jac
    norm = np.linalg.norm(J, axis=0)
    cond = float(np.linalg.cond(J / np.maximum(norm, 1e-14)))
    wid, ht, sx, sy, _ = unpack(best.x)
    rms = float(np.sqrt(np.mean((best.fun * error) ** 2)))
    # A boundary solution for optical blur can still occur on an actual sharp
    # simulated image; boundary CD or a near-null CD direction is unusable.
    n_cd = 2 if ellipse else 1
    projected = (
        J[:, :n_cd]
        - J[:, n_cd:] @ np.linalg.lstsq(J[:, n_cd:], J[:, :n_cd], rcond=None)[0]
    )
    sensitivity = float(np.linalg.svd(projected, compute_uv=False)[-1])
    status = "ok"
    if (
        not best.success
        or np.any(best.active_mask[:n_cd])
        or cond > 1e4
        or sensitivity < 7.0
    ):
        status = "size_blur_unresolved"
    elif rms > 0.095:
        status = "transfer_model_mismatch"
    return wid, ht, sx, sy, cond, rms, status


def _tile_orders(patch, freq, anchor_local):
    h, w = patch.shape
    wy, wx = np.hanning(h), np.hanning(w)
    window = wy[:, None] * wx[None, :]
    patch = (patch - np.sum(patch * window) / window.sum()) * window / window.sum()
    q = np.vstack((freq, freq + [3.6 / w, 2.4 / h], freq + [-2.3 / w, 3.7 / h]))
    # Native pixel centers. Separable direct DFT avoids leakage from choosing
    # the nearest FFT bin when rotated periods do not fit an integer tile.
    if max(h, w) > 192:
        values = _interpolated_dft(patch, q)
    else:

        def dft(orders):
            tx = np.exp(-2j * np.pi * np.outer(orders[:, 0], np.arange(w)))
            ty = np.exp(-2j * np.pi * np.outer(orders[:, 1], np.arange(h)))
            return np.sum((ty @ patch.astype(np.complex128, copy=False)) * tx, axis=1)

        values = np.concatenate([dft(orders) for orders in np.split(q, 3)])
    # Hann offsets of several bins avoid counting a main lobe as its noise.
    A, side1, side2 = np.split(values, 3)
    # Robust floor from off-lattice samples, less sensitive to neighboring orders.
    noise = max(1e-12, float(np.median(np.minimum(abs(side1), abs(side2)))))
    A *= np.exp(2j * np.pi * (freq @ anchor_local))
    return A, noise


def _interpolated_dft(patch, q):
    """Sample a 3x zero-padded FFT using local cubic interpolation.

    Shift the spatial kernel to zero before interpolation so its phase changes
    slowly between adjacent bins, then restore the native pixel-center phase.
    Cost scales with tile area even for oblique arrays with many distinct orders.
    """
    h, w = patch.shape
    size = np.array([fft.next_fast_len(3 * w), fft.next_fast_len(3 * h)])
    center = np.array([w // 2, h // 2])
    start = size // 2 - center
    padded = np.zeros(size[::-1], dtype="f4")
    padded[start[1] : start[1] + h, start[0] : start[0] + w] = patch
    spectrum = fft.fft2(fft.ifftshift(padded))
    pos = q * size
    lower = np.floor(pos).astype(int)
    t = (pos - lower).T
    weights = np.array(
        [
            -0.5 * t * (t - 1) ** 2,
            1 - 2.5 * t**2 + 1.5 * t**3,
            0.5 * t + 2 * t**2 - 1.5 * t**3,
            0.5 * t**2 * (t - 1),
        ]
    )
    offsets = np.arange(-1, 3)[:, None]
    ix = (lower[:, 0] + offsets) % size[0]
    iy = (lower[:, 1] + offsets) % size[1]
    local = (
        spectrum[iy[:, None, :], ix[None, :, :]]
        * weights[:, 1, None, :]
        * weights[None, :, 0, :]
    )
    return local.sum(axis=(0, 1)) * np.exp(-2j * np.pi * (q @ center))


def compare_shape_profiles(image, region, ellipse_angle, square_angle):
    """Fit signed orders and test predictions on reserved orders and spatial support.

    The detector PSF can make the outline of a small square nearly circular.
    Relative response across low and high harmonics carries extra information
    when multiple diffraction orders are visible. The gates here are
    qualification heuristics and the per-family evidence is retained for audit.
    """
    xy_min = np.maximum(region.bounds[:2], np.min(region.centers, axis=0))
    xy_max = np.minimum(region.bounds[2:], np.max(region.centers, axis=0))
    ranges = [(xy_min, xy_max)]
    axis = int(np.argmax(xy_max - xy_min))
    if (xy_max - xy_min)[axis] >= 13 * region.pitch:
        mid = np.floor((xy_min[axis] + xy_max[axis]) / 2)
        first_hi, second_lo = xy_max.copy(), xy_min.copy()
        first_hi[axis], second_lo[axis] = mid, mid
        ranges = [(xy_min, first_hi), (second_lo, xy_max)]
    if np.min(xy_max - xy_min) < 5.5 * region.pitch:
        return None
    freq = _frequencies(region.basis)
    if len(freq) < 15:
        return None
    ids = np.lexsort((freq[:, 0], np.linalg.norm(freq, axis=1)))
    ranks = np.arange(len(ids))
    train, test = ids[ranks % 3 != 1], ids[ranks % 3 == 1]
    anchor = np.asarray(region.diagnostics["anchor_xy"])
    cell_area = abs(np.linalg.det(region.basis))
    spectra, boxes = [], []
    for start, end in ranges:
        center = (start + end) / 2
        lo = np.ceil(np.maximum(start, center - 96)).astype(int)
        hi = np.floor(np.minimum(end, center + 96)).astype(int)
        boxes.append([*lo.tolist(), *hi.tolist()])
        inside = np.all((region.centers >= lo) & (region.centers < hi), axis=1)
        sites = region.centers[inside]
        if len(sites) < max(12, 0.65 * np.prod(hi - lo) / cell_area):
            spectra.append(None)
            continue
        # Account for a different local phase origin using only detected site
        # centers, leaving every reserved harmonic available for CD validation.
        coords = (sites - anchor) @ np.linalg.inv(region.basis)
        delta = np.median((coords - np.rint(coords)) @ region.basis, axis=0)
        amp, floor = _tile_orders(
            image[lo[1] : hi[1], lo[0] : hi[0]], freq, anchor + delta - lo
        )
        scale = max(abs(amp[train]).max(), 1e-15)
        amp, floor = amp / scale, floor / scale
        spectra.append(
            (amp, floor)
            if np.count_nonzero(abs(amp) > max(0.009, 5 * floor)) >= 9
            else None
        )
    if spectra[0] is None:
        return None
    z, noise = spectra[0]
    spatial = spectra[1] if len(spectra) > 1 else None
    lattice_angle = float(np.arctan2(*region.basis[0, ::-1]))
    # Angles from the observed aperture moments are independent of the lattice;
    # the lattice angle is a second hypothesis, not a required alignment.
    candidates = {
        "square": (square_angle, lattice_angle),
        "ellipse": (ellipse_angle, lattice_angle),
        "circle": (0.0,),
    }
    scores = {}
    spatial_scores = {}
    solutions = {}
    for family, angles in candidates.items():
        for angle in angles:
            candidate = replace(region, shape=family, angle=angle)
            w, h, sx, sy, _, _, status = _fit_orders(
                z.real[train],
                freq[train],
                np.full(len(train), noise),
                region.pitch,
                candidate,
            )
            if status != "ok":
                continue
            predicted = (
                _form_factor(freq, w, h, angle, family)
                * np.exp(
                    -2 * np.pi**2 * ((sx * freq[:, 0]) ** 2 + (sy * freq[:, 1]) ** 2)
                )
                * np.sinc(freq[:, 0])
                * np.sinc(freq[:, 1])
            )
            gain = (predicted[train] @ z.real[train]) / max(
                1e-12, predicted[train] @ predicted[train]
            )
            error = float(
                np.sqrt(np.mean((z.real[test] - gain * predicted[test]) ** 2))
            )
            if family not in scores or error < scores[family]:
                scores[family] = error
                # Freeze contour, orientation, and x/y blur. Fit a single
                # radiometric gain on the independent patch's training orders.
                if spatial is not None:
                    other = spatial[0].real
                    gain2 = (predicted[train] @ other[train]) / max(
                        1e-12, predicted[train] @ predicted[train]
                    )
                    spatial_scores[family] = float(
                        np.sqrt(np.mean((other[test] - gain2 * predicted[test]) ** 2))
                    )
                solutions[family] = (
                    float(angle if w >= h else angle + np.pi / 2),
                    float(max(w / h, h / w)),
                )
    eligible = dict(scores)
    # An ellipse with equal axes is precisely a circle. Select its simpler
    # nested family unless there is measurable anisotropy and predictive gain.
    if (
        "ellipse" in eligible
        and "circle" in eligible
        and (
            solutions["ellipse"][1] < 1.13
            or scores["circle"] - scores["ellipse"]
            < max(2 * noise, 0.002, 0.20 * scores["circle"])
        )
    ):
        del eligible["ellipse"]
    if not eligible:
        return None
    ranked = sorted(eligible, key=eligible.get)
    winner = ranked[0]
    margin = eligible[ranked[1]] - eligible[winner] if len(ranked) > 1 else 0.0
    resolved = (
        len(ranked) > 1
        and eligible[winner] < 0.045
        and margin > max(0.0035, 3 * noise, 0.4 * eligible[winner])
    )
    spatial_status = "small_region_reserved_orders_only"
    if len(ranges) == 2:
        spatial_status = "insufficient_spatial_signal"
        spatial_agrees = False
        if spatial is not None and len(ranked) > 1:
            alternatives = min(spatial_scores[family] for family in ranked[1:])
            score = spatial_scores[winner]
            spatial_agrees = score < 0.06 and alternatives - score > max(
                0.0015, 1.5 * spatial[1], 0.15 * score
            )
            spatial_status = (
                "validated" if spatial_agrees else "inconclusive_spatial_holdout"
            )
        resolved = resolved and spatial_agrees
    return (
        winner,
        solutions[winner][0],
        bool(resolved),
        {
            "heldout_shape_rms": scores,
            "spatial_holdout_rms": spatial_scores,
            "spatial_holdout_status": spatial_status,
            "shape_patch_bounds_xyxy": boxes,
            "tested_orders": len(test),
            "noise_fraction": float(noise),
            "shape_margin": float(margin),
        },
    )


def _frequencies(basis):
    dual = np.linalg.inv(basis).T
    orders = np.array(
        [(i, j) for i in range(6) for j in range(-5, 6) if i > 0 or j > 0]
    )
    freq = orders @ dual
    return freq[np.all(np.abs(freq) < 0.445, axis=1)]


def harmonic_tiles(
    image, region, shift_xy=(0.0, 0.0), tile_px=128, origin_xy=(0.0, 0.0)
):
    """Return one record per *complete* reference-aligned tile for this region.

    tile_i and tile_j are integer x/y indices relative to origin_xy.
    Area-averaged x/y CD are axes of the shape in pixels; ellipse CD is the
    major axis. dx_px/dy_px are independent phase residuals around the nominal
    reference lattice after removing the supplied global acquisition shift.
    """
    if not np.isfinite(tile_px) or tile_px < 4:
        raise ValueError("tile_px must be a finite tile width of at least four pixels")
    data = np.asarray(image)
    shift = np.asarray(shift_xy, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    x0, y0, x1, y1 = np.asarray(region.bounds) - 0.5
    start = np.ceil((np.array((x0, y0)) - origin) / tile_px - 1e-10).astype(int)
    stop = np.floor((np.array((x1, y1)) - origin) / tile_px + 1e-10).astype(int)
    sites = np.floor((region.centers - origin) / tile_px).astype(int)
    unique, counts = np.unique(sites, axis=0, return_counts=True)
    populations = dict(zip(map(tuple, unique), counts))
    cell_area = abs(np.linalg.det(region.basis))
    # Up to five primitive orders on each axis and their mixed orders. Beyond
    # the componentwise Nyquist limit the sampled coefficients alias.
    freq = _frequencies(region.basis)
    anchor = np.asarray(region.diagnostics["anchor_xy"])
    out = []
    for j in range(start[1], stop[1]):
        for i in range(start[0], stop[0]):
            lower = origin + np.array((i, j)) * tile_px
            lo = np.ceil(lower + shift - 1e-10).astype(int)
            hi = np.ceil(lower + tile_px + shift - 1e-10).astype(int)
            x, y = lower + tile_px / 2
            count = int(populations.get((i, j), 0))
            row = {
                "tile_i": int(i),
                "tile_j": int(j),
                "x": float(x),
                "y": float(y),
                "region_id": region.region_id,
                "x0": float(lower[0]),
                "y0": float(lower[1]),
                "x1": float(lower[0] + tile_px),
                "y1": float(lower[1] + tile_px),
                "cd_px": np.nan,
                "cd_x_px": np.nan,
                "cd_y_px": np.nan,
                "area_diameter_px": np.nan,
                "dx_px": np.nan,
                "dy_px": np.nan,
                "fit_rms": np.nan,
                "condition": np.nan,
                "n_orders": 0,
                "n_sites": count,
                "phase_status": "unavailable",
                "status": "insufficient_orders",
            }
            out.append(row)
            if np.any(lo < 0) or hi[0] > data.shape[1] or hi[1] > data.shape[0]:
                row["status"] = "incomplete_native_footprint"
                continue
            if (
                count < max(16, 0.74 * tile_px**2 / cell_area)
                or min(hi - lo) < 5 * region.pitch
            ):
                row["status"] = "insufficient_pattern_coverage"
                continue
            if len(freq) < 5:
                continue
            amplitude, noise = _tile_orders(
                data[lo[1] : hi[1], lo[0] : hi[0]], freq, anchor + shift - lo
            )
            radii = np.linalg.norm(freq, axis=1)
            fundamental = (radii <= 1.26 / region.pitch) & (abs(amplitude) > 5 * noise)
            if fundamental.sum() < 2 or np.linalg.matrix_rank(freq[fundamental]) < 2:
                row["status"] = "two_dimensional_phase_unresolved"
                continue
            f = freq[fundamental]
            w = np.abs(amplitude[fundamental])
            d = np.linalg.lstsq(
                f * w[:, None],
                -np.angle(amplitude[fundamental]) * w / (2 * np.pi),
                rcond=None,
            )[0]
            # Localization by phase requires residuals smaller than half a
            # period and a common displacement across visible fundamental orders.
            phase_err = np.angle(amplitude[fundamental] * np.exp(2j * np.pi * (f @ d)))
            if (
                np.linalg.norm(d) > 0.35 * region.pitch
                or np.average(phase_err**2, weights=w**2) > 0.18**2
            ):
                row["status"] = "inconsistent_lattice_phase"
                continue
            row.update(dx_px=float(d[0]), dy_px=float(d[1]), phase_status="ok")
            if region.shape not in ("ellipse", "circle", "square"):
                row["status"] = "ambiguous_shape"
                continue
            real = (amplitude * np.exp(2j * np.pi * (freq @ d))).real
            good = abs(amplitude) > max(4 * noise, 0.008 * abs(amplitude).max())
            # Signed near-zero Fourier orders also constrain size, while a
            # field with only the first diffracted order cannot identify blur.
            n_orders = int(good.sum())
            row["n_orders"] = n_orders
            if (
                n_orders < (7 if region.shape == "ellipse" else 5)
                or radii[good].max() < 1.8 / region.pitch
            ):
                continue
            include = radii < min(0.46, 1.22 * radii[good].max())
            if include.sum() < 6:
                continue
            wid, ht, sx, sy, cond, rms, status = _fit_orders(
                real[include],
                freq[include],
                np.full(include.sum(), noise),
                region.pitch,
                region,
            )
            row.update(
                condition=cond,
                fit_rms=rms,
                status=status,
                blur_x_px=float(sx),
                blur_y_px=float(sy),
            )
            if status == "ok":
                area_diameter = (
                    2 * np.sqrt(wid * ht / np.pi)
                    if region.shape == "square"
                    else np.sqrt(wid * ht)
                )
                row.update(
                    cd_px=float(max(wid, ht)),
                    cd_x_px=float(wid),
                    cd_y_px=float(ht),
                    area_diameter_px=float(area_diameter),
                )
    return out
