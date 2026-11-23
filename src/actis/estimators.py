"""Native-pixel CD and center estimates for independently acquired images."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .fit import ApertureLibrary, array_backend, solve_stamps, to_host

METHOD_NAMES = ("contour", "model", "differential", "flux")
N_RAYS = 40


def empty_measurement(i, center, shape, flag="unmeasured"):
    return dict(
        site_index=int(i),
        x_px=float(center[0]),
        y_px=float(center[1]),
        shape=shape,
        cd_px=np.nan,
        area_diameter_px=np.nan,
        width_px=np.nan,
        height_px=np.nan,
        angle_rad=np.nan,
        contour_xy=None,
        valid=False,
        flag=flag,
        quality=0.0,
        flux_dn_px2=np.nan,
    )


def _polygon_metrics(vertices):
    # Center before taking moments of small contours far from the image origin.
    origin = vertices.mean(axis=1)
    x, y = np.moveaxis(vertices - origin[:, None], 2, 0)
    xx, yy = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yy - xx * y
    area = cross.sum(axis=1) / 2
    safe_area = np.maximum(area, 1e-12)
    cx = ((x + xx) * cross).sum(axis=1) / (6 * safe_area)
    cy = ((y + yy) * cross).sum(axis=1) / (6 * safe_area)
    vx = ((x * x + x * xx + xx * xx) * cross).sum(axis=1) / (12 * safe_area) - cx * cx
    vy = ((y * y + y * yy + yy * yy) * cross).sum(axis=1) / (12 * safe_area) - cy * cy
    vxy = ((2 * x * y + x * yy + xx * y + 2 * xx * yy) * cross).sum(axis=1) / (
        24 * safe_area
    ) - cx * cy
    half_gap = np.hypot((vx - vy) / 2, vxy)
    mean = (vx + vy) / 2
    eig = np.column_stack((mean + half_gap, mean - half_gap))
    return (
        origin + np.column_stack((cx, cy)),
        area,
        4 * np.sqrt(np.maximum(eig, 0)),
        np.arctan2(2 * vxy, vx - vy) / 2,
    )


def _outline(center, width, height, angle, shape):
    if shape == "square":
        uv = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (width / 2)
    else:
        t = np.arange(48) * (2 * np.pi / 48)
        uv = np.column_stack((width * np.cos(t), height * np.sin(t))) / 2
    c, s = np.cos(angle), np.sin(angle)
    return uv @ np.array([[c, s], [-s, c]]) + center


def _shape_from_contour(points, center, axis_ratio):
    if axis_ratio > 1.17:
        return "ellipse"
    q = points - center
    radii = np.linalg.norm(q, axis=1)
    theta = np.arctan2(q[:, 1], q[:, 0])
    fourth = abs(np.mean((radii / radii.mean() - 1) * np.exp(4j * theta)))
    return "square" if fourth > 0.037 else "circle"


def _isocontours(norm, level, sites, dirs, rays, distance, xp=np):
    # First downward crossing on each ray prevents inclusion of a neighbor.
    falls = (norm[:, :, :-1] >= level) & (norm[:, :, 1:] < level)
    crossing = falls.any(axis=2) & (norm[:, :, 0] >= level)
    ir = falls.argmax(axis=2)
    a = xp.take_along_axis(norm, ir[:, :, None], axis=2)[:, :, 0]
    b = xp.take_along_axis(norm, ir[:, :, None] + 1, axis=2)[:, :, 0]
    stable = abs(a - b) > 1e-8
    frac = xp.where(stable, (a - level) / xp.where(stable, a - b, 1), 0)
    dist = to_host(xp, (ir + xp.clip(frac, 0, 1)) * (distance[1] - distance[0]))
    crossing = to_host(xp, crossing)
    coverage = crossing.mean(axis=1)
    for k in np.flatnonzero((coverage >= 0.94) & (coverage < 1.0)):
        ok = crossing[k]
        dist[k, ~ok] = np.interp(rays[~ok], rays[ok], dist[k, ok], period=2 * np.pi)
    points = sites[:, None] + dirs[None] * dist[:, :, None]
    centroids, areas, axes, major_angles = _polygon_metrics(points)
    return coverage, points, centroids, areas, axes, major_angles


def _contours(image, centers, pitch, shape, angle, threshold_diagnostics=False, xp=np):
    if xp is np:
        map_coordinates = ndimage.map_coordinates
    else:
        from cupyx.scipy.ndimage import map_coordinates
    rays = np.arange(N_RAYS) * 2 * np.pi / N_RAYS
    dirs = np.column_stack((np.cos(rays), np.sin(rays)))
    radius = 0.64 * pitch
    distance = np.linspace(0, radius, max(15, int(np.ceil(radius * 4)) + 1))
    offsets = xp.asarray(dirs[:, None, :] * distance[None, :, None])
    results = []
    rows_per_batch = max(1, min(256, 150000 // (len(distance) * N_RAYS)))
    for start in range(0, len(centers), rows_per_batch):
        sites = centers[start : start + rows_per_batch]
        positions = xp.asarray(sites)[:, None, None, :] + offsets
        values = map_coordinates(
            image,
            xp.stack((positions[..., 1], positions[..., 0])),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        finite = xp.isfinite(values).all(axis=(1, 2))
        # Rays include all four cardinal directions. Exclude sites whose
        # physical support extends beyond the frame, before using the samples.
        last_pixel = np.array(image.shape[::-1]) - 1
        finite &= xp.asarray(
            ((sites >= radius) & (sites <= last_pixel - radius)).all(axis=1)
        )
        values = xp.where(finite[:, None, None], values, 0)
        dark, dark_q32 = xp.quantile(
            values[:, :, distance.searchsorted(0.65 * radius) :].reshape(
                len(sites), -1
            ),
            xp.asarray([0.16, 0.32]),
            axis=1,
        )
        bright = xp.quantile(
            values[
                :, :, : distance.searchsorted(max(0.7, 0.15 * pitch), side="right")
            ].reshape(len(sites), -1),
            0.96,
            axis=1,
        )
        amplitude = bright - dark
        noise_floor = xp.maximum(
            1.90 * (dark_q32 - dark), 1e-6 * xp.maximum(abs(dark), 1.0)
        )
        snr = amplitude / noise_floor
        norm = (values - dark[:, None, None]) / xp.maximum(
            amplitude[:, None, None], 1e-12
        )
        finite, dark, amplitude, snr = (
            to_host(xp, a) for a in (finite, dark, amplitude, snr)
        )
        contrast_ok = snr >= 3.5
        coverage, points, centroids, areas, axes, major_angles = _isocontours(
            norm, 0.5, sites, dirs, rays, distance, xp
        )
        sensitivities = {}
        if threshold_diagnostics:
            for level in (0.4, 0.6):
                cov, _, _, area, lengths, _ = _isocontours(
                    norm, level, sites, dirs, rays, distance, xp
                )
                closed = (cov >= 0.94) & (area > 0) & (lengths[:, 1] > 0)
                sensitivities[level] = np.where(
                    closed, 2 * np.sqrt(np.maximum(area, 0) / np.pi), np.nan
                )
        for k, center in enumerate(sites):
            row = empty_measurement(start + k, center, shape)
            if threshold_diagnostics:
                row.update(
                    contour_area_diameter40_px=np.nan, contour_area_diameter60_px=np.nan
                )
            results.append(row)
            if not finite[k]:
                row["flag"] = "border"
                continue
            row.update(
                background_dn=float(dark[k]),
                contrast_dn=float(amplitude[k]),
                contrast_snr=float(snr[k]),
            )
            if not contrast_ok[k]:
                row["flag"] = "low_contrast"
                continue
            if coverage[k] < 0.94:
                row["flag"] = "open_contour"
                continue
            if areas[k] <= 0 or axes[k, 1] <= 0:
                row["flag"] = "degenerate_contour"
                continue
            resolved_shape = (
                _shape_from_contour(points[k], centroids[k], axes[k, 0] / axes[k, 1])
                if shape == "auto"
                else shape
            )
            diameter = 2 * np.sqrt(areas[k] / np.pi)
            if resolved_shape == "square":
                width = height = np.sqrt(areas[k])
                theta = angle
                if theta is None:
                    vector = points[k] - centroids[k]
                    r = np.linalg.norm(vector, axis=1)
                    phi = np.arctan2(vector[:, 1], vector[:, 0])
                    corner = np.mean((r - r.mean()) * np.exp(4j * phi))
                    theta = np.angle(-corner) / 4
            elif resolved_shape == "ellipse":
                width, height = axes[k]
                theta = major_angles[k]
            else:
                width = height = diameter
                theta = 0.0
            row.update(
                x_px=float(centroids[k, 0]),
                y_px=float(centroids[k, 1]),
                width_px=float(width),
                height_px=float(height),
                cd_px=float(width),
                area_diameter_px=float(diameter),
                angle_rad=float(theta),
                shape=resolved_shape,
                contour_xy=points[k],
                valid=True,
                quality=float(coverage[k]),
                flag="ok" if coverage[k] == 1.0 else "interpolated_rays",
            )
            if threshold_diagnostics:
                row["contour_area_diameter40_px"] = float(sensitivities[0.4][k])
                row["contour_area_diameter60_px"] = float(sensitivities[0.6][k])
    return results


def _flux(image, centers, pitch, shape, contours, bright, dark, xp=np):
    """Aperture photometry on native pixels, with a local surround background."""
    results = [empty_measurement(i, center, shape) for i, center in enumerate(centers)]
    rad = pitch / 2
    half = int(np.ceil(0.64 * pitch + 0.7))
    grid_y, grid_x = np.mgrid[-half : half + 1, -half : half + 1]
    xx, yy = grid_x.ravel(), grid_y.ravel()
    rounded = np.rint(centers).astype(int)
    ok = ((rounded >= half) & (rounded < np.array(image.shape[::-1]) - half)).all(
        axis=1
    )
    for i in np.flatnonzero(~ok):
        results[i]["flag"] = "border"
    indices = np.flatnonzero(ok)
    batch_size = max(1, min(1024, 180000 // len(xx)))
    for k in range(0, len(indices), batch_size):
        ids = indices[k : k + batch_size]
        x = xp.asarray(xx) + xp.asarray(rounded[ids, 0, None])
        y = xp.asarray(yy) + xp.asarray(rounded[ids, 1, None])
        dx, dy = (
            x - xp.asarray(centers[ids, 0, None]),
            y - xp.asarray(centers[ids, 1, None]),
        )
        dist2 = dx * dx + dy * dy
        values = image[y, x]
        ring = (dist2 >= (0.48 * pitch) ** 2) & (dist2 <= (0.64 * pitch) ** 2)
        # A low percentile used directly as background would generate positive
        # flux from zero-mean read noise. Use the ring median or a dark reference.
        ordered = xp.sort(xp.where(ring, values, np.inf), axis=1)
        last = xp.maximum(ring.sum(axis=1) - 1, 0)
        qi = xp.rint(last[:, None] * xp.asarray([0.25, 0.5, 0.75])).astype(int)
        q25, median, q75 = xp.take_along_axis(ordered, qi, axis=1).T
        bg = median if dark is None else xp.full(len(ids), dark)
        # Sample only boundary pixels at 4x4 subpixel positions.
        weight = (dist2 < rad * rad).astype(np.float32)
        edge = (dist2 >= max(0, rad - 0.72) ** 2) & (dist2 < (rad + 0.72) ** 2)
        ex, ey = dx[edge], dy[edge]
        partial = xp.zeros(len(ex), dtype=np.float32)
        for sx in (-0.375, -0.125, 0.125, 0.375):
            for sy in (-0.375, -0.125, 0.125, 0.375):
                partial += (ex + sx) ** 2 + (ey + sy) ** 2 <= rad * rad
        weight[edge] = partial / 16
        weighted = weight * (values - bg[:, None])
        flux = weighted.sum(axis=1)
        sigma = (q75 - q25) / 1.349
        variance_factor = (weight**2).sum(axis=1)
        if dark is None:
            variance_factor += 1.57 * weight.sum(axis=1) ** 2 / (last + 1)
        noise = xp.maximum(sigma * xp.sqrt(variance_factor), 1e-9)
        snr = flux / noise
        positive = xp.isfinite(flux) & (flux > 0)
        safe_flux = xp.where(positive, flux, 1)
        offsets = (
            xp.column_stack(((weighted * dx).sum(axis=1), (weighted * dy).sum(axis=1)))
            / safe_flux[:, None]
        )
        positive, flux, offsets, snr, bg = (
            to_host(xp, a) for a in (positive, flux, offsets, snr, bg)
        )
        for j, i in enumerate(ids):
            row = results[i]
            if not positive[j]:
                row["flag"] = "no_positive_flux"
                continue
            if snr[j] < 3.5:
                row["flag"] = "low_flux_snr"
                continue
            if np.linalg.norm(offsets[j]) > rad * 0.5:
                row["flag"] = "unstable_centroid"
                continue
            xy = centers[i] + offsets[j]
            resolved_shape = (
                contours[i]["shape"] if shape == "auto" and contours else shape
            )
            row.update(
                x_px=float(xy[0]),
                y_px=float(xy[1]),
                flux_dn_px2=float(flux[j]),
                valid=True,
                shape=resolved_shape,
                flag="uncalibrated_flux",
                quality=float(min(1.0, snr[j] / 10)),
                flux_snr=float(snr[j]),
                background_dn=float(bg[j]),
            )
            if bright is None or dark is None:
                continue
            # Calibration also assumes adequate encircled energy and a linear
            # relation of integrated contrast to mask opening area.
            area = flux[j] / (bright - dark)
            eq_diameter = 2 * np.sqrt(area / np.pi)
            if resolved_shape == "square":
                width = height = np.sqrt(area)
            elif resolved_shape == "ellipse":
                if not contours or not contours[i]["valid"]:
                    row["flag"] = "flux_only_no_axis_ratio"
                    continue
                ratio = contours[i]["width_px"] / contours[i]["height_px"]
                width, height = (
                    eq_diameter * np.sqrt(ratio),
                    eq_diameter / np.sqrt(ratio),
                )
            else:
                width = height = eq_diameter
            row.update(
                cd_px=float(width),
                width_px=float(width),
                height_px=float(height),
                area_diameter_px=float(eq_diameter),
                angle_rad=contours[i]["angle_rad"] if contours else 0.0,
                flag="reference_normalized_flux",
            )
    return results


@dataclass
class _Scene:
    image: np.ndarray
    centers: np.ndarray
    width: np.ndarray
    height: np.ndarray
    contrast: np.ndarray
    near: np.ndarray
    xx: np.ndarray
    yy: np.ndarray

    def samples(self, indices):
        ij = np.rint(self.centers[indices]).astype(int)
        x = ij[:, 0, None] + self.xx
        y = ij[:, 1, None] + self.yy
        obs = self.image[y, x]
        off_x = (x - self.centers[indices, 0, None]).astype(np.float32)
        off_y = (y - self.centers[indices, 1, None]).astype(np.float32)
        return obs, off_x, off_y

    def fit(self, indices, library, steps, *, profile_only=False):
        obs, off_x, off_y = self.samples(indices)
        neighbors = []
        halo = library.radius + np.max(abs(self.xx))
        for adj in self.near[indices].T:
            delta = self.centers[adj] - self.centers[indices]
            active = np.max(abs(delta), axis=1) < halo
            if not np.any(active):
                continue
            rel = np.where(active, self.contrast[adj] / self.contrast[indices], 0)
            neighbors.append(
                (
                    delta[:, 0],
                    delta[:, 1],
                    self.width[adj],
                    self.height[adj],
                    np.clip(rel, 0.0, 4.0),
                )
            )
        return solve_stamps(
            library,
            obs,
            off_x,
            off_y,
            self.width[indices],
            self.height[indices],
            neighbors,
            steps=steps,
            profile_only=profile_only,
        )


def _representative(ids, limit=10):
    return ids[np.linspace(0, len(ids) - 1, min(limit, len(ids)), dtype=int)]


def _model_methods(
    image,
    centers,
    pitch,
    shape,
    angle,
    contours,
    names,
    psf,
    tile_size,
    backend,
    fit_seconds=None,
):
    result = {
        name: [empty_measurement(i, center, shape) for i, center in enumerate(centers)]
        for name in names
    }
    valid = np.array([r["valid"] for r in contours])
    if shape == "auto" and valid.any():
        shapes = [r["shape"] for r in contours if r["valid"]]
        shape = max(("circle", "ellipse", "square"), key=shapes.count)
    if not valid.any() or shape == "ambiguous":
        for rows in result.values():
            for row in rows:
                row["flag"] = (
                    "ambiguous_shape"
                    if shape == "ambiguous"
                    else "no_resolved_contours"
                )
        return result
    xp = array_backend(backend)
    cw = np.array([r["width_px"] for r in contours])
    ch = np.array([r["height_px"] for r in contours])
    width, height = np.nanmedian(cw[valid]), np.nanmedian(ch[valid])
    if shape != "ellipse":
        height = width
    # A relative intensity contour is an initialization only. Each repeat's
    # sites retain their own fitted dimensions and center positions.
    cw = np.where(valid, cw, width)
    ch = np.where(valid, ch, height)
    contrast = np.array([r.get("contrast_dn", np.nan) for r in contours])
    contrast = np.where(
        (contrast > 0) & np.isfinite(contrast), contrast, np.nanmedian(contrast[valid])
    )
    if shape in {"ellipse", "square"} and angle is None:
        angles = np.array([r["angle_rad"] for r in contours if r["valid"]])
        symmetry = 4 if shape == "square" else 2
        angle = np.angle(np.mean(np.exp(symmetry * 1j * angles))) / symmetry
    angle = 0.0 if angle is None else angle
    half = max(3, int(np.ceil(min(0.54 * pitch, 0.7 * max(width, height) + 3))))
    iy, ix = np.mgrid[-half : half + 1, -half : half + 1]
    near = cKDTree(centers).query(centers, k=min(9, len(centers)))[1]
    near = np.reshape(near, (len(centers), -1))[:, 1:]
    scene = _Scene(image, centers, cw, ch, contrast, near, ix.ravel(), iy.ravel())
    bins = defaultdict(list)
    for i, (cx, cy) in enumerate(np.rint(centers).astype(int)):
        if (
            cx - half < 0
            or cy - half < 0
            or cx + half >= image.shape[1]
            or cy + half >= image.shape[0]
        ):
            for rows in result.values():
                rows[i]["flag"] = "border"
            continue
        if not valid[i]:
            for rows in result.values():
                rows[i]["flag"] = contours[i]["flag"]
            continue
        bins[(cx // tile_size, cy // tile_size)].append(i)
    bins = {key: np.asarray(ids) for key, ids in bins.items()}
    if not bins:
        return result
    if psf is None:
        # Profile tile-shared effective blur with independent geometry and
        # contrast per selected hole. Include x/y blur differences from TDI.
        candidates = [
            (0.45, 0.45),
            (0.85, 0.85),
            (1.25, 1.25),
            (1.75, 1.75),
            (0.7, 1.15),
            (1.15, 0.7),
        ]
    else:
        candidates = [psf]
    profiles = defaultdict(list)

    def profile_sigma(sigma, keys):
        library = ApertureLibrary(shape, angle, width, height, sigma, xp)
        for key in keys:
            sample = _representative(bins[key])
            rmse, width_px = scene.fit(sample, library, steps=3, profile_only=True)
            profiles[key].append((float(rmse), sigma, float(width_px)))

    if psf is None:
        for sigma in candidates:
            profile_sigma(sigma, bins)
        # Refine both detector-axis sigmas near the winning coarse setting.
        # Profiles from multiple tiles at identical settings reuse the library.
        delta = 0.16
        trials = defaultdict(list)
        coarse = {}
        for key in bins:
            _, (sx, sy), _ = min(profiles[key])
            coarse[key] = (sx, sy)
            for ax, ay in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                trials[(round(sx + delta * ax, 3), round(sy + delta * ay, 3))].append(
                    key
                )
        for sigma, keys in trials.items():
            profile_sigma(sigma, keys)
        interpolated = defaultdict(list)
        for key, (sx, sy) in coarse.items():
            costs = {s: score**2 for score, s, _ in profiles[key]}
            base = costs[sx, sy]
            vertex = []
            for ax, ay in ((1, 0), (0, 1)):
                left = costs[round(sx - delta * ax, 3), round(sy - delta * ay, 3)]
                right = costs[round(sx + delta * ax, 3), round(sy + delta * ay, 3)]
                curvature = left + right - 2 * base
                if curvature > 1e-10:
                    move = 0.5 * delta * np.clip((left - right) / curvature, -2, 2)
                else:
                    move = -delta if left < right else delta
                vertex.append(round(((sx if ax else sy) + move) * 20) / 20)
            extra = tuple(vertex)
            if extra not in costs:
                interpolated[extra].append(key)
        for sigma, keys in interpolated.items():
            profile_sigma(sigma, keys)
    else:
        for key in bins:
            profiles[key].append((0.0, psf, width))

    by_sigma = defaultdict(list)
    for key, profile in profiles.items():
        best = min(profile)
        # Compare the free per-site widths at nuisance settings giving almost
        # the same predictive image residual. This is an empirical sensitivity
        # diagnostic, not a confidence interval for physical mask CD.
        tolerance = max(0.0025, best[0] * 0.05)
        supported = [w for r, _, w in profile if r <= best[0] + tolerance]
        weak = any(abs(w - best[2]) >= max(0.3, best[2] * 0.07) for w in supported)
        at_limit = psf is None and (max(best[1]) >= 1.74 or min(best[1]) <= 0.31)
        diagnostic = dict(
            psf_source="fixed" if psf is not None else "tile_profile",
            blur_profile_candidate_count=len(profile),
            blur_profile_support_sites=0
            if psf is not None
            else len(_representative(bins[key])),
            blur_profile_best_rmse=np.nan if psf is not None else best[0],
            blur_profile_width_span_px=np.nan
            if psf is not None
            else max(supported) - min(supported),
        )
        by_sigma[best[1]].append((bins[key], weak, at_limit, diagnostic))
    batch_size = max(1, min(384, 80000 // len(scene.xx)))
    for sigma, groups in by_sigma.items():
        library = ApertureLibrary(shape, angle, width, height, sigma, xp)
        for ids, weak, at_limit, diagnostic in groups:
            for j in range(0, len(ids), batch_size):
                group = ids[j : j + batch_size]
                for name in names:
                    if fit_seconds is not None:
                        xp.cuda.get_current_stream().synchronize()
                        fit_start = perf_counter()
                    # The differential estimate linearizes at the contour seed.
                    # It uses one displacement/size update, independently of the
                    # full fit. A fixed PSF makes this path especially cheap.
                    steps = 1 if name == "differential" else 6
                    fitted = scene.fit(group, library, steps=steps)
                    for idx, params in zip(group, zip(*fitted)):
                        dx, dy, w, h, err, cond, bound, amp, base = params
                        theta = angle
                        if shape == "ellipse" and h > w:
                            w, h = h, w
                            theta += np.pi / 2
                        if at_limit:
                            flag = "blur_at_search_limit"
                        elif weak:
                            flag = "blur_cd_confounded"
                        elif bound:
                            flag = "width_at_search_limit"
                        elif cond > 2e4:
                            flag = "ill_conditioned"
                        elif err > 0.16 or abs(dx) > 1.98 or abs(dy) > 1.98:
                            flag = "poor_fit"
                        else:
                            flag = "ok"
                        center = centers[idx] + (dx, dy)
                        diameter = (
                            2 * w / np.sqrt(np.pi)
                            if shape == "square"
                            else np.sqrt(w * h)
                        )
                        row = result[name][idx]
                        row.update(
                            x_px=float(center[0]),
                            y_px=float(center[1]),
                            cd_px=float(w),
                            width_px=float(w),
                            height_px=float(h),
                            angle_rad=float(theta),
                            area_diameter_px=float(diameter),
                            shape=shape,
                            valid=flag == "ok",
                            flag=flag,
                            quality=float(np.clip(1 - err, 0, 1)),
                            rmse_fraction=float(err),
                            condition_number=float(cond),
                            psf_sigma_x_px=float(sigma[0]),
                            psf_sigma_y_px=float(sigma[1]),
                            contrast_dn=float(amp),
                            background_dn=float(base),
                            contour_xy=_outline(center, w, h, theta, shape),
                        )
                        row.update(diagnostic)
                    if fit_seconds is not None:
                        fit_seconds[name] += perf_counter() - fit_start
        del library
    return result


def estimate_holes(
    image,
    centers_xy,
    *,
    pitch_px,
    shape="auto",
    psf_sigma_px=None,
    bright_dn=None,
    dark_dn=None,
    methods=METHOD_NAMES,
    tile_size_px=256,
    angle_rad=None,
    backend="cpu",
):
    """Return one ordered record per input site for each requested method.

    Centers are (x, y) in native mask-image pixels. The primary CD is circle
    diameter, ellipse major axis, or square side. Ambiguous contours use the
    equal-area diameter. Flux uses signed background-subtracted digital counts
    times pixel area. Converting it to width requires independent bright and
    dark references. `valid` applies to the method's native observable.

    Shape orientation is shared within this input region. A supplied sigma is
    a detector-axis effective optical blur in native pixels, before pixel-area
    integration. If omitted, a profiled blur is shared within each tile.
    """
    frame = np.asarray(image, dtype=np.float32)
    sites = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    if (
        frame.ndim != 2
        or not np.all(np.isfinite(frame))
        or not np.all(np.isfinite(sites))
    ):
        raise ValueError("Expected a finite 2-D grayscale image and finite xy sites")
    pitch = float(np.min(np.asarray(pitch_px, dtype=float)))
    if not np.isfinite(pitch) or pitch < 3:
        raise ValueError("Nearest-neighbor pitch must be at least three native pixels")
    if shape not in {"auto", "square", "circle", "ellipse", "ambiguous"}:
        raise ValueError("Unsupported hole shape: " + str(shape))
    if set(methods) - set(METHOD_NAMES):
        raise ValueError("Unsupported method name")
    if int(tile_size_px) <= 0:
        raise ValueError("tile_size_px must be positive")
    if bright_dn is not None and dark_dn is not None and not bright_dn > dark_dn:
        raise ValueError("The independent bright level must exceed the dark level")
    if psf_sigma_px is None:
        psf = None
    else:
        sigma = np.broadcast_to(np.asarray(psf_sigma_px, dtype=float), (2,))
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Effective PSF sigma must be finite and positive")
        psf = tuple(sigma)
    if not len(sites):
        return {name: [] for name in methods}
    angle = None if angle_rad is None else float(angle_rad)
    requires_contour = bool(set(methods) - {"flux"}) or (
        bright_dn is not None and dark_dn is not None and shape in {"ellipse", "auto"}
    )
    contours = (
        _contours(
            frame,
            sites,
            pitch,
            shape,
            angle,
            threshold_diagnostics="contour" in methods,
        )
        if requires_contour
        else []
    )
    result = {}
    if "contour" in methods:
        result["contour"] = contours
    if "flux" in methods:
        result["flux"] = _flux(frame, sites, pitch, shape, contours, bright_dn, dark_dn)
    model_names = [m for m in ("model", "differential") if m in methods]
    if model_names:
        result.update(
            _model_methods(
                frame,
                sites,
                pitch,
                shape,
                angle,
                contours,
                model_names,
                psf,
                int(tile_size_px),
                backend,
            )
        )
    return result
