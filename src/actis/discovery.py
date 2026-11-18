"""Detect illuminated islands and estimate a lattice in native (column, row) pixels.

All public positions use pixel centers (the first pixel is at 0, 0). A moving
image displacement is added to a reference position to locate its observation.
No measured image is interpolated for feature metrology.
"""

from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from scipy import fft, ndimage
from scipy.spatial import cKDTree

from .harmonics import compare_shape_profiles


@dataclass
class Region:
    region_id: int
    centers: np.ndarray
    pitch: float
    basis: np.ndarray
    shape: str
    best_shape: str
    angle: float
    lattice: str
    bounds: tuple
    confidence: float
    diagnostics: dict = field(default_factory=dict)


@dataclass
class Registration:
    shift_xy: np.ndarray
    score: float
    ambiguous: bool
    reason: str
    candidates: list
    periodic_ambiguity: bool


def _background_maps(a):
    # Estimate local background and contrast across many periods, interpolating
    # the tile statistics to avoid imposing analysis-tile boundaries on holes.
    h, w = a.shape
    rows = np.array_split(np.arange(h), max(1, h // 64))
    cols = np.array_split(np.arange(w), max(1, w // 64))
    coarse = np.array(
        [
            [
                np.percentile(a[y[0] : y[-1] + 1, x[0] : x[-1] + 1], (10, 98))
                for x in cols
            ]
            for y in rows
        ]
    )
    xc = [np.mean(x) for x in cols]
    yc = [np.mean(y) for y in rows]

    def expand(grid):
        strips = np.stack([np.interp(np.arange(w), xc, row) for row in grid])
        ry = np.interp(np.arange(h), yc, np.arange(len(rows)))
        low = np.floor(ry).astype(int)
        high = np.minimum(low + 1, len(rows) - 1)
        return (
            strips[low] + (ry - low)[:, None] * (strips[high] - strips[low])
        ).astype("f4")

    return expand(coarse[:, :, 0]), expand(coarse[:, :, 1])


def _islands(image):
    a = np.asarray(image, dtype="f4")
    smooth = ndimage.gaussian_filter(a, 0.65)
    low, high = _background_maps(smooth)
    # Median absolute mixed difference rejects smooth illumination and edges.
    diff = np.diff(np.diff(a[::2], axis=0), axis=1)
    noise = np.median(np.abs(diff - np.median(diff))) / 1.349
    contrast = np.maximum(high - low, 4.5 * noise)
    excursion = smooth - low
    mask = excursion > np.maximum(0.47 * contrast, 2.1 * noise)
    labels, n = ndimage.label(mask)
    y, x = np.nonzero(mask)
    idx = labels[y, x]
    peak = np.zeros(n + 1, dtype="f4")
    np.maximum.at(peak, idx, excursion[y, x])
    core = excursion[y, x] >= 0.50 * peak[idx]
    y, x, idx = y[core], x[core], idx[core]
    count = np.bincount(idx, minlength=n + 1)
    weight = np.maximum(excursion[y, x], 0)

    def accum(v):
        return np.bincount(idx, weights=v, minlength=n + 1)

    mass = np.maximum(accum(weight), 1e-20)
    cx, cy = accum(weight * x) / mass, accum(weight * y) / mass
    x = x - cx[idx]
    y = y - cy[idx]
    mu20, mu02, mu11 = [accum(weight * v) / mass for v in (x * x, y * y, x * y)]
    m4 = accum(weight * (x * x + y * y) ** 2) / mass
    h4 = (
        accum(weight * (x**4 - 6 * x * x * y * y + y**4))
        + 4j * accum(weight * (x**3 * y - x * y**3))
    ) / np.maximum(mass * m4, 1e-20)
    tr = mu20 + mu02
    delta = np.hypot(mu20 - mu02, 2 * mu11)
    ratio = np.sqrt(np.maximum(tr + delta, 0) / np.maximum(tr - delta, 1e-8))
    angle = np.arctan2(2 * mu11, mu20 - mu02) / 2
    boundary = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    keep = (count >= 4) & (ratio < 8) & (count < image.size * 0.065)
    keep[boundary] = False
    return np.column_stack((cx[keep], cy[keep])), np.column_stack(
        (count[keep], ratio[keep], angle[keep], h4[keep].real, h4[keep].imag)
    )


def _basis(xy, pitch):
    tree = cKDTree(xy)
    sample = xy[:: max(1, len(xy) // 1500)]
    _, idx = tree.query(sample, k=min(18, len(xy)))
    delta = (xy[idx[:, 1:]] - sample[:, None, :]).reshape(-1, 2)
    step = max(0.15, 0.052 * pitch)
    # Bound the histogram even in the presence of far-away outliers.
    lim = int(np.ceil(4.2 * pitch / step))
    bins = np.arange(-lim - 0.5, lim + 1.5) * step
    hist = np.histogram2d(delta[:, 0], delta[:, 1], bins=(bins, bins))[0]
    density = ndimage.gaussian_filter(hist, 1)
    peaks = (density == ndimage.maximum_filter(density, size=5)) & (
        density > 0.10 * density.max()
    )
    modes = []
    vectors = cKDTree(delta)
    for ij in np.argwhere(peaks):
        v = (ij - lim) * step
        if v[0] < -0.11 * pitch or (abs(v[0]) <= 0.11 * pitch and v[1] < 0):
            continue
        lengths = np.linalg.norm(v)
        if not 0.62 * pitch < lengths < 4 * pitch:
            continue
        ids = vectors.query_ball_point(v, 0.16 * pitch)
        if len(ids) < 0.20 * len(sample):
            continue
        v = np.median(delta[ids], axis=0)
        modes.append(v)
    options = []
    for a, b in combinations(modes, 2):
        det = abs(np.linalg.det([a, b]))
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if det < 0.45 * la * lb:
            continue
        options.append((round(det / (pitch * pitch), 1), la + lb, a, b))
    if not options:
        raise ValueError("No two independent repeating lattice directions")
    _, _, a, b = min(options, key=lambda v: v[:2])
    # Pick the more horizontal primitive as first. Canonical orientation
    # makes the reported basis and tile phase reproducible.
    if abs(a[0]) / np.linalg.norm(a) < abs(b[0]) / np.linalg.norm(b):
        a, b = b, a
    a = a * np.sign(a[0] or 1)
    b = b * np.sign(np.linalg.det([a, b]))
    basis = np.array([a, b])
    anchor = np.median(xy, axis=0)
    for _ in range(5):
        coords = (xy - anchor) @ np.linalg.inv(basis)
        offset = np.angle(np.mean(np.exp(2j * np.pi * coords), axis=0)) / (2 * np.pi)
        indices = np.rint(coords - offset)
        residual = np.linalg.norm((coords - offset - indices) @ basis, axis=1)
        good = residual < 0.19 * pitch
        if good.sum() < 8:
            break
        design = np.column_stack((np.ones(good.sum()), indices[good]))
        if np.linalg.matrix_rank(design) < 3:
            break
        fit = np.linalg.lstsq(design, xy[good], rcond=None)[0]
        anchor, basis = fit[0], fit[1:]
    coords = (xy - anchor) @ np.linalg.inv(basis)
    residual = np.linalg.norm((coords - np.rint(coords)) @ basis, axis=1)
    fraction = float(np.mean(residual < 0.19 * pitch))
    return basis, anchor, fraction


def detect_centers(image):
    """Independently detect native xy centers for reciprocal cross-run matching."""
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 12 or not np.isfinite(image).all():
        raise ValueError("Center detection needs a finite two-dimensional image")
    centers, _ = _islands(image)
    return centers


def _subregions(xy, feat, pitch, bounds, depth=0):
    """Find axis-aligned population changes or blank gaps before lattice fitting."""
    if len(xy) < 40 or depth >= 5:
        return [(np.arange(len(xy)), bounds)]
    nearest, ix = cKDTree(xy).query(xy, k=min(7, len(xy)))
    along = xy[ix[:, 1:]] - xy[:, None, :]
    theta = np.arctan2(along[:, :, 1], along[:, :, 0])
    fourth = np.mean(np.exp(4j * theta[:, :4]), axis=1)
    desc = np.column_stack(
        (
            np.log(nearest[:, 1].clip(0.5)),
            np.log(feat[:, 0]),
            np.log(feat[:, 1].clip(1, 5)),
            fourth.real,
            fourth.imag,
            feat[:, 3],
            feat[:, 4],
        )
    )
    best = None
    for axis in (0, 1):
        pos = xy[:, axis]
        order = np.argsort(pos)
        gaps = np.diff(pos[order])
        candidates = list(np.linspace(pos.min(), pos.max(), 22)[1:-1])
        candidates.extend((pos[order][:-1] + gaps / 2)[gaps > 2.2 * pitch])
        for cut in candidates:
            left = pos < cut
            if min(left.sum(), (~left).sum()) < 18:
                continue
            gap = pos[~left].min() - pos[left].max()
            # Compare near the cut to locate a boundary between distinct arrays.
            slab = max(4 * pitch, 0.20 * np.ptp(pos))
            lband, rband = left & (pos >= cut - slab), (~left) & (pos <= cut + slab)
            if min(lband.sum(), rband.sum()) < 9:
                lband, rband = left, ~left
            l, r = desc[lband], desc[rband]
            shift = np.abs(np.median(l, axis=0) - np.median(r, axis=0))
            spread = (
                np.percentile(l, 80, axis=0)
                - np.percentile(l, 20, axis=0)
                + np.percentile(r, 80, axis=0)
                - np.percentile(r, 20, axis=0)
            ) / 2
            score = float(
                np.max(
                    shift
                    / (spread + np.array([0.17, 0.37, 0.24, 0.75, 0.75, 0.21, 0.21]))
                )
            )
            if gap > 2.2 * pitch:
                score = max(score, gap / (1.7 * pitch))
            if score > 1.24 and (best is None or score > best[0]):
                best = (
                    score,
                    axis,
                    float((pos[left].max() + pos[~left].min()) / 2),
                    left,
                )
    if best is None:
        return [(np.arange(len(xy)), bounds)]
    _, axis, cut, left = best
    pos = xy[:, axis]
    # A cut on a coarse search grid can split one extra column of sites
    # from the true population. Minimize assignment errors in its neighborhood.
    l = np.median(desc[pos < cut], axis=0)
    r = np.median(desc[pos >= cut], axis=0)
    spread = np.percentile(desc, 90, axis=0) - np.percentile(desc, 10, axis=0)
    metric = abs(l - r) / np.maximum(spread, 0.025) ** 2
    side = np.sum((np.abs(desc - l) - np.abs(desc - r)) * metric, axis=1)
    if np.max(np.abs(side)) > 1e-4:
        cut = _best_boundary(pos, side, cut, 2.4 * pitch)
        left = pos < cut
    regions = []
    for side, sel in enumerate((left, ~left)):
        ids = np.flatnonzero(sel)
        b = list(bounds)
        b[axis if side else axis + 2] = int(np.ceil(cut))
        for local_ids, local_b in _subregions(
            xy[ids], feat[ids], pitch, tuple(b), depth + 1
        ):
            regions.append((ids[local_ids], local_b))
    return regions


def _best_boundary(pos, side, initial, radius):
    order = np.argsort(pos)
    edges = (pos[order][:-1] + pos[order][1:]) / 2
    valid = (abs(edges - initial) <= radius) & (np.diff(pos[order]) > 1e-5)
    if not valid.any():
        return initial
    # A good boundary puts negative (left-like) scores before positive ones.
    costs = np.cumsum(side[order])[:-1]
    return float(edges[np.argmin(np.where(valid, costs, np.inf))])


def _profile_partition(image, region, ellipse_angle, square_angle):
    """Compare widely separated strips where the center crop can hide mixtures."""
    xy = region.centers
    extent = np.ptp(xy, axis=0)
    for ax in np.argsort(-extent):
        if extent[ax] < 13 * region.pitch:
            continue
        position = xy[:, ax]
        cut = float(np.median(position))
        evidence = []
        for side in (position < cut, position >= cut):
            points = xy[side]
            if len(points) < 24:
                break
            evidence.append(
                compare_shape_profiles(
                    image, replace(region, centers=points), ellipse_angle, square_angle
                )
            )
        if len(evidence) != 2 or any(e is None or not e[2] for e in evidence):
            continue
        if evidence[0][0] == evidence[1][0]:
            continue
        # Profiles are used only to localize a family transition for discovery.
        # Keep the original samples for every subsequent CD and placement fit.
        n = max(3, min(12, int(region.pitch * 0.45)))
        grid = np.arange(-n, n + 1)
        dy, dx = np.meshgrid(grid, grid, indexing="ij")
        chosen = np.linspace(0, len(xy) - 1, min(1600, len(xy)), dtype=int)
        xc, yc = xy[chosen].T
        stamps = ndimage.map_coordinates(
            image.astype("f4", copy=False),
            [yc[:, None] + dy.ravel(), xc[:, None] + dx.ravel()],
            order=1,
            mode="nearest",
        )
        stamps -= np.percentile(stamps, 12, axis=1)[:, None]
        stamps /= np.maximum(np.max(stamps, axis=1)[:, None], 1e-6)
        p = position[chosen]
        q0, q1 = np.percentile(p, (24, 76))
        l, r = np.median(stamps[p <= q0], axis=0), np.median(stamps[p >= q1], axis=0)
        score = np.sum((stamps - (l + r) / 2) * (r - l), axis=1)
        cut = _best_boundary(p, score, cut, 0.36 * extent[ax])
        return int(ax), cut
    return None


def discover(image, pitch_px=None, shape="auto", regions=None):
    """Discover bright islands, locally homogeneous domains, and lattice bases.

    Explicit reference-coordinate rectangles use exclusive right/bottom bounds.
    Optional region keys are bounds, shape, pitch_px, and angle_deg.
    """
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 12 or not np.isfinite(image).all():
        raise ValueError("Discovery needs a finite 2D image of at least 12x12 pixels")
    h, w = image.shape
    configs = regions or [{"bounds": (0, 0, w, h)}]
    out = []
    for cfg in configs:
        x0, y0, x1, y1 = map(int, cfg["bounds"])
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Region bounds outside the native reference image")
        xy, feat = _islands(image[y0:y1, x0:x1])
        if len(xy) < 9:
            continue
        xy += [x0, y0]
        auto_pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
        given_pitch = cfg.get("pitch_px", pitch_px)
        if given_pitch is not None and (
            not np.isfinite(given_pitch) or given_pitch < 2
        ):
            raise ValueError(
                "pitch_px must be a finite distance of at least two pixels"
            )
        groups = (
            [(np.arange(len(xy)), (x0, y0, x1, y1))]
            if regions
            else _subregions(xy, feat, given_pitch or auto_pitch, (x0, y0, x1, y1))
        )
        for ids, bounds in groups:
            points, f = xy[ids], feat[ids]
            pitch = float(
                given_pitch or np.median(cKDTree(points).query(points, k=2)[0][:, 1])
            )
            try:
                basis, anchor, coherence = _basis(points, pitch)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if coherence < 0.45:
                continue
            length = np.linalg.norm(basis, axis=1)
            dot = np.dot(*basis)
            shear = abs(dot) / (length[0] * length[1])
            ratios = abs(dot) / length**2
            if shear < 0.075:
                lattice = (
                    "square"
                    if abs(np.log(length[0] / length[1])) < 0.09
                    else "rectangular"
                )
            elif np.min(np.abs(ratios - 0.5)) < 0.10:
                lattice = "staggered"
            else:
                lattice = "oblique"
            ecc = float(np.median(f[:, 1]))
            fourth = np.mean(f[:, 3] + 1j * f[:, 4])
            elongation = np.mean((f[:, 1] - 1) * np.exp(2j * f[:, 2]))
            ellipse_angle = float(np.angle(elongation) / 2)
            square_angle = float(
                (np.angle(-fourth) / 4 + np.pi / 4) % (np.pi / 2) - np.pi / 4
            )
            if ecc > 1.18 and abs(elongation) > 0.14:
                hypothesis = "ellipse"
                angle = ellipse_angle
            elif abs(fourth) > 0.205:
                hypothesis = "square"
                angle = square_angle
            else:
                hypothesis = "circle"
                angle = 0.0
            resolved = np.median(f[:, 0]) >= 40 and (
                hypothesis == "ellipse" or hypothesis == "square" and abs(fourth) > 0.27
            )
            choice = cfg.get("shape", shape)
            if choice not in ("auto", "circle", "ellipse", "square", "ambiguous"):
                raise ValueError(f"Unknown shape: {choice}")
            diagnostics = {
                "anchor_xy": anchor.tolist(),
                "lattice_inlier_fraction": coherence,
                "median_component_area_px": float(np.median(f[:, 0])),
                "apparent_axis_ratio": ecc,
                "fourfold_moment": float(abs(fourth)),
                "shape_resolved": bool(resolved),
            }
            region = Region(
                len(out),
                points,
                pitch,
                basis,
                choice,
                hypothesis,
                angle,
                lattice,
                bounds,
                coherence,
                diagnostics,
            )
            if regions is None and choice == "auto":
                split = _profile_partition(image, region, ellipse_angle, square_angle)
                if split is not None:
                    ax, cut = split
                    left = points[:, ax] < cut
                    for side, sel in enumerate((left, ~left)):
                        sub = list(bounds)
                        sub[ax if side else ax + 2] = int(np.ceil(cut))
                        groups.append((ids[sel], tuple(sub)))
                    continue
            evidence = compare_shape_profiles(
                image, region, ellipse_angle, square_angle
            )
            if evidence is not None:
                hypothesis, angle, resolved, scores = evidence
                region.diagnostics.update(scores, shape_resolved=resolved)
            region.best_shape = hypothesis
            region.shape = (
                (hypothesis if resolved else "ambiguous")
                if choice == "auto"
                else choice
            )
            if choice in ("square", "ellipse") and hypothesis != choice:
                angle = {"square": square_angle, "ellipse": ellipse_angle}[choice]
            region.angle = (
                float(np.deg2rad(cfg["angle_deg"])) if "angle_deg" in cfg else angle
            )
            out.append(region)
    return out


def match_centers(reference_xy, moving_xy, shift_xy, pitch_px):
    """Return reciprocal, gated reference/moving index pairs without reuse."""
    predicted = np.asarray(reference_xy) + np.asarray(shift_xy)
    moving = np.asarray(moving_xy)
    if len(predicted) == 0 or len(moving) == 0:
        return np.empty((0, 2), dtype=int)
    dist, nearest = cKDTree(moving).query(predicted)
    _, back = cKDTree(predicted).query(moving)
    idx = np.flatnonzero(
        (dist < 0.32 * pitch_px) & (back[nearest] == np.arange(len(predicted)))
    )
    return np.column_stack((idx, nearest[idx]))


def _registration_band(image, pitch):
    image = np.asarray(image, dtype="f4")
    width = int(np.clip(round(pitch * 1.6), 5, 101))
    image = ndimage.gaussian_filter(image, 0.65) - ndimage.uniform_filter(image, width)
    scale = np.percentile(np.abs(image), 98)
    return np.clip(image / max(scale, 1e-10), -4, 4)


def register(reference, moving, pitch_px, max_shift_px=None):
    """Estimate a bounded translation; report alternate peaks beyond the prior.

    The radial prior defaults to 0.45 of the nearest-neighbor pitch because
    these are nominally same-area crops. Setting a larger bound tests whether
    landmarks actually disambiguate neighboring lattice-period solutions.
    """
    if np.shape(reference) != np.shape(moving) or np.ndim(reference) != 2:
        raise ValueError("Registration requires equal-shape 2D images")
    if not np.isfinite(pitch_px) or pitch_px <= 0:
        raise ValueError("Registration requires positive pitch_px")
    bound = 0.45 * pitch_px if max_shift_px is None else float(max_shift_px)
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("max_shift_px must be positive")
    a, b = (_registration_band(i, pitch_px) for i in (reference, moving))
    h, w = a.shape
    win = (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype("f4")
    fa, fb = (fft.rfft2(i * win) for i in (a, b))
    cross = fb * fa.conj()
    # Partial whitening preserves useful low-frequency landmarks under blur.
    cross /= np.sqrt(np.abs(cross) + max(1e-12, float(np.abs(cross).max()) * 1e-7))
    corr = fft.fftshift(fft.irfft2(cross, s=(h, w)))
    radius = int(min(np.ceil(max(bound + 2, 2.3 * pitch_px)), (min(h, w) - 5) // 2))
    if radius < 2 or max(a.std(), b.std()) < 1e-9 or min(a.std(), b.std()) < 1e-9:
        return Registration(np.zeros(2), 0.0, True, "insufficient_contrast", [], True)
    corr = corr[
        h // 2 - radius : h // 2 + radius + 1, w // 2 - radius : w // 2 + radius + 1
    ]
    peaks = corr == ndimage.maximum_filter(corr, size=3)
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    # Keep the strongest candidates inside and outside the requested prior.
    ypk, xpk = np.nonzero(peaks)
    inside = np.hypot(ypk - radius, xpk - radius) <= bound + 1
    chosen = []
    for group in (np.flatnonzero(inside), np.flatnonzero(~inside)):
        chosen.extend(group[np.argsort(corr[ypk[group], xpk[group]])[-16:]])
    # Score competing translations on the same native interior with NCC.
    border = min(radius + 2, (min(h, w) - 3) // 2)
    step = max(1, int(np.sqrt((h - 2 * border) * (w - 2 * border) / 45000)))
    yy, xx = np.mgrid[border : h - border : step, border : w - border : step]
    base = a[yy, xx].ravel()
    base = base - base.mean()
    norm = np.linalg.norm(base)
    candidates = []
    for j in chosen:
        y, x = ypk[j], xpk[j]
        offsets = []
        for lo, hi in (
            (corr[y, x - 1], corr[y, x + 1]),
            (corr[y - 1, x], corr[y + 1, x]),
        ):
            curvature = lo + hi - 2 * corr[y, x]
            offsets.append(
                float(np.clip(0.5 * (lo - hi) / curvature, -0.7, 0.7))
                if curvature < 0
                else 0.0
            )
        dx, dy = np.array((x - radius, y - radius)) + offsets
        vals = ndimage.map_coordinates(
            b, [yy + dy, xx + dx], order=1, mode="nearest"
        ).ravel()
        vals -= vals.mean()
        score = float(np.dot(base, vals) / max(1e-15, norm * np.linalg.norm(vals)))
        candidates.append((float(dx), float(dy), score))
    candidates.sort(key=lambda v: -v[2])
    allowed = [p for p in candidates if np.hypot(*p[:2]) <= bound + 1e-6]
    if not allowed:
        return Registration(
            np.zeros(2), 0.0, True, "no_in_bounds_peak", candidates, False
        )
    best = allowed[0]

    def alternative(peaks):
        return any(
            np.linalg.norm(np.subtract(p[:2], best[:2])) > 0.5 * pitch_px
            and p[2] >= best[2] - 0.003
            for p in peaks
        )

    prior_conflict = candidates[0][2] > best[2] + max(0.012, 0.08 * (1 - best[2]))
    ambiguous = best[2] < 0.35 or alternative(allowed) or prior_conflict
    alias = alternative(candidates)
    reason = (
        "low_correlation"
        if best[2] < 0.35
        else "stronger_peak_outside_prior"
        if prior_conflict
        else "ambiguous_in_bounds"
        if ambiguous
        else "bounded_periodic_solution"
        if alias
        else "ok"
    )
    return Registration(
        np.asarray(best[:2]), best[2], ambiguous, reason, candidates, alias
    )
