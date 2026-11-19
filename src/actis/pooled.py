"""Native relative-intensity aperture area and placement pooled over lattice tiles."""

from dataclasses import dataclass

import numpy as np

from .fit import to_host

try:
    from numba import njit
except ImportError:
    njit = None


def _moments_numpy(image, xy, low, scale, half, xp=np):
    """Vectorized fallback with bounded stamp batches."""
    offsets = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    border = (abs(dy) == half) | (abs(dx) == half)
    basis = np.stack(
        (np.ones_like(dx), dx, dy, dx * dx, dy * dy, dx * dy, border), axis=-1
    )
    basis = xp.asarray(basis.reshape(-1, 7), dtype=np.float64)
    dx, dy = xp.asarray(dx.ravel()), xp.asarray(dy.ravel())
    out = xp.empty((len(xy), 7))
    batch = max(1, 120000 // dx.size)
    for i in range(0, len(xy), batch):
        sites = xy[i : i + batch]
        stamp = image[sites[:, 1, None] + dy.ravel(), sites[:, 0, None] + dx.ravel()]
        weights = xp.clip(
            (stamp - low[i : i + batch, None]) * scale[i : i + batch, None], 0, 1
        )
        out[i : i + batch] = weights @ basis
    return out


def _native_moments(image, xy, low, scale, half):
    out = np.zeros((len(xy), 7))
    for k in range(len(xy)):
        cx, cy = xy[k]
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                weight = min(1.0, (image[cy + y, cx + x] - low[k]) * scale[k])
                if weight <= 0:
                    continue
                out[k, 0] += weight
                out[k, 1] += weight * x
                out[k, 2] += weight * y
                out[k, 3] += weight * x * x
                out[k, 4] += weight * y * y
                out[k, 5] += weight * x * y
                if abs(x) == half or abs(y) == half:
                    out[k, 6] += weight
    return out


_moments = (
    njit(nogil=True, cache=True)(_native_moments)
    if njit is not None
    else _moments_numpy
)


def _measure(image, centers, half, xp=np):
    """Independent local contrast and soft pixel occupancy for each seed."""
    xy = xp.asarray(np.rint(centers).astype(np.int64))
    mid = max(1, half // 2)
    offsets = np.arange(-1, 2)
    cy, cx = np.meshgrid(offsets, offsets, indexing="ij")
    core = image[
        xy[:, 1, None] + xp.asarray(cy.ravel()), xy[:, 0, None] + xp.asarray(cx.ravel())
    ]
    # Second brightest in the central 3x3 suppresses isolated hot pixels.
    bright = xp.partition(core, 7, axis=1)[:, 7].astype(float)
    u = np.array(
        [
            -half,
            -mid,
            0,
            mid,
            half,
            half,
            half,
            half,
            half,
            mid,
            0,
            -mid,
            -half,
            -half,
            -half,
            -half,
        ]
    )
    v = np.array(
        [
            -half,
            -half,
            -half,
            -half,
            -half,
            -mid,
            0,
            mid,
            half,
            half,
            half,
            half,
            half,
            mid,
            0,
            -mid,
        ]
    )
    ring = image[xy[:, 1, None] + xp.asarray(v), xy[:, 0, None] + xp.asarray(u)]
    q = xp.partition(ring, (3, 7, 8, 12), axis=1).astype(float)
    dark = 0.5 * (q[:, 7] + q[:, 8])
    contrast = bright - dark
    noise = xp.maximum((q[:, 12] - q[:, 3]) / 1.349, 1e-6 * xp.maximum(abs(dark), 1))
    good = contrast > 3.5 * noise
    saturated = xp.zeros(len(xy), bool)
    if image.dtype.kind in "ui":
        saturated = (core == np.iinfo(image.dtype).max).sum(axis=1) >= 2
    # A finite ramp integrates fractional occupancy around a local half-height
    # threshold. Its endpoints are fixed before testing against the CD sweep.
    low = dark + 0.28 * contrast
    scale = 1.0 / xp.maximum(0.44 * contrast, 1e-12)
    m = (
        _moments(image, xy, low, scale, half)
        if xp is np
        else _moments_numpy(image, xy, low, scale, half, xp)
    )
    area = m[:, 0]
    denominator = xp.maximum(area, 1e-12)
    delta = m[:, 1:3] / denominator[:, None]
    cx, cy = delta.T
    good &= ~saturated & (area >= 2.0) & (m[:, 6] <= 0.035 * area)
    good &= xp.hypot(cx, cy) <= 0.58 * half
    cov = m[:, 3:6] / denominator[:, None] - xp.column_stack(
        (cx * cx, cy * cy, cx * cy)
    )
    return tuple(to_host(xp, a) for a in (good, area, xy + delta, cov, saturated))


@dataclass
class _Grid:
    start: np.ndarray
    nx: int
    ny: int
    cell_area: float
    bounds: np.ndarray
    half: int
    centers: np.ndarray
    bins: np.ndarray
    ideal: np.ndarray
    counts: np.ndarray


def _grid(region, tile_px, origin):
    if not np.isfinite(tile_px) or tile_px < 4:
        raise ValueError("Tile width must be at least four native pixels")
    bounds = np.asarray(region.bounds) - 0.5
    start = np.ceil((bounds[:2] - origin) / tile_px - 1e-10).astype(int)
    stop = np.floor((bounds[2:] - origin) / tile_px + 1e-10).astype(int)
    nx, ny = stop - start
    if nx <= 0 or ny <= 0:
        return None
    cell_area = abs(np.linalg.det(region.basis))
    if cell_area <= 0 or not np.isfinite(cell_area):
        raise ValueError("Nondegenerate lattice basis required for pooled geometry")
    half = max(2, int(np.floor((region.pitch - 1.0) / 2)))
    centers = np.asarray(region.centers)
    grid = np.floor((centers - origin) / tile_px).astype(int)
    use = ((grid >= start) & (grid < stop)).all(axis=1)
    centers, grid = centers[use], grid[use] - start
    ids = grid[:, 1] * nx + grid[:, 0]
    counts = np.bincount(ids, minlength=nx * ny)
    anchor = np.asarray(region.diagnostics["anchor_xy"])
    ideal = (
        np.rint((centers - anchor) @ np.linalg.inv(region.basis)) @ region.basis
        + anchor
    )
    return _Grid(start, nx, ny, cell_area, bounds, half, centers, ids, ideal, counts)


@dataclass
class _Exposure:
    valid: np.ndarray
    moments: np.ndarray
    saturated: np.ndarray
    shift: np.ndarray
    shape: tuple


def _exposure(data, grid, shift, xp=np):
    centers, half, bounds = grid.centers, grid.half, grid.bounds
    rounded = np.rint(centers + shift).astype(int)
    usable = ((rounded >= half) & (rounded < np.array(data.shape[::-1]) - half)).all(
        axis=1
    )
    # Do not collect a stamp from an adjacent design domain.
    usable &= ((centers - half >= bounds[:2]) & (centers + half < bounds[2:])).all(
        axis=1
    )
    sites = centers[usable]
    good, area, measured, cov, saturated = _measure(data, sites + shift, half, xp)
    valid = np.zeros(len(centers), bool)
    valid[usable] = good
    sat = np.zeros(len(centers), bool)
    sat[usable] = saturated
    moments = np.zeros((len(centers), 6))
    moments[usable] = np.column_stack(
        (area, measured - (grid.ideal[usable] + shift), cov)
    )
    return _Exposure(valid, moments, sat, shift, data.shape)


def _rows(region, grid, exposure, common, tile_px, origin):
    nx, start, cell_area = grid.nx, grid.start, grid.cell_area
    nt = nx * grid.ny
    shift = exposure.shift
    bins = grid.bins[common]
    accepted = np.bincount(bins, minlength=nt)
    native = np.bincount(grid.bins[exposure.valid], minlength=nt)
    sat = np.bincount(grid.bins[exposure.saturated], minlength=nt)
    averaged = np.stack(
        [
            np.bincount(bins, weights=col, minlength=nt)
            for col in exposure.moments[common].T
        ],
        axis=1,
    ).astype(float)
    averaged /= np.maximum(1, accepted[:, None])
    out = []
    for index in range(nt):
        j, i = divmod(index, nx)
        i, j = i + start[0], j + start[1]
        x0, y0 = origin + [i * tile_px, j * tile_px]
        x1, y1 = x0 + tile_px, y0 + tile_px
        count, valid = int(grid.counts[index]), int(accepted[index])
        row = dict(
            tile_i=int(i),
            tile_j=int(j),
            region_id=region.region_id,
            x=x0 + tile_px / 2,
            y=y0 + tile_px / 2,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            n_sites=count,
            valid_sites=valid,
            native_valid_sites=int(native[index]),
            valid_fraction=valid / max(count, 1),
            saturated_sites=int(sat[index]),
            cd_px=np.nan,
            cd_x_px=np.nan,
            cd_y_px=np.nan,
            area_diameter_px=np.nan,
            dx_px=np.nan,
            dy_px=np.nan,
            axis_angle_rad=np.nan,
            phase_status="unavailable",
            status="insufficient_pattern_coverage",
            cd_basis={"ellipse": "major diameter", "square": "square side"}.get(
                region.shape, "area-equivalent diameter"
            ),
        )
        out.append(row)
        # Test coverage both against expected lattice occupancy and actual
        # contrast/neighbor/saturation checks, so invisible sites cost coverage.
        if (
            min(x0 + shift[0], y0 + shift[1]) < -0.500001
            or x1 + shift[0] > exposure.shape[1] - 0.499999
            or y1 + shift[1] > exposure.shape[0] - 0.499999
        ):
            row["status"] = "incomplete_native_footprint"
            continue
        if count < max(9, 0.70 * tile_px**2 / cell_area):
            continue
        if valid < max(8, 0.65 * count):
            row["status"] = "insufficient_valid_apertures"
            continue
        mean_area, dx, dy, vx, vy, vxy = averaged[index]
        if np.hypot(dx, dy) > 0.33 * region.pitch:
            row["status"] = "inconsistent_lattice_placement"
            continue
        diameter = 2 * np.sqrt(mean_area / np.pi)
        row.update(dx_px=dx, dy_px=dy, phase_status="ok", area_diameter_px=diameter)
        if region.shape == "ellipse":
            gap = np.hypot((vx - vy) / 2, vxy)
            mean_var = (vx + vy) / 2
            if mean_var - gap <= 0.05:
                row["status"] = "ellipse_axes_unresolved"
                continue
            aspect = np.sqrt((mean_var + gap) / (mean_var - gap))
            major, minor = diameter * np.sqrt(aspect), diameter / np.sqrt(aspect)
            angle = 0.5 * np.arctan2(2 * vxy, vx - vy)
        elif region.shape == "square":
            major = minor = np.sqrt(mean_area)
            angle = region.angle
        else:
            major = minor = diameter
            angle = np.nan
        row.update(
            cd_px=major,
            cd_x_px=major if region.shape != "ambiguous" else np.nan,
            cd_y_px=minor if region.shape != "ambiguous" else np.nan,
            axis_angle_rad=angle,
            status="ok",
        )
    return out


def pooled_group_tiles(images, region, shifts, tile_px, origin_xy, groups, *, xp=np):
    """Measure once per acquisition; intersect stable site IDs inside each group."""
    origin = np.asarray(origin_xy, dtype=float)
    grid = _grid(region, tile_px, origin)
    if len(images) != len(shifts):
        raise ValueError("Each acquisition needs a native shift")
    if grid is None:
        return {tuple(group): [[] for _ in group] for group in groups}
    samples = [
        _exposure(xp.asarray(image), grid, np.asarray(shift), xp)
        for image, shift in zip(images, shifts)
    ]
    out = {}
    for group in groups:
        group = tuple(group)
        common = np.logical_and.reduce([samples[i].valid for i in group])
        out[group] = [
            _rows(region, grid, samples[i], common, tile_px, origin) for i in group
        ]
    return out


def pooled_common_tiles(images, region, shifts, tile_px=128, origin_xy=(0.0, 0.0)):
    """Return [repeat][tile] rows measuring exactly the same apertures in each repeat."""
    group = tuple(range(len(images)))
    if not group:
        return []
    return pooled_group_tiles(images, region, shifts, tile_px, origin_xy, [group])[
        group
    ]


def pooled_tiles(
    image, region, shift_xy=(0.0, 0.0), tile_px=128, origin_xy=(0.0, 0.0), *, xp=np
):
    """Mean area and within-hole moments of complete locally normalized apertures.

    Holes belong to tiles by reference center. Displacements are relative to
    ideal lattice sites after removing the global acquisition translation.
    """
    return pooled_group_tiles(
        [image], region, [shift_xy], tile_px, origin_xy, [(0,)], xp=xp
    )[(0,)][0]
