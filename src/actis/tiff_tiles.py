"""Reference-guided selection of complete bright-hole array tiles on CUDA."""

from dataclasses import dataclass, replace

import numpy as np

from .discovery import Region, discover
from .harmonics import _frequencies


@dataclass
class ReferenceTile:
    region: Region
    pixels: object
    spectrum: object
    search: object
    window: object
    dft_x: object
    dft_y: object

    def geometry(self, shift):
        """Transfer native reference seeds and lattice phase to the matched tile."""
        side = self.pixels.shape[0]
        centers = self.region.centers + shift
        inside = ((centers >= -0.5) & (centers < side - 0.5)).all(axis=1)
        return replace(
            self.region,
            centers=centers[inside],
            diagnostics={
                **self.region.diagnostics,
                "anchor_xy": self.region.diagnostics["anchor_xy"] + shift,
            },
        )

    def sample_orders(self, patch, freq, anchor):
        """Reuse the separable native DFT and off-lattice noise probes on CUDA."""
        weighted = (patch - (patch * self.window).sum(dtype=np.float64)) * self.window
        values = ((self.dft_y @ weighted.astype(self.dft_y.dtype)) * self.dft_x).sum(
            axis=1
        )
        amplitude, side1, side2 = np.split(values.get(), 3)
        noise = max(1e-12, float(np.median(np.minimum(abs(side1), abs(side2)))))
        return amplitude * np.exp(2j * np.pi * (freq @ anchor)), noise


def prepare_reference(image, side, shape, cp):
    templates = []
    for region in discover(image, shape=shape):
        x0, y0, x1, y1 = region.bounds
        if min(x1 - x0, y1 - y0) < side or side < 5 * region.pitch:
            continue
        # A homogeneous reference domain needs only one exemplar, irrespective
        # of its position in a scan. Retain a halo of seeds for translated tiles.
        x, y = (x0 + x1 - side) // 2, (y0 + y1 - side) // 2
        radius = min(side // 4, int(np.ceil(max(np.linalg.norm(region.basis, axis=1)))))
        centers = region.centers - [x, y]
        in_tile = ((centers >= -0.5) & (centers < side - 0.5)).all(axis=1)
        if in_tile.sum() < max(16, 0.74 * side**2 / abs(np.linalg.det(region.basis))):
            continue
        nearby = ((centers >= -radius - 0.5) & (centers < side + radius - 0.5)).all(
            axis=1
        )
        local = replace(
            region,
            bounds=(0, 0, side, side),
            centers=centers[nearby],
            diagnostics={
                **region.diagnostics,
                "anchor_xy": np.asarray(region.diagnostics["anchor_xy"]) - [x, y],
            },
        )
        pixels = cp.asarray(image[y : y + side, x : x + side], dtype=cp.float32)
        window = np.outer(np.hanning(side), np.hanning(side))
        window /= window.sum()
        weights = cp.asarray(window, dtype=cp.float32)
        spectrum = cp.fft.rfft2((pixels - (pixels * weights).sum()) * weights).conj()
        freq = _frequencies(region.basis)
        probes = np.vstack(
            (freq, freq + [3.6 / side, 2.4 / side], freq + [-2.3 / side, 3.7 / side])
        )
        axis = np.arange(side)
        tx, ty = [cp.asarray(np.exp(-2j * np.pi * f[:, None] * axis)) for f in probes.T]
        templates.append(
            ReferenceTile(
                local,
                pixels,
                spectrum,
                cp.arange(-radius, radius + 1),
                weights,
                tx,
                ty,
            )
        )
    if not templates:
        raise ValueError(
            "Reference needs a complete tile of bright holes with a resolved "
            "two-dimensional lattice and at least five periods across the tile"
        )
    return templates


def _match(tiles, spectra, template, cp):
    from cupyx.scipy.ndimage import map_coordinates

    n, side, _ = tiles.shape
    cross = cp.fft.irfft2(spectra * template.spectrum, s=(side, side), axes=(-2, -1))
    # Whole-pitch shifts are equivalent for periodic metrology. Search nearby
    # translations so wraparound cannot displace most of the physical tile.
    search = template.search
    local = cross[:, search[:, None], search]
    highest = local.max(axis=(1, 2), keepdims=True)
    # Favor the nearest equivalent peak on a perfectly periodic array. An
    # arbitrary whole-pitch alias would unnecessarily discard boundary seeds.
    near_peak = local >= highest - abs(highest) * 1e-5
    distance2 = search[:, None] ** 2 + search**2
    peak = cp.where(near_peak, distance2, side**2).reshape(n, -1).argmin(axis=1)
    dy, dx = search[peak // len(search)], search[peak % len(search)]
    ids = cp.arange(n)
    center = cross[ids, dy, dx]

    def interpolate(left, right):
        curvature = 2 * center - left - right
        return cp.where(
            curvature > 0,
            cp.clip((right - left) / cp.maximum(2 * curvature, 1e-20), -0.5, 0.5),
            0,
        )

    shift_x = dx + interpolate(
        cross[ids, dy, (dx - 1) % side], cross[ids, dy, (dx + 1) % side]
    )
    shift_y = dy + interpolate(
        cross[ids, (dy - 1) % side, dx], cross[ids, (dy + 1) % side, dx]
    )
    axis = cp.arange(side)
    rx, ry = axis - shift_x[:, None], axis - shift_y[:, None]
    overlap = ((rx >= 0) & (rx <= side - 1))[:, None, :] & (
        (ry >= 0) & (ry <= side - 1)
    )[:, :, None]
    coords = cp.stack(cp.broadcast_arrays(ry[:, :, None], rx[:, None, :]))
    # Only the overlap contributes. Clamp the interpolation stencil at its
    # last pixel, including when a fractional coordinate rounds to the edge.
    aligned = map_coordinates(
        template.pixels, coords, order=1, mode="nearest", prefilter=False
    )

    def quadrant_sum(a):
        return a.reshape(n, 2, side // 2, 2, side // 2).sum(axis=(2, 4))

    # Center each quadrant before accumulating squares, including for 16-bit
    # pages with weak contrast on a large DC offset.
    count = quadrant_sum(overlap)

    def centered(a):
        mean = quadrant_sum(a * overlap) / count
        return (
            a.reshape(n, 2, side // 2, 2, side // 2) - mean[:, :, None, :, None]
        ).reshape(tiles.shape) * overlap

    a, b = centered(tiles), centered(aligned)
    energy = cp.sqrt(quadrant_sum(a * a) * quadrant_sum(b * b))
    corr = cp.where(energy > 0, quadrant_sum(a * b) / cp.maximum(energy, 1e-20), 0)
    corr = cp.where(corr >= 1 - 1e-12, 1, corr)
    return corr.min(axis=(1, 2)), shift_x, shift_y


def select_tiles(tiles, templates, min_corr, cp):
    """Return (batch index, reference index, native dx, dy) for accepted tiles."""
    data = tiles.astype(cp.float32, copy=False)
    window = templates[0].window
    centered = data - (data * window).sum(axis=(1, 2), keepdims=True)
    spectra = cp.fft.rfft2(centered * window)
    best = cp.full(len(data), -cp.inf, dtype=cp.float32)
    match = cp.zeros((len(data), 3))
    for i, template in enumerate(templates):
        score, dx, dy = _match(data, spectra, template, cp)
        better = score > best
        match[:, 0] = cp.where(better, i, match[:, 0])
        match[:, 1] = cp.where(better, dx, match[:, 1])
        match[:, 2] = cp.where(better, dy, match[:, 2])
        best = cp.maximum(score, best)
    keep = cp.flatnonzero(best >= min_corr)
    return cp.column_stack((keep, match[keep])).get()
