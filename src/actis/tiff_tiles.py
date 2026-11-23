"""Reference-guided selection of complete bright-hole array tiles on CUDA."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import repeat

import numpy as np

from .discovery import Region, discover
from .harmonics import _frequencies, measure_orders


def _harmonic_cds(region, freq, amplitudes, floors):
    values = []
    for amplitude, floor in zip(amplitudes, floors):
        row = measure_orders(
            amplitude, floor, freq, region, {"status": "insufficient_orders"}
        )
        values.append(row.get("cd_px", np.nan))
    return values


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

    def harmonic_cds(self, tiles, shifts, pool, workers):
        side = tiles.shape[-1]
        freq = _frequencies(self.region.basis)
        centers = self.region.centers + shifts[:, None, :]
        counts = ((centers >= -0.5) & (centers < side - 0.5)).all(axis=2).sum(axis=1)
        enough = counts >= max(
            16, 0.74 * side**2 / abs(np.linalg.det(self.region.basis))
        )
        cds = np.full(len(tiles), np.nan)
        if len(freq) < 5 or side < 5 * self.region.pitch or not enough.any():
            return cds
        means = (tiles * self.window).sum(axis=(1, 2), keepdims=True, dtype=np.float64)
        weighted = ((tiles - means) * self.window).astype(self.dft_y.dtype)
        orders = ((self.dft_y @ weighted) * self.dft_x).sum(axis=2).get()[enough]
        amp, off1, off2 = np.split(orders, 3, axis=1)
        floors = np.maximum(1e-12, np.median(np.minimum(abs(off1), abs(off2)), axis=1))
        anchors = self.region.diagnostics["anchor_xy"] + shifts[enough]
        amp *= np.exp(2j * np.pi * (anchors @ freq.T))
        if pool is None or len(amp) < workers:
            cds[enough] = _harmonic_cds(self.region, freq, amp, floors)
        else:
            jobs = pool.map(
                _harmonic_cds,
                repeat(self.region),
                repeat(freq),
                np.array_split(amp, workers),
                np.array_split(floors, workers),
            )
            cds[enough] = np.concatenate(list(jobs))
        return cds


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


@lru_cache(maxsize=1)
def _quadrant_kernels(cp):
    # Calculate coordinates in registers. Match CuPy's order-1 sampling of a
    # float32 reference, including how it rounds the stencil near integer edges.
    align = cp.ElementwiseKernel(
        "float32 tile, raw float32 ref, raw float64 dx, raw float64 dy, int32 side",
        "float32 target, float32 aligned, bool overlap",
        r"""
        const int tile_id = i / side / side;
        const double x = i % side - dx[tile_id], y = (i / side) % side - dy[tile_id];
        overlap = x >= 0 && x <= side - 1 && y >= 0 && y <= side - 1;
        target = overlap ? tile : 0;
        aligned = 0;
        if (overlap) {
            const int x0 = (int)floorf((float)x), y0 = (int)floorf((float)y);
            const int x1 = min(x0 + 1, side - 1), y1 = min(y0 + 1, side - 1);
            const double fx = x - x0, fy = y - y0;
            aligned = ref[(size_t)y0 * side + x0] * (float)((1 - fy) * (1 - fx));
            aligned = fmaf(ref[(size_t)y0 * side + x1], (float)((1 - fy) * fx), aligned);
            aligned = fmaf(ref[(size_t)y1 * side + x0], (float)(fy * (1 - fx)), aligned);
            aligned = fmaf(ref[(size_t)y1 * side + x1], (float)(fy * fx), aligned);
        }
        """,
        "actis_align_overlap",
    )
    moments = cp.ElementwiseKernel(
        "raw float32 target, raw float32 aligned, raw bool overlap, "
        "raw float64 mean_a, raw float64 mean_b, int32 side, int32 columns",
        "float64 aa, float64 bb, float64 ab",
        r"""
        const int half_side = side / 2, tile = i / columns / 4;
        const int quadrant = (i / columns) % 4, column = i % columns;
        const int top = (quadrant / 2) * half_side;
        const int x = (quadrant % 2) * half_side + column % half_side;
        const int y0 = top + (column / half_side) * 32;
        const double ma = mean_a[tile * 4 + quadrant], mb = mean_b[tile * 4 + quadrant];
        aa = bb = ab = 0;
        for (int y = y0; y < min(y0 + 32, top + half_side); ++y) {
            const size_t p = ((size_t)tile * side + y) * side + x;
            if (overlap[p]) {
                const double a = target[p] - ma, b = aligned[p] - mb;
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
        }
        """,
        "actis_centered_quadrant_moments",
    )
    return align, moments


def _quadrant_scores(tiles, pixels, shift_x, shift_y, cp):
    n, side, _ = tiles.shape
    align, moments = _quadrant_kernels(cp)
    # Keep the same per-quadrant means and double-precision centered products
    # for weak 16-bit contrast. Fuse each product with its reduction to avoid
    # full-size centered arrays and square/product temporaries.
    half = side // 2
    shape = (n, 2, half, 2, half)
    target, aligned, overlap = (
        a.reshape(shape) for a in align(tiles, pixels, shift_x, shift_y, side)
    )
    axes = (2, 4)
    count = overlap.sum(axis=axes, keepdims=True)
    mean_a = target.sum(axis=axes, keepdims=True) / count
    mean_b = aligned.sum(axis=axes, keepdims=True) / count
    # Each thread covers one column of a 32-row strip. Adjacent threads
    # read adjacent pixels, including when only one large tile fits a batch.
    columns = half * ((half + 31) // 32)
    partial = cp.empty((3, n, 4, columns), dtype=cp.float64)
    moments(target, aligned, overlap, mean_a, mean_b, side, columns, *partial)
    aa, bb, ab = partial.sum(axis=-1)
    energy = cp.sqrt(aa * bb)
    corr = cp.where(energy > 0, ab / cp.maximum(energy, 1e-20), 0)
    return cp.where(corr >= 1 - 1e-12, 1, corr).min(axis=1)


def _match(tiles, spectra, template, cp):
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
    scores = _quadrant_scores(tiles, template.pixels, shift_x, shift_y, cp)
    return scores, shift_x, shift_y


def select_tiles(tiles, templates, min_corr, cp):
    """Return (batch index, reference index, native dx, dy) for accepted tiles."""
    data = tiles.astype(cp.float32, copy=False)
    window = templates[0].window
    centered = data - (data * window).sum(axis=(1, 2), keepdims=True)
    spectra = cp.fft.rfft2(centered * window)
    best, match_x, match_y = _match(data, spectra, templates[0], cp)
    reference_ids = cp.zeros(len(data), dtype=cp.int32)
    for i, template in enumerate(templates[1:], 1):
        score, dx, dy = _match(data, spectra, template, cp)
        better = score > best
        reference_ids = cp.where(better, i, reference_ids)
        match_x = cp.where(better, dx, match_x)
        match_y = cp.where(better, dy, match_y)
        best = cp.maximum(score, best)
    keep = cp.flatnonzero(best >= min_corr)
    return cp.column_stack(
        (keep, reference_ids[keep], match_x[keep], match_y[keep])
    ).get()
