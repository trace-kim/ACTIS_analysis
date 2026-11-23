"""Batched native-site measurements for the reference-matched tiles of one TIFF page."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .estimators import (
    BLUR_CANDIDATES,
    BLUR_STEP,
    N_RAYS,
    _blur_result,
    _blur_vertex,
)


@lru_cache(maxsize=2)
def _kernels(cp, max_fit_threads=512):
    source = Path(__file__).with_suffix(".cu").read_text(encoding="ascii")
    return cp.RawModule(
        code=source,
        options=(f"-DACTIS_FIT_THREADS={max_fit_threads}", f"-DACTIS_RAYS={N_RAYS}"),
    )


def _power_two(n):
    return 1 << (int(n) - 1).bit_length()


@dataclass
class Sites:
    tiles: object
    centers: np.ndarray
    tile_id: np.ndarray
    seed: np.ndarray
    positions: object
    owners: object
    region: object

    @classmethod
    def matched(cls, tiles, region, shifts, cp):
        side = tiles.shape[-1]
        coords = region.centers + shifts[:, None, :]
        tile_id, seed = np.nonzero(
            ((coords >= -0.5) & (coords < side - 0.5)).all(axis=2)
        )
        centers = coords[tile_id, seed]
        return cls(
            tiles.astype(cp.float32),
            centers,
            tile_id,
            seed,
            cp.asarray(centers),
            cp.asarray(tile_id, dtype=cp.int32),
            region,
        )

    def contours(self, cp):
        pitch = self.region.pitch
        radius = 0.64 * pitch
        distance = np.linspace(0, radius, max(15, int(np.ceil(radius * 4)) + 1))
        rays = np.arange(N_RAYS) * 2 * np.pi / N_RAYS
        dirs = np.column_stack((np.cos(rays), np.sin(rays)))
        offsets = dirs[:, None] * distance[None, :, None]
        dark = distance.searchsorted(0.65 * radius)
        bright = distance.searchsorted(max(0.7, 0.15 * pitch), side="right")
        ordered = _power_two(N_RAYS * (len(distance) - dark))
        count = N_RAYS * len(distance) + max(ordered, _power_two(N_RAYS * bright))
        out = cp.empty((len(self.centers), 3), dtype=cp.float64)
        if len(out):
            _kernels(cp).get_function("contours")(
                (len(out),),
                (128,),
                (
                    self.tiles,
                    self.positions,
                    self.owners,
                    np.int32(self.tiles.shape[-1]),
                    cp.asarray(offsets),
                    cp.asarray(dirs),
                    np.int32(len(distance)),
                    np.int32(dark),
                    np.int32(bright),
                    np.float64(radius),
                    np.float64(distance[1]),
                    np.int32(ordered),
                    np.int32({"square": 1, "ellipse": 2}.get(self.region.shape, 0)),
                    out,
                ),
                shared_mem=count * 4,
            )
        return out.get()

    def flux(self, cp):
        pitch = self.region.pitch
        half = int(np.ceil(0.64 * pitch + 0.7))
        ordered = _power_two((half * 2 + 1) ** 2)
        out = cp.empty(len(self.centers), dtype=cp.float64)
        if len(out):
            _kernels(cp).get_function("flux")(
                (len(out),),
                (128,),
                (
                    self.tiles,
                    self.positions,
                    self.owners,
                    np.int32(self.tiles.shape[-1]),
                    np.float64(pitch),
                    np.int32(half),
                    np.int32(ordered),
                    out,
                ),
                shared_mem=4 * ordered,
            )
        return out.get()


def supported(region):
    # Larger supports use the general bounded-memory estimators.
    return region.pitch <= 27 and region.shape in (
        "circle",
        "square",
        "ellipse",
        "ambiguous",
    )


def _tile_medians(values, valid, sites, count):
    """Median of every column over each tile's valid sites. NaN for a tile without any."""
    # One row of lattice seeds per tile. Unused places sort to the end.
    grid = np.full((count, len(sites.region.centers), values.shape[1]), np.inf)
    grid[sites.tile_id[valid], sites.seed[valid]] = values[valid]
    grid.sort(axis=1)
    sizes = np.bincount(sites.tile_id[valid], minlength=count)
    tiles = np.arange(count)
    medians = 0.5 * (
        grid[tiles, np.maximum(sizes - 1, 0) // 2] + grid[tiles, sizes // 2]
    )
    medians[sizes == 0] = np.nan
    return medians


def tile_means(values, tile_id, count):
    """Mean of the finite values in each tile, and how many there were."""
    finite = np.isfinite(values)
    sites = np.bincount(tile_id[finite], minlength=count)
    total = np.bincount(tile_id[finite], weights=values[finite], minlength=count)
    return np.where(sites > 0, total / np.maximum(sites, 1), np.nan), sites


@lru_cache(maxsize=512)
def _filter_matrix(sigma, radius, support):
    # Separable Gaussian and native pixel integration, on the original 5x grid.
    # The aperture is surrounded by the full finite Gaussian and box support.
    s = sigma * 5
    r = int(4 * s + 0.5)
    weights = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    weights = np.convolve(weights / weights.sum(), np.ones(5) / 5)
    delta = np.arange(-radius, radius + 1)[:, None] - np.arange(-support, support + 1)
    idx = delta + r + 2
    return np.where(
        (idx >= 0) & (idx < len(weights)), weights[idx.clip(0, len(weights) - 1)], 0
    ).astype("f4")


class Library:
    """Blurred, pixel-integrated openings at nine sizes (81 for an ellipse).

    The sizes span the page's typical contour size, so one library serves every
    tile of the page. Only the blur step is repeated during the blur search.
    """

    def __init__(self, width, height, region, cp):
        self.cp, self.shape = cp, region.shape
        self.nh = 9 if self.shape == "ellipse" else 1
        lo_w, hi_w = max(0.6, width * 0.48), max(width * 1.72, 2.5)
        lo_h, hi_h = max(0.6, height * 0.48), max(height * 1.72, 2.5)
        self.bounds = np.array((lo_w, hi_w, lo_h, hi_h), dtype="f4")
        self.max_size = max(hi_w, hi_h if self.nh == 9 else height)
        self.support = int(np.ceil((0.72 * self.max_size + 0.2) * 5))
        axis = cp.arange(-self.support, self.support + 1, dtype=cp.float32) / 5
        y, x = cp.meshgrid(axis, axis, indexing="ij")
        u = np.cos(region.angle) * x + np.sin(region.angle) * y
        v = -np.sin(region.angle) * x + np.cos(region.angle) * y
        widths = np.repeat(np.linspace(lo_w, hi_w, 9), self.nh)
        wi = cp.asarray(widths, dtype=cp.float32)[:, None, None]
        if self.shape == "square":
            inside = cp.maximum(abs(u) - wi / 2, abs(v) - wi / 2)
        else:
            hi = wi
            if self.nh == 9:
                heights = np.tile(np.linspace(lo_h, hi_h, 9), 9)
                hi = cp.asarray(heights, dtype=cp.float32)[:, None, None]
            r = cp.hypot(2 * u / wi, 2 * v / hi)
            grad = cp.hypot(4 * u / wi**2, 4 * v / hi**2) / cp.maximum(r, 1e-8)
            inside = (r - 1) / cp.maximum(grad, 1 / cp.maximum(wi, hi))
        self.mask = cp.clip(0.5 - inside * 5, 0, 1).astype(cp.float32)

    def blur(self, sigma):
        cp = self.cp
        self.radius = int(np.ceil(0.72 * self.max_size + 4.5 * max(sigma) + 2))
        across, down = (
            cp.asarray(_filter_matrix(float(s), 5 * self.radius, self.support))
            for s in sigma
        )
        # Two matrix products apply the separable filters to all size planes.
        self.data = cp.ascontiguousarray((down @ self.mask) @ across.T)


class PageFits:
    """Model and differential fits for one page, with one blur found on that page.

    Every opening keeps its own contour seed, position and fitted size. The blur
    is the one setting they share: it is searched once, on up to ten openings
    from each tile, by how well the blurred templates reproduce their pixels.
    """

    def __init__(self, sites, contours, neighbors, cp):
        self.cp, self.sites = cp, sites
        region, side, count = sites.region, sites.tiles.shape[-1], len(sites.tiles)
        valid = np.isfinite(contours[:, 0])
        typical = np.median(contours[valid], axis=0)
        local = _tile_medians(contours, valid, sites, count)[sites.tile_id]
        local = np.where(np.isfinite(local), local, typical)
        seeds = contours.copy()
        seeds[~valid, :2] = local[~valid, :2]
        # Even an open contour can supply the contrast of a neighboring hole.
        faint = ~np.isfinite(seeds[:, 2]) | (seeds[:, 2] <= 0)
        seeds[faint, 2] = local[faint, 2]
        self.half = max(
            3, int(np.ceil(min(0.54 * region.pitch, 0.7 * max(typical[:2]) + 3)))
        )
        rounded = np.rint(sites.centers)
        inside = ((rounded >= self.half) & (rounded < side - self.half)).all(axis=1)
        self.use = np.flatnonzero(valid & inside)
        # The eight nearest lattice sites that lie in the same tile. The extra
        # column absorbs the padding index of lattices with fewer neighbors.
        index = np.full((count, len(region.centers) + 1), -1, dtype=np.int32)
        index[sites.tile_id, sites.seed] = np.arange(len(seeds))
        self.near = cp.asarray(index[sites.tile_id[:, None], neighbors[sites.seed]])
        self.seeds = cp.asarray(seeds, dtype="f4")
        self.ids = cp.asarray(self.use, dtype="i4")
        self.library = Library(typical[0], typical[1], region, cp)
        self.bounds = cp.asarray(self.library.bounds)
        sizes = np.bincount(sites.tile_id[self.use], minlength=count)
        first = np.cumsum(sizes) - sizes
        picks = [
            first[sizes == size][:, None]
            + np.linspace(0, size - 1, min(10, size), dtype=int)
            for size in np.unique(sizes[sizes > 0])
        ]
        if picks:
            picks = self.use[np.sort(np.concatenate(picks, axis=None))]
        self.picks = cp.asarray(picks, dtype="i4")
        self.sigma, self.usable = None, True

    def solve(self, ids, steps, profile):
        library, sites = self.library, self.sites
        threads = _power_two((2 * self.half + 1) ** 2)
        result = self.cp.empty((len(ids), 2), dtype="f4")
        _kernels(self.cp, max(512, threads)).get_function("fit_apertures")(
            (len(ids),),
            (threads,),
            (
                sites.tiles,
                np.int32(sites.tiles.shape[-1]),
                sites.positions,
                self.seeds,
                self.near,
                sites.owners,
                ids,
                np.int32(self.half),
                library.data,
                self.bounds,
                np.int32(library.radius),
                np.int32(library.nh),
                np.int32(steps),
                np.int32(profile),
                result,
            ),
            shared_mem=threads * 4,
        )
        return result.get()

    def prepare(self, psf):
        """Set the page's blur: the supplied one, or the best of a short search."""
        if psf is not None:
            self.sigma = tuple(psf)
            self.library.blur(self.sigma)
            return
        profile = []

        def trial(sigma):
            self.library.blur(sigma)
            rmse, width = np.median(self.solve(self.picks, 3, True), axis=0)
            profile.append((float(rmse), sigma, float(width)))

        for sigma in BLUR_CANDIDATES:
            trial(sigma)
        coarse = min(profile)[1]
        for ax, ay in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            trial(
                (
                    round(coarse[0] + BLUR_STEP * ax, 3),
                    round(coarse[1] + BLUR_STEP * ay, 3),
                )
            )
        vertex = _blur_vertex(coarse, profile)
        if vertex not in {sigma for _, sigma, _ in profile}:
            trial(vertex)
        best, _, weak, at_limit = _blur_result(profile, None)
        self.sigma, self.usable = best[1], not (weak or at_limit)
        self.library.blur(self.sigma)

    def measure(self, name):
        """One size per usable opening, NaN where the fit or the page's blur fails."""
        size = self.solve(self.ids, 6 if name == "model" else 1, False)[:, 1]
        return size if self.usable else np.full(len(size), np.nan, dtype=size.dtype)


def measure_group(tiles, template, shifts, psf, run, cp):
    """Measure the openings of one page's tiles that matched one reference.

    Returns {method: (mean per tile, valid openings per tile)} and, when the
    aperture fits ran, the blur they used as (sigma x, sigma y, usable).
    """
    sites = Sites.matched(tiles, template.region, shifts, cp)
    attempted, count = len(sites.centers), len(tiles)
    names = [m for m in ("model", "differential") if m in run.selected]
    per_tile, blur = {}, None
    if names or "contour" in run.selected:
        with run.time("contour"):
            contours = sites.contours(cp)
        if "contour" in run.selected:
            widths = contours[:, 0]
            per_tile["contour"] = tile_means(widths, sites.tile_id, count)
            run.record("contour", widths[np.isfinite(widths)], attempted)
    if "flux" in run.selected:
        with run.time("flux"):
            signal = sites.flux(cp)
        per_tile["flux"] = tile_means(signal, sites.tile_id, count)
        run.record("flux", signal[np.isfinite(signal)], attempted)
    for name in names:
        run.record(name, [], attempted)
    if (
        not names
        or template.region.shape == "ambiguous"
        or not np.isfinite(contours[:, 0]).any()
    ):
        return per_tile, blur
    with run.time("fit_setup"):
        fits = PageFits(sites, contours, template.neighbors, cp)
        if len(fits.use):
            fits.prepare(psf)
            blur = (*fits.sigma, fits.usable)
    if not len(fits.use):
        return per_tile, blur
    for name in names:
        with run.time(name):
            values = fits.measure(name)
        per_tile[name] = tile_means(values, sites.tile_id[fits.use], count)
        run.record(name, values[np.isfinite(values)], 0)
    return per_tile, blur
