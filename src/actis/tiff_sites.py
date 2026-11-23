"""Batched native-site measurements for independent, reference-matched TIFF tiles."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .estimators import (
    BLUR_CANDIDATES,
    BLUR_STEP,
    N_RAYS,
    _blur_result,
    _blur_vertex,
    _representative,
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


@dataclass
class FitTile:
    index: int
    median: np.ndarray
    half: int
    centers: np.ndarray
    seeds: np.ndarray
    neighbors: np.ndarray
    use: np.ndarray


def _fit_tiles(sites, contours):
    side = sites.tiles.shape[-1]
    groups = {}
    ends = np.searchsorted(sites.tile_id, np.arange(len(sites.tiles) + 1))
    for i, (lo, hi) in enumerate(pairwise(ends)):
        seeds = contours[lo:hi].copy()
        valid = np.isfinite(seeds[:, 0])
        if not valid.any():
            continue
        median = np.median(seeds[valid], axis=0)
        seeds[~valid, :2] = median[:2]
        # Even an open contour can supply the contrast of a neighboring hole.
        contrast = seeds[:, 2]
        contrast[~np.isfinite(contrast) | (contrast <= 0)] = median[2]
        half = max(
            3, int(np.ceil(min(0.54 * sites.region.pitch, 0.7 * max(median[:2]) + 3)))
        )
        centers = sites.centers[lo:hi]
        rounded = np.rint(centers)
        ids = np.flatnonzero(
            valid & ((rounded >= half) & (rounded < side - half)).all(axis=1)
        )
        if len(ids):
            near = (
                cKDTree(centers)
                .query(centers, k=min(9, len(centers)))[1]
                .reshape(len(centers), -1)[:, 1:]
            )
            groups.setdefault(half, []).append(
                FitTile(i, median[:2], half, centers, seeds, near, ids)
            )
    return groups


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


class TileLibraries:
    """Each tile retains its own contour-derived aperture size interpolation grid."""

    def __init__(self, tiles, region, cp):
        self.cp, self.shape = cp, region.shape
        self.nh = 9 if self.shape == "ellipse" else 1
        w, h = np.array([t.median for t in tiles]).T
        self.lo_w, self.hi_w = np.maximum(0.6, w * 0.48), np.maximum(w * 1.72, 2.5)
        self.lo_h, self.hi_h = np.maximum(0.6, h * 0.48), np.maximum(h * 1.72, 2.5)
        self.max_size = np.maximum(self.hi_w, self.hi_h if self.nh == 9 else h)
        self.support = int(np.ceil((0.72 * max(self.max_size) + 0.2) * 5))
        axis = cp.arange(-self.support, self.support + 1, dtype=cp.float32) / 5
        y, x = cp.meshgrid(axis, axis, indexing="ij")
        u = np.cos(region.angle) * x + np.sin(region.angle) * y
        v = -np.sin(region.angle) * x + np.cos(region.angle) * y
        sizes = np.linspace(self.lo_w, self.hi_w, 9).T
        wi = cp.asarray(np.repeat(sizes, self.nh, axis=1), dtype=cp.float32)[
            :, :, None, None
        ]
        if self.shape == "square":
            inside = cp.maximum(abs(u) - wi / 2, abs(v) - wi / 2)
        else:
            hi = (
                wi
                if self.nh == 1
                else cp.asarray(
                    np.tile(np.linspace(self.lo_h, self.hi_h, 9).T, (1, 9)),
                    dtype=cp.float32,
                )[:, :, None, None]
            )
            r = cp.hypot(2 * u / wi, 2 * v / hi)
            grad = cp.hypot(4 * u / wi**2, 4 * v / hi**2) / cp.maximum(r, 1e-8)
            inside = (r - 1) / cp.maximum(grad, 1 / cp.maximum(wi, hi))
        self.mask = cp.clip(0.5 - inside * 5, 0, 1).astype(cp.float32)

    def filter(self, sigmas):
        cp = self.cp
        self.radii = np.ceil(
            0.72 * self.max_size + 4.5 * np.max(sigmas, axis=1) + 2
        ).astype(int)
        self.radius = int(max(self.radii))
        sx, sy = np.asarray(sigmas).T

        def matrix(s):
            return cp.asarray(
                np.stack(
                    [_filter_matrix(float(v), 5 * self.radius, self.support) for v in s]
                )
            )

        # Two matrix products apply the separable filters to all size planes.
        self.data = cp.ascontiguousarray(
            (matrix(sy)[:, None] @ self.mask) @ matrix(sx)[:, None].swapaxes(-1, -2)
        )


class TileFits:
    """Profile blur per tile and solve its native stamps in parallel on CUDA."""

    def __init__(self, pixels, group, region, cp):
        self.cp, self.group = cp, group
        self.tables = TileLibraries(group, region, cp)
        self.half = group[0].half
        centers = np.concatenate([t.centers for t in group])
        self.centers = cp.asarray(centers)
        self.seeds = cp.asarray(np.concatenate([t.seeds for t in group]), dtype="f4")
        starts = np.cumsum([0] + [len(t.centers) for t in group[:-1]])
        near = np.full((len(centers), 8), -1, dtype="i4")
        for t, start in zip(group, starts):
            near[start : start + len(t.centers), : t.neighbors.shape[1]] = (
                t.neighbors + start
            )
        self.near = cp.asarray(near)
        self.ids = [t.use + s for t, s in zip(group, starts)]
        owner = np.repeat(np.arange(len(group)), [len(t.centers) for t in group])
        self.owners = cp.asarray(owner, dtype="i4")
        self.full_ids = cp.asarray(np.concatenate(self.ids), dtype="i4")
        self.pixels = pixels[cp.asarray([t.index for t in group])]
        self.bounds = cp.asarray(
            np.stack(
                (
                    self.tables.lo_w,
                    self.tables.hi_w,
                    self.tables.lo_h,
                    self.tables.hi_h,
                ),
                axis=1,
            ),
            dtype="f4",
        )
        picks = [_representative(ids) for ids in self.ids]
        self.picks = cp.asarray(np.concatenate(picks), dtype="i4")
        self.ends = np.cumsum([0] + list(map(len, picks)))
        self.profiles = [[] for t in group]

    def solve(self, ids, steps, profile):
        tables = self.tables
        threads = _power_two((2 * self.half + 1) ** 2)
        result = self.cp.empty((len(ids), 2), dtype="f4")
        _kernels(self.cp, max(512, threads)).get_function("fit_apertures")(
            (len(ids),),
            (threads,),
            (
                self.pixels,
                np.int32(self.pixels.shape[-1]),
                self.centers,
                self.seeds,
                self.near,
                self.owners,
                ids,
                np.int32(self.half),
                tables.data,
                self.bounds,
                self.cp.asarray(tables.radii, dtype="i4"),
                np.int32(tables.radius),
                np.int32(tables.nh),
                np.int32(steps),
                np.int32(profile),
                result,
            ),
            shared_mem=threads * 4,
        )
        return result.get()

    def profile(self, sigmas, include=None):
        self.tables.filter(sigmas)
        result = self.solve(self.picks, 3, True)
        if np.all(np.diff(self.ends) == self.ends[1]):
            medians = np.median(result.reshape(len(self.group), -1, 2), axis=1)
        else:
            medians = [
                np.median(result[a:b], axis=0) for a, b in pairwise(self.ends)
            ]
        for i, (rmse, width) in enumerate(medians):
            if include is None or include[i]:
                self.profiles[i].append((float(rmse), tuple(sigmas[i]), float(width)))

    def prepare(self, psf):
        nt = len(self.group)
        if psf is not None:
            self.tables.filter(np.tile(psf, (nt, 1)))
            self.exclude = np.zeros(nt, dtype=bool)
            return
        for sigma in BLUR_CANDIDATES:
            self.profile(np.tile(sigma, (nt, 1)))
        coarse = np.array([min(p)[1] for p in self.profiles])
        for direction in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.profile(np.round(coarse + BLUR_STEP * np.array(direction), 3))
        extra = [
            _blur_vertex(tuple(base), profile)
            for base, profile in zip(coarse, self.profiles)
        ]
        include = [s not in {v[1] for v in p} for s, p in zip(extra, self.profiles)]
        if any(include):
            self.profile(extra, include)
        results = [_blur_result(profile, psf) for profile in self.profiles]
        self.tables.filter(np.array([best[1] for best, _, _, _ in results]))
        self.exclude = np.array([weak or at_limit for _, _, weak, at_limit in results])

    def measure(self, name):
        cd = self.solve(self.full_ids, 6 if name == "model" else 1, False)[:, 1]
        valid = np.isfinite(cd) & ~np.repeat(self.exclude, list(map(len, self.ids)))
        return cd[valid]


def measure_group(tiles, template, shifts, psf, run, cp):
    sites = Sites.matched(tiles, template.region, shifts, cp)
    attempted = len(sites.centers)
    names = [m for m in ("model", "differential") if m in run.selected]
    if names or "contour" in run.selected:
        with run.time("contour"):
            contours = sites.contours(cp)
        if "contour" in run.selected:
            widths = contours[:, 0]
            run.record("contour", widths[np.isfinite(widths)], attempted)
    if "flux" in run.selected:
        with run.time("flux"):
            signal = sites.flux(cp)
        run.record("flux", signal[np.isfinite(signal)], attempted)
    if not names:
        return
    for name in names:
        run.record(name, [], attempted)
    if template.region.shape == "ambiguous":
        return
    with run.time("fit_setup"):
        groups = _fit_tiles(sites, contours)
    for group in groups.values():
        # Bound the largest possible PSF library plus its filter workspace.
        planes = 81 if template.region.shape == "ellipse" else 9
        max_size = max(max(t.median) for t in group) * 1.72
        max_sigma = max(psf) if psf is not None else 1.91
        side = 10 * int(np.ceil(0.72 * max(max_size, 2.5) + 4.5 * max_sigma + 2)) + 1
        limit = max(1, min(64, 2**25 // (side * side * planes)))
        for start in range(0, len(group), limit):
            with run.time("fit_setup"):
                fits = TileFits(
                    sites.tiles, group[start : start + limit], template.region, cp
                )
                fits.prepare(psf)
            for name in names:
                with run.time(name):
                    values = fits.measure(name)
                run.record(name, values, 0)
            del fits
