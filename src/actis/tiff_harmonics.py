"""Lattice-harmonic CD for many reference-matched tiles at once.

The tests and the fit are those of harmonics.measure_orders, applied to every
tile of a page together. The nonlinear fit runs on CUDA, one thread per start.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

SHAPES = {"circle": 0, "square": 1, "ellipse": 2}
STARTS = (0.23, 0.46, 0.74)
LOW = {False: (0.075, 0.004, 0.004, -2.0), True: (0.075, 0.075, 0.004, 0.004, -2.0)}
HIGH = {False: (1.20, 0.35, 0.35, 4.0), True: (1.20, 1.20, 0.35, 0.35, 4.0)}
TOLERANCE, MAX_EVALUATIONS = 1e-7, 100


@lru_cache(maxsize=1)
def _kernel(cp):
    source = Path(__file__).with_suffix(".cu").read_text(encoding="ascii")
    return cp.RawKernel(source, "fit_harmonics")


def _fit(target, use, error, freq, region, cp):
    """Best of the three starts for each tile: parameters, cost, state, Gauss-Newton matrix."""
    tiles, orders = target.shape
    jobs = tiles * len(STARTS)
    solution = cp.empty((jobs, 6))
    cost = cp.empty(jobs)
    state = cp.empty(jobs, dtype=cp.int32)
    gram = cp.empty((jobs, 6, 6))
    _kernel(cp)(
        ((jobs + 63) // 64,),
        (64,),
        (
            cp.asarray(freq, dtype=cp.float64),
            cp.asarray(target, dtype=cp.float64),
            cp.asarray(use, dtype=cp.uint8),
            cp.asarray(error, dtype=cp.float64),
            cp.asarray(STARTS, dtype=cp.float64),
            np.int32(orders),
            np.int32(tiles),
            np.int32(len(STARTS)),
            np.float64(region.pitch),
            np.float64(np.cos(region.angle)),
            np.float64(np.sin(region.angle)),
            np.int32(SHAPES[region.shape]),
            np.int32(MAX_EVALUATIONS),
            np.float64(TOLERANCE),
            solution,
            cost,
            state,
            gram,
        ),
    )
    cost = cost.get().reshape(tiles, -1)
    # The first of equal costs wins, as in the sequential reference.
    best = np.where(np.isfinite(cost), cost, np.inf).argmin(axis=1)
    pick = np.arange(tiles) * len(STARTS) + best
    return (
        solution.get()[pick],
        cost[np.arange(tiles), best],
        state.get()[pick],
        gram.get()[pick],
    )


def matched_cds(amplitude, noise, freq, region, cp):
    """One CD per tile from its anchored complex orders. NaN where a test fails."""
    cds = np.full(len(amplitude), np.nan)
    if not len(amplitude):
        return cds
    pitch = region.pitch
    radii = np.linalg.norm(freq, axis=1)
    size = np.abs(amplitude)
    fundamental = (radii <= 1.26 / pitch) & (size > 5 * noise[:, None])
    # A 2-D displacement needs two fundamental orders in different directions.
    first = freq[fundamental.argmax(axis=1)]
    cross = first[:, :1] * freq[:, 1] - first[:, 1:] * freq[:, 0]
    apart = np.abs(cross) > 1e-9 * np.linalg.norm(first, axis=1)[:, None] * radii
    resolved = (fundamental & apart).any(axis=1)
    weight = np.where(fundamental, size**2, 0.0)
    phase = -np.angle(amplitude) / (2 * np.pi)
    fx, fy = freq.T
    a, b, c = weight @ (fx * fx), weight @ (fx * fy), weight @ (fy * fy)
    rx, ry = (weight * phase) @ fx, (weight * phase) @ fy
    det = np.where(resolved, a * c - b * b, 1.0)
    shift = np.stack(((c * rx - b * ry) / det, (a * ry - b * rx) / det), axis=1)
    turn = np.exp(2j * np.pi * (shift @ freq.T))
    # Localization by phase requires residuals smaller than half a period and
    # a common displacement across the visible fundamental orders.
    error = np.angle(amplitude * turn)
    spread = (weight * error**2).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-300)
    resolved &= ~(np.linalg.norm(shift, axis=1) > 0.35 * pitch) & ~(spread > 0.18**2)
    if region.shape not in SHAPES:
        return cds
    ellipse = region.shape == "ellipse"
    signed = (amplitude * turn).real
    good = size > np.maximum(4 * noise, 0.008 * size.max(axis=1))[:, None]
    reach = np.where(good, radii, 0.0).max(axis=1)
    include = radii < np.minimum(0.46, 1.22 * reach)[:, None]
    fit = (
        resolved
        & (good.sum(axis=1) >= (7 if ellipse else 5))
        & (reach >= 1.8 / pitch)
        & (include.sum(axis=1) >= 6)
    )
    if not fit.any():
        return cds
    signed, include, floor = signed[fit], include[fit], noise[fit]
    scale = np.maximum(np.abs(np.where(include, signed, 0.0)).max(axis=1), 1e-15)
    floor = np.maximum(floor / scale, 0.008)
    z, cost, state, gram = _fit(
        signed / scale[:, None], include, floor, freq, region, cp
    )
    dim = 5 if ellipse else 4
    z, gram = z[:, :dim], gram[:, :dim, :dim]
    n_cd = dim - 3
    low, high = np.array(LOW[ellipse]), np.array(HIGH[ellipse])
    at_bound = (z - low <= TOLERANCE * np.maximum(1, np.abs(low))) | (
        high - z <= TOLERANCE * np.maximum(1, np.abs(high))
    )
    # Conditioning of the column-normalized Jacobian, from its Gram matrix.
    norm = np.maximum(np.sqrt(np.einsum("tii->ti", gram)), 1e-14)
    eig = np.linalg.eigvalsh(gram / (norm[:, :, None] * norm[:, None, :]))
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(eig[:, 0] > 0, np.sqrt(eig[:, -1] / eig[:, 0]), np.inf)
    # What is left of the size columns once blur and gain have explained all they can.
    nuisance = np.linalg.pinv(gram[:, n_cd:, n_cd:], hermitian=True)
    mixed = gram[:, :n_cd, n_cd:]
    left = gram[:, :n_cd, :n_cd] - mixed @ nuisance @ mixed.transpose(0, 2, 1)
    sensitivity = np.sqrt(np.maximum(np.linalg.eigvalsh(left)[:, 0], 0.0))
    rms = floor * np.sqrt(2 * cost / include.sum(axis=1))
    ok = (
        (state != 0)
        & ~at_bound[:, :n_cd].any(axis=1)
        & ~(cond > 1e4)
        & ~(sensitivity < 7.0)
        & ~(rms > 0.095)
    )
    wide = z[:, 0] * pitch
    tall = z[:, 1] * pitch if ellipse else wide
    cds[np.flatnonzero(fit)[ok]] = np.maximum(wide, tall)[ok]
    return cds
