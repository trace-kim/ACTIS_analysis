"""Precision on matched mask locations. The first array axis is acquisition."""

from itertools import combinations

import numpy as np


def precision(values):
    """Return sample 3σ and 300s/|mean| on sites/bins present in every repeat.

    Values must already be aggregated within each site or bin in each image.
    Missing or infinite measurements invalidate the comparison for that location.
    The count array retains the number of finite observations for diagnostics.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or len(values) < 2:
        raise ValueError("precision requires at least two acquisitions")
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    complete = count == len(values)
    safe = np.where(complete, values, 0.0)
    mean = np.where(complete, safe.mean(axis=0), np.nan)
    sigma3 = np.where(complete, 3.0 * safe.std(axis=0, ddof=1), np.nan)
    percent = np.full(mean.shape, np.nan, dtype=np.float64)
    np.divide(100.0 * sigma3, np.abs(mean), out=percent, where=complete & (mean != 0))
    return {"mean": mean, "sigma3": sigma3, "percent": percent, "count": count}


def triplet_scores(values):
    """Enumerate three-acquisition scores on fixed, all-run common support.

    Each row contains the zero-based acquisition indices, arithmetic mean of
    binwise 3σ, arithmetic mean of defined relative scores, and bin count.
    Overlapping subsets describe sensitivity to choosing three acquisitions.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("triplet scores require an acquisition axis")
    if len(values) < 3:
        return []
    flat = values.reshape((len(values), -1))
    common = flat[:, np.isfinite(flat).all(axis=0)]
    rows = []
    for repeats in combinations(range(len(values)), 3):
        stats = precision(common[list(repeats)])
        relative = stats["percent"][np.isfinite(stats["percent"])]
        rows.append(
            {
                "repeats": repeats,
                "sigma3": float(stats["sigma3"].mean()) if common.shape[1] else np.nan,
                "percent": float(relative.mean()) if relative.size else np.nan,
                "bins": common.shape[1],
            }
        )
    return rows


def tile_indices(xy_px, bounds_xyxy_px, pixel_nm=21.0, tile_um=5.0):
    """Assign points to complete, nonoverlapping tiles in a common mask frame.

    Bounds are (xmin, ymin, xmax, ymax) in pixels, with exclusive upper edges.
    The origin is the supplied lower edge. IDs increase in row-major order.
    A point outside full tiles receives -1. Return IDs and (rows, columns).
    """
    xy = np.asarray(xy_px, dtype=float)
    bounds = np.asarray(bounds_xyxy_px, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_px must have shape (n, 2)")
    if (
        bounds.shape != (4,)
        or not np.isfinite(bounds).all()
        or np.any(bounds[2:] < bounds[:2])
    ):
        raise ValueError("invalid tile bounds")
    if not (
        np.isfinite(pixel_nm) and np.isfinite(tile_um) and pixel_nm > 0 and tile_um > 0
    ):
        raise ValueError("pixel_nm and tile_um must be finite and positive")
    tile_px = 1000.0 * tile_um / pixel_nm
    nx, ny = np.floor((bounds[2:] - bounds[:2]) / tile_px + 1e-12).astype(int)
    ids = np.full(len(xy), -1, dtype=np.int64)
    inside = (
        np.isfinite(xy).all(axis=1)
        & (xy >= bounds[:2]).all(axis=1)
        & (xy < bounds[2:]).all(axis=1)
    )
    points = np.flatnonzero(inside)
    columns, rows = np.floor((xy[points] - bounds[:2]) / tile_px).astype(int).T
    full = (columns < nx) & (rows < ny)
    ids[points[full]] = rows[full] * nx + columns[full]
    return ids, (int(ny), int(nx))


def response_fit(truth_nm, measured):
    """Fit measured = intercept + slope × truth and report error vs identity.

    ``mean_error`` is mean(measured - truth) and ``rmse`` is the RMS of that
    difference, meaningful for commensurate inputs. Slope and intercept are
    undefined when fewer than two distinct truth values remain.
    """
    truth = np.asarray(truth_nm, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if truth.shape != measured.shape:
        raise ValueError("truth and measured must have identical shapes")
    valid = np.isfinite(truth) & np.isfinite(measured)
    truth, measured = truth[valid], measured[valid]
    slope, intercept, mean_error, rmse = [np.nan] * 4
    if truth.size:
        error = measured - truth
        mean_error = float(np.mean(error))
        rmse = float(np.sqrt(np.mean(error**2)))
    if truth.size >= 2 and np.ptp(truth) > 0:
        centered = truth - truth.mean()
        slope = float(centered @ (measured - measured.mean()) / (centered @ centered))
        intercept = float(measured.mean() - slope * truth.mean())
    return {
        "slope": slope,
        "intercept": intercept,
        "mean_error": mean_error,
        "rmse": rmse,
        "count": int(truth.size),
    }
