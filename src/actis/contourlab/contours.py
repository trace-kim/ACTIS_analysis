"""Contours of one image as packed arrays, and their geometry.

A page of dense contact holes has tens of thousands of contours of a few dozen
points each. All of them share one (P, 2) point array plus the index at which
each begins, so that tracing, moving and measuring them are array operations.

Coordinates are (x, y) = (column, row) with pixel centres at integers and y
pointing down. Direction convention: walking along a contour, the outward
normal is (ty, -tx) of the unit tangent (tx, ty), and the region the contour
bounds lies on the other side. A closed contour around its region therefore
has positive signed area, and the boundary of a hole in a region negative.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.signal import savgol_coeffs

# what measure() gives of every contour, besides its centre and its number of points
MEASURES = ("area", "diameter", "width", "height", "major", "minor", "angle", "length")


@dataclass(frozen=True)
class Contours:
    xy: np.ndarray  # (P, 2) float64, the points of every contour in turn
    start: np.ndarray  # (N + 1,) index of each contour's first point; start[N] = P
    closed: np.ndarray  # (N,) bool: the last point joins the first
    left_out: dict = field(default_factory=dict)  # candidates dropped, per reason

    def __len__(self):
        return len(self.closed)

    @cached_property
    def counts(self):
        return np.diff(self.start)

    @cached_property
    def owner(self):
        """Contour number of every point."""
        return np.repeat(np.arange(len(self)), self.counts)

    @cached_property
    def neighbours(self):
        """One step forward and one step back along each contour: for each, the
        index of the point reached, and whether there is one (the ends of an
        open contour have none on one side, and stay where they are)."""
        first, last = self.start[:-1], self.start[1:] - 1
        ahead = np.arange(1, len(self.xy) + 1)
        behind = np.arange(-1, len(self.xy) - 1)
        ahead[last] = np.where(self.closed, first, last)
        behind[first] = np.where(self.closed, last, first)
        joined_ahead = np.ones(len(self.xy), bool)
        joined_behind = np.ones(len(self.xy), bool)
        joined_ahead[last] = joined_behind[first] = self.closed
        return (ahead, joined_ahead), (behind, joined_behind)

    def split(self):
        """The contours as a list of (M, 2) arrays."""
        return np.split(self.xy, self.start[1:-1])

    def moved(self, xy):
        """The same contours with their points at new places."""
        return Contours(xy, self.start, self.closed, self.left_out)

    def select(self, keep, **dropped):
        """The contours where keep (N,) is True; dropped counts add to left_out."""
        keep = np.asarray(keep, bool)
        counts = self.counts
        left_out = dict(self.left_out)
        for reason, count in dropped.items():
            if count:
                left_out[reason] = left_out.get(reason, 0) + int(count)
        return Contours(
            self.xy[np.repeat(keep, counts)],
            np.concatenate([[0], np.cumsum(counts[keep])]).astype(np.int64),
            self.closed[keep],
            left_out,
        )


def empty(left_out=None):
    return Contours(
        np.empty((0, 2)), np.zeros(1, np.int64), np.zeros(0, bool), left_out or {}
    )


def from_list(points, closed=True, left_out=None):
    """Contours from a list of (M, 2) arrays; closed is one bool or one per contour."""
    points = [np.asarray(p, float).reshape(-1, 2) for p in points]
    if not points:
        return empty(left_out)
    start = np.concatenate([[0], np.cumsum([len(p) for p in points])])
    flags = np.broadcast_to(np.asarray(closed, bool), (len(points),)).copy()
    return Contours(
        np.concatenate(points), start.astype(np.int64), flags, left_out or {}
    )


def step_index(contours, k):
    """(index, valid): for every point, the point k steps on along its contour.

    A closed contour wraps round. An open one stops at its ends: the index is
    clamped there and valid is False.
    """
    step, joined = contours.neighbours[0 if k > 0 else 1]
    index = np.arange(len(contours.xy))
    valid = np.ones(len(index), bool)
    for _ in range(abs(k)):
        valid &= joined[index]
        index = step[index]
    return index, valid


def lengths(contours):
    """(N,) the path length of each contour."""
    following, joined = step_index(contours, 1)
    segment = contours.xy[following] - contours.xy
    segment = np.where(joined, np.hypot(segment[:, 0], segment[:, 1]), 0.0)
    return np.add.reduceat(segment, contours.start[:-1])


def areas(contours):
    """(N,) the signed area of each contour (measure()), NaN for an open one."""
    xy = contours.xy
    origin = xy[contours.start[:-1]][contours.owner]  # keeps the products small
    x, y = xy[:, 0] - origin[:, 0], xy[:, 1] - origin[:, 1]
    ring = np.arange(1, len(xy) + 1)
    ring[contours.start[1:] - 1] = contours.start[:-1]
    cross = x * y[ring] - x[ring] * y
    area = np.add.reduceat(cross, contours.start[:-1]) / 2
    return np.where(contours.closed & (np.abs(area) > 1e-12), area, np.nan)


def normals(contours, span=1):
    """(P, 2) unit outward normals from the tangent across +-span points.

    Points where the two coincide have no direction and get (0, 0).
    """
    ahead, _ = step_index(contours, span)
    behind, _ = step_index(contours, -span)
    tangent = contours.xy[ahead] - contours.xy[behind]
    length = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 0)
    return np.column_stack([tangent[:, 1], -tangent[:, 0]])


def group_median(values, start):
    """Median of the finite values in each group start[i]:start[i + 1].

    A group without a finite value gives NaN.
    """
    counts = np.diff(start)
    if not len(counts) or not len(values):
        return np.full(len(counts), np.nan)
    owner = np.repeat(np.arange(len(counts)), counts)
    finite = np.isfinite(values)
    order = np.lexsort((np.where(finite, values, np.inf), owner))
    ranked = values[order]
    found = np.add.reduceat(finite.astype(np.int64), start[:-1])
    low = start[:-1] + np.maximum(found - 1, 0) // 2
    high = np.minimum(start[:-1] + found // 2, len(values) - 1)
    return np.where(found > 0, (ranked[low] + ranked[high]) / 2, np.nan)


def measure(contours):
    """Per-contour geometry as a dict of (N,) arrays (centre: (N, 2)).

    area is signed (negative: the boundary of a hole) and NaN for an open
    contour, as are diameter (of the circle of equal area), major and minor
    (four standard deviations of the enclosed area along its principal axes:
    the axes of an ellipse) and angle (of the major axis, radians, from x
    towards y). centre is the area centroid, or the mean point of an open
    contour. width and height are the extents along x and y; length is the
    path length.
    """
    n = len(contours)
    if n == 0:
        zero = np.empty(0)
        names = MEASURES
        return dict(
            {name: zero for name in names},
            centre=np.empty((0, 2)),
            points=np.empty(0, np.int64),
        )
    first, counts, owner = contours.start[:-1], contours.counts, contours.owner
    xy = contours.xy

    def total(values):
        return np.add.reduceat(values, first)

    following, joined = step_index(contours, 1)
    segment = np.where(joined, np.linalg.norm(xy[following] - xy, axis=1), 0.0)
    mean = np.add.reduceat(xy, first, axis=0) / counts[:, None]
    p = xy - mean[owner]  # moments about a nearby origin stay well conditioned
    # The polygon sums take every contour as closed: its first point follows its last.
    ring = np.arange(len(xy)) + 1
    ring[contours.start[1:] - 1] = first
    q = p[ring]
    x, y, xx, yy = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
    cross = x * yy - xx * y
    area = total(cross) / 2
    safe = np.where(np.abs(area) > 1e-12, area, 1.0)
    cx = total((x + xx) * cross) / (6 * safe)
    cy = total((y + yy) * cross) / (6 * safe)
    vx = total((x * x + x * xx + xx * xx) * cross) / (12 * safe) - cx * cx
    vy = total((y * y + y * yy + yy * yy) * cross) / (12 * safe) - cy * cy
    vxy = total((2 * x * y + x * yy + xx * y + 2 * xx * yy) * cross) / (24 * safe)
    vxy -= cx * cy
    spread = np.hypot((vx - vy) / 2, vxy)
    middle = (vx + vy) / 2
    whole = contours.closed & (np.abs(area) > 1e-12)
    nan = np.where(whole, 1.0, np.nan)
    low = np.minimum.reduceat(xy, first, axis=0)
    high = np.maximum.reduceat(xy, first, axis=0)
    return dict(
        area=area * nan,
        centre=np.where(whole[:, None], mean + np.column_stack([cx, cy]), mean),
        diameter=2 * np.sqrt(np.abs(area) / np.pi) * nan,
        width=high[:, 0] - low[:, 0],
        height=high[:, 1] - low[:, 1],
        major=4 * np.sqrt(np.maximum(middle + spread, 0)) * nan,
        minor=4 * np.sqrt(np.maximum(middle - spread, 0)) * nan,
        angle=np.arctan2(2 * vxy, vx - vy) / 2 * nan,
        length=total(segment),
        points=counts,
    )


def _respaced(contours, count):
    """Contour i with count[i] points at equal steps along it.

    The first point of a closed contour stays; an open one keeps both ends.
    """
    xy, first, owner = contours.xy, contours.start[:-1], contours.owner
    closed = contours.closed
    following, joined = step_index(contours, 1)
    segment = np.where(joined, np.linalg.norm(xy[following] - xy, axis=1), 0.0)
    length = np.add.reduceat(segment, first)
    along = np.cumsum(segment) - segment
    along -= along[first][owner]  # distance of every point along its own contour
    gap = float(length.max()) + 1.0  # keeps the contours apart on one axis
    numbers = np.arange(len(contours))
    # a closed contour returns to its first point at its full length
    position = np.concatenate([along + owner * gap, (length + numbers * gap)[closed]])
    values = np.concatenate([xy, xy[first][closed]])
    order = np.argsort(position, kind="stable")
    position, values = position[order], values[order]
    start = np.concatenate([[0], np.cumsum(count)]).astype(np.int64)
    new_owner = np.repeat(numbers, count)
    index = np.arange(start[-1]) - start[:-1][new_owner]
    parts = np.where(closed, count, np.maximum(count - 1, 1))[new_owner]
    target = index * (length[new_owner] / parts) + new_owner * gap
    moved = np.column_stack([np.interp(target, position, values[:, k]) for k in (0, 1)])
    return Contours(moved, start, closed, contours.left_out)


def resample(contours, spacing, min_points=8):
    """Each contour with its points at equal steps of about `spacing` along it.

    A closed contour gets at least min_points points.
    """
    if len(contours) == 0:
        return contours
    steps = np.maximum(np.rint(lengths(contours) / spacing).astype(np.int64), 1)
    count = np.where(contours.closed, np.maximum(steps, min_points), steps + 1)
    return _respaced(contours, count)


def smooth(contours, window):
    """The contours with point-to-point scatter removed (Savitzky-Golay).

    Every point is put on a cubic fitted to the `window` points around it along
    its contour, which follows a curved edge without shrinking a closed
    contour. Points nearer than half a window to the end of an open contour,
    and contours of fewer points than the window, stay as they are.
    """
    window = max(5, int(window) | 1)
    half = window // 2
    weights = savgol_coeffs(window, 3)
    total = np.zeros_like(contours.xy)
    usable = contours.counts[contours.owner] >= window
    for k in range(-half, half + 1):
        index, valid = step_index(contours, k)
        total += weights[half + k] * contours.xy[index]
        usable &= valid
    return contours.moved(np.where(usable[:, None], total, contours.xy))


def fourier_shape(contours, harmonics, points=64):
    """Closed contours reduced to a simple shape of the same area.

    harmonics 1: the ellipse with the contour's centroid, the orientation and
    axis ratio of its second moments, and its area. More: the contour as a
    periodic curve of `points` equally spaced points, of which only the mean
    position and the first `harmonics` Fourier harmonics are kept (a rounded
    polygon), scaled about its centroid back to the contour's area, which the
    harmonics left out would otherwise take with them. Open contours stay as
    they are.
    """
    whole = contours.closed
    if not whole.any():
        return contours
    rings = contours.select(whole)
    wanted = np.abs(measure(rings)["area"])  # of the contours as they are
    rings = _respaced(rings, np.full(len(rings), points, np.int64))
    before = measure(rings)
    if harmonics <= 1:
        turn = np.arange(points) * (2 * np.pi / points)
        turn = turn[None, :] * np.sign(before["area"])[:, None]  # same way round
        u = before["major"][:, None] / 2 * np.cos(turn)
        v = before["minor"][:, None] / 2 * np.sin(turn)
        cos, sin = np.cos(before["angle"])[:, None], np.sin(before["angle"])[:, None]
        rebuilt = np.stack([u * cos - v * sin, u * sin + v * cos], axis=2)
        rebuilt = rebuilt.reshape(-1, 2) + np.repeat(before["centre"], points, axis=0)
    else:
        curve = (rings.xy[:, 0] + 1j * rings.xy[:, 1]).reshape(-1, points)
        spectrum = np.fft.fft(curve, axis=1)
        order = np.abs(np.fft.fftfreq(points, 1 / points))
        spectrum[:, order > harmonics] = 0
        curve = np.fft.ifft(spectrum, axis=1)
        rebuilt = np.column_stack([curve.real.ravel(), curve.imag.ravel()])
    after = measure(rings.moved(rebuilt))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(wanted / np.abs(after["area"]))
    scale = np.repeat(np.where(np.isfinite(scale), scale, 1.0), points)[:, None]
    centre = np.repeat(after["centre"], points, axis=0)
    rebuilt = iter(np.split(centre + scale * (rebuilt - centre), len(rings)))
    merged = [next(rebuilt) if w else p for p, w in zip(contours.split(), whole)]
    return from_list(merged, whole, contours.left_out)


def without_isolated(shift, contours, neighbours=3, tolerance=0.3):
    """Shifts (P,) along the normals with the isolated ones replaced.

    A shift that is NaN, or further from the median of its `neighbours` on
    each side than 4 robust sigma (of its contour) + tolerance, takes that
    median (0 if its neighbours have none). A real deviation of an edge spans
    several neighbouring points and is kept.
    """
    if not len(shift):
        return shift
    around = []
    for k in range(-neighbours, neighbours + 1):
        if k:
            index, valid = step_index(contours, k)
            around.append(np.where(valid, shift[index], np.nan))
    around = np.sort(np.column_stack(around), axis=1)  # NaN sort last
    found = np.isfinite(around).sum(axis=1, keepdims=True)
    low = np.take_along_axis(around, np.maximum(found - 1, 0) // 2, axis=1)
    high = np.take_along_axis(
        around, np.minimum(found // 2, around.shape[1] - 1), axis=1
    )
    median = np.where(found[:, 0] > 0, (low[:, 0] + high[:, 0]) / 2, np.nan)
    sigma = 1.4826 * group_median(np.abs(shift - median), contours.start)
    sigma = np.nan_to_num(sigma)[contours.owner]
    isolated = ~np.isfinite(shift) | (np.abs(shift - median) > 4 * sigma + tolerance)
    return np.nan_to_num(np.where(isolated, median, shift))
