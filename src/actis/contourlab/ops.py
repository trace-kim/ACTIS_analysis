"""Image operations of the contour lab.

Grey levels are float32 in the detector's own numbers (DN); coordinates are
(x, y) = (column, row) with pixel centres at integers.

An ACTIS contact hole is about four pixels across, so the outline of a pixel
mask says little about it. Outlines here are level sets instead: the zero line
of a "field" (the image minus a threshold level, which may vary across the
image), first found between the pixels by marching squares and then moved onto
the zero of the field's cubic-spline interpolation.
"""

import os
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from contourpy import LineType, contour_generator
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from . import contours as geometry
from .contours import Contours

UPSCALERS = ("cubic spline", "lanczos", "linear")
RULES = ("half level", "steepest gradient", "threshold level")
_BLOCK = 400_000  # contour points handled at a time where a profile is kept per point
_workers = None


def _pool():
    global _workers
    if _workers is None:
        _workers = ThreadPoolExecutor(min(8, os.cpu_count() or 1))
    return _workers


# --- Noise and levels -------------------------------------------------------


def noise_sigma(image, tile=128, band=0.30):
    """Standard deviation of the image's pixel noise, in grey levels.

    The median of the windowed periodogram over the four corners of the
    spectrum (both spatial frequencies above `band` cycles per pixel), in up to
    sixteen tiles. The optics pass little there, a repeating pattern puts only
    isolated harmonics there, and row or column stripes none at all, so a dense
    hole array does not inflate it the way a difference of neighbouring pixels
    would. Too small an image: the latter, as a fallback.
    """
    a = np.asarray(image, dtype=np.float32)
    h, w = a.shape
    side = int(min(tile, h, w))
    if side < 16:
        d = np.diff(np.diff(a, axis=0), axis=1)
        return float(np.median(np.abs(d - np.median(d))) / 1.349) if d.size else 0.0
    window = np.outer(np.hanning(side), np.hanning(side))
    window /= np.sqrt((window**2).sum())
    frequency = np.abs(np.fft.fftfreq(side))
    corner = (frequency[:, None] > band) & (frequency[None, :] > band)
    found = []
    for y in np.linspace(0, h - side, max(1, min(4, h // side))).astype(int):
        for x in np.linspace(0, w - side, max(1, min(4, w // side))).astype(int):
            part = a[y : y + side, x : x + side]
            power = np.abs(np.fft.fft2((part - part.mean()) * window)) ** 2
            found.append(np.sqrt(np.median(power[corner]) / np.log(2)))
    return float(np.median(found))


def blur(image, sigma):
    """Gaussian blur; sigma 0 gives the image itself."""
    return ndimage.gaussian_filter(image, sigma, mode="nearest") if sigma > 0 else image


def blur_noise_gain(sigma, band=None):
    """Factor by which a Gaussian blur of this sigma lowers pixel noise: white
    noise, or (band = (cutoff along x, along y), cycles per pixel) noise that a
    low pass has confined to those frequencies already, which a blur lowers
    much less."""
    if sigma <= 0:
        return 1.0
    if band is not None:
        f = (np.arange(128) + 0.5) / 128 - 0.5
        fx, fy = np.meshgrid(f, f)
        inside = (fx / band[0]) ** 2 + (fy / band[1]) ** 2 <= 1
        if not inside.any():
            return 1.0
        passed = np.exp(-4 * np.pi**2 * sigma**2 * (fx**2 + fy**2))
        return float(np.sqrt(passed[inside].mean()))
    impulse = np.zeros(2 * int(4 * sigma + 1) + 1)
    impulse[len(impulse) // 2] = 1
    return float((ndimage.gaussian_filter1d(impulse, sigma) ** 2).sum())


def tile_levels(image, tile, low_percent, high_percent):
    """Dark and bright level of every tile: (low, high, y centres, x centres).

    The image is cut into tiles of about `tile` pixels; low and high are the
    two percentiles of each tile's grey levels.
    """
    h, w = image.shape
    rows = np.array_split(np.arange(h), max(1, round(h / tile)))
    cols = np.array_split(np.arange(w), max(1, round(w / tile)))
    levels = np.array(
        [
            [
                np.percentile(
                    image[y[0] : y[-1] + 1, x[0] : x[-1] + 1],
                    (low_percent, high_percent),
                )
                for x in cols
            ]
            for y in rows
        ],
        dtype=np.float32,
    )
    centres_y = np.array([y.mean() for y in rows])
    centres_x = np.array([x.mean() for x in cols])
    return levels[:, :, 0], levels[:, :, 1], centres_y, centres_x


def expand_tiles(grid, centres_y, centres_x, shape):
    """A tile grid as a smooth map of the image: bilinear between tile centres."""
    h, w = shape
    strips = np.stack([np.interp(np.arange(w), centres_x, row) for row in grid])
    at = np.interp(np.arange(h), centres_y, np.arange(len(centres_y)))
    low = np.floor(at).astype(int)
    high = np.minimum(low + 1, len(centres_y) - 1)
    mixed = strips[low] + (at - low)[:, None] * (strips[high] - strips[low])
    return mixed.astype(np.float32)


def feature_levels(image, window):
    """Dark and bright level around every pixel within `window` pixels.

    The minimum and maximum over a window of about one feature spacing, each
    averaged over the same window: every feature is compared with its own
    peak and its own surround.
    """
    size = max(3, int(round(window)) | 1)
    low = ndimage.minimum_filter(image, size=size, mode="nearest")
    high = ndimage.maximum_filter(image, size=size, mode="nearest")
    low = ndimage.uniform_filter(low, size=size, mode="nearest")
    high = ndimage.uniform_filter(high, size=size, mode="nearest")
    return low, high


def estimate_pitch(image, limit=512):
    """Spacing in pixels of a repeating pattern, or None if nothing repeats.

    The distance of the nearest strong off-centre peak of the autocorrelation
    of the image's central part.
    """
    a = np.asarray(image, dtype=np.float32)
    h, w = a.shape
    side_y, side_x = min(h, limit), min(w, limit)
    if min(side_y, side_x) < 16:
        return None
    y0, x0 = (h - side_y) // 2, (w - side_x) // 2
    part = a[y0 : y0 + side_y, x0 : x0 + side_x]
    part = part - ndimage.uniform_filter(part, 31, mode="nearest")
    part = part * np.outer(np.hanning(side_y), np.hanning(side_x))
    power = np.abs(np.fft.rfft2(part)) ** 2
    correlation = np.fft.fftshift(np.fft.irfft2(power, s=part.shape))
    peak = correlation.max()
    if peak <= 0:
        return None
    correlation /= peak
    cy, cx = side_y // 2, side_x // 2
    yy, xx = np.mgrid[0:side_y, 0:side_x]
    radius = np.hypot(yy - cy, xx - cx)
    local = correlation == ndimage.maximum_filter(correlation, size=5)
    local &= (radius >= 2.5) & (radius <= min(side_y, side_x) / 3)
    local[[0, -1], :] = False
    local[:, [0, -1]] = False
    if not local.any():
        return None
    strongest = correlation[local].max()
    if strongest < 0.15:
        return None
    ys, xs = np.nonzero(local & (correlation >= 0.5 * strongest))
    nearest = np.argmin(radius[ys, xs])
    y, x = ys[nearest], xs[nearest]

    def vertex(before, at, after):
        bend = before - 2 * at + after
        return 0.5 * (before - after) / bend if bend < 0 else 0.0

    dy = (
        y - cy + vertex(correlation[y - 1, x], correlation[y, x], correlation[y + 1, x])
    )
    dx = (
        x - cx + vertex(correlation[y, x - 1], correlation[y, x], correlation[y, x + 1])
    )
    return float(np.hypot(dy, dx))


def otsu_levels(image, classes=2):
    """Otsu's thresholds between `classes` (2 or 3) classes of grey level.

    The levels that divide the histogram into classes of least variance
    within each: one level for two classes, two for three.
    """
    values = np.asarray(image, dtype=np.float32).ravel()
    low, high = float(values.min()), float(values.max())
    if not high > low:
        return [low] * (classes - 1)
    bins = 512 if classes == 2 else 160
    histogram, edges = np.histogram(values, bins, (low, high))
    mids = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(histogram).astype(np.float64)
    moment = np.cumsum(histogram * mids)
    if classes == 2:
        rest = weight[-1] - weight
        both = (weight > 0) & (rest > 0)
        gap = moment / np.maximum(weight, 1) - (moment[-1] - moment) / np.maximum(
            rest, 1
        )
        between = np.where(both, weight * rest * gap**2, -1.0)
        # Across a gap in the histogram every level divides it equally well:
        # take the middle of the gap, not its first grey level.
        best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
        return [float(edges[best[len(best) // 2] + 1])]
    # three classes: bins [0, i], (i, j], (j, end]
    i, j = np.triu_indices(bins - 1, 1)
    w0, w1, w2 = weight[i], weight[j] - weight[i], weight[-1] - weight[j]
    m0, m1, m2 = moment[i], moment[j] - moment[i], moment[-1] - moment[j]
    filled = (w0 > 0) & (w1 > 0) & (w2 > 0)
    between = (
        m0**2 / np.maximum(w0, 1)
        + m1**2 / np.maximum(w1, 1)
        + m2**2 / np.maximum(w2, 1)
    )
    between = np.where(filled, between, -1.0)
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    lower, upper = int(np.median(i[best])), int(np.median(j[best]))
    return [float(edges[lower + 1]), float(edges[upper + 1])]


# --- Image changes ----------------------------------------------------------


def upscale(image, factor, method="cubic spline"):
    """The image interpolated to `factor` times as many pixels per side.

    Both cover the same area: pixel (x, y) of the result lies at
    ((x + 0.5) / factor - 0.5, (y + 0.5) / factor - 0.5) of the image.
    """
    if factor == 1:
        return image
    if method == "lanczos":
        h, w = image.shape
        picture = Image.fromarray(np.ascontiguousarray(image, dtype=np.float32))
        picture = picture.resize((w * factor, h * factor), Image.Resampling.LANCZOS)
        return np.asarray(picture, dtype=np.float32)
    order = 1 if method == "linear" else 3
    return ndimage.zoom(
        image, factor, order=order, mode="reflect", grid_mode=True
    ).astype(np.float32, copy=False)


def richardson_lucy(image, sigma_x, sigma_y, iterations):
    """The image with a Gaussian blur taken out again (Richardson-Lucy).

    Each iteration sharpens the estimate so that, blurred with the Gaussian
    (sigma_x along the rows, sigma_y down the columns), it comes closer to the
    image. The grey levels are counted from the image's darkest value, which
    the algorithm needs to be positive.
    """
    data = np.asarray(image, dtype=np.float32)
    if iterations < 1 or max(sigma_x, sigma_y) <= 0:
        return data
    # counted from just below the darkest value, so that every pixel is positive
    floor = np.float32(float(data.min()) - 1e-3 * max(float(np.ptp(data)), 1e-6))
    data = data - floor
    sigma = (sigma_y, sigma_x)
    estimate = data.copy()
    for _ in range(int(iterations)):
        blurred = ndimage.gaussian_filter(estimate, sigma, mode="nearest")
        ratio = data / np.maximum(blurred, 1e-12)
        estimate *= ndimage.gaussian_filter(ratio, sigma, mode="nearest")
    return estimate + floor


def non_local_means(image, noise, strength=1.0, patch=3, search=11):
    """Each pixel as a mean of the pixels whose surroundings look like its own.

    The mean is over a window of `search` pixels, weighted by how alike the
    `patch`-pixel surroundings are: exp(-d2 / h^2), where d2 is the mean
    squared difference of the two patches less the 2 noise^2 that noise alone
    gives, and h = strength x noise (grey levels).
    """
    a = np.asarray(image, dtype=np.float32)
    reach = max(1, int(search) // 2)
    size = max(1, int(patch) | 1)
    padded = np.pad(a, reach, mode="reflect")
    h, w = a.shape
    total = np.zeros_like(a)
    weights = np.zeros_like(a)
    scale = np.float32(1.0 / max(strength * noise, 1e-12) ** 2)
    alone = np.float32(2 * noise**2)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            moved = padded[reach + dy : reach + dy + h, reach + dx : reach + dx + w]
            apart = ndimage.uniform_filter((a - moved) ** 2, size=size, mode="nearest")
            weight = np.exp(-np.maximum(apart - alone, 0) * scale)
            total += weight * moved
            weights += weight
    return total / weights


def remove_stripes(image, rows=True, columns=True, smooth_px=65):
    """The image without offsets that differ from line to line.

    The median of every row (column) is compared with the medians of the rows
    (columns) around it, over smooth_px lines, and the difference subtracted.
    """
    a = np.asarray(image, dtype=np.float32).copy()
    size = max(3, int(smooth_px) | 1)
    for axis, wanted in ((1, rows), (0, columns)):
        if wanted:
            level = np.median(a, axis=axis)
            offset = level - ndimage.median_filter(level, size=size, mode="nearest")
            a -= offset[:, None] if axis == 1 else offset[None, :]
    return a


def disk(size):
    """A disk of this width in pixels (odd) as a boolean footprint."""
    radius = max(int(size) | 1, 1) // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius + radius * 0.5


# --- Level sets: outlines between the pixels --------------------------------


def level_lines(field):
    """The zero lines of a field as contours, by marching squares.

    Each line runs with the positive side of the field on its inside (contour
    module docstring), closed unless it leaves the image. Lines of fewer than
    three points are dropped.
    """
    if field.shape[0] < 2 or field.shape[1] < 2:
        return geometry.empty()
    generator = contour_generator(z=field, line_type=LineType.ChunkCombinedOffset)
    points, offsets = generator.lines(0.0)
    if points[0] is None or not len(points[0]):
        return geometry.empty()
    xy, start = points[0], offsets[0].astype(np.int64)
    first, last = start[:-1], start[1:] - 1
    closed = np.all(xy[first] == xy[last], axis=1) & (last - first >= 3)
    keep = np.ones(len(xy), bool)
    keep[last[closed]] = False  # a closed line repeats its first point
    counts = np.diff(start) - closed
    lines = Contours(xy[keep], np.concatenate([[0], np.cumsum(counts)]), closed)
    return lines.select(counts >= 3)


def spline(image):
    """Cubic-spline coefficients of an image, for sample()."""
    kind = np.float64 if image.size <= 2**24 else np.float32
    return ndimage.spline_filter(image, order=3, mode="mirror", output=kind)


def sample(coefficients, xy):
    """(M,) the spline of spline() at points (M, 2) between the pixels."""

    def part(points):
        return ndimage.map_coordinates(
            coefficients,
            [points[:, 1], points[:, 0]],
            order=3,
            mode="mirror",
            prefilter=False,
        )

    if len(xy) < 100_000:
        return part(xy)
    # scipy releases the interpreter lock while it interpolates
    pieces = np.array_split(xy, 16)
    return np.concatenate(list(_pool().map(part, pieces)))


def to_level(field, lines, probe=0.25, max_step=0.75):
    """Contours moved along their normals onto the field's zero level.

    The field is taken between the pixels as its cubic spline. Each point is
    moved by two secant steps, at most max_step pixels; a point where the
    field does not fall going outward stays where it is.
    """
    if not len(lines):
        return lines
    coefficients = spline(field)
    normal = geometry.normals(lines)
    xy = lines.xy

    def value(offset):
        return sample(coefficients, xy + offset[:, None] * normal)

    zero = np.zeros(len(xy))
    at_start = value(zero)
    slope = (value(zero + probe) - at_start) / probe
    falls = slope < 0
    first = np.where(falls, at_start / np.where(falls, -slope, 1.0), 0.0)
    first = np.clip(first, -max_step, max_step)
    at_first = value(first)
    change = at_first - at_start
    usable = falls & (np.abs(change) > 1e-12)
    second = np.where(
        usable, first - at_first * first / np.where(usable, change, 1), first
    )
    second = np.clip(second, -max_step, max_step)
    at_second = value(second)
    change = at_second - at_first
    usable = falls & (np.abs(change) > 1e-12) & (second != first)
    third = np.where(
        usable,
        second - at_second * (second - first) / np.where(usable, change, 1),
        second,
    )
    third = np.clip(third, -max_step, max_step)
    return lines.moved(xy + third[:, None] * normal)


def trace_level(field, spacing=0.25, subpixel=True):
    """The zero level set of a field as contours with points every `spacing`."""
    lines = level_lines(field)
    if not len(lines):
        return lines
    lines = geometry.resample(lines, spacing)
    return to_level(field, lines) if subpixel else lines


# --- Edges measured along the normals ---------------------------------------


def _blocks(contours, limit=_BLOCK):
    """The contours in turn as pieces of at most about `limit` points."""
    start = contours.start
    i = 0
    while i < len(contours):
        j = int(np.searchsorted(start, start[i] + limit, side="right")) - 1
        j = min(max(j, i + 1), len(contours))
        piece = Contours(
            contours.xy[start[i] : start[j]],
            start[i : j + 1] - start[i],
            contours.closed[i:j],
        )
        yield slice(start[i], start[j]), piece
        i = j


def _span(contours, reach):
    """How many contour points lie within `reach` pixels along a contour."""
    measures = geometry.measure(contours)
    steps = np.maximum(measures["points"] - ~contours.closed, 1)
    spacing = float(np.median(measures["length"] / steps))
    return max(1, int(round(reach / spacing))) if spacing > 0 else 1


def _crossings(profiles, offsets, level):
    """Offset where each profile falls through its level, nearest to zero.

    profiles (M, T) are sampled at evenly spaced offsets going outward; level
    is (M,). NaN where a profile does not cross.
    """
    before, after = profiles[:, :-1], profiles[:, 1:]
    level = level[:, None]
    fall = before - after
    step = offsets[1] - offsets[0]
    at = offsets[:-1] + step * (before - level) / np.where(fall != 0, fall, 1)
    distance = np.where((before >= level) & (after < level), np.abs(at), np.inf)
    nearest = distance.argmin(axis=1)
    found = np.isfinite(distance.min(axis=1))
    return np.where(found, at[np.arange(len(at)), nearest], np.nan)


def _steepest(profiles, offsets):
    """Offset of the steepest fall along each profile; NaN where it has no peak."""
    step = offsets[1] - offsets[0]
    slope = -np.gradient(profiles, step, axis=1)
    k = np.argmax(slope, axis=1)
    rows = np.arange(len(k))
    last = slope.shape[1] - 1
    before = slope[rows, np.clip(k - 1, 0, None)]
    after = slope[rows, np.clip(k + 1, None, last)]
    bend = before - 2 * slope[rows, k] + after
    peak = (k > 0) & (k < last) & (bend < 0)
    shift = np.where(peak, 0.5 * (before - after) / np.where(peak, bend, 1.0), 0.0)
    return np.where(peak, offsets[0] + (k + shift) * step, np.nan)


def refine(
    image,
    contours,
    rule="half level",
    search=1.5,
    step=0.125,
    blur_sigma=0.5,
    fraction=0.5,
    level=np.nan,
    bright=True,
    reach=0.75,
    tolerance=0.3,
):
    """Contours with every point moved along its normal to its edge.

    The image, blurred by blur_sigma, is sampled (cubic spline) from -search to
    +search pixels along each point's outward normal, in steps of `step`.
    "half level": where it passes `fraction` of the way from the contour's own
    surround level (the median over its points of the lowest sample outward) up
    to its own inside level (the median of the highest sample inward, or the
    level at its centroid if that is higher: the peak of a feature too small
    for the search to reach from its edge).
    "steepest gradient": where it changes fastest. "threshold level": where it
    passes `level`. A point without such an edge, or far off its neighbours',
    takes the median of its neighbours (contours.without_isolated). bright
    False: the regions are the darker side. reach: each normal is taken across
    the contour points this far before and after its point, which evens out
    the steps of an outline traced along pixels.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    if not len(contours):
        return contours
    sign = 1.0 if bright else -1.0
    smooth = blur(np.asarray(image, dtype=np.float32), blur_sigma) * np.float32(sign)
    coefficients = spline(smooth)
    offsets = np.arange(-search, search + step / 2, step)
    inward, outward = offsets <= 0, offsets >= 0
    moved = np.empty_like(contours.xy)
    middle = sample(coefficients, geometry.measure(contours)["centre"])
    done = 0
    span = _span(contours, reach)
    for where, piece in _blocks(contours):
        normal = geometry.normals(piece, span)
        profiles = np.stack(
            [sample(coefficients, piece.xy + t * normal) for t in offsets], axis=1
        )
        if rule == "steepest gradient":
            shift = _steepest(profiles, offsets)
        else:
            if rule == "threshold level":
                edge = np.full(len(piece.xy), sign * level)
            else:
                inside = geometry.group_median(
                    profiles[:, inward].max(axis=1), piece.start
                )
                inside = np.fmax(inside, middle[done : done + len(piece)])
                around = geometry.group_median(
                    profiles[:, outward].min(axis=1), piece.start
                )
                edge = (around + fraction * (inside - around))[piece.owner]
            shift = _crossings(profiles, offsets, edge)
        shift = geometry.without_isolated(shift, piece, tolerance=tolerance)
        moved[where] = piece.xy + shift[:, None] * normal
        done += len(piece)
    return contours.moved(moved)


def edge_contrast(image, contours, distance=1.0, reach=0.75):
    """(N,) mean grey level just inside each contour minus just outside it."""
    if not len(contours):
        return np.empty(0)
    coefficients = spline(np.asarray(image, dtype=np.float32))
    normal = geometry.normals(contours, _span(contours, reach))
    step = sample(coefficients, contours.xy - distance * normal)
    step = step - sample(coefficients, contours.xy + distance * normal)
    return np.add.reduceat(step, contours.start[:-1]) / contours.counts


# --- Edges without a threshold ----------------------------------------------


def gradient_edges(
    image,
    sigma=0.8,
    oversample=3,
    high=0.3,
    low_ratio=0.5,
    min_length=6.0,
    spacing=0.25,
    floor=0.0,
):
    """Edges where the brightness changes fastest, as sub-pixel contours.

    The zero crossings of the second derivative of the image along its
    gradient (Gaussian derivatives of the given sigma, on the image
    interpolated `oversample` times): the same edges Canny's detector marks,
    found as lines between the pixels. A line is kept along the stretches
    where the gradient exceeds low_ratio x the upper threshold, if somewhere
    on the stretch it exceeds the upper threshold itself (`high` x the 99.9th
    percentile of the gradient, and not less than `floor` grey levels per
    pixel) and the stretch is at least min_length pixels long. Each line runs
    with the brighter side on its inside.
    Returns (contours in the image's pixels, the upper threshold in grey levels
    per pixel).
    """
    fine = upscale(np.asarray(image, dtype=np.float32), oversample)
    s = sigma * oversample
    dx = ndimage.gaussian_filter(fine, s, order=(0, 1), mode="nearest")
    dy = ndimage.gaussian_filter(fine, s, order=(1, 0), mode="nearest")
    curve = dx * dx * ndimage.gaussian_filter(fine, s, order=(0, 2), mode="nearest")
    curve += (
        2 * dx * dy * ndimage.gaussian_filter(fine, s, order=(1, 1), mode="nearest")
    )
    curve += dy * dy * ndimage.gaussian_filter(fine, s, order=(2, 0), mode="nearest")
    del fine
    size = np.hypot(dx, dy)
    del dx, dy
    np.negative(curve, out=curve)  # positive on the concave, brighter side of an edge
    lines = level_lines(curve)
    del curve
    upper = float(high * np.percentile(size[::2, ::2], 99.9))
    upper = max(upper, floor / oversample)  # per pixel of the interpolated image
    if not len(lines) or not upper > 0:
        return geometry.empty({"weak": len(lines)}), upper * oversample
    lines = geometry.resample(lines, spacing * oversample)
    strength = ndimage.map_coordinates(
        size, [lines.xy[:, 1], lines.xy[:, 0]], order=1, mode="nearest"
    )
    pieces, origin = runs(lines, strength >= low_ratio * upper)
    faint = len(lines) - len(np.unique(origin))  # lines without one strong enough point
    if not len(pieces):
        return geometry.empty({"weak": faint}), upper * oversample
    piece_strength = ndimage.map_coordinates(
        size, [pieces.xy[:, 1], pieces.xy[:, 0]], order=1, mode="nearest"
    )
    strongest = np.maximum.reduceat(piece_strength, pieces.start[:-1])
    length = geometry.measure(pieces)["length"] / oversample
    strong, long = strongest >= upper, length >= min_length
    kept = pieces.select(
        strong & long, weak=faint + (~strong).sum(), short=(strong & ~long).sum()
    )
    return kept.moved((kept.xy + 0.5) / oversample - 0.5), upper * oversample


def runs(contours, keep):
    """The stretches of the contours along which keep (P,) is True.

    Returns (contours, the contour each came from). A closed contour all of
    whose points are kept stays closed; any other stretch is an open contour.
    """
    if not len(contours):
        return contours, np.empty(0, np.int64)
    owner, counts, first = contours.owner, contours.counts, contours.start[:-1]
    local = np.arange(len(keep)) - first[owner]
    entire = np.logical_and.reduceat(keep, first)
    # begin a closed contour at a dropped point, so that no stretch runs past its end
    dropped_at = np.minimum.reduceat(np.where(keep, counts[owner], local), first)
    turn = np.where(contours.closed & ~entire, dropped_at, 0)
    source = first[owner] + (local + turn[owner]) % counts[owner]
    kept = keep[source]
    begins = kept & ((local == 0) | ~np.concatenate([[False], kept[:-1]]))
    points = contours.xy[source][kept]
    if not len(points):
        return geometry.empty(contours.left_out), np.empty(0, np.int64)
    start = np.flatnonzero(begins[kept])
    origin = owner[kept][start]
    start = np.concatenate([start, [len(points)]]).astype(np.int64)
    closed = entire[origin] & contours.closed[origin]
    pieces = Contours(points, start, closed, contours.left_out)
    enough = np.diff(start) >= np.where(closed, 3, 2)
    return pieces.select(enough), origin[enough]


# --- What the optics pass, and what is noise --------------------------------
#
# An actinic image holds no detail finer than its optics resolve: beyond their
# cutoff the spectrum of the image is detector noise only. ACTIS samples about
# twice as finely as that cutoff needs, so most of the frequency plane carries
# noise and no pattern, and can be removed without touching an edge.


def _tile_power(a, side, most=4):
    """Mean windowed periodogram of up to most x most tiles of `side` pixels,
    and how many tiles it is the mean of."""
    h, w = a.shape
    taper = np.hanning(side) ** 2  # its leakage falls off fast enough for 16 bits
    window = np.outer(taper, taper).astype(np.float32)
    total, count = np.zeros((side, side)), 0
    for y in np.linspace(0, h - side, max(1, min(most, h // side))).astype(int):
        for x in np.linspace(0, w - side, max(1, min(most, w // side))).astype(int):
            part = a[y : y + side, x : x + side]
            total += np.abs(sfft.fft2((part - part.mean()) * window)) ** 2
            count += 1
    return total / count, count


def pass_band(image, keep=0.1):
    """(along x, along y): the highest spatial frequencies, in cycles per pixel,
    at which the image holds pattern above its noise. (0.5, 0.5): up to the
    sampling limit, nothing to remove.

    The averaged periodogram of up to sixteen tiles is compared with its noise
    floor (the corners of the spectrum, as noise_sigma). Off the two axes, the
    band ends after the last ring in which the mean power exceeds the floor by
    more than `keep` of it, or in which a single harmonic of a repeating
    pattern stands out of the noise. On the axes, where line patterns aligned with the
    pixel grid put all their harmonics and where stripe noise lies too, it ends
    after the last isolated peak far above that axis's own level.
    """
    a = np.asarray(image, dtype=np.float32)
    side = int(min(256, *a.shape)) // 2 * 2
    if side < 48:
        return 0.5, 0.5
    power, count = _tile_power(a, side)
    frequency = np.abs(np.fft.fftfreq(side))
    fy, fx = frequency[:, None], frequency[None, :]
    corner = (fx > 0.3) & (fy > 0.3)
    # the median of a mean of `count` exponentials, as a share of their mean
    floor = float(np.median(power[corner])) / stats.gamma.median(count, scale=1 / count)
    if not floor > 0:
        return 0.5, 0.5
    radius = np.hypot(fx, fy)
    # the power noise alone reaches in one of a thousand spectra, in any of its values
    harmonic = floor * stats.gamma.isf(1e-3 / power.size, count, scale=1 / count)
    near_axis = 1.5 / side
    edges = np.arange(0.06, 0.5 + 1e-9, 1 / 64)
    found = []
    for along_x in (True, False):
        sector = (fx > near_axis) & (fy > near_axis)
        sector = sector & ((fx >= fy) if along_x else (fy > fx))
        ring = np.digitize(radius[sector], edges)
        total = np.bincount(ring, power[sector], len(edges) + 1)[1:-1]
        counts = np.bincount(ring, minlength=len(edges) + 1)[1:-1]
        # a ring's mean is only known to about 1 / sqrt(its independent values): on
        # a small image noise alone would look like pattern in some ring
        unsure = 4 / np.sqrt(np.maximum(counts, 1) * count / 2)
        above = total / np.maximum(counts, 1) / floor - 1 > np.maximum(keep, unsure)
        last = np.flatnonzero(above[:-1] & above[1:])  # two rings in a row
        cut = edges[last[-1] + 2] if len(last) else edges[0]
        # a repeating pattern puts its power in a few harmonics, which a ring's
        # mean hides: the band reaches as far as any harmonic stands out
        peaks = np.bincount(ring, power[sector] > harmonic, len(edges) + 1)[1:-1]
        outer = np.flatnonzero(peaks)
        if len(outer):
            cut = max(cut, edges[outer[-1] + 1])
        # the axis itself: the strongest of its three lines of the spectrum
        lines = power[[0, 1, -1], :] if along_x else power[:, [0, 1, -1]].T
        line = lines.max(axis=0)[: side // 2]
        along = frequency[: side // 2]
        level = float(np.median(line[along > 0.1]))
        peaks = np.flatnonzero((line > 30 * level) & (along > 0.06))
        if len(peaks):
            cut = max(cut, along[peaks[-1]] + 2 / side)
        found.append(float(min(cut, 0.5)))
    return found[0], found[1]


def optical_filter(
    image, cutoff_x, cutoff_y, deblur_x=0.0, deblur_y=0.0, max_gain=3.0, edge=0.15
):
    """(the image without what lies beyond the cutoffs, the factor by which its
    pixel noise changed).

    A low pass in the frequency plane: all of the ellipse with half axes
    cutoff_x and cutoff_y (cycles per pixel), nothing beyond 1 + `edge` times
    it, a smooth step between. deblur_x, deblur_y: sigma in pixels of a
    Gaussian blur divided out inside the band, by at most max_gain at any
    frequency. The image is mirrored at its borders first, so that they do not
    ring.
    """
    a = np.asarray(image, dtype=np.float32)
    h, w = a.shape
    pad_y, pad_x = min(32, h), min(32, w)
    rows = sfft.next_fast_len(h + 2 * pad_y, real=True)
    cols = sfft.next_fast_len(w + 2 * pad_x, real=True)
    padded = np.pad(
        a, ((pad_y, rows - h - pad_y), (pad_x, cols - w - pad_x)), mode="symmetric"
    )
    spectrum = sfft.rfft2(padded, workers=-1)
    fy = np.abs(sfft.fftfreq(rows)).astype(np.float32)[:, None]
    fx = sfft.rfftfreq(cols).astype(np.float32)[None, :]
    reach = np.hypot(fx / np.float32(cutoff_x), fy / np.float32(cutoff_y))
    gain = np.clip((1 + edge - reach) / edge, 0, 1)
    gain = gain * gain * (3 - 2 * gain)
    if deblur_x > 0 or deblur_y > 0:
        blur = np.exp(-2 * np.pi**2 * ((deblur_x * fx) ** 2 + (deblur_y * fy) ** 2))
        gain = gain / np.maximum(blur, 1 / max_gain)
    gain = gain.astype(np.float32)
    spectrum *= gain
    out = sfft.irfft2(spectrum, s=(rows, cols), workers=-1)
    # Each column of the half plane stands for two of the whole, but the first
    # (and the last, if the width is even).
    weight = np.full(gain.shape[1], 2.0)
    weight[0] = 1.0
    if cols % 2 == 0:
        weight[-1] = 1.0
    squared = (gain.astype(np.float64) ** 2 * weight).sum()
    noise_gain = float(np.sqrt(squared / (rows * cols)))
    return np.ascontiguousarray(out[pad_y : pad_y + h, pad_x : pad_x + w]), noise_gain


def noise_by_level(image, bins=24):
    """How the pixel noise depends on the grey level: (levels, sigmas), rising
    levels, for np.interp. Shot noise makes bright areas noisier than dark
    ones, so one sigma for a page that holds both fits neither.

    The noise is what the image holds in the corners of its spectrum (both
    frequencies above 0.3 cycles per pixel, without the lattice harmonics that
    stand out there); its strength is taken per class of local brightness.
    """
    a = np.asarray(image, dtype=np.float32)
    h, w = a.shape
    overall = noise_sigma(a)
    flat = np.array([float(a.min()), float(a.max()) + 1.0]), np.array([overall] * 2)
    if min(h, w) < 48 or not overall > 0:
        return flat
    spectrum = sfft.rfft2(a, workers=-1)
    fy = np.abs(sfft.fftfreq(h))[:, None]
    fx = sfft.rfftfreq(w)[None, :]
    corner = (fx > 0.3) & (fy > 0.3)
    power = spectrum.real**2 + spectrum.imag**2
    typical = float(np.median(power[corner]))

    # the corners come in gently (from 0.25 to 0.4 cycles per pixel), so that the
    # noise of a bright area does not ring far into a dark one beside it
    def rise(f):
        t = np.clip((f - 0.25) / 0.15, 0, 1)
        return t * t * (3 - 2 * t)

    weight = rise(fx) * rise(fy) * (power <= 12 * typical)
    share = 2.0 * float((weight**2).sum()) / (h * w)  # of the whole frequency plane
    if not share > 0.01:
        return flat
    spectrum *= weight
    rest = sfft.irfft2(spectrum, s=(h, w), workers=-1)
    local = ndimage.gaussian_filter(a, 1.5, mode="nearest")
    stride = max(1, int(np.sqrt(a.size / 1_000_000)))
    level = local[::stride, ::stride].ravel()
    square = (rest[::stride, ::stride] ** 2).ravel()
    order = np.argsort(level)
    groups = np.array_split(order, max(2, min(bins, len(order) // 2000)))
    levels = np.array([np.median(level[g]) for g in groups], dtype=float)
    # the median of a squared normal deviate is 0.4549 of its variance
    sigmas = np.sqrt([np.median(square[g]) / 0.4549 / share for g in groups])
    sigmas = np.maximum(sigmas, 0.1 * overall)  # a clipped area has no noise to measure
    keep = np.concatenate([[True], np.diff(levels) > 0])
    return levels[keep], sigmas[keep]


_noise_maps = {}  # id(image) -> (weak reference to it, its noise map, its overall noise)


def noise_map(image):
    """(sigma of the pixel noise at every pixel, the one sigma of the whole
    image): the noise each pixel's local brightness goes with (noise_by_level).
    Kept for as long as the image is, so that asking again costs nothing."""
    kept = _noise_maps.get(id(image))
    if kept is not None and kept[0]() is image:
        return kept[1], kept[2]
    a = np.asarray(image, dtype=np.float32)
    levels, sigmas = noise_by_level(a)
    local = ndimage.gaussian_filter(a, 1.5, mode="nearest")
    sigma = np.interp(local, levels, sigmas).astype(np.float32)
    overall = noise_sigma(a)
    key = id(image)
    try:
        watch = weakref.ref(image, lambda _: _noise_maps.pop(key, None))
    except TypeError:  # not an array that can be watched: made again next time
        return sigma, overall
    _noise_maps[key] = (watch, sigma, overall)
    return sigma, overall


# --- Levels of each feature's own ------------------------------------------

_QUADRIC = np.linalg.pinv(
    np.array(
        [[1, x, y, x * x, x * y, y * y] for y in (-1, 0, 1) for x in (-1, 0, 1)],
        dtype=float,
    )
)


_GATHER = 0.8  # sigma of the smoothing a feature is told from noise in, pixels


def peak_values(image, ys, xs, base):
    """The image's value at the top of the peak at each of the pixels (ys, xs),
    between the pixels. A hole only a few pixels wide is as much as a tenth
    brighter at its top than at the centre of the pixel nearest to it, by an
    amount that depends on where the hole lies on the pixel grid.

    base (M,): the dark level under each peak. The top is that of the quadric
    fitted to the logarithm of the 3 x 3 values above the base, which is exact
    for a Gaussian peak and close for a blurred opening.
    """
    h, w = image.shape
    values = np.stack(
        [
            image[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ],
        axis=1,
    ).astype(float)
    above = values - base[:, None]
    height = above[:, 4]
    usable = height > 0
    scale = np.where(usable, height, 1.0)
    c = np.log(np.maximum(above / scale[:, None], 1e-3)) @ _QUADRIC.T
    bend = 4 * c[:, 3] * c[:, 5] - c[:, 4] ** 2
    dome = usable & (bend > 0) & (c[:, 3] < 0)
    safe = np.where(dome, bend, 1.0)
    dx = np.clip((c[:, 4] * c[:, 2] - 2 * c[:, 5] * c[:, 1]) / safe, -0.75, 0.75)
    dy = np.clip((c[:, 4] * c[:, 1] - 2 * c[:, 3] * c[:, 2]) / safe, -0.75, 0.75)
    # the fitted surface at its top, less what it is at the centre pixel
    rise = np.where(dome, 0.5 * (c[:, 1] * dx + c[:, 2] * dy), 0.0)
    return base + height * np.exp(np.clip(rise, 0.0, 0.5))


def feature_field(
    image,
    sigma,
    fraction=0.5,
    peers=1,
    min_contrast=10.0,
    reach=48,
    smooth=1.0,
    gathered_gain=None,
    scale=1,
):
    """(field, how many tops set levels, how many were ignored as noise): the
    image minus a threshold level that the features set for themselves.

    A feature is anything with a top: a hole, a line, a pad.
    Which tops count. A top must stand out: above the pass to any higher
    ground within eight pixels by four noise sigma and by a fifth of its
    height over its surround, so that a noise bump on the ridge between two
    holes, or a second bump on one hole, is no top. It must not lie on the
    border of the image, where a feature's top is outside. It must have an edge
    within reach: in the image smoothed by 0.8 px (which gathers a feature's
    light, so that a faint hole is told from noise by all of its pixels), the
    lowest value around it falls by min_contrast noise sigma between one
    distance and four times it, for distances of 2, 4, 8 ... pixels up to
    about `reach`, by more than twice the fall over the distances before (as
    brightness that slopes evenly would). A bump on an empty area fails that,
    also where the area is bright, uneven, and noisier than the dark. And its
    dark level must not lie
    above the middle level of its eight nearest tops: one that does is a
    ripple on top of a feature.
    Its two levels. Bright: the image at the top, between the pixels. Dark:
    the lowest value of the smoothed image around the top (a valley read from
    single pixels depends on where the pixels fall in it), taken further and
    further out until going twice as far no longer finds anything lower by a
    tenth of the contrast. peers > 1: both are the medians over the top's
    `peers` nearest tops of like contrast (within a factor of 1.4 of its own).
    The field. Every pixel takes the level, dark + fraction x (bright - dark),
    of the top nearest to it, so a wide feature is inside all the way to its
    middle and an empty area is outside; `smooth` (sigma, pixels) rounds the
    steps between neighbouring levels. Without any top the field is negative
    everywhere.
    sigma: the pixel noise of the image, a number or a map. gathered_gain: the
    factor by which the 0.8 px smoothing lowers that noise (None: as for
    white noise; less is gained on an image that was blurred already).
    scale: pixels of the image per pixel of the detector (an upscaled image):
    every distance above but `reach` is in detector pixels.
    """
    a = np.asarray(image, dtype=np.float32)
    nothing = np.full(a.shape, -1.0, np.float32)
    span = float(a.max()) - float(a.min())
    if not span > 0:
        return nothing, 0, 0
    noise = np.maximum(np.asarray(sigma, dtype=np.float32), np.float32(1e-4 * span))
    noise = np.broadcast_to(noise, a.shape)
    if gathered_gain is None:
        gathered_gain = blur_noise_gain(_GATHER)
    scale = int(scale)
    gathered = ndimage.gaussian_filter(a, _GATHER * scale, mode="nearest")
    first = 4 * scale + 1  # a window of 5 detector pixels, then 9, then 17
    lows = [ndimage.minimum_filter(gathered, size=first, mode="nearest")]
    for size in (first, 2 * first - 1):
        lows.append(ndimage.minimum_filter(lows[-1], size=size, mode="nearest"))
    # tops that stand out: what is left of a top after the ground around it has
    # risen to within `stand` of every higher top near it
    stand = np.maximum(4 * noise, np.float32(0.2) * (a - lows[2]))
    sunk = a - stand
    flood = sunk
    for _ in range(8):
        risen = ndimage.maximum_filter(flood, size=2 * scale + 1, mode="nearest")
        flood = np.minimum(risen, a)
    # equal neighbours (8-bit flats) take turns, by an amount far below a grey level
    jitter = np.random.default_rng(0).random(a.shape, dtype=np.float32)
    order = a + jitter * np.float32(1e-5 * span)
    tops = order == ndimage.maximum_filter(order, size=3, mode="nearest")
    tops &= flood <= sunk + np.float32(1e-6 * span)
    # What is brightest at the very border of the image has its top outside it:
    # it sets no level of its own and takes its neighbours'.
    tops[:scale] = tops[-scale:] = False
    tops[:, :scale] = tops[:, -scale:] = False
    del jitter, order, flood, sunk, stand
    ys, xs = np.nonzero(tops)
    if not len(ys):
        return nothing, 0, 0
    steps = [gathered[ys, xs].astype(float)] + [
        low[ys, xs].astype(float) for low in lows
    ]
    low, size = lows[2], 4 * first - 3
    del lows, gathered
    while size < 2 * reach + 1:
        low = ndimage.minimum_filter(low, size=size, mode="nearest")
        size = 2 * size - 1
        steps.append(low[ys, xs].astype(float))
    del low
    steps = np.stack(steps, axis=1)  # the top, then its surround ever further out
    fall = steps[:, :-2] - steps[:, 2:]
    # Brightness that slopes evenly falls twice as far over twice the distance:
    # only what a fall exceeds that by is an edge.
    fall[:, 1:] -= 2 * fall[:, :-1].copy()
    needed = min_contrast * gathered_gain * noise[ys, xs]
    real = fall.max(axis=1) >= needed
    ignored = int((~real).sum())
    ys, xs, steps, needed = ys[real], xs[real], steps[real], needed[real]
    if not len(ys):
        return nothing, 0, ignored
    base, settled = steps[:, 1].copy(), np.zeros(len(ys), bool)
    for k in range(2, steps.shape[1]):
        # not before the edge is reached: on a flat top, noise alone looks settled
        below = steps[:, 0] - base >= needed
        now = ~settled & below
        now &= base - steps[:, k] <= 0.1 * (steps[:, 0] - steps[:, k])
        # the floor is what the wider look found: the same valley, seen whole
        base = np.where(settled, base, steps[:, k])
        settled |= now
    top = peak_values(a, ys, xs, base)
    if len(ys) > 2:
        # A top that stands on ground its neighbours call the inside of a feature is a
        # ripple on that feature (a bright area is never quite flat), not a feature.
        tree = cKDTree(np.column_stack([xs, ys]))
        _, near = tree.query(tree.data, k=min(9, len(ys)), workers=-1)
        middle = np.median(((base + top) / 2)[near[:, 1:]], axis=1)
        stands = base <= middle
        ignored += int((~stands).sum())
        ys, xs, top, base = ys[stands], xs[stands], top[stands], base[stands]
    if peers > 1 and len(ys) > 1:
        tree = cKDTree(np.column_stack([xs, ys]))
        _, near = tree.query(tree.data, k=min(int(peers), len(ys)), workers=-1)
        near = near.reshape(len(ys), -1)
        contrast = np.maximum(top - base, 1e-9 * span)
        ratio = contrast[near] / contrast[:, None]
        alike = (ratio >= 1 / 1.4) & (ratio <= 1.4)
        top = np.nanmedian(np.where(alike, top[near], np.nan), axis=1)
        base = np.nanmedian(np.where(alike, base[near], np.nan), axis=1)
    level = (base + fraction * (top - base)).astype(np.float32)
    number = np.zeros(a.shape, np.int32)
    number[ys, xs] = np.arange(len(ys), dtype=np.int32)
    unmarked = np.ones(a.shape, bool)
    unmarked[ys, xs] = False
    nearest = ndimage.distance_transform_edt(
        unmarked, return_distances=False, return_indices=True
    )
    levels = level[number[nearest[0], nearest[1]]]
    del nearest, number, unmarked
    if smooth > 0:
        levels = ndimage.gaussian_filter(levels, smooth * scale, mode="nearest")
    return a - levels, len(ys), ignored
