"""The modules of a contour workflow: single operations on an ACTIS image.

Each module is a function `name(state, **options) -> State`: it takes what the
modules before it made of the image and returns what it makes of that. There
are three kinds (STAGES), usually run in this order:
  Image     changes the grey-level image: denoise, flatten, deblur, upscale.
  Mask      marks the pattern's pixels, usually by a threshold. A threshold
            also keeps its "field" (the image minus the threshold level, which
            may vary across the image): the outline between the pixels is
            where the field is zero.
  Contours  makes contours, and chooses, moves or smooths them.
@module registers a function in MODULES under its own name (the name workflow
files use) with what a window needs to offer it: a title, its stage, and for
each option its default, its limits and what it does. The function's docstring
says what the module does and when to use it; the window shows it.

Options in pixels (`_px`) are pixels of the image as loaded: after Upscale a
module multiplies them by State.scale, so that putting Upscale into a workflow
needs no other option changed. A module never changes the arrays it is given:
States share them.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import contours as geometry
from . import ops
from .contours import Contours

STAGES = ("Image", "Mask", "Contours")
MAX_PIXELS = 120_000_000  # of one working image: about half a gigabyte as float32


class WorkflowError(ValueError):
    """A module cannot work on what the modules before it made; says what is missing."""


@dataclass(frozen=True)
class State:
    """What a workflow has made of an image so far."""

    image: np.ndarray  # float32 grey levels: what the pattern is found in
    scale: int = 1  # pixels of `image` per pixel of the image as loaded, per side
    edge_image: np.ndarray | None = None  # what edges are measured on, if not `image`
    mask: np.ndarray | None = None  # bool, the pixels of `image`: True = region
    field: np.ndarray | None = None  # float32, > 0 on the mask; zero level = outline
    bright: bool = True  # the regions are the brighter side
    level: float = np.nan  # the one grey level the mask was cut at; NaN: none
    contours: Contours | None = None  # in the pixels of `image`
    noise: float = np.nan  # pixel noise (sigma, grey levels) of the image as loaded
    note: str = ""  # what the last module says of what it did

    @property
    def measured(self):
        """The image the edges are measured on."""
        return self.image if self.edge_image is None else self.edge_image


def image_contours(state):
    """The state's contours in the pixels of the image as loaded (None: none yet)."""
    if state.contours is None or state.scale == 1:
        return state.contours
    return state.contours.moved((state.contours.xy + 0.5) / state.scale - 0.5)


@dataclass(frozen=True)
class Option:
    """One option of a module: a bool, a number (int or float) or a choice."""

    default: bool | int | float | str
    help: str
    low: float = 0.0
    high: float = 0.0
    step: float = 1.0
    choices: tuple = ()
    unit: str = ""

    def checked(self, value, name):
        """The value as the option's type; a ValueError says why it cannot be."""
        if isinstance(self.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, not {value!r}")
            return value
        if isinstance(self.default, str):
            if value not in self.choices:
                options = ", ".join(self.choices)
                raise ValueError(f"{name} must be one of {options}, not {value!r}")
            return value
        whole = isinstance(self.default, int)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not numeric or not np.isfinite(value) or (whole and value != int(value)):
            kind = "whole number" if whole else "number"
            raise ValueError(f"{name} must be a {kind}, not {value!r}")
        if not self.low <= value <= self.high:
            raise ValueError(
                f"{name} must be between {self.low:g} and {self.high:g}, not {value!r}"
            )
        return type(self.default)(value)


def number(default, low, high, step, help, unit=""):
    return Option(default, help, low, high, step, unit=unit)


def choice(default, choices, help):
    return Option(default, help, choices=tuple(choices))


def flag(default, help):
    return Option(default, help)


@dataclass(frozen=True)
class Module:
    name: str  # its name in MODULES and in workflow files: the function's
    title: str  # its name in a window
    stage: str  # one of STAGES
    help: str  # what it does and when to use it: the function's docstring
    options: dict  # Option by name, in the function's order
    function: Callable  # function(state, **options) -> State


MODULES = {}  # by name, in the order a window lists them


def module(title, stage, **options):
    """Register the decorated function as a module; its arguments after the
    state are its options."""

    def register(function):
        arguments = list(inspect.signature(function).parameters)[1:]
        assert stage in STAGES and arguments == list(options), function
        text = " ".join(function.__doc__.split())
        MODULES[function.__name__] = Module(
            function.__name__, title, stage, text, options, function
        )
        return function

    return register


def need_mask(state, title):
    if state.mask is None:
        raise WorkflowError(f"{title} needs a mask: put a threshold before it")
    return state.mask


def need_contours(state, title):
    if state.contours is None:
        raise WorkflowError(
            f"{title} needs contours: put Trace contours (or another module that "
            "makes contours) before it"
        )
    return state.contours


def before_the_mask(state, title):
    if state.mask is not None or state.contours is not None:
        raise WorkflowError(
            f"{title} changes the size of the image: put it before the module "
            "that makes the mask or the contours"
        )


def room_for(pixels, title):
    if pixels > MAX_PIXELS:
        raise WorkflowError(
            f"{title} would work on {pixels / 1e6:.0f} million pixels: choose a "
            "smaller region of the image, or a smaller factor"
        )


def grey(value):
    """A grey level in words: whole from a thousand up, else four digits."""
    return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.4g}"


def noise_of(state):
    """Pixel noise (sigma) of the state's image, as far as the modules kept track
    of it; measured on the image as it is if unknown."""
    return state.noise if np.isfinite(state.noise) else ops.noise_sigma(state.image)


def measured_noise(state, image):
    """The noise of an image a module sharpened, measured again where it can be
    (the detector's own pixels): sharpening amplifies noise."""
    return ops.noise_sigma(image) if state.scale == 1 else state.noise


# --- Image modules ----------------------------------------------------------

BRIGHT = (
    "The pattern is the brighter side (openings in the absorber, as ACTIS shows "
    "contact holes); off: the darker side"
)


@module(
    "Gaussian blur",
    "Image",
    sigma_px=number(0.5, 0, 20, 0.1, "Sigma of the Gaussian; 0: none", "px"),
)
def gaussian_blur(state, sigma_px):
    """Average each pixel with its neighbours, weighted by a Gaussian: the plain
    way to remove pixel noise. It adds to the blur of the optics, so a hole only
    a few pixels across looks wider and fainter: keep sigma well under a pixel
    unless the image is noisy. (The threshold modules have a blur of their own,
    which leaves this image as it is.)"""
    noise = state.noise * ops.blur_noise_gain(sigma_px)
    image = ops.blur(state.image, sigma_px * state.scale)
    return replace(state, image=image, noise=noise, note="")


@module(
    "Median filter",
    "Image",
    size_px=number(3, 3, 31, 2, "Width of the window (odd; even: one wider)", "px"),
)
def median_filter(state, size_px):
    """Replace each pixel by the median of the window around it: removes single
    bright or dark pixels and keeps straight edges where they are. It flattens
    the top of a feature narrower than about twice the window: a hole of four
    pixels loses its peak even at 3, so look at what it does to the contours."""
    size = int(round(size_px * state.scale)) | 1
    image = ndimage.median_filter(state.image, size=size, mode="nearest")
    return replace(state, image=image, note="")


@module(
    "Remove stripes",
    "Image",
    rows=flag(True, "Remove offsets that differ from row to row"),
    columns=flag(True, "Remove offsets that differ from column to column"),
    smooth_px=number(
        65,
        5,
        2001,
        2,
        "Each line's median is compared with the medians of this many lines "
        "around it: more than a few periods of the pattern",
        "px",
    ),
)
def remove_stripes(state, rows, columns, smooth_px):
    """Subtract a brightness offset that differs from one row or column to the
    next (TDI line noise, column response). Caution on an array aligned with the
    pixel grid: the lines through the hole centres and the lines between them
    have different medians by themselves, and are then levelled as if that were
    a stripe. Look at the image this step makes before trusting it."""
    image = ops.remove_stripes(state.image, rows, columns, smooth_px * state.scale)
    return replace(state, image=image, note="")


@module(
    "Flatten background",
    "Image",
    tile_px=number(
        128, 8, 4096, 8, "Size of the tiles the background level is taken from", "px"
    ),
    bright=flag(True, BRIGHT),
)
def flatten_background(state, tile_px, bright):
    """Subtract the background level where it varies slowly across the image
    (flare, a detector offset that drifts), so that one grey level can divide
    pattern from background everywhere: the 5th percentile of each tile (the
    95th for a dark pattern), interpolated between the tiles. The contrast is
    not changed: where the illumination itself varies, use the Mid-level
    threshold with local tiles instead."""
    percent = 5.0 if bright else 95.0
    tile = tile_px * state.scale
    low, _, ys, xs = ops.tile_levels(state.image, tile, percent, percent)
    background = ops.expand_tiles(low, ys, xs, state.image.shape)
    image = state.image - background + np.float32(background.mean())
    spread = float(low.max() - low.min())
    return replace(state, image=image, note=f"background varied by {grey(spread)}")


@module(
    "Normalize contrast",
    "Image",
    tile_px=number(
        128, 8, 4096, 8, "Size of the tiles the levels are taken from", "px"
    ),
    low_percent=number(
        5.0, 0, 50, 0.5, "Percentile of a tile taken as its dark level", "%"
    ),
    high_percent=number(
        99.5, 50, 100, 0.5, "Percentile of a tile taken as its bright level", "%"
    ),
    min_contrast=number(
        16.0,
        0,
        1000,
        0.5,
        "A tile whose two levels differ by less than this many noise sigma has no "
        "pattern: its contrast is taken as this much, so that its noise is not "
        "stretched to full scale (at 16 it stays below 0.5)",
        "sigma",
    ),
)
def normalize_contrast(state, tile_px, low_percent, high_percent, min_contrast):
    """Rescale the grey levels so that everywhere the local dark level becomes 0
    and the local bright level 1: removes illumination that varies across the
    image (or from page to page) from the image itself. A Fixed threshold at
    0.5 after it is the same as the Mid-level threshold with local tiles."""
    tile = tile_px * state.scale
    low, high, ys, xs = ops.tile_levels(state.image, tile, low_percent, high_percent)
    noise = noise_of(state)
    contrast = np.maximum(high - low, min_contrast * noise)
    contrast = np.maximum(contrast, 1e-12)
    blank = int((high - low < min_contrast * noise).sum())
    shape = state.image.shape
    image = state.image - ops.expand_tiles(low, ys, xs, shape)
    image /= ops.expand_tiles(contrast, ys, xs, shape)
    note = f"contrast {grey(contrast.min())} to {grey(contrast.max())}"
    if blank:
        note += f", {blank} of {contrast.size} tiles without pattern"
    return replace(
        state, image=image, noise=noise / float(np.median(contrast)), note=note
    )


@module(
    "Deblur",
    "Image",
    sigma_x_px=number(
        0.7, 0, 5, 0.05, "Sigma of the blur along the rows (across the scan)", "px"
    ),
    sigma_y_px=number(
        0.7, 0, 5, 0.05, "Sigma of the blur down the columns (the TDI scan)", "px"
    ),
    iterations=number(
        8, 1, 100, 1, "Richardson-Lucy iterations: more sharpens more, and noise too"
    ),
)
def deblur(state, sigma_x_px, sigma_y_px, iterations):
    """Take part of the optical blur out of the image again (Richardson-Lucy
    deconvolution with a Gaussian): holes come out with steeper edges and
    flatter tops, so a threshold is less sensitive to how blurred an
    acquisition was. Too large a sigma or too many iterations ring around the
    holes and amplify noise: compare the contours with the step unticked."""
    f = state.scale
    image = ops.richardson_lucy(state.image, sigma_x_px * f, sigma_y_px * f, iterations)
    noise = measured_noise(state, image)
    grown = f"noise x {noise / state.noise:.1f}" if state.noise > 0 else ""
    return replace(state, image=image, noise=noise, note=grown)


@module(
    "Unsharp mask",
    "Image",
    sigma_px=number(1.0, 0.1, 20, 0.1, "Sigma of the blur that is subtracted", "px"),
    amount=number(1.0, 0, 10, 0.1, "How much of the difference is added back: 0: none"),
)
def unsharp_mask(state, sigma_px, amount):
    """Sharpen the image by adding the difference between it and a blurred copy
    of itself: steeper edges at the price of more noise and of an overshoot
    beside every edge. Simpler and faster than Deblur, and less faithful."""
    soft = ops.blur(state.image, sigma_px * state.scale)
    image = state.image + np.float32(amount) * (state.image - soft)
    return replace(state, image=image, noise=measured_noise(state, image), note="")


@module(
    "Band pass",
    "Image",
    small_px=number(
        0.5, 0, 20, 0.1, "Sigma that removes detail finer than the pattern", "px"
    ),
    large_px=number(
        5.0, 0.5, 500, 0.5, "Sigma that removes changes wider than the pattern", "px"
    ),
)
def band_pass(state, small_px, large_px):
    """Keep only detail of the pattern's own size: a lightly blurred copy of the
    image minus a strongly blurred one (difference of Gaussians). Pixel noise
    and slow brightness changes both go. The grey levels are no longer those of
    the detector and dark rings appear around bright features: put Edge image
    before it, to find the pattern here and measure its edges there."""
    f = state.scale
    image = ops.blur(state.image, small_px * f) - ops.blur(state.image, large_px * f)
    noise = state.noise * ops.blur_noise_gain(small_px)
    return replace(state, image=image, noise=noise, note="")


@module(
    "Non-local means",
    "Image",
    strength=number(
        1.0, 0.1, 10, 0.1, "Filtering strength, in times the image's noise sigma"
    ),
    patch_px=number(3, 1, 9, 2, "Width of the surroundings compared (odd)", "px"),
    search_px=number(11, 3, 31, 2, "Width of the window searched (odd)", "px"),
)
def non_local_means(state, strength, patch_px, search_px):
    """Replace each pixel by a mean of the pixels whose surroundings look like
    its own, which removes noise and keeps the edges of features much larger
    than a pixel. Holes of a few pixels at a spacing that is no whole number of
    pixels have no exact look-alikes (each is shifted by a fraction of a
    pixel), and on synthetic hole arrays a plain blur of 0.5 to 0.8 px gave
    the more repeatable contours: compare before relying on it. Slow: its time
    grows with the square of the search window."""
    if state.scale != 1:
        raise WorkflowError(
            "Non-local means works on the detector's own pixels: put it before Upscale"
        )
    image = ops.non_local_means(
        state.image, noise_of(state), strength, patch_px, search_px
    )
    return replace(state, image=image, note="")


@module(
    "Upscale",
    "Image",
    factor=number(4, 1, 8, 1, "Times as many pixels per side; 1: none"),
    method=choice(
        "cubic spline",
        ops.UPSCALERS,
        "cubic spline: smooth and passes through every pixel value. lanczos: "
        "a little sharper, with slight ringing. linear: no overshoot, blunter",
    ),
)
def upscale(state, factor, method):
    """Interpolate the image to more pixels. Trace contours already places its
    outline between the pixels, so this is only needed for modules that work
    on pixels: Morphology on a finer mask, or thresholds whose mask you want to
    look at in detail. The options in pixels of the modules after it stay
    pixels of the image as loaded. Memory and time grow with the square of the
    factor."""
    before_the_mask(state, "Upscale")
    room_for(state.image.size * factor**2, "Upscale")
    image = ops.upscale(state.image, factor, method)
    kept = state.edge_image
    if kept is not None:
        kept = image if kept is state.image else ops.upscale(kept, factor, method)
    note = f"{image.shape[1]} x {image.shape[0]} px"
    return replace(
        state, image=image, edge_image=kept, scale=state.scale * factor, note=note
    )


@module("Invert", "Image")
def invert(state):
    """The negative of the image: dark becomes bright, within the same range of
    grey levels. (The threshold modules find a dark pattern without it: their
    option bright.)"""
    low, high = float(state.image.min()), float(state.image.max())
    return replace(state, image=(low + high) - state.image, note="")


@module("Edge image", "Image")
def edge_image(state):
    """Keep the image as it is here as the one edges are measured on (Refine
    edges, the contrast of Filter contours). The image modules after this one
    then only help to find the pattern: strong denoising or a band pass there
    cannot move the edges. Without this module the edges are measured on the
    image as it is when they are refined."""
    return replace(
        state, edge_image=state.image, note="the edges are measured on this image"
    )


# --- Mask modules -----------------------------------------------------------

REFERENCES = ("local tiles", "each feature", "whole image")
TILE_PX = 128  # the tile size of the local levels unless told otherwise
OTSU = ("two classes", "three classes: lower", "three classes: upper")
MORPHOLOGY = ("open", "close", "dilate", "erode")
ABOVE_NOISE = (
    "so that where there is no pattern it lies above the noise instead of "
    "through it; 0: off"
)
THRESHOLD_BLUR = (
    "Gaussian blur (sigma) of the image for this threshold only: steadier levels "
    "and outlines, while the image itself stays as it is; 0: none"
)


def with_mask(state, field, note, **changed):
    """The state with the mask of a field (> 0), without the contours of the
    mask before it; its note ends with the share of the image that is mask."""
    mask = field > 0
    share = f"{100 * float(mask.mean()):.1f} % of the image"
    return replace(
        state,
        mask=mask,
        field=field,
        contours=None,
        note=f"{note}, {share}" if note else share,
        **changed,
    )


def changed_mask(state, mask):
    """The state with a mask that is no longer a threshold's: no field, no
    contours."""
    share = f"{100 * float(mask.mean()):.1f} % of the image"
    return replace(state, mask=mask, field=None, contours=None, note=share)


def sampled(image, limit=1_000_000):
    """About `limit` pixels of an image, evenly spread: for percentiles."""
    stride = max(1, int(np.sqrt(image.size / limit)))
    return image[::stride, ::stride]


@module(
    "Mid-level threshold",
    "Mask",
    reference=choice(
        "local tiles",
        REFERENCES,
        "Where the dark and bright levels come from. local tiles: two percentiles "
        "of every tile of the image, interpolated between the tiles: follows "
        "illumination that varies across the image, and compares every hole with "
        "the bright level of its neighbourhood, so a smaller, fainter hole gives a "
        "smaller contour. each feature: the darkest and brightest value within "
        "about one pattern spacing: every hole against its own peak, which hides "
        "part of a size difference but needs no neighbours. whole image: two "
        "percentiles of the image, one level everywhere",
    ),
    fraction=number(
        0.5,
        0.05,
        0.95,
        0.05,
        "The threshold is this share of the way from the background level to the "
        "pattern's level: 0.5 is the half-maximum outline; lower follows the foot "
        "of the profile, higher its top",
    ),
    window_px=number(
        0,
        0,
        4096,
        1,
        "local tiles: size of a tile (0: 128 px). Smaller follows illumination "
        "more closely, and takes more of a regional difference in hole size for "
        "illumination. each "
        "feature: width of the neighbourhood (0: the pattern's spacing, measured "
        "from the image)",
        "px",
    ),
    background_percent=number(
        5.0,
        0,
        50,
        0.5,
        "Percentile taken as the background level (tiles, whole image), counted "
        "from the background's end of the grey levels",
        "%",
    ),
    pattern_percent=number(
        99.5,
        50,
        100,
        0.5,
        "Percentile taken as the pattern's level, counted the same way: high "
        "enough to reach the top of the holes even where they are sparse",
        "%",
    ),
    above_noise=number(
        6.0,
        0,
        1000,
        0.5,
        "The threshold is never set less than this many noise sigma above the "
        f"background level, {ABOVE_NOISE}. A pattern so faint that its threshold "
        "would be lower is drawn smaller, or lost",
        "sigma",
    ),
    blur_px=number(0.5, 0, 20, 0.1, THRESHOLD_BLUR, "px"),
    bright=flag(True, BRIGHT),
)
def mid_level_threshold(
    state,
    reference,
    fraction,
    window_px,
    background_percent,
    pattern_percent,
    above_noise,
    blur_px,
    bright,
):
    """Make the mask where the image is above a share of the way from its dark
    level to its bright level, both taken from the image itself: the threshold
    follows gain, offset and illumination, which a fixed grey level does not.
    Images of the same pattern taken at different brightness give the same
    outline. The module for ACTIS hole arrays: start here."""
    f = state.scale
    sign = 1.0 if bright else -1.0
    # A dark pattern is found as a bright one in the negative of the image.
    signed = ops.blur(state.image, blur_px * f) * np.float32(sign)
    # the noise the threshold sees: the image's, lowered by this module's blur
    floor = np.float32(above_noise * noise_of(state) * ops.blur_noise_gain(blur_px))
    percents = (background_percent, pattern_percent)
    share = np.float32(fraction)
    level = np.nan
    if reference == "whole image":
        low, high = np.percentile(sampled(signed), percents)
        cut = np.float32(low + max(fraction * (high - low), floor))
        field = signed - cut
        level = sign * float(cut)
        note = f"level {grey(level)}"
    elif reference == "local tiles":
        tile = (window_px or TILE_PX) * f
        low, high, ys, xs = ops.tile_levels(signed, tile, *percents)
        rise = share * (high - low)
        cuts = low + np.maximum(rise, floor)
        field = signed - ops.expand_tiles(cuts, ys, xs, signed.shape)
        shown = np.sort(sign * np.array([cuts.min(), cuts.max()]))
        note = f"levels {grey(shown[0])} to {grey(shown[1])}"
        blank = int((rise < floor).sum())
        if blank:
            note += f", {blank} of {low.size} tiles without pattern"
    else:
        window = window_px * f
        if not window:
            window = ops.estimate_pitch(state.image) or 12 * f
        low, high = ops.feature_levels(signed, window)
        field = signed - (low + np.maximum(share * (high - low), floor))
        note = f"neighbourhood of {window / f:.1f} px"
    return with_mask(state, field, note, bright=bright, level=level)


@module(
    "Otsu threshold",
    "Mask",
    classes=choice(
        "two classes",
        OTSU,
        "two classes: pattern | background, the threshold between them. three "
        "classes: lower / upper: the grey levels fall into three groups (for "
        "instance background | edges and faint features | bright cores) and the "
        "threshold is the lower or the upper of the two between them",
    ),
    above_noise=number(
        6.0,
        0,
        1000,
        0.5,
        "The threshold is never set less than this many noise sigma away from "
        f"the mean of the background's pixels, {ABOVE_NOISE}. (Otsu divides any "
        "image in two, an empty one too.)",
        "sigma",
    ),
    blur_px=number(0.5, 0, 20, 0.1, THRESHOLD_BLUR, "px"),
    bright=flag(True, BRIGHT),
)
def otsu_threshold(state, classes, above_noise, blur_px, bright):
    """Make the mask by Otsu's threshold: the one grey level that best divides
    the image's pixels into classes, found from its histogram. One level for
    the whole image, so it follows a difference in hole size most directly, and
    a difference in illumination too: on an uneven image put Flatten background
    before it, or use the Mid-level threshold."""
    sign = 1.0 if bright else -1.0
    # A dark pattern is found as a bright one in the negative of the image.
    signed = ops.blur(state.image, blur_px * state.scale) * np.float32(sign)
    values = sampled(signed)
    levels = ops.otsu_levels(values, 2 if classes == OTSU[0] else 3)
    cut = levels[-1 if (classes == OTSU[2]) == bright else 0]
    below = values[values <= cut]
    floor = above_noise * noise_of(state) * ops.blur_noise_gain(blur_px)
    guarded = max(cut, float(below.mean()) + floor) if below.size else cut
    level = sign * guarded
    note = f"level {grey(level)}" + (
        " (raised above the noise)" if guarded > cut else ""
    )
    field = signed - np.float32(guarded)
    return with_mask(state, field, note, bright=bright, level=level)


@module(
    "Fixed threshold",
    "Mask",
    level=number(
        128.0,
        -1e9,
        1e9,
        1,
        "The grey level; the readout under the image shows the level under the cursor",
        "DN",
    ),
    blur_px=number(0.0, 0, 20, 0.1, THRESHOLD_BLUR, "px"),
    bright=flag(True, BRIGHT),
)
def fixed_threshold(state, level, blur_px, bright):
    """Make the mask at a grey level you choose. The same level for every
    image: for images of the same brightness and contrast, or after Normalize
    contrast (then 0.5 is the half-maximum outline)."""
    smooth = ops.blur(state.image, blur_px * state.scale)
    field = (smooth - np.float32(level)) * np.float32(1.0 if bright else -1.0)
    return with_mask(state, field, "", bright=bright, level=float(level))


@module(
    "Adaptive threshold",
    "Mask",
    block_px=number(
        15.0,
        2,
        2000,
        1,
        "Width of the neighbourhood whose mean a pixel is compared with (a "
        "Gaussian of a quarter this sigma): wider than a feature",
        "px",
    ),
    offset_percent=number(
        5.0,
        -100,
        100,
        0.5,
        "How much brighter than that mean a pixel must be (a dark pattern: "
        "darker), in percent of the image's range of grey levels: more than the "
        "noise",
        "%",
    ),
    bright=flag(True, BRIGHT),
)
def adaptive_threshold(state, block_px, offset_percent, bright):
    """Make the mask by comparing each pixel with the mean of its neighbourhood:
    a threshold that follows any brightness change, and that has no memory of
    what dark and bright are. In flat areas wider than the neighbourhood it
    cuts regions out of the noise unless the offset is above the noise."""
    f = state.scale
    low, high = np.percentile(sampled(state.image), (1, 99))
    offset = np.float32(offset_percent / 100 * (high - low))
    mean = ops.blur(state.image, block_px * f / 4)
    sign = np.float32(1.0 if bright else -1.0)
    field = (state.image - mean) * sign - offset
    return with_mask(state, field, "", bright=bright, level=np.nan)


@module(
    "Morphology",
    "Mask",
    operation=choice(
        "open",
        MORPHOLOGY,
        "open: remove what is narrower than the disk (specks, thin bridges "
        "between regions). close: fill what is narrower (pinholes, gaps). "
        "dilate: grow every region by the disk's radius. erode: shrink it",
    ),
    size_px=number(3.0, 0.25, 999, 0.25, "Width of the disk", "px"),
)
def morphology(state, operation, size_px):
    """Change the mask with a disk before its contours are traced: clean it
    (open, close), or grow and shrink its regions (dilate, erode). The mask is
    then no longer a threshold's, so Trace contours follows its pixels: at four
    pixels per hole that is coarse. Put Upscale first for a finer mask, and
    Refine edges after the tracing to put the outline back on the edges."""
    mask = need_mask(state, "Morphology")
    side = max(3, int(round(size_px * state.scale)) | 1)
    shape = ops.disk(side)
    act = {
        "open": ndimage.binary_opening,
        "close": ndimage.binary_closing,
        "dilate": ndimage.binary_dilation,
        "erode": ndimage.binary_erosion,
    }[operation]
    return changed_mask(state, act(mask, structure=shape))


@module("Fill holes", "Mask")
def fill_holes(state):
    """Fill every hole in the mask (the pixels a region encloses on all
    sides), so that each region has an outer contour only. As after Morphology,
    Trace contours then follows the mask's pixels."""
    mask = need_mask(state, "Fill holes")
    return changed_mask(state, ndimage.binary_fill_holes(mask))


# --- Contours modules -------------------------------------------------------

LEFT_OUT = {  # the reasons a candidate contour is left out, in words
    "small": "too small",
    "large": "too large",
    "border": "cut by the image border",
    "inner": "inner boundaries",
    "open": "open",
    "weak": "weak edges",
    "short": "too short",
    "elongated": "elongated",
    "size outlier": "of unusual size",
    "low contrast": "of low contrast",
    "near border": "near the image border",
    "degenerate": "without a closed outline",
}
SPACING = "Distance between neighbouring points of a contour"


def described(contours):
    """How many contours there are, closed and open, and how many candidates
    were left out and why, in words."""
    closed = int(contours.closed.sum())
    text = f"{len(contours)} contours ({closed} closed, {len(contours) - closed} open)"
    left = [
        f"{count} {LEFT_OUT.get(why, why)}"
        for why, count in contours.left_out.items()
        if count
    ]
    return text + (f"; left out: {', '.join(left)}" if left else "")


def first_reason(contours, reasons):
    """Contours without those a reason applies to; each one left out is counted
    under the first of its reasons, in the order given."""
    keep = np.ones(len(contours), bool)
    counts = {}
    for name, applies in reasons.items():
        applies = applies & keep
        counts[name] = int(applies.sum())
        keep &= ~applies
    return contours.select(keep, **counts)


@module(
    "Trace contours",
    "Contours",
    min_area_px=number(
        3.0, 0, 1e8, 1, "Closed contours of less area are left out: noise", "px²"
    ),
    max_area_px=number(
        0.0, 0, 1e9, 10, "Closed contours of more area are left out; 0: no limit", "px²"
    ),
    holes=flag(
        True,
        "The boundaries of holes inside a region too (a ring gives two contours); "
        "off: outer boundaries only",
    ),
    open_at_border=flag(
        True,
        "A region that runs out of the image gives open contours ending at the "
        "image border (lines, features cut off); off: it is left out (whole "
        "features only)",
    ),
    spacing_px=number(0.25, 0.05, 10, 0.05, SPACING, "px"),
)
def trace_contours(state, min_area_px, max_area_px, holes, open_at_border, spacing_px):
    """Make contours of the mask's regions. After a threshold the outline is
    traced between the pixels: where the threshold's field (the image minus its
    level) is zero, on the cubic spline through the pixel values, so a
    four-pixel hole gets a smooth outline placed to a few hundredths of a
    pixel with no upscaling. After a module that changed the mask's pixels
    (Morphology, Fill holes) the outline follows the pixels instead."""
    mask, f = need_mask(state, "Trace contours"), state.scale
    if state.field is not None:
        lines = ops.trace_level(state.field, spacing_px * f)
        how = "between the pixels"
    else:
        outline = mask.astype(np.float32) - np.float32(0.5)
        lines = ops.trace_level(outline, spacing_px * f, subpixel=False)
        how = "along the mask's pixels"
    closed, area = lines.closed, np.nan_to_num(geometry.areas(lines))
    size = np.abs(area) / f**2
    short = geometry.lengths(lines) / f < 2 * np.sqrt(np.pi * min_area_px)
    kept = first_reason(
        lines,
        {
            "border": ~closed & (not open_at_border),
            "small": np.where(closed, size < min_area_px, short),
            "inner": closed & (area < 0) & (not holes),
            "large": closed & (size > max_area_px) & (max_area_px > 0),
        },
    )
    return replace(state, contours=kept, note=f"{described(kept)}; {how}")


@module(
    "Trace gradient edges",
    "Contours",
    blur_px=number(
        0.8,
        0.3,
        20,
        0.1,
        "Sigma of the Gaussian the derivatives are taken with: larger finds less "
        "noise and rounds small features more",
        "px",
    ),
    oversample=number(
        3,
        1,
        6,
        1,
        "The image is interpolated to this many times as many pixels per side "
        "before the derivatives are taken; more is finer and slower",
    ),
    high=number(
        0.3,
        0.01,
        1,
        0.01,
        "An edge needs a point where the brightness changes by more than this "
        "share of the steepest changes in the image",
    ),
    low_ratio=number(
        0.5,
        0.05,
        1,
        0.05,
        "The other points of the edge need this share of `high`: lower follows an "
        "edge further into the noise",
    ),
    above_noise=number(
        6.0,
        0,
        1000,
        0.5,
        "`high` is never taken less than this many times the change that noise "
        "alone gives, so that an image without pattern gives no edges; 0: off",
        "sigma",
    ),
    min_length_px=number(6.0, 0, 1e5, 1, "Shorter edges are left out: noise", "px"),
    closed_only=flag(True, "Keep only edges that close on themselves (whole features)"),
    bright=flag(True, BRIGHT),
    spacing_px=number(0.25, 0.05, 10, 0.05, SPACING, "px"),
)
def trace_gradient_edges(
    state,
    blur_px,
    oversample,
    high,
    low_ratio,
    above_noise,
    min_length_px,
    closed_only,
    bright,
    spacing_px,
):
    """Make contours without any threshold: the lines where the brightness
    changes fastest (the zero crossings of the second derivative across the
    edge, which is where Canny's detector puts its edges), found between the
    pixels and kept where the change is strong. It needs no dark or bright
    level, so offset and illumination do not matter; for holes only a few
    pixels across the steepest slope is set by the blur of the optics more than
    by the hole, so these contours follow hole size less than a threshold's."""
    f = state.scale
    room_for(state.measured.size * oversample**2, "Trace gradient edges")
    # Noise alone gives a Gaussian derivative of this sigma, in grey levels per pixel.
    by_noise = noise_of(state) / (np.sqrt(8 * np.pi) * blur_px**2) / f
    lines, upper = ops.gradient_edges(
        state.measured,
        blur_px * f,
        oversample,
        high,
        low_ratio,
        min_length_px * f,
        spacing_px * f,
        floor=above_noise * by_noise,
    )
    if not bright:  # the darker side is the region: walk every line the other way
        pieces = [points[::-1] for points in lines.split()]
        lines = geometry.from_list(pieces, lines.closed, lines.left_out)
    kept = first_reason(lines, {"open": ~lines.closed & closed_only})
    note = f"{described(kept)}; above {grey(upper * f)} grey levels / px"
    return replace(
        state, contours=kept, mask=None, field=None, bright=bright, note=note
    )


@module(
    "Radial contours",
    "Contours",
    pitch_px=number(
        0.0,
        0,
        1000,
        0.1,
        "Spacing of neighbouring holes; 0: the median distance between "
        "neighbouring centres",
        "px",
    ),
)
def radial_contours(state, pitch_px):
    """The contour method of this repository's benchmark (actis.estimators),
    for bright holes: around every hole centre the image is sampled along 40
    rays, and the contour is where each ray falls through half-way between the
    hole's own bright core and its own dark surround. The centres are those of
    the contours so far, or, with none, the bright islands the benchmark's
    discovery finds. For comparing the benchmark's contours with the others."""
    from ..discovery import detect_centers
    from ..estimators import estimate_holes

    f = state.scale
    image = state.measured
    if state.contours is not None and state.contours.closed.any():
        centres = geometry.measure(state.contours)["centre"][state.contours.closed]
    else:
        try:
            centres = detect_centers(image)
        except ValueError as error:
            raise WorkflowError(f"Radial contours: {error}") from error
    if len(centres) < 2:
        raise WorkflowError("Radial contours found fewer than two hole centres")
    pitch = pitch_px * f
    if not pitch:
        pitch = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]))
    if pitch < 3:
        raise WorkflowError(
            f"Radial contours needs holes at least 3 px apart, not {pitch:.2f}: "
            "set pitch_px"
        )
    rows = estimate_holes(image, centres, pitch_px=pitch, methods=("contour",))
    rows = rows["contour"]
    flags = {
        "border": "near border",
        "low_contrast": "low contrast",
        "open_contour": "open",
        "degenerate_contour": "degenerate",
    }
    left_out = {}
    for row in rows:
        if not row["valid"]:
            reason = flags.get(row["flag"], row["flag"])
            left_out[reason] = left_out.get(reason, 0) + 1
    points = [row["contour_xy"] for row in rows if row["valid"]]
    kept = geometry.from_list(points, True, left_out)
    note = f"{described(kept)}; spacing {pitch / f:.2f} px"
    return replace(state, contours=kept, bright=True, note=note)


@module(
    "Refine edges",
    "Contours",
    rule=choice(
        "half level",
        ops.RULES,
        "Where the edge is. half level: where the image is `fraction` of the way "
        "from the contour's own surround up to its own inside level, both found "
        "along its normals: every feature by its own two levels. steepest "
        "gradient: where the brightness changes fastest: no levels needed, but "
        "on holes of a few pixels the blur decides it more than the hole. "
        "threshold level: where the image passes the grey level the mask was cut "
        "at (Otsu, Fixed, or Mid-level of the whole image)",
    ),
    search_px=number(
        1.5,
        0.25,
        100,
        0.25,
        "Each point is searched this far in and out along its normal: about the "
        "distance from a hole's edge to its centre, and less than half the gap "
        "to the next feature",
        "px",
    ),
    step_px=number(0.125, 0.01, 5, 0.005, "Sampling step along the normal", "px"),
    blur_px=number(
        0.5, 0, 20, 0.1, "Gaussian blur of the image the edges are found on", "px"
    ),
    fraction=number(
        0.5, 0.05, 0.95, 0.05, "half level: the share of the way up to the inside level"
    ),
)
def refine_edges(state, rule, search_px, step_px, blur_px, fraction):
    """Move every contour point along its normal to its edge in the grey-level
    image, between the pixels. Needed after a module that left the outline on
    the mask's pixels (Morphology, Fill holes), to measure the edges on another
    image than the pattern was found in (Edge image), or to change what counts
    as the edge. A point without an edge, or far off its neighbours', takes its
    neighbours' place."""
    contours, f = need_contours(state, "Refine edges"), state.scale
    if rule == "threshold level" and not np.isfinite(state.level):
        raise WorkflowError(
            "Refine edges: threshold level needs the grey level of an Otsu or "
            "Fixed threshold, or of a Mid-level threshold of the whole image: "
            "use half level or steepest gradient"
        )
    refined = ops.refine(
        state.measured,
        contours,
        rule,
        search_px * f,
        step_px * f,
        blur_px * f,
        fraction,
        state.level,
        state.bright,
        reach=0.75 * f,
        tolerance=0.3 * f,
    )
    return replace(state, contours=refined, note=rule)


@module(
    "Filter contours",
    "Contours",
    closed_only=flag(
        False, "Leave out the open contours (features cut by the image border)"
    ),
    min_diameter_px=number(
        0.0,
        0,
        1e5,
        0.1,
        "Leave out closed contours whose equal-area diameter is less; 0: off",
        "px",
    ),
    max_diameter_px=number(
        0.0,
        0,
        1e5,
        0.1,
        "Leave out closed contours whose diameter is more; 0: off",
        "px",
    ),
    max_aspect=number(
        0.0,
        0,
        100,
        0.1,
        "Leave out closed contours longer than this many times their width "
        "(merged neighbours, streaks); 0: off",
    ),
    size_outliers=number(
        0.0,
        0,
        100,
        0.5,
        "Leave out closed contours whose diameter is further than this many "
        "robust sigma from the median of all (merged pairs, fragments, "
        "particles); 0: off. In an image with patterns of several sizes this "
        "leaves out the rarer sizes",
        "sigma",
    ),
    min_contrast=number(
        0.0,
        0,
        1000,
        0.5,
        "Leave out contours across which the grey level changes by less than "
        "this many noise sigma (1 px inside against 1 px outside); 0: off",
        "sigma",
    ),
    border_px=number(
        0.0, 0, 1e4, 1, "Leave out contours nearer than this to the image border", "px"
    ),
)
def filter_contours(
    state,
    closed_only,
    min_diameter_px,
    max_diameter_px,
    max_aspect,
    size_outliers,
    min_contrast,
    border_px,
):
    """Leave out the contours that are not the pattern you want: open ones,
    ones of the wrong size or shape, faint ones. Each one left out is counted
    under the first of these reasons that applies."""
    contours, f = need_contours(state, "Filter contours"), state.scale
    measures = geometry.measure(contours)
    closed = contours.closed
    diameter = np.nan_to_num(measures["diameter"]) / f
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = np.nan_to_num(measures["major"] / measures["minor"], posinf=np.inf)
    reasons = {
        "open": ~closed & closed_only,
        "small": closed & (diameter < min_diameter_px),
        "large": closed & (diameter > max_diameter_px) & (max_diameter_px > 0),
        "elongated": closed & (aspect > max_aspect) & (max_aspect > 0),
    }
    if size_outliers > 0 and closed.sum() > 2:
        middle = np.median(diameter[closed])
        sigma = 1.4826 * np.median(np.abs(diameter[closed] - middle))
        far = np.abs(diameter - middle) > size_outliers * max(sigma, 1e-6)
        reasons["size outlier"] = closed & far
    if min_contrast > 0 and len(contours):
        step = ops.edge_contrast(state.measured, contours, f, 0.75 * f)
        step = step if state.bright else -step
        reasons["low contrast"] = step < min_contrast * noise_of(state)
    if border_px > 0 and len(contours):
        rows, cols = state.image.shape
        first = contours.start[:-1]
        low = np.minimum.reduceat(contours.xy, first, axis=0).min(axis=1)
        high = np.maximum.reduceat(contours.xy, first, axis=0)
        room = np.minimum(low, np.minimum(cols - 1 - high[:, 0], rows - 1 - high[:, 1]))
        reasons["near border"] = room < border_px * f
    kept = first_reason(contours, reasons)
    return replace(state, contours=kept, note=described(kept))


@module(
    "Smooth contours",
    "Contours",
    window=number(
        9,
        5,
        999,
        2,
        "Contour points each cubic is fitted to (odd): scatter over fewer points "
        "than about half of it goes. At the default spacing 9 points are 2 px",
    ),
)
def smooth_contours(state, window):
    """Take the point-to-point scatter out of the contours: each point is put
    on a cubic fitted to the points around it along its contour
    (Savitzky-Golay), which follows a curved edge without cutting it short, so
    closed contours do not shrink. It smooths real roughness shorter than the
    window too."""
    contours = need_contours(state, "Smooth contours")
    smooth = geometry.smooth(contours, window)
    return replace(state, contours=smooth, note=f"cubics over {window | 1} points")


@module(
    "Simplify shape",
    "Contours",
    harmonics=number(
        3,
        1,
        30,
        1,
        "1: the ellipse with the outline's centre, orientation and axis ratio. "
        "More: that many Fourier harmonics of the outline are kept: 2 to 4 give "
        "a rounded polygon, more follow the outline in more detail",
    ),
)
def simplify_shape(state, harmonics):
    """Reduce every closed contour to a simple shape of the same area: an
    ellipse, or the lowest harmonics of its outline (Fourier descriptors). A
    hole of four pixels carries only a few numbers' worth of shape; the rest of
    a free outline is noise. The area, and so the diameter, stays what it was;
    width, height and orientation become those of the simple shape. Open
    contours stay as they are."""
    contours = need_contours(state, "Simplify shape")
    simple = geometry.fourier_shape(contours, harmonics)
    note = "ellipses" if harmonics <= 1 else f"{harmonics} harmonics"
    return replace(state, contours=simple, note=note)
