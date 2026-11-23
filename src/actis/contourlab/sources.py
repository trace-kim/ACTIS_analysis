"""Images for the contour lab: PNG files, and TIFF files page by page or stacked.

A board stack of the instrument is one TIFF of up to 1,500 pages, each page the
next stretch of a scan. page() gives one of them; stack() joins consecutive
pages below each other into one long strip. Grey levels are kept as stored
(8 or 16 bit, or float), with brighter always the higher number.
"""

import contextlib
import threading
from pathlib import Path

import numpy as np
from PIL import Image

SUFFIXES = (".png", ".tif", ".tiff")
MAX_STACK_PIXELS = 400_000_000  # of one stacked strip: 0.4 GB at 8 bit


def _grey(pixels):
    """A page as a 2-D array: colour is averaged, a lone channel unwrapped."""
    pixels = np.asarray(pixels)
    if pixels.ndim == 3:
        if pixels.shape[2] in (2, 4):  # with an alpha channel: not grey level
            pixels = pixels[:, :, :-1]
        kind = pixels.dtype
        pixels = pixels.mean(axis=2)
        if kind.kind in "ui":
            pixels = np.rint(pixels).astype(kind)
    if pixels.ndim != 2:
        raise ValueError("a page must be a single grey-level image")
    if pixels.dtype == bool:
        return pixels.astype(np.uint8) * 255
    if pixels.dtype.kind == "f":
        return np.nan_to_num(pixels.astype(np.float32, copy=False))
    if pixels.dtype.kind == "i":
        low, high = int(pixels.min()), int(pixels.max())
        fits = np.uint8 if high <= 255 else np.uint16 if high <= 65535 else None
        if low >= 0 and fits is not None:  # 16-bit PNG read as 32-bit integers
            return pixels.astype(fits)
        return pixels.astype(np.float32)
    if pixels.dtype.kind != "u" or pixels.dtype.itemsize > 2:
        return pixels.astype(np.float32)
    return pixels.astype(pixels.dtype.newbyteorder("="), copy=False)


class ImageFile:
    """An image file opened to read its pages; close() it when done.

    pages: how many it has (a PNG: 1). shape, dtype: of its first page, which
    stack() needs the others to share. notes: what was changed on reading.
    """

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"there is no file {self.path}")
        if self.path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"{self.path.name}: open a PNG or TIFF image")
        self.name = self.path.name
        self.notes = []
        self._lock = threading.Lock()  # one reader at a time on the open file
        self._tiff = self._picture = None
        if self.path.suffix.lower() != ".png":
            try:
                import tifffile
            except ImportError:  # Pillow reads plain TIFFs too
                tifffile = None
            if tifffile is not None:
                # Whatever a reader raises on a file it cannot make sense of is
                # told as one kind of error, with the file's name.
                try:
                    self._tiff = tifffile.TiffFile(self.path)
                    self.pages = len(self._tiff.pages)
                except Exception as error:
                    self.close()
                    reason = f"{self.name}: not a readable TIFF ({error})"
                    raise ValueError(reason) from error
        if self._tiff is None:
            try:
                self._picture = Image.open(self.path)
                self.pages = int(getattr(self._picture, "n_frames", 1))
            except Exception as error:
                self.close()
                reason = f"{self.name}: not a readable image ({error})"
                raise ValueError(reason) from error
        if self.pages < 1:
            self.close()
            raise ValueError(f"{self.name}: not a readable image (it has no pages)")
        try:
            first = self.page(0)
        except ValueError:
            self.close()
            raise
        self.shape, self.dtype = first.shape, first.dtype

    def page(self, index):
        """Page `index` (from 0) as a 2-D array, brighter = higher.

        A ValueError says so if the file has no such page, or the page cannot
        be read (a compression the installed readers do not know, a damaged
        file).
        """
        if not 0 <= index < self.pages:
            raise ValueError(
                f"{self.name} has pages 1 to {self.pages}, not {index + 1}"
            )
        with self._lock:
            try:
                pixels = self._read(index)
            except Exception as error:
                reason = f"page {index + 1} of {self.name} cannot be read: {error}"
                raise ValueError(reason) from error
        return np.ascontiguousarray(pixels)

    def _read(self, index):
        if self._tiff is not None:
            page = self._tiff.pages[index]
            pixels = _grey(page.asarray())
            if getattr(page, "photometric", None) == 0 and pixels.dtype.kind == "u":
                pixels = np.iinfo(pixels.dtype).max - pixels  # white is zero
                self._note("stored with white as zero: shown inverted")
            return pixels
        self._picture.seek(index)
        picture = self._picture
        if picture.mode in ("P", "PA", "CMYK", "YCbCr", "LAB", "HSV"):
            picture = picture.convert("RGB")
        return _grey(np.asarray(picture))

    def stack(self, first, count, reverse=False, flip=False):
        """`count` pages from page `first`, joined below each other into a strip.

        reverse: the last of them at the top (a stripe scanned upwards stores
        its pages in that order). flip: every page upside down (as it stores
        them). A ValueError says so if the pages differ in width or kind, or
        if the strip would be too large.
        """
        if count < 1 or first < 0 or first + count > self.pages:
            raise ValueError(
                f"{self.name} has pages 1 to {self.pages}: cannot stack {count} "
                f"from page {first + 1}"
            )
        rows, cols = self.shape
        if rows * cols * count > MAX_STACK_PIXELS:
            most = max(1, MAX_STACK_PIXELS // (rows * cols))
            raise ValueError(
                f"{count} pages of {cols} x {rows} px are too many to stack at "
                f"once: stack {most} or fewer"
            )
        order = range(first, first + count)
        strip = np.empty((rows * count, cols), self.dtype)
        for place, index in enumerate(reversed(order) if reverse else order):
            page = self.page(index)
            if page.shape != self.shape or page.dtype != self.dtype:
                raise ValueError(
                    f"page {index + 1} of {self.name} is {page.shape[1]} x "
                    f"{page.shape[0]} px {page.dtype}, not {cols} x {rows} px "
                    f"{self.dtype} as page 1: such pages cannot be stacked"
                )
            strip[place * rows : (place + 1) * rows] = page[::-1] if flip else page
        return strip

    def _note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def close(self):
        for handle in (self._tiff, self._picture):
            if handle is not None:
                with contextlib.suppress(Exception):  # closed already, or never whole
                    handle.close()
        self._tiff = self._picture = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_image(path, page=0):
    """One page of an image file as a 2-D array (a PNG: its only page)."""
    with ImageFile(path) as file:
        return file.page(page)


def display_range(image, low=0.5, high=99.5):
    """(black, white): the grey levels a display shows as black and white, two
    percentiles of about a million of the image's pixels."""
    stride = max(1, int(np.sqrt(image.size / 1_000_000)))
    part = image[::stride, ::stride]
    black, white = np.percentile(part, (low, high))
    if not white > black:
        black, white = float(part.min()), float(part.max())
    if not white > black:
        white = black + 1
    return float(black), float(white)
