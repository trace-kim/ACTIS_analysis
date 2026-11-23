"""The contour lab's session: the images that are open, and for each what the
newest run of a workflow made of it.

A Tab is one image file as it is shown: one of its pages, or several stacked
into a strip. It runs workflows on that image (or a region of it) in a
background thread, always the newest one asked for, and keeps the result in the
form a window draws: every image and mask as tiles at halving resolutions
(Layer), every set of contours as one packed buffer with its statistics
(ContourSet). Nothing here knows about HTTP; server.py only passes it on.
"""

import csv
import io
import json
import os
import shutil
import string
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np

from .. import __version__
from . import contours as geometry
from .contours import MEASURES, Contours
from .modules import LEFT_OUT, MODULES, STAGES
from .sources import SUFFIXES, ImageFile, display_range
from .workflow import PRESETS, START, Run, run_workflow, workflow_data, workflow_steps

TILE = 256  # pixels per side of a tile
MAX_REGION_PIXELS = 24_000_000  # the most a workflow is run on at once
WHOLE_IMAGE_PIXELS = 6_000_000  # up to here a workflow runs on the whole image unasked
MAX_UPLOAD_BYTES = 2**30


class LabError(ValueError):
    """Something asked of the session that it cannot do; the message is for the user."""


# --- Layers: arrays as tiles ------------------------------------------------


def halved(array):
    """An array at half the resolution: the mean of every 2 x 2 pixels."""
    rows, cols = array.shape
    if rows % 2 or cols % 2:
        array = np.pad(array, ((0, rows % 2), (0, cols % 2)), mode="edge")
    rows, cols = array.shape
    blocks = array.reshape(rows // 2, 2, cols // 2, 2)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


class Layer:
    """A 2-D array a window shows, cut into tiles at halving resolutions.

    scale: pixels of the array per pixel of the image as loaded; origin: the
    (x, y) of that image the array begins at (a region). A mask is served as
    0 and 255, so that its coarser levels say how much of a pixel is mask.
    """

    def __init__(self, key, array, scale=1, origin=(0, 0), mask=False):
        self.id, self.scale, self.origin, self.mask = key, int(scale), origin, mask
        self.of = array  # kept: a tab finds the layer of an array by the array itself
        if mask:
            array = array.astype(np.uint8) * np.uint8(255)
        elif array.dtype not in (np.uint8, np.uint16, np.float32):
            array = array.astype(np.float32)
        self.shape = array.shape
        self._levels = [array]
        self._lock = threading.Lock()
        self.levels = 1
        side = max(array.shape)
        while side > TILE:
            side = (side + 1) // 2
            self.levels += 1
        self.range = (0.0, 255.0) if mask else display_range(array)

    def level(self, number):
        """The array at resolution 1 / 2^number."""
        if not 0 <= number < self.levels:
            raise LabError(f"layer {self.id} has no level {number}")
        with self._lock:
            while len(self._levels) <= number:
                smaller = halved(self._levels[-1])
                if self.mask:
                    smaller = np.rint(smaller).astype(np.uint8)
                self._levels.append(smaller)
            return self._levels[number]

    def tile(self, number, tx, ty):
        """(bytes, dtype, width, height) of tile (tx, ty) of a level, row by row."""
        array = self.level(number)
        rows, cols = array.shape
        if not (0 <= tx * TILE < cols and 0 <= ty * TILE < rows):
            raise LabError(f"layer {self.id} has no tile {tx}, {ty} at level {number}")
        part = array[ty * TILE : (ty + 1) * TILE, tx * TILE : (tx + 1) * TILE]
        part = np.ascontiguousarray(
            part.astype(part.dtype.newbyteorder("<"), copy=False)
        )
        return part.tobytes(), part.dtype.name, part.shape[1], part.shape[0]

    def describe(self):
        rows, cols = self.shape
        return {
            "id": self.id,
            "width": cols,
            "height": rows,
            "scale": self.scale,
            "x": self.origin[0],
            "y": self.origin[1],
            "levels": self.levels,
            "mask": self.mask,
            "black": self.range[0],
            "white": self.range[1],
            "tile": TILE,
        }

    @property
    def nbytes(self):
        return sum(level.nbytes for level in self._levels)


# --- Contour sets -----------------------------------------------------------


def finite(value, digits=6):
    """A number for JSON: rounded, None if it is not finite."""
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


class ContourSet:
    """The contours a step made, in the pixels of the image as loaded, with
    their measures, as a window and the exports need them."""

    def __init__(self, key, contours, of=None):
        self.id, self.contours = key, contours
        self.of = of  # the contours of the workflow's state these were made from
        self.measures = geometry.measure(contours)
        self._payload = None
        self._lock = threading.Lock()

    def payload(self):
        """The contours as one buffer for a window (little endian):

        uint32 n, P; uint32 start[n + 1]; uint8 closed[n], padded to 4 bytes;
        float32 xy[2 P]; float32 centre[2 n]; float32 diameter[n];
        float32 box[4 n] (least x, least y, most x, most y).
        """
        with self._lock:
            if self._payload is None:
                c, n = self.contours, len(self.contours)
                first = c.start[:-1]
                if n:
                    low = np.minimum.reduceat(c.xy, first, axis=0)
                    high = np.maximum.reduceat(c.xy, first, axis=0)
                else:
                    low = high = np.empty((0, 2))
                closed = np.zeros(-(-n // 4) * 4, np.uint8)
                closed[:n] = c.closed
                parts = [
                    np.array([n, len(c.xy)], "<u4"),
                    c.start.astype("<u4"),
                    closed,
                    c.xy.astype("<f4"),
                    self.measures["centre"].astype("<f4"),
                    self.measures["diameter"].astype("<f4"),
                    np.hstack([low, high]).astype("<f4"),
                ]
                self._payload = b"".join(part.tobytes() for part in parts)
            return self._payload

    def statistics(self):
        """What a window says of the set: how many, how large, how alike."""
        c, m = self.contours, self.measures
        closed = c.closed & np.isfinite(m["diameter"])
        result = {
            "count": len(c),
            "closed": int(c.closed.sum()),
            "open": int((~c.closed).sum()),
            "points": int(len(c.xy)),
            "left_out": {
                LEFT_OUT.get(why, why): int(count)
                for why, count in c.left_out.items()
                if count
            },
        }
        if not closed.any():
            return result
        for name in ("diameter", "width", "height", "major", "minor"):
            values = m[name][closed]
            middle = float(np.median(values))
            result[name] = {
                "median": finite(middle),
                "mean": finite(values.mean()),
                "sigma": finite(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "robust_sigma": finite(1.4826 * np.median(np.abs(values - middle))),
                "min": finite(values.min()),
                "max": finite(values.max()),
            }
        diameter = m["diameter"][closed]
        spread = result["diameter"]["robust_sigma"] or result["diameter"]["sigma"] or 0
        middle = result["diameter"]["median"]
        low = max(float(diameter.min()), middle - 6 * spread)
        high = min(float(diameter.max()), middle + 6 * spread)
        if not high > low:
            low, high = middle - 0.5, middle + 0.5
        counts, edges = np.histogram(
            diameter, bins=min(48, max(8, len(diameter) // 4)), range=(low, high)
        )
        result["histogram"] = {
            "edges": [finite(edge) for edge in edges],
            "counts": counts.tolist(),
            "below": int((diameter < low).sum()),
            "above": int((diameter > high).sum()),
        }
        return result

    def detail(self, index):
        """Every measure of one contour."""
        if not 0 <= index < len(self.contours):
            raise LabError(f"there is no contour {index}")
        m = self.measures
        row = {name: finite(m[name][index]) for name in MEASURES}
        row.update(
            index=index,
            closed=bool(self.contours.closed[index]),
            points=int(m["points"][index]),
            x=finite(m["centre"][index, 0]),
            y=finite(m["centre"][index, 1]),
        )
        if row["angle"] is not None:
            row["angle"] = finite(np.degrees(m["angle"][index]), 3)
        return row

    def table(self, pixel_nm=None):
        """(header, rows): one row of measures per contour, in pixels, with the
        lengths in nanometres too if the pixel size is given."""
        m, c = self.measures, self.contours
        header = ["contour", "closed", "points", "x_px", "y_px"]
        header += [
            f"{name}_px" + ("2" if name == "area" else "") for name in MEASURES[:6]
        ]
        header += ["angle_deg", "length_px"]
        lengths = ("diameter", "width", "height", "major", "minor")
        if pixel_nm:
            header += [f"{name}_nm" for name in lengths]
        columns = [
            np.arange(len(c)),
            c.closed.astype(int),
            m["points"],
            np.round(m["centre"][:, 0], 4),
            np.round(m["centre"][:, 1], 4),
        ]
        columns += [np.round(m[name], 5) for name in MEASURES[:6]]
        columns += [np.round(np.degrees(m["angle"]), 3), np.round(m["length"], 4)]
        if pixel_nm:
            columns += [np.round(m[name] * pixel_nm, 3) for name in lengths]
        rows = []
        for values in zip(*columns):
            rows.append(
                [
                    "" if isinstance(v, float) and not np.isfinite(v) else v
                    for v in (x.item() for x in values)
                ]
            )
        return header, rows

    def csv(self, pixel_nm=None):
        header, rows = self.table(pixel_nm)
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        return out.getvalue().encode("utf-8")

    def npz(self, **extra):
        """The contours and their measures as the bytes of a NumPy .npz file:
        xy (P, 2), start (N + 1,), closed (N,) and one array per measure;
        np.split(xy, start[1:-1]) gives the contours one by one."""
        out = io.BytesIO()
        arrays = {name: self.measures[name] for name in (*MEASURES, "centre", "points")}
        arrays.update(extra)
        np.savez_compressed(
            out,
            xy=self.contours.xy,
            start=self.contours.start,
            closed=self.contours.closed,
            **arrays,
        )
        return out.getvalue()

    def json(self, **extra):
        """The contours with their points and measures, as a JSON document."""
        header, rows = self.table()
        points = [np.round(p, 4).tolist() for p in self.contours.split()]
        records = []
        for row, xy in zip(rows, points):
            record = {
                name: (None if value == "" else value)
                for name, value in zip(header, row)
            }
            record["closed"] = bool(record["closed"])
            record["xy"] = xy
            records.append(record)
        return json.dumps(dict(extra, contours=records), allow_nan=False).encode(
            "utf-8"
        )


# --- A tab: one image, and the newest run on it -----------------------------


def made_by(run, viewed):
    """The state (0: the image as loaded) whose step made the contours of
    run.states[viewed] what they are; None: it has none."""
    wanted = run.states[viewed].contours
    if wanted is None:
        return None
    return next(k for k in range(viewed + 1) if run.states[k].contours is wanted)


class Tab:
    def __init__(self, key, file, temporary=None, published=None):
        self.id, self.file, self.temporary = key, file, temporary
        self.published = published  # called with the tab after every result
        self.view = dict(
            page=0, stack=False, first=0, count=1, reverse=False, flip=False
        )
        self.changed = threading.Condition()
        self.serial = 0  # counts the results published
        self.result = None  # the newest, as a window gets it
        self.used = time.monotonic()
        self._wanted = None  # (request, steps, region): the newest run asked for
        self._running = False
        self._closed = False
        self._run = None  # the newest finished Run
        self._ran = None  # (steps, region) it ran
        self._crop = (None, None)  # (region, the array of it)
        self._layers = {}  # id(array) -> Layer, of the newest result
        self._sets = {}  # id(Contours) -> ContourSet
        self._old = ({}, {})  # the layers and sets of the result before
        self._number = 0
        self.source = self.source_layer = None
        self._load()

    # - what is shown: a page, or pages stacked -

    def set_view(
        self, page=0, stack=False, first=0, count=1, reverse=False, flip=False
    ):
        """Show page `page`, or (stack) `count` pages from `first` as one strip."""
        view = dict(
            page=int(page),
            stack=bool(stack),
            first=int(first),
            count=int(count),
            reverse=bool(reverse),
            flip=bool(flip),
        )
        if view == self.view:
            return
        previous, self.view = self.view, view
        try:
            self._load()
        except Exception:
            self.view = previous
            raise

    def _load(self):
        view = self.view
        if view["stack"]:
            source = self.file.stack(
                view["first"], view["count"], view["reverse"], view["flip"]
            )
        else:
            source = self.file.page(view["page"])
        extent = (float(source.min()), float(source.max()))
        with self.changed:
            self.source, self.extent = source, extent
            self._number += 1
            self.source_layer = Layer(f"S{self._number}", source)
            self._run = self._ran = self.result = None
            self._crop = (None, None)
            self._old = (self._layers, self._sets)
            self._layers, self._sets = {}, {}
            self.serial += 1
            self.changed.notify_all()

    def describe(self):
        rows, cols = self.source.shape
        view = self.view
        if view["stack"]:
            order = range(view["first"], view["first"] + view["count"])
            pages = list(reversed(order)) if view["reverse"] else list(order)
        else:
            pages = [view["page"]]
        low, high = self.extent
        whole = rows * cols <= WHOLE_IMAGE_PIXELS
        return {
            "id": self.id,
            "name": self.file.name,
            "path": None if self.temporary else str(self.file.path),
            "pages": self.file.pages,
            "page_height": self.file.shape[0],
            "view": dict(view),
            "shown_pages": pages,  # the pages of the strip, from its top
            "width": cols,
            "height": rows,
            "dtype": str(self.source.dtype),
            "min": low,
            "max": high,
            "notes": list(self.file.notes),
            "source": self.source_layer.describe(),
            "serial": self.serial,
            # the region a workflow runs on unless told: the whole image, or the
            # top of a large one
            "region": None if whole else self.default_region(),
            "max_region_pixels": MAX_REGION_PIXELS,
        }

    def default_region(self):
        rows, cols = self.source.shape
        height = min(rows, self.file.shape[0], max(2, WHOLE_IMAGE_PIXELS // cols))
        return [0, 0, cols, height]

    # - running workflows -

    def region(self, region):
        """A region [x0, y0, x1, y1] (None: the whole image) as whole pixels
        inside the image; a LabError if it is empty or too large to run on."""
        rows, cols = self.source.shape
        if region is None:
            region = [0, 0, cols, rows]
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in region)
        except (TypeError, ValueError):
            raise LabError("a region is four numbers: x0, y0, x1, y1") from None
        x0, x1 = sorted((min(max(x0, 0), cols), min(max(x1, 0), cols)))
        y0, y1 = sorted((min(max(y0, 0), rows), min(max(y1, 0), rows)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise LabError("the region is smaller than 8 x 8 px")
        if (x1 - x0) * (y1 - y0) > MAX_REGION_PIXELS:
            raise LabError(
                f"the region is {(x1 - x0) * (y1 - y0) / 1e6:.0f} million pixels: a "
                f"workflow runs on at most {MAX_REGION_PIXELS / 1e6:.0f} million at "
                "once. Choose a smaller region (Shift + drag on the image)"
            )
        return (x0, y0, x1, y1)

    def request(self, number, steps, region=None):
        """Ask for a run of these steps on a region; the newest request wins.
        The result is published when it is done (wait())."""
        region = self.region(region)
        with self.changed:
            if self._closed:
                raise LabError("this image was closed")
            self.used = time.monotonic()
            fresh = self._ran == (steps, region) and self.result is not None
            if fresh and not self._running and self._wanted is None:
                # nothing changed: say so, without running it again
                self.result = dict(self.result, request=number)
                self.serial += 1
                self.changed.notify_all()
                return
            self._wanted = (number, steps, region)
            if not self._running:
                self._running = True
                threading.Thread(target=self._work, daemon=True).start()

    def _newer(self, number):
        wanted = self._wanted
        return self._closed or (wanted is not None and wanted[0] != number)

    def _work(self):
        while True:
            with self.changed:
                if self._wanted is None or self._closed:
                    self._running = False
                    return
                number, steps, region = self._wanted
                self._wanted = None
                source, previous = self.source, self._run
                if self._crop[0] != region or self._crop[1] is None:
                    x0, y0, x1, y1 = region
                    self._crop = (region, source[y0:y1, x0:x1])
                crop = self._crop[1]
            began = time.perf_counter()
            try:
                run = run_workflow(crop, steps, previous, partial(self._newer, number))
            except Exception as error:  # noqa: BLE001
                # Anything unforeseen is shown as a run that failed at once: this
                # thread must go on to the next run asked for.
                run = Run(crop, steps, [], [], error)
            seconds = time.perf_counter() - began
            with self.changed:
                if source is not self.source or self._closed:
                    continue  # another page was chosen meanwhile
                if run.states:
                    self._run = run  # its first steps may serve the next run
                if run.stopped or self._newer(number):
                    continue
                self._ran = (steps, region)
                self.result = self._publish(number, run, region, seconds)
                self.serial += 1
                self.changed.notify_all()
            if self.published is not None:
                self.published(self)

    def _publish(self, number, run, region, seconds):
        x0, y0 = region[:2]
        layers, sets = {}, {}

        def layer(array, scale, mask=False):
            found = self._layers.get(id(array)) or layers.get(id(array))
            if found is None:
                self._number += 1
                found = Layer(f"L{self._number}", array, scale, (x0, y0), mask)
            layers[id(array)] = found
            return found.id

        def contour_set(state):
            found = self._sets.get(id(state.contours)) or sets.get(id(state.contours))
            if found is None:
                self._number += 1
                c = state.contours
                xy = (c.xy + 0.5) / state.scale - 0.5 + [x0, y0]
                placed = Contours(xy, c.start, c.closed, c.left_out)
                found = ContourSet(f"C{self._number}", placed, of=c)
            sets[id(state.contours)] = found
            return found.id

        states = []
        for index, state in enumerate(run.states):
            after = made_by(run, index)
            before = None if after is None else made_by(run, after - 1)
            states.append(
                {
                    "image": layer(state.image, state.scale),
                    "mask": None
                    if state.mask is None
                    else layer(state.mask, state.scale, True),
                    "contours": None
                    if after is None
                    else contour_set(run.states[after]),
                    "made_by": after,
                    "before": None
                    if before is None
                    else contour_set(run.states[before]),
                    "before_made_by": before,
                    "scale": state.scale,
                    "note": state.note,
                    "noise": finite(state.noise),
                }
            )
        self._old = (self._layers, self._sets)  # a window may still be drawing them
        self._layers, self._sets = layers, sets
        failed = None if run.error is None else len(run.states)
        return {
            "request": number,
            "seconds": round(seconds, 4),
            "region": list(region),
            "steps": [
                {
                    "note": run.states[k + 1].note if k + 1 < len(run.states) else None,
                    "seconds": round(run.seconds[k], 4)
                    if k < len(run.seconds)
                    else None,
                }
                for k in range(len(run.steps))
            ],
            "ran": len(run.states) - 1,
            "error": run.failure or (str(run.error) if run.error is not None else None),
            "failed_step": failed,
            "states": states,
            "layers": {found.id: found.describe() for found in layers.values()},
            "contours": {found.id: found.statistics() for found in sets.values()},
        }

    def wait(self, after, timeout=25.0):
        """The tab's state once its serial is past `after` (at once if it is):
        {"serial", "result" (None: none yet, or another page was chosen),
        "busy"}; after `timeout` seconds without news, the same as it is."""
        with self.changed:
            self.changed.wait_for(lambda: self.serial > after or self._closed, timeout)
            return {
                "serial": self.serial,
                "result": self.result,
                "busy": self._running or self._wanted is not None,
                "closed": self._closed,
            }

    # - what a window fetches -

    def layer(self, key):
        if self.source_layer.id == key:
            return self.source_layer
        with self.changed:
            for table in (self._layers, self._old[0]):
                for found in table.values():
                    if found.id == key:
                        return found
        raise LabError(f"there is no layer {key} (any more)")

    def contour_set(self, key):
        with self.changed:
            for table in (self._sets, self._old[1]):
                for found in table.values():
                    if found.id == key:
                        return found
        raise LabError(f"there are no contours {key} (any more)")

    def export(self, key, kind, pixel_nm=None):
        """(bytes, file name, content type) of a set of contours as a file."""
        found = self.contour_set(key)
        steps, region = self._ran if self._ran else ([], None)
        about = {
            "image": self.file.name,
            "view": dict(self.view),
            "region": list(region) if region else None,
            "pixel_nm": pixel_nm,
            "workflow": workflow_data(steps),
        }
        stem = Path(self.file.name).stem + "-contours"
        if kind == "csv":
            return found.csv(pixel_nm), stem + ".csv", "text/csv; charset=utf-8"
        if kind == "json":
            return found.json(**about), stem + ".json", "application/json"
        if kind == "npz":
            data = found.npz(about=np.array(json.dumps(about)))
            return data, stem + ".npz", "application/octet-stream"
        raise LabError(f"contours are exported as csv, json or npz, not {kind!r}")

    def memory(self):
        """Roughly the bytes this tab's results hold (its image not counted)."""
        with self.changed:
            arrays = {}
            if self._run is not None:
                for state in self._run.states:
                    for array in (
                        state.image,
                        state.mask,
                        state.field,
                        state.edge_image,
                    ):
                        if array is not None:
                            arrays[id(array)] = array.nbytes
                    if state.contours is not None:
                        arrays[id(state.contours)] = state.contours.xy.nbytes * 2
            layers = sum(found.nbytes for found in self._layers.values())
            return sum(arrays.values()) + layers

    def forget(self):
        """Let go of the run and its layers (to make room); the tab's image stays,
        and the next request runs the workflow again."""
        with self.changed:
            if self._running or self._wanted is not None:
                return
            self._run = self._ran = self.result = None
            self._old = ({}, {})
            self._layers, self._sets = {}, {}
            self.serial += 1
            self.changed.notify_all()

    def close(self):
        with self.changed:
            self._closed = True
            self._wanted = None
            self.changed.notify_all()
        self.file.close()
        if self.temporary is not None:
            try:
                self.temporary.unlink()
            except OSError:
                pass


# --- The session ------------------------------------------------------------


class Session:
    """The tabs that are open. memory_limit: when the results of all tabs hold
    more bytes than this, those of the tabs longest not used are let go."""

    def __init__(self, memory_limit=3 * 2**30):
        self.tabs = {}
        self.memory_limit = memory_limit
        self._lock = threading.Lock()
        self._count = 0
        self._folder = None  # of the files dropped on the window

    def _add(self, file, temporary=None):
        with self._lock:
            self._count += 1
            key = f"t{self._count}"
        try:
            tab = Tab(key, file, temporary, lambda tab: self.make_room(keep=tab.id))
        except Exception:
            file.close()
            raise
        with self._lock:
            self.tabs[key] = tab
        return tab

    def open(self, path):
        """Open an image file in a new tab."""
        try:
            return self._add(ImageFile(Path(path).expanduser()))
        except (OSError, ValueError) as error:
            raise LabError(str(error)) from error

    def upload(self, name, data):
        """Open an image given as bytes (a file dropped on the window)."""
        name = Path(name).name or "image.png"
        if Path(name).suffix.lower() not in SUFFIXES:
            raise LabError(f"{name}: open a PNG or TIFF image")
        if len(data) > MAX_UPLOAD_BYTES:
            raise LabError(f"{name} is too large to drop here: open it by its path")
        with self._lock:
            if self._folder is None:
                self._folder = Path(tempfile.mkdtemp(prefix="actis-contour-lab-"))
            self._count += 1
            path = self._folder / f"{self._count}-{name}"
        path.write_bytes(data)
        try:
            tab = self._add(ImageFile(path), temporary=path)
        except (OSError, ValueError) as error:
            path.unlink(missing_ok=True)
            # without the place it was kept at, which says nothing to the user
            reason = str(error).replace(path.name, name).split(" (")[0]
            raise LabError(reason) from error
        tab.file.name = name
        return tab

    def duplicate(self, key):
        """The same file, showing the same pages, in another tab."""
        tab = self.get(key)
        if tab.temporary is not None:
            copy = self.upload(tab.file.name, tab.temporary.read_bytes())
        else:
            copy = self.open(tab.file.path)
        copy.set_view(**tab.view)
        return copy

    def get(self, key):
        tab = self.tabs.get(key)
        if tab is None:
            raise LabError(f"there is no open image {key}")
        tab.used = time.monotonic()
        return tab

    def close(self, key):
        with self._lock:
            tab = self.tabs.pop(key, None)
        if tab is not None:
            tab.close()

    def shutdown(self):
        for key in list(self.tabs):
            self.close(key)
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)

    def make_room(self, keep=None):
        """Let go of the results of the tabs longest not used while all of them
        together hold more than the memory limit (never the tab `keep`)."""
        tabs = sorted(self.tabs.values(), key=lambda tab: tab.used)
        held = {tab.id: tab.memory() for tab in tabs}
        for tab in tabs:
            if sum(held.values()) <= self.memory_limit:
                break
            if tab.id != keep and held[tab.id]:
                tab.forget()
                held[tab.id] = tab.memory()

    def describe(self):
        return {"tabs": [tab.describe() for tab in self.tabs.values()]}


def parse_steps(data):
    """The steps a window sends (the "steps" of a workflow file), checked."""
    try:
        return workflow_steps(workflow_data([]) | {"steps": data})
    except ValueError as error:
        raise LabError(f"the workflow cannot run: {error}") from error


def meta():
    """What a window needs to offer the modules and the workflows to start from."""
    modules = []
    for module in MODULES.values():
        options = []
        for name, option in module.options.items():
            kind = (
                "flag"
                if isinstance(option.default, bool)
                else "choice"
                if isinstance(option.default, str)
                else "whole"
                if isinstance(option.default, int)
                else "number"
            )
            options.append(
                {
                    "name": name,
                    "kind": kind,
                    "default": option.default,
                    "help": option.help,
                    "low": option.low,
                    "high": option.high,
                    "step": option.step,
                    "choices": list(option.choices),
                    "unit": option.unit,
                }
            )
        modules.append(
            {
                "name": module.name,
                "title": module.title,
                "stage": module.stage,
                "help": module.help,
                "options": options,
            }
        )
    presets = [
        {"name": name, "help": text, "steps": workflow_data(steps)["steps"]}
        for name, (text, steps) in PRESETS.items()
    ]
    return {
        "version": __version__,
        "stages": list(STAGES),
        "modules": modules,
        "presets": presets,
        "start": START,
        "format": workflow_data([]),
        "suffixes": list(SUFFIXES),
        "max_region_pixels": MAX_REGION_PIXELS,
    }


def browse(folder=None):
    """The folders and image files of a folder, for a window's Open dialog.

    folder None or "": the user's home. On Windows the drives are listed too.
    """
    path = Path(folder).expanduser() if folder else Path.home()
    try:
        path = path.resolve()
        if not path.is_dir():
            raise LabError(f"{path} is not a folder")
        entries = []
        with os.scandir(path) as found:
            for entry in found:
                try:
                    if entry.name.startswith("."):
                        continue
                    if entry.is_dir():
                        entries.append({"name": entry.name, "folder": True})
                    elif Path(entry.name).suffix.lower() in SUFFIXES:
                        status = entry.stat()
                        entries.append(
                            {
                                "name": entry.name,
                                "folder": False,
                                "size": status.st_size,
                                "modified": status.st_mtime,
                            }
                        )
                except OSError:
                    continue  # no permission: not listed
    except PermissionError:
        raise LabError(f"{path} cannot be read (no permission)") from None
    except OSError as error:
        raise LabError(f"{path} cannot be read: {error.strerror or error}") from None

    def natural(entry):
        text, parts, number = entry["name"].lower(), [], ""
        for letter in text:
            if letter.isdigit():
                number += letter
            else:
                if number:
                    parts.append((0, int(number), ""))
                    number = ""
                parts.append((1, 0, letter))
        if number:
            parts.append((0, int(number), ""))
        return (not entry["folder"], parts)

    entries.sort(key=natural)
    drives = []
    if os.name == "nt":
        drives = [
            f"{letter}:\\"
            for letter in string.ascii_uppercase
            if os.path.exists(f"{letter}:\\")
        ]
    parent = None if path.parent == path else str(path.parent)
    return {"folder": str(path), "parent": parent, "drives": drives, "entries": entries}
