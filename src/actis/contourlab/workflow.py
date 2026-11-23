"""A workflow: a list of modules, each with its options, run in turn on an image.

    steps = [step("mid_level_threshold"), step("trace_contours", holes=False)]
    contours = workflow_contours(image, steps)   # Contours, in the image's pixels
    save_workflow("my.json", steps)              # load_workflow("my.json") another day

A workflow is designed by trying (the contour lab's window): run_workflow()
gives the State after every step, for a window that shows each, and takes up an
earlier run where the steps begin alike, so that changing a late step does not
run the early ones again. PRESETS are workflows to start from.
"""

import copy
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import ops
from .modules import MODULES, State, WorkflowError, image_contours

FORMAT, VERSION = "actis contour workflow", 1  # of a workflow file


@dataclass
class Step:
    """One module of a workflow, with its options."""

    module: str  # its name in MODULES
    options: dict = field(default_factory=dict)  # every option of the module
    enabled: bool = True  # False: skipped; what the steps before it made goes on


def step(module, enabled=True, **options):
    """A step of this module with these options, the others at their defaults.

    A ValueError names a module or an option that does not exist, or a value
    the option cannot have.
    """
    if module not in MODULES:
        raise ValueError(f"there is no module {module!r}")
    known = MODULES[module].options
    for name in options:
        if name not in known:
            raise ValueError(f"{module} has no option {name!r}")
    checked = {
        name: option.checked(options.get(name, option.default), f"{module}.{name}")
        for name, option in known.items()
    }
    return Step(module, checked, bool(enabled))


@dataclass
class Run:
    """What a workflow made of an image, step by step."""

    image: np.ndarray  # the image it was given
    steps: list  # its steps, as they were when it ran
    states: list  # states[0]: the image as given; states[k]: after step k
    seconds: list  # the time each step took
    error: Exception | None = None  # why step len(states) failed
    stopped: bool = False  # given up before its last step, for a newer run

    @property
    def final(self):
        """What the last step that ran made."""
        return self.states[-1]

    @property
    def failure(self):
        """Which step failed and why, in words ("": none)."""
        if self.error is None:
            return ""
        number = len(self.states)
        name = self.steps[number - 1].module
        title = MODULES[name].title if name in MODULES else name
        reason = str(self.error)
        if not isinstance(self.error, WorkflowError):
            reason = f"{type(self.error).__name__}: {reason}"
        return f"Step {number} ({title}) failed: {reason}"


def first_state(image):
    """The state a workflow starts from: the image as float32 grey levels."""
    grey = np.asarray(image, dtype=np.float32)
    if grey.ndim != 2 or min(grey.shape) < 2:
        raise WorkflowError("a workflow needs a grey-level image of 2 x 2 px or more")
    return State(grey, noise=ops.noise_sigma(grey), origin=grey)


def run_workflow(image, steps, previous=None, stop=None):
    """Run the steps in turn on an image (2-D, any numeric type).

    A step that fails ends the run (Run.error); a step that is not enabled
    passes on what it was given.
    previous: an earlier run. If it was of this very array, the states of the
    steps the two begin with alike are taken from it instead of made again.
    stop: called before every step; True gives the run up there (Run.stopped).
    """
    steps = copy.deepcopy(list(steps))
    alike = previous is not None and previous.image is image
    states, seconds = [previous.states[0] if alike else first_state(image)], []
    for number, this in enumerate(steps):
        alike = (
            alike
            and number < len(previous.states) - 1
            and previous.steps[number] == this
        )
        if alike:
            states.append(previous.states[number + 1])
            seconds.append(previous.seconds[number])
            continue
        if stop is not None and stop():
            return Run(image, steps, states, seconds, stopped=True)
        start = time.perf_counter()
        try:
            if this.enabled:
                state = MODULES[this.module].function(states[-1], **this.options)
            else:
                state = replace(states[-1], note="off")
        except Exception as error:  # noqa: BLE001
            # Whatever a module raises ends the run here; Run.failure tells it.
            return Run(image, steps, states, seconds, error)
        states.append(state)
        seconds.append(time.perf_counter() - start)
    return Run(image, steps, states, seconds)


def workflow_contours(image, steps):
    """The contours a workflow finds in an image, in the image's own pixels.

    A WorkflowError says which step failed and why, or that the workflow makes
    no contours.
    """
    run = run_workflow(image, steps)
    if run.error is not None:
        raise WorkflowError(run.failure) from run.error
    if run.final.contours is None:
        raise WorkflowError("the workflow makes no contours: it needs Trace contours")
    return image_contours(run.final)


# --- Workflow files (JSON) --------------------------------------------------


def workflow_data(steps):
    """A workflow as the data of its file."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "steps": [
            {"module": s.module, "enabled": s.enabled, "options": dict(s.options)}
            for s in steps
        ],
    }


def workflow_steps(data):
    """The workflow in the data of a workflow file; a ValueError says what is
    wrong with it. An option the file does not give has its default."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("not a contour workflow file")
    if not isinstance(data.get("version"), int) or data["version"] > VERSION:
        raise ValueError(
            f"a workflow file of version {data.get('version')!r}: this program "
            f"reads up to {VERSION}"
        )
    if not isinstance(data.get("steps"), list):
        raise ValueError("the workflow file has no list of steps")
    steps = []
    for number, entry in enumerate(data["steps"], 1):
        try:
            options = entry.get("options", {})
            steps.append(step(entry["module"], entry.get("enabled", True), **options))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"step {number}: {error}") from error
    return steps


def save_workflow(path, steps):
    text = json.dumps(workflow_data(steps), indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def load_workflow(path):
    """The workflow of a file save_workflow wrote."""
    return workflow_steps(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Workflows to start from ------------------------------------------------
#
# The first group was designed for ACTIS images: holes about four pixels wide,
# optics that pass no detail finer than about two pixels, shot noise, 8-bit
# pages that mix patterns. The second group are the general-purpose routes the
# lab began with, kept to compare against. scripts/evaluate_contour_flows.py
# scores all of them on synthetic images.

WHOLE = {"holes": False, "open_at_border": False}  # whole features only
GROUPS = ("Designed for ACTIS images", "General-purpose routes, to compare")
PRESETS = {}  # by name: (what it is for, its steps); preset() gives a copy
GUIDE = {}  # by name: its group, what it is best for, what to watch


def example(name, group, text, best, watch, *steps):
    """A workflow to start from."""
    assert group in GROUPS and name not in PRESETS, name
    PRESETS[name] = (text, list(steps))
    GUIDE[name] = {"group": group, "best": best, "watch": watch}


example(
    "Shared half level",
    GROUPS[0],
    "Every feature is outlined half way between its top and the dark level "
    "around it. Both levels are shared with its nearest features of the same "
    "kind, so no tile is laid over the image: holes, lines and pads of any size "
    "and brightness are found on one page, and empty areas stay empty.",
    "Any ACTIS page: the place to start",
    "Follows a size difference about one to one only among features of one kind",
    step("feature_threshold"),
    step("trace_contours"),
)
example(
    "Own half level",
    GROUPS[0],
    "As the shared half level, with every feature measured against its own top "
    "alone. For features that have no neighbours of their kind: isolated holes, "
    "one-off shapes, holes of very different brightness side by side.",
    "Isolated or one-of-a-kind features",
    "Hides part of a size difference, because a smaller hole is also fainter; "
    "the least steady in noise",
    step("feature_threshold", peers=1),
    step("trace_contours"),
)
example(
    "Optical low-pass first",
    GROUPS[0],
    "The spatial frequencies beyond what the optics deliver are cut off before "
    "the shared half level, in place of a blur: the noise they carry goes and "
    "the edges stay as sharp as they are. The step reports the cutoff it found "
    "and how much noise went.",
    "Images sampled more finely than their optics resolve",
    "Removes nothing where the pattern reaches the sampling limit; on noisy "
    "images set the cutoff by hand, the same for every image",
    step("optical_filter"),
    step("feature_threshold", blur_px=0.0),
    step("trace_contours"),
)
example(
    "Deblurred half level",
    GROUPS[0],
    "The blur of the optics is divided out inside their pass band before the "
    "features are outlined: steeper edges and flatter tops. Set the two sigmas "
    "to the blur of your images.",
    "Images whose blur is known",
    "Brings back noise; too large a sigma rings around the features",
    step("optical_filter", deblur_x_px=0.7, deblur_y_px=0.7),
    step("feature_threshold", blur_px=0.0),
    step("trace_contours"),
)
example(
    "Steepest edge",
    GROUPS[0],
    "No dark or bright level at all: the outline is where the brightness "
    "changes fastest, found between the pixels. Gain, offset and illumination "
    "cannot move it.",
    "Wide features with real edges; images whose levels cannot be trusted",
    "On holes of a few pixels the blur sets the steepest slope more than the "
    "hole does, so sizes follow the hole weakly",
    step("trace_gradient_edges"),
)
example(
    "One ellipse per hole",
    GROUPS[0],
    "The shared half level, with every outline reduced to the ellipse of the "
    "same area, centre, elongation and orientation. A hole four pixels wide "
    "carries little more than that; the rest of a free outline is noise.",
    "Width, height and orientation of small holes; a calmer picture",
    "Hides a real irregular shape; features wider than 16 px keep their outline",
    step("feature_threshold"),
    step("trace_contours"),
    step("simplify_shape", harmonics=1, max_diameter_px=16.0),
)
example(
    "Dark features",
    GROUPS[0],
    "The shared half level for a pattern that is the darker side: absorber dots "
    "or lines on the bright multilayer.",
    "Dark-on-bright patterns",
    "On a page of bright holes it outlines the gaps between them",
    step("feature_threshold", bright=False),
    step("trace_contours"),
)
example(
    "Local tiles",
    GROUPS[1],
    "The image is cut into tiles of 128 pixels; the outline is half way between "
    "two percentiles of each tile's grey levels. The lab's first default.",
    "Even hole arrays that fill the image",
    "A tile that holds few holes, two kinds of pattern, or only a bright area "
    "gets the wrong level",
    step("mid_level_threshold"),
    step("trace_contours", **WHOLE),
)
example(
    "One level for the page",
    GROUPS[1],
    "A single grey level for the whole image (Otsu's), after the background is "
    "flattened.",
    "Even, comparable images",
    "Follows illumination and gain as strongly as it follows size",
    step("flatten_background"),
    step("otsu_threshold"),
    step("trace_contours", **WHOLE),
)
example(
    "Pixel mask route",
    GROUPS[1],
    "The classical route: upscale four times, threshold, clean the mask, trace "
    "its pixels, put the outline back on the edges. A base for trying mask "
    "operations.",
    "When the mask itself needs work",
    "Slower, and no more exact than tracing between the pixels",
    step("upscale"),
    step("mid_level_threshold"),
    step("morphology", operation="open", size_px=1.0),
    step("trace_contours", **WHOLE),
    step("refine_edges"),
    step("smooth_contours"),
)
example(
    "Benchmark radial contour",
    GROUPS[1],
    "The contour method of this repository's benchmark: 40 rays around every "
    "discovered hole centre, each cut at the hole's own half level.",
    "Comparing the benchmark's contours with the others",
    "Bright holes in arrays only",
    step("radial_contours"),
)
START = "Shared half level"  # the workflow a window opens with


def preset(name):
    """The steps of a workflow to start from (PRESETS), yours to change."""
    return copy.deepcopy(PRESETS[name][1])
