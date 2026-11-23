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
    return State(grey, noise=ops.noise_sigma(grey))


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

HOLES = {"holes": False, "open_at_border": False}  # whole features only


def example(text, *steps):
    """A workflow to start from: (what it is for, its steps)."""
    return text, list(steps)


PRESETS = {  # by name; preset() gives a copy of the steps to change
    "Bright holes: local half level": example(
        "Contact holes as ACTIS shows them, bright on dark. The outline is where "
        "the image is half-way between the dark and the bright level of its "
        "neighbourhood, traced between the pixels. Gain, offset and illumination "
        "that vary across the image or between acquisitions drop out, and a "
        "smaller hole gives a smaller contour. The most repeatable flow on "
        "synthetic acquisitions: the place to start.",
        step("mid_level_threshold"),
        step("trace_contours", **HOLES),
    ),
    "Bright holes: deblurred": example(
        "As the local half level, on an image with part of the optical blur taken "
        "out first (Richardson-Lucy). Steeper edges make the outline depend less "
        "on how blurred an acquisition was. Set the two sigmas to the blur of "
        "your images; too much rings.",
        step("deblur"),
        step("mid_level_threshold"),
        step("trace_contours", **HOLES),
    ),
    "Bright holes: each hole's own half maximum": example(
        "Every hole against its own peak and its own surround, with no reference "
        "to its neighbours: works for isolated holes and for holes of very "
        "different brightness in one image. A fainter hole is not drawn smaller "
        "for being fainter, so this follows a difference in hole size less than "
        "the local half level does.",
        step("mid_level_threshold", reference="each feature"),
        step("trace_contours", **HOLES),
    ),
    "Bright holes: one level for the image (Otsu)": example(
        "One grey level for the whole image, after the background has been "
        "flattened. It follows a difference in hole size most strongly, and a "
        "difference in illumination or gain just as strongly: for even, "
        "comparable images.",
        step("flatten_background"),
        step("otsu_threshold"),
        step("trace_contours", **HOLES),
    ),
    "Bright holes: steepest gradient": example(
        "The holes are found by the local half level, then every contour point "
        "is moved to where the brightness changes fastest. Needs no levels; on "
        "holes of a few pixels the steepest slope is set by the blur more than "
        "by the hole, so compare it with the half level before relying on it.",
        step("mid_level_threshold"),
        step("trace_contours", **HOLES),
        step("refine_edges", rule="steepest gradient"),
    ),
    "Edges without a threshold": example(
        "No dark or bright level at all: the lines where the brightness changes "
        "fastest, traced between the pixels (the zero crossings of the second "
        "derivative across the edge, where Canny's detector puts its edges).",
        step("trace_gradient_edges"),
    ),
    "Noisy pages": example(
        "Low signal or 8-bit scan pages: the threshold is taken on a more "
        "strongly blurred image, and what is left of the noise in the outlines "
        "is taken out by keeping only the lowest harmonics of each. Holes are "
        "found reliably down to a contrast of about ten noise sigma.",
        step("mid_level_threshold", blur_px=0.8),
        step("trace_contours", **HOLES),
        step("simplify_shape", harmonics=3),
    ),
    "Any pattern (lines, pads, mixed pages)": example(
        "Nothing assumed about the pattern: every region's whole boundary, the "
        "boundaries of holes inside it too, open where a region runs out of the "
        "image. For lines and spaces, pads, and pages that mix patterns.",
        step("mid_level_threshold"),
        step("trace_contours", min_area_px=6.0),
    ),
    "Dark features on a bright background": example(
        "The pattern is the darker side: absorber dots or lines on the bright "
        "multilayer. Otherwise the local half level.",
        step("mid_level_threshold", bright=False),
        step("trace_contours", **HOLES),
    ),
    "Benchmark radial contour": example(
        "What the benchmark of this repository measures as its contour method "
        "(actis.estimators): 40 rays around every discovered hole centre, each "
        "cut at the hole's own half level. To compare it with the other flows on "
        "the same image.",
        step("radial_contours"),
    ),
    "Pixel mask route (upscaled)": example(
        "The classical route, for when the mask itself needs work: the image is "
        "interpolated to 4 times as many pixels, thresholded, the mask cleaned "
        "by an opening, its pixel outline traced and then put back on the edges. "
        "Slower and no more exact than tracing between the pixels; a base for "
        "trying mask operations.",
        step("upscale"),
        step("mid_level_threshold"),
        step("morphology", operation="open", size_px=1.0),
        step("trace_contours", **HOLES),
        step("refine_edges"),
        step("smooth_contours"),
    ),
    "Bright holes: one ellipse per hole": example(
        "The local half level, with every outline reduced to the ellipse that "
        "fits it: for holes of a few pixels a free outline carries little more, "
        "and an ellipse gives width, height and orientation directly.",
        step("mid_level_threshold"),
        step("trace_contours", **HOLES),
        step("simplify_shape", harmonics=1),
    ),
}
START = "Bright holes: local half level"  # the workflow a window opens with


def preset(name):
    """The steps of a workflow to start from (PRESETS), yours to change."""
    return copy.deepcopy(PRESETS[name][1])
