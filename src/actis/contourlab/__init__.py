"""Contour lab: workflows of image-processing modules that find contours in ACTIS images.

    python -m actis.contourlab [images ...]    # the window, in the browser

designs a workflow on real images: open PNG or TIFF files in tabs, put modules
in a list, change their options and see what each makes of the image. A
workflow saved there runs without the window:

    from actis.contourlab import load_image, load_workflow, measure, workflow_contours

    contours = workflow_contours(load_image("scan.tiff", page=17), load_workflow("my.json"))
    diameters = measure(contours)["diameter"]  # px, one per contour

modules.py holds the modules, workflow.py the runner and the workflows to start
from, contours.py the contours and their geometry, ops.py the image operations,
sources.py the files; session.py and server.py are the window's other half.
"""

from .contours import Contours, measure
from .modules import MODULES, STAGES, State, WorkflowError, image_contours
from .sources import ImageFile, load_image
from .workflow import (
    PRESETS,
    START,
    Run,
    Step,
    load_workflow,
    preset,
    run_workflow,
    save_workflow,
    step,
    workflow_contours,
)

__all__ = [
    "MODULES",
    "PRESETS",
    "STAGES",
    "START",
    "Contours",
    "ImageFile",
    "Run",
    "State",
    "Step",
    "WorkflowError",
    "image_contours",
    "load_image",
    "load_workflow",
    "measure",
    "preset",
    "run_workflow",
    "save_workflow",
    "step",
    "workflow_contours",
]
