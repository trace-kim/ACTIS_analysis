"""The modules of a contour workflow, the workflows to start from, and their files."""

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from actis.contourlab import contours as geometry
from actis.contourlab.modules import (
    LEFT_OUT,
    MODULES,
    STAGES,
    State,
    WorkflowError,
    image_contours,
)
from actis.contourlab.workflow import (
    FORMAT,
    PRESETS,
    START,
    Step,
    load_workflow,
    preset,
    run_workflow,
    save_workflow,
    step,
    workflow_contours,
    workflow_data,
    workflow_steps,
)
from actis.synthetic import render_repeats

HALF = np.sqrt(2 * np.log(2))
WHOLE = dict(holes=False, open_at_border=False)


@pytest.fixture(scope="module")
def page():
    """One synthetic acquisition of a hole array with detector noise, and its truth."""
    frames, truth = render_repeats(
        image_shape=(160, 192), shape="circle", pitch_px=8.0, cd_nm=90, n_repeats=1
    )
    return frames[0], truth


def blobs(shape=(96, 120), pitch=12.0, sigma=1.6, height=1000.0, base=100.0):
    """Gaussian blobs centred on pixels: at half height each is a circle of
    radius sigma * HALF, and the brightest pixel is base + height exactly."""
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    centres = np.array(
        [
            (cx, cy)
            for cy in np.arange(pitch, rows - pitch / 2, pitch)
            for cx in np.arange(pitch, cols - pitch / 2, pitch)
        ]
    )
    image = np.full(shape, float(base))
    for cx, cy in centres:
        image += height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    return image.astype(np.float32), centres


def matched(contours, centres, within=1.0):
    """For each true centre the number of the contour found there (-1: none)."""
    found = geometry.measure(contours)["centre"]
    if not len(found):
        return np.full(len(centres), -1)
    distance, index = cKDTree(found).query(centres)
    return np.where(distance < within, index, -1)


# --- What the modules are ---------------------------------------------------


def test_every_module_says_what_it_does_and_what_its_options_do():
    assert len(MODULES) >= 20
    for name, module in MODULES.items():
        assert module.name == name and module.stage in STAGES
        assert len(module.help) > 60 and "\n" not in module.help, name
        for option_name, option in module.options.items():
            assert len(option.help) > 15, (name, option_name)
            assert option.checked(option.default, option_name) == option.default
            if isinstance(option.default, str):
                assert option.default in option.choices
            elif not isinstance(option.default, bool):
                assert option.low <= option.default <= option.high and option.step > 0
    titles = [module.title for module in MODULES.values()]
    assert len(set(titles)) == len(titles)
    stages = [STAGES.index(module.stage) for module in MODULES.values()]
    assert stages == sorted(stages)  # listed by stage, in the order they run


def test_every_reason_to_leave_a_contour_out_has_its_words(page):
    image, _ = page
    used = set()
    for steps in [s for _, s in PRESETS.values()] + [
        [
            step("mid_level_threshold"),
            step("trace_contours", max_area_px=30.0),
            step("filter_contours", closed_only=True, max_aspect=1.01, border_px=20.0),
        ]
    ]:
        run = run_workflow(image, steps)
        used |= set(run.final.contours.left_out)
    assert used and used <= set(LEFT_OUT)


BEFORE = {  # what a module needs before it, to run with its defaults
    "morphology": ["mid_level_threshold"],
    "fill_holes": ["mid_level_threshold"],
    "trace_contours": ["mid_level_threshold"],
    "refine_edges": ["mid_level_threshold", "trace_contours"],
    "filter_contours": ["mid_level_threshold", "trace_contours"],
    "smooth_contours": ["mid_level_threshold", "trace_contours"],
    "simplify_shape": ["mid_level_threshold", "trace_contours"],
}


GIVEN = {"fixed_threshold": {"level": 8000.0}}  # no default can suit every image


@pytest.mark.parametrize("name", list(MODULES))
def test_every_module_runs_with_its_defaults(name, page):
    image, _ = page
    steps = [step(before) for before in BEFORE.get(name, [])]
    steps.append(step(name, **GIVEN.get(name, {})))

    run = run_workflow(image, steps)

    assert run.error is None, run.failure
    before, after = run.states[-2], run.final
    assert after.image.dtype == np.float32 and np.isfinite(after.image).all()
    assert after.image.shape == tuple(np.array(image.shape) * after.scale)
    stage = MODULES[name].stage
    if stage == "Image":
        assert after.mask is None and after.contours is None
    elif stage == "Mask":
        assert after.mask.dtype == bool and after.mask.shape == after.image.shape
        assert 0.02 < after.mask.mean() < 0.6 and after.contours is None
    else:
        assert len(after.contours) > 100 and after.note
    # a module leaves the arrays it was given as they were
    again = run_workflow(image, steps[:-1])
    np.testing.assert_array_equal(before.image, again.final.image)
    if before.contours is not None:
        np.testing.assert_array_equal(before.contours.xy, again.final.contours.xy)


# --- The mid-level threshold and the tracing between the pixels -------------


def test_the_half_level_outline_of_a_blob_is_where_it_should_be():
    image, centres = blobs()
    steps = [
        step(
            "mid_level_threshold",
            reference="whole image",
            background_percent=0.0,
            pattern_percent=100.0,
            blur_px=0.0,
        ),
        step("trace_contours", **WHOLE),
    ]

    contours = workflow_contours(image, steps)

    measures = geometry.measure(contours)
    assert len(contours) == len(centres) and contours.closed.all()
    # the cubic spline through pixels 1.6 sigma apart is this close to the blob
    np.testing.assert_allclose(measures["diameter"], 2 * 1.6 * HALF, atol=0.008)
    index = matched(contours, centres, 0.1)
    assert (index >= 0).all()
    np.testing.assert_allclose(measures["centre"][index], centres, atol=0.002)
    assert (measures["area"] > 0).all()  # each runs round its region


@pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7])
def test_the_fraction_moves_the_outline_up_and_down_the_profile(fraction):
    image, _centres = blobs()
    steps = [
        step(
            "mid_level_threshold",
            reference="whole image",
            fraction=fraction,
            background_percent=0.0,
            pattern_percent=100.0,
            blur_px=0.0,
        ),
        step("trace_contours", **WHOLE),
    ]

    diameter = geometry.measure(workflow_contours(image, steps))["diameter"]

    radius = 1.6 * np.sqrt(2 * np.log(1 / fraction))
    np.testing.assert_allclose(diameter, 2 * radius, atol=0.012)


@pytest.mark.parametrize("reference", ["local tiles", "each feature", "whole image"])
def test_gain_offset_and_illumination_do_not_move_the_outline(reference):
    image, centres = blobs((128, 192), pitch=8)
    ramp = np.linspace(0.8, 1.25, 192, dtype=np.float32)[None, :]
    window = 32 if reference == "local tiles" else 0  # each feature: the spacing
    steps = [
        step("mid_level_threshold", reference=reference, window_px=window),
        step("trace_contours", **WHOLE),
    ]

    plain = workflow_contours(image, steps)
    scaled = workflow_contours(image * 3.7 + 900, steps)
    shaded = workflow_contours(image * ramp, steps)

    assert len(plain) == len(scaled) == len(shaded) == len(centres)
    np.testing.assert_allclose(scaled.xy, plain.xy, atol=2e-3)
    change = geometry.measure(shaded)["diameter"] - geometry.measure(plain)["diameter"]
    # one level for the whole image follows the illumination; local levels do not
    if reference == "whole image":
        assert abs(change).max() > 0.2
    else:
        # (the tiles' levels stay flat from the outermost tile centres to the border)
        assert np.median(abs(change)) < 0.02 and abs(change).max() < 0.08


def test_a_dark_pattern_is_found_as_the_bright_one_in_the_negative(page):
    image, _ = page
    negative = (60000 - image.astype(np.int64)).astype(np.uint16)

    bright = workflow_contours(image, preset(START))
    dark = workflow_contours(negative, preset("Dark features"))

    assert len(dark) == len(bright) > 300
    np.testing.assert_allclose(dark.xy, bright.xy, atol=1e-3)
    assert (geometry.measure(dark)["area"][dark.closed] > 0).all()


def test_an_image_without_pattern_gives_no_contours():
    rng = np.random.default_rng(0)
    blank = rng.normal(1200, 40, (200, 240))
    blank += rng.normal(0, 6, (200, 1))  # row offsets, as a TDI sensor has

    for name in (START, "Own half level", "Optical low-pass first", "Local tiles"):
        run = run_workflow(blank, preset(name))
        assert run.error is None and len(run.final.contours) == 0, name
    assert "0 tops set levels" in run_workflow(blank, preset(START)).states[1].note
    assert "tiles without pattern" in run.states[1].note


def test_a_page_that_is_partly_blank_gives_contours_only_on_its_pattern():
    rng = np.random.default_rng(1)
    image, centres = blobs((192, 320), pitch=8, height=4000, base=300)
    image[:, 150:] = 300
    image = image + rng.normal(0, 30, image.shape).astype(np.float32)
    whole = centres[centres[:, 0] < 146]  # those not cut by the edge of the pattern

    contours = workflow_contours(image, preset(START))

    centre = geometry.measure(contours)["centre"]
    assert centre[:, 0].max() < 152 and (matched(contours, whole) >= 0).all()
    # the slivers left of the blobs that the edge of the pattern cut are features too
    assert len(contours) <= (centres[:, 0] < 156).sum()


def test_lines_that_run_out_of_the_image_give_open_contours():
    _y, x = np.mgrid[0:120, 0:160].astype(np.float32)
    lines = 500 + 3000 / (1 + np.exp(-4 * np.cos(2 * np.pi * x / 20)))

    contours = workflow_contours(lines, preset(START))
    whole_only = run_workflow(lines, preset("Local tiles"))

    assert len(contours) == 16 and not contours.closed.any()  # two edges per line
    assert sorted(set(np.round(contours.xy[:, 0] % 20, 1))) == [5.0, 15.0]
    ends = contours.xy[contours.start[:-1], 1]
    np.testing.assert_allclose(np.minimum(ends, 119 - ends), 0, atol=1e-3)  # the border
    assert len(whole_only.final.contours) == 0
    assert whole_only.final.contours.left_out["border"] == 16


def test_inner_boundaries_are_traced_only_if_asked_for():
    y, x = np.mgrid[0:80, 0:80]
    radius = np.hypot(x - 40.2, y - 39.7)
    ring = (1000 * ((radius < 25) & (radius > 10))).astype(np.float32)

    def traced(**options):
        steps = [
            step("fixed_threshold", level=500.0, blur_px=1.0),
            step("trace_contours", **options),
        ]
        return workflow_contours(ring, steps)

    both = traced()
    outer = traced(holes=False)

    assert len(both) == 2 and len(outer) == 1 and outer.left_out["inner"] == 1
    np.testing.assert_allclose(
        sorted(geometry.measure(both)["diameter"]), [20, 50], atol=0.3
    )
    assert geometry.measure(outer)["area"][0] > 0
    assert len(traced(min_area_px=500.0)) == 1  # the hole's boundary is too small
    assert len(traced(max_area_px=500.0)) == 1


def test_the_outline_follows_the_pixels_once_the_mask_was_changed():
    image, _centres = blobs()
    threshold = step("fixed_threshold", level=600.0)

    smooth = run_workflow(image, [threshold, step("trace_contours")])
    coarse = run_workflow(
        image, [threshold, step("morphology", size_px=1.0), step("trace_contours")]
    )
    refined = run_workflow(
        image,
        [
            threshold,
            step("morphology", size_px=1.0),
            step("trace_contours"),
            step("refine_edges", rule="threshold level", blur_px=0.0),
        ],
    )

    assert "between the pixels" in smooth.final.note
    assert "along the mask's pixels" in coarse.final.note and coarse.final.field is None

    def worst(run):
        return abs(
            geometry.measure(run.final.contours)["diameter"] - 2 * 1.6 * HALF
        ).max()

    assert worst(smooth) < 0.008 and worst(coarse) > 0.05
    assert worst(refined) < 0.015  # put back on the edge


def test_upscaling_changes_no_option_and_gives_the_contours_in_the_images_pixels(page):
    image, _truth = page
    steps = [step("feature_threshold"), step("trace_contours", **WHOLE)]

    plain = run_workflow(image, steps)
    fine = run_workflow(image, [step("upscale", factor=3)] + steps)

    assert fine.final.scale == 3 and fine.final.image.shape == (480, 576)
    a, b = image_contours(plain.final), image_contours(fine.final)
    assert len(a) == len(b)
    index = matched(b, geometry.measure(a)["centre"], 0.2)
    assert (index >= 0).all()
    np.testing.assert_allclose(
        geometry.measure(b)["diameter"][index],
        geometry.measure(a)["diameter"],
        atol=0.05,
    )
    assert abs(len(b.xy) - len(a.xy)) < 0.02 * len(a.xy)  # the same spacing of points
    with pytest.raises(WorkflowError, match="before the module that makes the mask"):
        workflow_contours(image, steps + [step("upscale")])


# --- The workflows to start from --------------------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_workflow_to_start_from_finds_the_holes_of_an_array(name, page):
    image, truth = page
    dark = name.startswith("Dark features")
    given = (60000 - image.astype(np.int64)).astype(np.uint16) if dark else image

    run = run_workflow(given, preset(name))

    assert run.error is None, run.failure
    contours = image_contours(run.final)
    centres, pixel_nm = truth["centers_xy_px"], truth["pixel_nm"]
    index = matched(contours, centres, 0.3 * truth["pitch_px"])
    # the benchmark's own method leaves out the holes nearest the image border
    assert (index >= 0).mean() >= (0.9 if name.startswith("Benchmark") else 1.0)
    found = index >= 0
    measures = geometry.measure(contours)
    offset = measures["centre"][index[found]] - centres[found]
    assert np.abs(offset).max() < 0.25
    diameter = measures["diameter"][index[found]] * pixel_nm
    assert np.all(np.abs(diameter / truth["area_diameter_nm"][found] - 1) < 0.3)
    kept_open = any(s.options.get("open_at_border") for s in preset(name))
    assert contours.closed.all() or kept_open
    assert PRESETS[name][0] and len(run.states) == len(preset(name)) + 1


def test_the_workflows_follow_a_difference_in_hole_size():
    frames, truth = render_repeats(
        image_shape=(192, 256), pitch_px=7.0, cd_nm=85, cd_variation_nm=8, n_repeats=1
    )
    true = truth["area_diameter_nm"]
    slopes = {}
    for name in (START, "Own half level"):
        steps = preset(name)
        steps[-1].options.update(WHOLE)  # a hole cut by the border has no diameter
        contours = workflow_contours(frames[0], steps)
        index = matched(contours, truth["centers_xy_px"], 2.0)
        assert (index >= 0).all()
        diameter = geometry.measure(contours)["diameter"][index] * truth["pixel_nm"]
        slopes[name] = np.polyfit(true, diameter, 1)[0]
        # (a hole against its own top alone is the noisier measure of its size)
        assert np.corrcoef(true, diameter)[0, 1] > (0.9 if name == START else 0.8)

    # every hole against the levels of its neighbours follows its size; against its
    # own peak, part of the difference is lost
    own = slopes["Own half level"]
    assert 0.7 < slopes[START] < 1.2 and own < 0.92 * slopes[START]


def test_the_workflow_a_window_opens_with_is_one_of_those_to_start_from():
    assert START in PRESETS and next(iter(PRESETS)) == START
    for name, (description, steps) in PRESETS.items():
        assert len(description) > 80 and steps == preset(name), name
        stages = [STAGES.index(MODULES[s.module].stage) for s in steps]
        assert stages == sorted(stages), name
    changed = preset(START)
    changed[0].options["fraction"] = 0.9
    assert preset(START)[0].options["fraction"] == 0.5  # a copy, yours to change


# --- Steps, runs and failures -----------------------------------------------


def test_a_step_has_every_option_of_its_module():
    made = step("mid_level_threshold", fraction=0.4, window_px=96.0)

    assert made.module == "mid_level_threshold" and made.enabled
    assert list(made.options) == list(MODULES["mid_level_threshold"].options)
    assert made.options["fraction"] == 0.4 and made.options["blur_px"] == 0.5
    assert made.options["window_px"] == 96 and type(made.options["window_px"]) is int
    assert step("upscale", enabled=False).enabled is False


@pytest.mark.parametrize(
    "module, options, message",
    [
        ("sharpen", {}, "there is no module 'sharpen'"),
        ("upscale", {"times": 2}, "upscale has no option 'times'"),
        ("upscale", {"factor": 2.5}, "upscale.factor must be a whole number"),
        ("upscale", {"factor": 99}, "upscale.factor must be between 1 and 8"),
        ("upscale", {"method": "magic"}, "upscale.method must be one of cubic spline"),
        ("invert", {"x": 1}, "invert has no option 'x'"),
        ("trace_contours", {"holes": 1}, "trace_contours.holes must be true or false"),
        ("gaussian_blur", {"sigma_px": "1"}, "gaussian_blur.sigma_px must be a number"),
        ("gaussian_blur", {"sigma_px": float("nan")}, "must be a number"),
    ],
)
def test_a_step_that_cannot_be_is_refused(module, options, message):
    with pytest.raises(ValueError, match=message):
        step(module, **options)


def test_a_run_keeps_the_state_after_every_step(page):
    image, _ = page
    steps = [step("gaussian_blur"), step("mid_level_threshold"), step("trace_contours")]

    run = run_workflow(image, steps)

    assert run.error is None and run.failure == "" and not run.stopped
    assert len(run.states) == 4 and len(run.seconds) == 3 and run.final is run.states[3]
    first = run.states[0]
    assert first.image.dtype == np.float32 and first.mask is None and first.noise > 0
    np.testing.assert_array_equal(first.image, image.astype(np.float32))
    assert run.states[1].mask is None and run.states[2].contours is None
    assert run.states[2].mask is run.states[3].mask  # shared, not copied
    assert run.steps == steps and run.steps is not steps and run.image is image


def test_a_step_that_is_off_passes_on_what_it_was_given(page):
    image, _ = page
    steps = [step("gaussian_blur", enabled=False, sigma_px=5.0)] + preset(START)

    run = run_workflow(image, steps)

    assert run.states[1].note == "off" and run.states[1].image is run.states[0].image
    np.testing.assert_array_equal(
        run.final.contours.xy, run_workflow(image, preset(START)).final.contours.xy
    )


def test_a_later_run_takes_up_the_steps_it_begins_with_alike(page):
    image, _ = page
    steps = [step("gaussian_blur"), step("mid_level_threshold"), step("trace_contours")]
    first = run_workflow(image, steps)

    steps[2].options["spacing_px"] = 0.5
    second = run_workflow(image, steps, first)
    steps[0].options["sigma_px"] = 0.8
    third = run_workflow(image, steps, second)
    other = run_workflow(image.copy(), steps, third)

    assert all(a is b for a, b in zip(second.states[:3], first.states[:3]))
    assert (
        second.states[3] is not first.states[3]
        and second.seconds[:2] == first.seconds[:2]
    )
    assert third.states[0] is first.states[0]
    assert all(a is not b for a, b in zip(third.states[1:], second.states[1:]))
    assert other.states[0] is not third.states[0]  # another array: nothing is taken up
    np.testing.assert_array_equal(other.final.contours.xy, third.final.contours.xy)


def test_a_step_that_fails_ends_the_run_and_says_why(page):
    image, _ = page

    run = run_workflow(
        image, [step("gaussian_blur"), step("trace_contours"), step("invert")]
    )

    assert len(run.states) == 2 and isinstance(run.error, WorkflowError)
    assert run.failure == (
        "Step 2 (Trace contours) failed: Trace contours needs a mask: put a "
        "threshold before it"
    )
    with pytest.raises(WorkflowError, match="Step 2 .Trace contours. failed"):
        workflow_contours(image, [step("gaussian_blur"), step("trace_contours")])
    with pytest.raises(WorkflowError, match="makes no contours"):
        workflow_contours(image, [step("gaussian_blur")])
    for name in (
        "refine_edges",
        "filter_contours",
        "smooth_contours",
        "simplify_shape",
    ):
        failed = run_workflow(image, [step("mid_level_threshold"), step(name)])
        assert "needs contours" in failed.failure, name
    level = run_workflow(
        image, preset(START) + [step("refine_edges", rule="threshold level")]
    )
    assert "threshold level needs the grey level" in level.failure
    with pytest.raises(WorkflowError, match="grey-level image"):
        run_workflow(np.zeros((4, 4, 3)), [])


def test_a_run_can_be_given_up_between_its_steps(page):
    image, _ = page
    asked = []

    def stop():
        asked.append(len(asked))
        return len(asked) == 2

    run = run_workflow(image, preset(START), stop=stop)

    assert run.stopped and run.error is None and len(run.states) == 2
    taken_up = run_workflow(image, preset(START), run)
    assert taken_up.states[1] is run.states[1] and len(taken_up.states) == 3


def test_an_image_too_large_for_a_module_is_refused(monkeypatch, page):
    image, _ = page
    monkeypatch.setattr("actis.contourlab.modules.MAX_PIXELS", 100_000)

    run = run_workflow(image, [step("upscale", factor=4)])
    edges = run_workflow(image, [step("trace_gradient_edges", oversample=4)])

    assert "choose a smaller region" in run.failure and "0 million" in run.failure
    assert "Trace gradient edges would work on" in edges.failure
    assert run_workflow(image, [step("upscale", factor=1)]).error is None


# --- Workflow files ---------------------------------------------------------


def test_a_workflow_is_saved_in_a_file_and_read_again(tmp_path):
    steps = preset("Pixel mask route")
    steps[2].enabled = False
    steps[1].options["fraction"] = 0.35

    save_workflow(tmp_path / "mine.json", steps)

    assert load_workflow(tmp_path / "mine.json") == steps
    data = json.loads((tmp_path / "mine.json").read_text(encoding="utf-8"))
    assert data["format"] == FORMAT and data["version"] == 1
    assert data["steps"][1] == {
        "module": "mid_level_threshold",
        "enabled": True,
        "options": steps[1].options,
    }
    assert workflow_steps(workflow_data(steps)) == steps


def test_an_option_a_file_does_not_give_has_its_default():
    data = {
        "format": FORMAT,
        "version": 1,
        "steps": [{"module": "mid_level_threshold", "options": {"fraction": 0.3}}],
    }

    assert workflow_steps(data) == [step("mid_level_threshold", fraction=0.3)]


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "not a contour workflow file"),
        ({"format": "other"}, "not a contour workflow file"),
        ({"format": FORMAT, "version": 2, "steps": []}, "version 2"),
        ({"format": FORMAT, "version": 1}, "no list of steps"),
        (
            {"format": FORMAT, "version": 1, "steps": [{"module": "x"}]},
            "step 1: there is no",
        ),
        ({"format": FORMAT, "version": 1, "steps": [{}]}, "step 1: 'module'"),
        ({"format": FORMAT, "version": 1, "steps": ["upscale"]}, "step 1"),
        (
            {
                "format": FORMAT,
                "version": 1,
                "steps": [{"module": "upscale", "options": {"factor": 0}}],
            },
            "step 1: upscale.factor must be between",
        ),
    ],
)
def test_a_file_that_is_no_workflow_says_what_is_wrong(data, message):
    with pytest.raises(ValueError, match=message):
        workflow_steps(data)


def test_steps_compare_by_what_they_are():
    assert step("invert") == Step("invert", {}, True)
    assert step("upscale") != step("upscale", factor=2)
    assert State(np.zeros((2, 2), np.float32)).measured.shape == (2, 2)
