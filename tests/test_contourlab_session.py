"""The contour lab's session and its server: open images, runs, tiles, contours."""

import http.client
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from actis.contourlab import contours as geometry
from actis.contourlab import session as lab
from actis.contourlab.server import start
from actis.contourlab.session import (
    ContourSet,
    LabError,
    Layer,
    Session,
    browse,
    meta,
    parse_steps,
)
from actis.contourlab.workflow import (
    PRESETS,
    START,
    preset,
    workflow_contours,
    workflow_data,
)
from actis.synthetic import render_repeats

STEPS = workflow_data(preset(START))["steps"]


@pytest.fixture(scope="module")
def frames():
    """Three synthetic acquisitions of one hole array (uint16, 160 x 192)."""
    rendered, _ = render_repeats(
        image_shape=(160, 192), shape="circle", pitch_px=8.0, cd_nm=90, n_repeats=3
    )
    return rendered


@pytest.fixture
def files(tmp_path, frames):
    """A PNG of the first acquisition and a TIFF with all three as its pages."""
    Image.fromarray(frames[0]).save(tmp_path / "one.png")
    pages = [Image.fromarray(frame) for frame in frames]
    pages[0].save(tmp_path / "scan.tiff", save_all=True, append_images=pages[1:])
    return tmp_path


@pytest.fixture
def session():
    made = Session()
    yield made
    made.shutdown()


def finished(tab, steps=STEPS, region=None, number=1):
    """Run steps on a tab and wait for the result."""
    before = tab.serial
    tab.request(number, parse_steps(steps), region)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = tab.wait(before, 5)
        if state["result"] is not None and state["result"]["request"] == number:
            return state["result"]
        before = state["serial"]
    raise AssertionError("the run did not finish")


def unpacked(payload):
    """The parts of ContourSet.payload()."""
    n, total = (int(value) for value in np.frombuffer(payload, "<u4", 2))
    at = 8
    start = np.frombuffer(payload, "<u4", n + 1, at)
    at += 4 * (n + 1)
    closed = np.frombuffer(payload, np.uint8, n, at).astype(bool)
    at += -(-n // 4) * 4
    xy = np.frombuffer(payload, "<f4", 2 * total, at).reshape(-1, 2)
    at += 8 * total
    centre = np.frombuffer(payload, "<f4", 2 * n, at).reshape(-1, 2)
    at += 8 * n
    diameter = np.frombuffer(payload, "<f4", n, at)
    at += 4 * n
    box = np.frombuffer(payload, "<f4", 4 * n, at).reshape(-1, 4)
    assert at + 16 * n == len(payload)
    return start, closed, xy, centre, diameter, box


# --- Layers -----------------------------------------------------------------


def test_a_layer_serves_an_array_as_tiles_at_halving_resolutions():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 60000, (700, 300)).astype(np.uint16)

    layer = Layer("L1", array, scale=2, origin=(5, 7))

    about = layer.describe()
    assert about["levels"] == 3 and (about["width"], about["height"]) == (300, 700)
    assert (about["scale"], about["x"], about["y"], about["tile"]) == (2, 5, 7, 256)
    assert about["black"] < about["white"] and not about["mask"]
    data, kind, width, height = layer.tile(0, 1, 2)
    assert (kind, width, height) == ("uint16", 44, 188)  # the last tile is smaller
    np.testing.assert_array_equal(
        np.frombuffer(data, "<u2").reshape(height, width), array[512:, 256:]
    )
    half = layer.level(1)
    assert half.shape == (350, 150) and half.dtype == np.float32
    np.testing.assert_allclose(half[0, 0], array[:2, :2].mean())
    data, kind, width, height = layer.tile(2, 0, 0)
    assert (kind, width, height) == ("float32", 75, 175)
    assert np.frombuffer(data, "<f4").mean() == pytest.approx(array.mean(), rel=1e-3)
    for level, tx, ty in ((3, 0, 0), (0, 2, 0), (0, 0, 3), (0, -1, 0)):
        with pytest.raises(LabError):
            layer.tile(level, tx, ty)


def test_an_odd_size_is_halved_to_the_next_whole_pixel():
    array = np.arange(15, dtype=np.float32).reshape(3, 5)

    half = lab.halved(array)

    assert half.shape == (2, 3)
    np.testing.assert_allclose(half[0], [3, 5, 6.5])  # the last column counts twice
    np.testing.assert_allclose(half[1], [10.5, 12.5, 14])


def test_a_mask_is_served_as_how_much_of_each_pixel_is_mask():
    mask = np.zeros((300, 300), bool)
    mask[:, :150] = True
    mask[0, 299] = True

    layer = Layer("L2", mask, mask=True)

    data, kind, _width, _height = layer.tile(0, 0, 0)
    assert kind == "uint8" and set(np.frombuffer(data, np.uint8)) == {0, 255}
    coarse = np.frombuffer(layer.tile(1, 0, 0)[0], np.uint8).reshape(150, 150)
    assert coarse[5, 5] == 255 and coarse[5, 100] == 0 and coarse[0, 149] == 64
    assert layer.describe()["mask"] and layer.describe()["white"] == 255


# --- Contour sets -----------------------------------------------------------


def rings():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    return geometry.from_list(
        [3 * circle + [10, 20], 2 * circle + [30, 5], circle[:10] * 5 + [50, 50]],
        [True, True, False],
        {"small": 4, "border": 0},
    )


def test_a_contour_set_is_one_buffer_for_a_window():
    made = ContourSet("C1", rings())

    start, closed, xy, centre, diameter, box = unpacked(made.payload())

    assert list(start) == [0, 40, 80, 90] and list(closed) == [True, True, False]
    np.testing.assert_allclose(xy, rings().xy, atol=1e-5)
    np.testing.assert_allclose(centre[:2], [[10, 20], [30, 5]], atol=1e-5)
    np.testing.assert_allclose(diameter[:2], [6, 4], rtol=0.01)
    assert np.isnan(diameter[2])
    np.testing.assert_allclose(box[0], [7, 17, 13, 23], atol=0.05)
    assert made.payload() is made.payload()  # made once
    assert len(ContourSet("C0", geometry.empty()).payload()) == 12


def test_the_statistics_of_a_contour_set():
    stats = ContourSet("C1", rings()).statistics()

    assert (stats["count"], stats["closed"], stats["open"], stats["points"]) == (
        3,
        2,
        1,
        90,
    )
    assert stats["left_out"] == {"too small": 4}
    assert stats["diameter"]["median"] == pytest.approx(5, rel=0.01)
    assert stats["diameter"]["min"] == pytest.approx(4, rel=0.01)
    assert sum(stats["histogram"]["counts"]) == 2
    assert len(stats["histogram"]["edges"]) == len(stats["histogram"]["counts"]) + 1
    json.dumps(stats, allow_nan=False)
    empty = ContourSet("C0", geometry.empty()).statistics()
    assert empty["count"] == 0 and "diameter" not in empty


def test_one_contour_is_told_in_full():
    made = ContourSet("C1", rings())

    ring, arc = made.detail(0), made.detail(2)

    assert (
        ring["closed"] and ring["points"] == 40 and (ring["x"], ring["y"]) == (10, 20)
    )
    assert ring["diameter"] == pytest.approx(6, rel=0.01) and ring["area"] > 0
    assert ring["major"] == pytest.approx(6, rel=0.01)
    assert not arc["closed"] and arc["area"] is None and arc["angle"] is None
    assert arc["length"] == pytest.approx(5 * 9 * 2 * np.pi / 40, rel=0.01)
    with pytest.raises(LabError, match="there is no contour 3"):
        made.detail(3)


def test_contours_are_exported_as_a_table_and_with_their_points():
    made = ContourSet("C1", rings())

    lines = made.csv(pixel_nm=21.0).decode().splitlines()
    data = json.loads(made.json(image="x.png"))
    arrays = np.load(io.BytesIO(made.npz(about=np.array("note"))))

    assert lines[0].startswith("contour,closed,points,x_px,y_px,area_px2,diameter_px")
    assert lines[0].endswith("diameter_nm,width_nm,height_nm,major_nm,minor_nm")
    first = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert float(first["diameter_nm"]) == pytest.approx(21 * 6, rel=0.01)
    assert first["closed"] == "1" and len(lines) == 4
    last = dict(zip(lines[0].split(","), lines[3].split(",")))
    assert last["closed"] == "0" and last["area_px2"] == "" and last["length_px"] != ""
    assert "diameter_nm" not in made.csv().decode()
    assert data["image"] == "x.png" and len(data["contours"]) == 3
    assert (
        data["contours"][2]["area_px2"] is None
        and data["contours"][0]["closed"] is True
    )
    np.testing.assert_allclose(data["contours"][1]["xy"], rings().split()[1], atol=1e-4)
    np.testing.assert_array_equal(arrays["xy"], rings().xy)
    assert list(arrays["start"]) == [0, 40, 80, 90] and str(arrays["about"]) == "note"
    assert arrays["diameter"].shape == (3,) and arrays["centre"].shape == (3, 2)


# --- A tab ------------------------------------------------------------------


def test_a_tab_shows_an_image_and_runs_the_newest_workflow_on_it(
    session, files, frames
):
    tab = session.open(files / "one.png")

    about = tab.describe()
    result = finished(tab)

    assert (
        about["name"] == "one.png" and about["pages"] == 1 and about["region"] is None
    )
    assert (about["width"], about["height"], about["dtype"]) == (192, 160, "uint16")
    assert about["min"] == frames[0].min() and about["source"]["id"].startswith("S")
    assert (
        result["error"] is None
        and result["ran"] == 2
        and result["region"] == [0, 0, 192, 160]
    )
    assert [s["note"] is not None for s in result["steps"]] == [True, True]
    assert all(s["seconds"] >= 0 for s in result["steps"])
    first, mask, traced = result["states"]
    assert (
        first["mask"] is None and first["contours"] is None and first["made_by"] is None
    )
    assert mask["mask"] in result["layers"] and mask["image"] == first["image"]
    assert traced["contours"] in result["contours"] and traced["made_by"] == 2
    assert traced["before"] is None and traced["mask"] == mask["mask"]
    stats = result["contours"][traced["contours"]]
    assert stats["closed"] == stats["count"] > 300
    assert 3.5 < stats["diameter"]["median"] < 5
    json.dumps(result, allow_nan=False)
    # what the lab's window draws is what the workflow finds by itself
    found = workflow_contours(frames[0], preset(START))
    _start, _closed, xy, *_ = unpacked(tab.contour_set(traced["contours"]).payload())
    np.testing.assert_allclose(xy, found.xy, atol=1e-4)
    data, _kind, width, height = tab.layer(first["image"]).tile(0, 0, 0)
    np.testing.assert_array_equal(
        np.frombuffer(data, "<f4").reshape(height, width), frames[0].astype(np.float32)
    )
    source = np.frombuffer(tab.layer(about["source"]["id"]).tile(0, 0, 0)[0], "<u2")
    np.testing.assert_array_equal(source.reshape(160, 192), frames[0])


def test_a_region_is_run_alone_and_its_contours_are_in_the_images_pixels(
    session, files, frames
):
    tab = session.open(files / "one.png")
    whole = finished(tab, number=1)
    part = finished(tab, region=[40.2, 30.0, 140.4, 121.0], number=2)

    assert part["region"] == [40, 30, 140, 121]
    layer = part["layers"][part["states"][0]["image"]]
    assert (layer["x"], layer["y"], layer["width"], layer["height"]) == (
        40,
        30,
        100,
        91,
    )
    *_, centre, _diameter, _box = unpacked(
        tab.contour_set(part["states"][2]["contours"]).payload()
    )
    assert centre[:, 0].min() > 40 and centre[:, 0].max() < 140
    assert centre[:, 1].min() > 30 and centre[:, 1].max() < 121
    *_, all_centres, _all_diameters, _ = unpacked(
        ContourSet("x", workflow_contours(frames[0], preset(START))).payload()
    )
    inside = (np.abs(all_centres - [90, 75.5]) < [40, 35]).all(axis=1)
    nearest = (
        np.abs(all_centres[inside][:, None] - centre[None]).sum(axis=2).argmin(axis=1)
    )
    np.testing.assert_allclose(centre[nearest], all_centres[inside], atol=0.05)
    assert whole["contours"] != part["contours"]
    for bad, message in (
        ([0, 0, 4, 100], "smaller than 8 x 8"),
        ("all", "four numbers"),
    ):
        with pytest.raises(LabError, match=message):
            tab.request(3, parse_steps(STEPS), bad)


def test_a_region_too_large_to_run_on_is_refused(session, files, monkeypatch):
    tab = session.open(files / "one.png")
    monkeypatch.setattr(lab, "MAX_REGION_PIXELS", 10_000)
    monkeypatch.setattr(lab, "WHOLE_IMAGE_PIXELS", 5_000)

    with pytest.raises(LabError, match="at most 0 million at once"):
        tab.request(1, parse_steps(STEPS))
    assert finished(tab, region=[0, 0, 100, 100])["error"] is None
    assert tab.describe()["region"] == [0, 0, 192, 26]  # what it runs on unless told


def test_the_steps_two_runs_begin_with_alike_are_not_run_again(session, files):
    tab = session.open(files / "one.png")
    first = finished(tab, number=1)
    steps = json.loads(json.dumps(STEPS))
    steps[1]["options"]["spacing_px"] = 0.5

    second = finished(tab, steps, number=2)
    again = finished(tab, steps, number=3)

    assert (
        second["states"][1] == first["states"][1]
    )  # the same layers: nothing to fetch
    assert second["states"][2]["contours"] != first["states"][2]["contours"]
    assert second["steps"][0]["seconds"] == first["steps"][0]["seconds"]
    assert again["states"] == second["states"] and again["request"] == 3
    assert tab.layer(second["states"][1]["mask"]).mask
    tab.contour_set(first["states"][2]["contours"])  # of the result before: still there
    finished(tab, STEPS, number=4)
    with pytest.raises(LabError, match="no contours C"):
        tab.contour_set(first["states"][2]["contours"])  # two results ago: gone


def test_a_step_that_fails_is_reported_with_the_steps_before_it(session, files):
    tab = session.open(files / "one.png")
    steps = [
        {"module": "gaussian_blur"},
        {"module": "trace_contours"},
        {"module": "invert"},
    ]

    result = finished(tab, steps)

    assert (
        result["failed_step"] == 2 and result["ran"] == 1 and len(result["states"]) == 2
    )
    assert result["error"].startswith(
        "Step 2 (Trace contours) failed: Trace contours needs a mask"
    )
    assert [s["note"] for s in result["steps"]] == ["", None, None]
    with pytest.raises(
        LabError, match="the workflow cannot run: step 1: there is no module"
    ):
        parse_steps([{"module": "sharpen"}])
    with pytest.raises(LabError, match="step 1: upscale.factor"):
        parse_steps([{"module": "upscale", "options": {"factor": 0}}])
    assert (
        finished(tab, [], number=2)["states"][0]["contours"] is None
    )  # no steps: the image


def test_contours_before_the_step_that_last_changed_them_are_kept(session, files):
    tab = session.open(files / "one.png")
    steps = STEPS + [
        {"module": "simplify_shape"},
        {"module": "gaussian_blur", "enabled": False},
    ]

    result = finished(tab, steps)

    traced, simple, off = result["states"][2:]
    assert simple["made_by"] == 3 and simple["before_made_by"] == 2
    assert simple["before"] == traced["contours"] != simple["contours"]
    assert off["contours"] == simple["contours"] and off["made_by"] == 3  # passed on
    assert result["steps"][3]["note"] == "off"


def test_the_pages_of_a_tiff_are_shown_one_at_a_time_or_stacked(session, files, frames):
    tab = session.open(files / "scan.tiff")
    about = tab.describe()
    assert (
        about["pages"] == 3
        and about["shown_pages"] == [0]
        and about["page_height"] == 160
    )
    first = finished(tab, number=1)

    tab.set_view(page=2)
    assert tab.wait(-1, 0)["result"] is None  # another image: its own run
    third = finished(tab, number=2)
    source = np.frombuffer(
        tab.layer(tab.describe()["source"]["id"]).tile(0, 0, 0)[0], "<u2"
    )
    np.testing.assert_array_equal(source.reshape(160, 192), frames[2])
    assert (
        tab.describe()["shown_pages"] == [2] and third["contours"] != first["contours"]
    )

    tab.set_view(stack=True, first=0, count=3, reverse=True, flip=True)
    strip = tab.describe()
    assert (strip["width"], strip["height"]) == (192, 480) and strip["shown_pages"] == [
        2,
        1,
        0,
    ]
    np.testing.assert_array_equal(tab.source, np.concatenate(frames)[::-1])
    stacked = finished(tab, number=3)
    count = stacked["contours"][stacked["states"][2]["contours"]]["count"]
    one = first["contours"][first["states"][2]["contours"]]["count"]
    assert 2.9 * one < count < 3.3 * one  # the holes of all three pages
    with pytest.raises(ValueError, match="cannot stack"):
        tab.set_view(stack=True, first=2, count=3)
    assert tab.describe()["view"]["count"] == 3 and tab.source.shape == (
        480,
        192,
    )  # as it was
    with pytest.raises(ValueError, match="pages 1 to 3, not 9"):
        tab.set_view(page=8)


def test_only_the_newest_workflow_asked_for_is_finished(session, files):
    tab = session.open(files / "scan.tiff")
    tab.set_view(stack=True, first=0, count=3)
    slow = parse_steps([{"module": "upscale", "options": {"factor": 4}}] + STEPS)
    before = tab.serial

    for number in range(1, 6):  # five in a row, as from a spin box held down
        tab.request(number, slow if number < 5 else parse_steps(STEPS))
    deadline = time.monotonic() + 120
    seen = []
    while time.monotonic() < deadline:
        state = tab.wait(before, 5)
        before = state["serial"]
        if state["result"] is not None:
            seen.append(state["result"]["request"])
        if not state["busy"] and seen:
            break

    assert seen[-1] == 5 and len(seen) <= 3  # not every one of them ran to its end
    assert state["result"]["states"][-1]["scale"] == 1


def test_a_closed_tab_stops_and_lets_go_of_its_file(session, files):
    tab = session.open(files / "scan.tiff")
    waiting = []
    thread = threading.Thread(target=lambda: waiting.append(tab.wait(tab.serial, 30)))
    thread.start()

    session.close(tab.id)
    thread.join(10)

    assert waiting and waiting[0]["closed"] and tab.id not in session.tabs
    (files / "scan.tiff").unlink()  # no longer held open
    with pytest.raises(LabError, match="this image was closed"):
        tab.request(1, parse_steps(STEPS))
    with pytest.raises(LabError, match="there is no open image"):
        session.get(tab.id)
    session.close(tab.id)  # twice is fine


# --- The session ------------------------------------------------------------


def test_a_session_opens_duplicates_and_closes_tabs(session, files, frames):
    one = session.open(files / "one.png")
    scan = session.open(files / "scan.tiff")
    scan.set_view(page=1)

    copy = session.duplicate(scan.id)
    dropped = session.upload("dropped.png", (files / "one.png").read_bytes())
    again = session.duplicate(dropped.id)

    assert [tab["name"] for tab in session.describe()["tabs"]] == [
        "one.png",
        "scan.tiff",
        "scan.tiff",
        "dropped.png",
        "dropped.png",
    ]
    assert len({one.id, scan.id, copy.id, dropped.id, again.id}) == 5
    assert copy.view == scan.view and copy.describe()["path"] == str(
        files / "scan.tiff"
    )
    np.testing.assert_array_equal(copy.source, frames[1])
    assert dropped.describe()["path"] is None  # it has none the user knows
    np.testing.assert_array_equal(again.source, frames[0])
    kept = dropped.temporary
    session.close(dropped.id)
    assert not kept.exists() and again.temporary.exists()
    for name, data, message in (
        ("notes.txt", b"hello", "open a PNG or TIFF image"),
        ("bad.png", b"hello", "bad.png: not a readable image"),
    ):
        with pytest.raises(LabError, match=message):
            session.upload(name, data)
    with pytest.raises(LabError, match="there is no file"):
        session.open(files / "missing.png")
    assert len(session.tabs) == 4


def test_the_results_of_tabs_long_not_used_make_room(files):
    session = Session(memory_limit=1)
    try:
        first = session.open(files / "one.png")
        second = session.open(files / "scan.tiff")
        finished(first)
        assert first.wait(-1, 0)["result"] is not None and first.memory() > 100_000

        finished(second)  # the newest stays; the other makes room

        assert second.wait(-1, 0)["result"] is not None
        assert first.wait(-1, 0)["result"] is None and first.memory() == 0
        assert finished(first, number=2)["error"] is None  # run again when asked
    finally:
        session.shutdown()


def test_the_window_is_told_every_module_and_workflow_to_start_from():
    told = meta()

    assert told["stages"] == ["Image", "Mask", "Contours"] and told["start"] == START
    assert [p["name"] for p in told["presets"]] == list(PRESETS)
    modules = {m["name"]: m for m in told["modules"]}
    threshold = {o["name"]: o for o in modules["mid_level_threshold"]["options"]}
    assert (
        threshold["fraction"]["kind"] == "number"
        and threshold["fraction"]["default"] == 0.5
    )
    assert (
        threshold["window_px"]["kind"] == "whole"
        and threshold["window_px"]["unit"] == "px"
    )
    assert (
        threshold["bright"]["kind"] == "flag"
        and threshold["reference"]["kind"] == "choice"
    )
    assert threshold["reference"]["choices"] == [
        "local tiles",
        "each feature",
        "whole image",
    ]
    assert parse_steps(told["presets"][0]["steps"]) == preset(START)
    json.dumps(told, allow_nan=False)


def test_a_folder_is_listed_for_the_open_dialog(tmp_path):
    for name in ("b10.png", "b9.png", "A.TIFF", "notes.txt", ".hidden.png"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "Zeta").mkdir()

    listed = browse(str(tmp_path))

    assert listed["folder"] == str(tmp_path.resolve()) and listed["parent"] == str(
        tmp_path.parent
    )
    names = [entry["name"] for entry in listed["entries"]]
    assert names == [
        "sub",
        "Zeta",
        "A.TIFF",
        "b9.png",
        "b10.png",
    ]  # folders, then by name
    assert listed["entries"][2]["size"] == 1 and not listed["entries"][2]["folder"]
    assert browse()["folder"] and isinstance(listed["drives"], list)
    with pytest.raises(LabError, match="is not a folder"):
        browse(str(tmp_path / "b9.png"))
    with pytest.raises(LabError, match="is not a folder"):
        browse(str(tmp_path / "nowhere"))


# --- The server -------------------------------------------------------------


@pytest.fixture
def served(files):
    server = start(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.session.shutdown()
    server.server_close()


def ask(server, path, data=None, key=True, raw=False, headers=None):
    """A request as the page makes it: (status, the reply)."""
    host, port = server.server_address[:2]
    request = urllib.request.Request(
        f"http://{host}:{port}{path}", method="POST" if data is not None else "GET"
    )
    if key:
        request.add_header("X-Lab-Key", server.key)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    if data is not None and not isinstance(data, bytes):
        data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(request, data, timeout=60) as reply:
            body = reply.read()
            return reply.status, (body if raw else json.loads(body)), reply.headers
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), error.headers


def test_a_second_lab_takes_the_next_free_port():
    first = start(port=0)
    port = first.server_address[1]
    try:
        second = start(port=port)  # taken: the next one that is free
        try:
            assert port < second.server_address[1] < port + 20
            assert second.key != first.key and second.server_address[0] == "127.0.0.1"
        finally:
            second.server_close()
        with pytest.raises(OSError, match=f"no free port from {port} to {port}"):
            start(port=port, tries=1)
    finally:
        first.server_close()


def test_the_command_line_opens_the_lab(monkeypatch, capsys):
    from actis import cli
    from actis.contourlab import server

    asked = []
    monkeypatch.setattr(server, "main", lambda argv: asked.append(argv) or 0)
    assert cli.main(["contours", "a.png", "b.tiff", "--port", "0", "--no-browser"]) == 0
    assert cli.main(["contours"]) == 0
    assert asked == [["a.png", "b.tiff", "--port", "0", "--no-browser"], []]

    monkeypatch.undo()

    def taken(port):
        raise OSError(f"no free port from {port}")

    monkeypatch.setattr(server, "start", taken)
    assert server.main(["--port", "9", "--no-browser"]) == 1  # said, not a traceback
    assert "Contour lab: no free port from 9" in capsys.readouterr().err


def test_the_page_is_served_only_with_the_key(served):
    host, port = served.server_address[:2]

    status, page, _ = ask(served, f"/?key={served.key}", key=False, raw=True)
    assert status == 200 and served.key.encode() in page and b"<canvas" in page
    assert served.url == f"http://{host}:{port}/?key={served.key}"
    assert ask(served, "/", key=False)[:2] == (
        403,
        {"error": "open the address the lab printed"},
    )
    assert ask(served, "/?key=wrong", key=False)[0] == 403
    assert ask(served, "/api/meta", key=False)[0] == 403
    assert (
        ask(served, "/api/meta", headers={"Host": "evil.example"})[0] == 403
    )  # rebinding
    kinds = {"app.js": "javascript", "view.js": "javascript", "app.css": "css"}
    for name, kind in kinds.items():
        status, body, headers = ask(served, f"/static/{name}", raw=True)
        assert status == 200 and len(body) > 1000
        # said here, not asked of the PC: a browser runs no script called plain text
        assert headers["Content-Type"] == f"text/{kind}; charset=utf-8"
        assert f"/static/{name}?key={served.key}".encode() in page  # the page uses it
    assert ask(served, "/static/..%2Fserver.py")[0] == 404
    assert (
        ask(served, "/static/missing.js")[0] == 404
        and ask(served, "/nothing")[0] == 404
    )
    assert ask(served, "/api/nothing")[0] == 404


def test_requests_on_one_connection_do_not_run_into_each_other(served, files):
    host, port = served.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=30)

    def one(method, path, body=None, key=served.key):
        connection.request(method, path, body=body, headers={"X-Lab-Key": key})
        reply = connection.getresponse()
        return reply.status, json.loads(reply.read())

    status, opened = one(
        "POST", "/api/open", json.dumps({"paths": [str(files / "one.png")]})
    )
    tab = opened["tabs"][0]["id"]
    # what these two are sent is not needed, and is read all the same: the page keeps
    # its connection, and its next request would otherwise begin with it
    assert one("POST", f"/api/tabs/{tab}/duplicate", "{}")[0] == 200
    assert one("GET", "/api/session")[0] == 200
    assert one("POST", f"/api/tabs/{tab}/close", '{"anything": 1}')[0] == 200
    status, session = one("GET", "/api/session")
    assert status == 200 and len(session["tabs"]) == 1
    assert (
        one("POST", "/api/nothing", "{}")[0] == 404
        and one("GET", "/api/meta")[0] == 200
    )
    assert (
        one("POST", "/api/open", "{}", key="wrong")[0] == 403
    )  # not read: connection closed
    connection.close()


def test_the_page_opens_images_runs_workflows_and_fetches_what_it_draws(
    served, files, frames
):
    status, meta_told, _ = ask(served, "/api/meta")
    assert status == 200 and meta_told["start"] == START

    status, opened, _ = ask(
        served,
        "/api/open",
        {"paths": [str(files / "scan.tiff"), str(files / "nothing.png")]},
    )
    assert (
        status == 200
        and len(opened["tabs"]) == 1
        and "there is no file" in opened["errors"][0]
    )
    tab = opened["tabs"][0]
    base = f"/api/tabs/{tab['id']}"
    assert ask(served, "/api/session")[1]["tabs"][0]["name"] == "scan.tiff"

    status, shown, _ = ask(served, base + "/view", {"page": 1})
    assert status == 200 and shown["view"]["page"] == 1
    status, started, _ = ask(served, base + "/run", {"steps": STEPS, "request": 7})
    assert status == 200
    after = started["serial"] - 1
    while True:
        status, state, _ = ask(served, f"{base}/result?after={after}&timeout=20")
        after = state["serial"]
        if state["result"] is not None and state["result"]["request"] == 7:
            break
    result = state["result"]
    assert result["error"] is None and not state["closed"]

    layer = result["states"][0]["image"]
    status, tile, headers = ask(served, f"{base}/tile/{layer}/0/0/0", raw=True)
    assert status == 200 and headers["X-Tile"] == "float32 192 160"
    np.testing.assert_array_equal(
        np.frombuffer(tile, "<f4").reshape(160, 192), frames[1].astype(np.float32)
    )
    assert ask(served, f"{base}/tile/{layer}/0/9/9")[0] == 400
    assert (
        ask(served, f"{base}/tile/L999/0/0/0")[1]["error"]
        == "there is no layer L999 (any more)"
    )

    contours = result["states"][2]["contours"]
    status, payload, _ = ask(served, f"{base}/contours/{contours}", raw=True)
    _start, closed, xy, _centre, diameter, _box = unpacked(payload)
    assert status == 200 and len(closed) == result["contours"][contours]["count"]
    status, detail, _ = ask(served, f"{base}/contours/{contours}/3")
    assert detail["diameter"] == pytest.approx(float(diameter[3]), rel=1e-4)
    assert ask(served, f"{base}/contours/{contours}/99999")[0] == 400

    status, table, headers = ask(
        served, f"{base}/export/{contours}/csv?pixel_nm=21", raw=True
    )
    assert (
        status == 200
        and b"diameter_nm" in table
        and "scan-contours.csv" in headers["Content-Disposition"]
    )
    status, document, _ = ask(served, f"{base}/export/{contours}/json")
    assert document["image"] == "scan.tiff" and document["view"]["page"] == 1
    assert parse_steps(document["workflow"]["steps"]) == preset(START)
    status, packed, _ = ask(served, f"{base}/export/{contours}/npz", raw=True)
    assert np.load(io.BytesIO(packed))["xy"].shape == xy.shape
    assert ask(served, f"{base}/export/{contours}/docx")[0] == 400

    assert ask(served, base + "/run", {"steps": [{"module": "nothing"}]})[0] == 400
    assert ask(served, base + "/view", {"page": 99})[0] == 400
    assert ask(served, base + "/view", {"pages": 1})[0] == 400
    assert ask(served, base + "/run", b"not json")[0] == 400

    status, copy, _ = ask(served, base + "/duplicate", {})
    assert status == 200 and copy["id"] != tab["id"] and copy["view"]["page"] == 1
    status, dropped, _ = ask(
        served, "/api/upload?name=dropped.png", (files / "one.png").read_bytes()
    )
    assert status == 200 and dropped["tabs"][0]["name"] == "dropped.png"
    assert ask(served, base + "/close", {})[0] == 200
    assert ask(served, base + "/result")[0] == 400  # there is no such image any more
    assert len(ask(served, "/api/session")[1]["tabs"]) == 2

    status, listing, _ = ask(
        served, f"/api/browse?folder={urllib.request.quote(str(files))}"
    )
    assert [entry["name"] for entry in listing["entries"]] == ["one.png", "scan.tiff"]
