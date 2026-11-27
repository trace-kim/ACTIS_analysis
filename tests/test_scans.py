import json
import sys
import textwrap

import pytest

import actis.scans as scans
from actis.scans import Lock, read_config, start, status, stop, supervise
from actis.stripes import MANIFEST
from actis.tiff_benchmark import METHODS

# Stands in for the stripe run of one segment. What it does on each attempt
# at a segment comes from plan.json: complete, failed, crash, or hang, which
# keeps saying the same thing until it is stopped.
STUB = textwrap.dedent(
    """
    import json, sys, time
    from pathlib import Path

    output, segment, plan = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    count = plan.with_name(f"attempts-{segment}")
    attempt = int(count.read_text()) if count.exists() else 0
    count.write_text(str(attempt + 1))
    steps = json.loads(plan.read_text())[segment]
    step = steps[min(attempt, len(steps) - 1)]
    print(f"0s | Segment {segment}, page 1 | reading the board files", flush=True)
    if step == "crash":
        print("Traceback: something broke", flush=True)
        raise SystemExit(3)
    if step == "hang":
        while True:
            print(f"9s | Segment {segment}, page 7 | model", flush=True)
            time.sleep(0.05)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "stripes.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {"segments": []}
    stripe, vertical = map(int, segment.split("_"))
    entry = {"stripe": stripe, "vertical": vertical, "status": step}
    if step == "complete":
        entry["result"] = f"{segment}.npz"
        entry["methods"] = {"raw": {"used_tiles": 10, "bins": 1, "failed_batches": 0}}
        (output / entry["result"]).write_bytes(b"")
    else:
        entry["error"] = f"{segment} cannot be read"
    data["segments"] = [
        s for s in data["segments"] if (s["stripe"], s["vertical"]) != (stripe, vertical)
    ] + [entry]
    time.sleep(0.02)  # so that the manifest's time stamp moves on
    manifest.write_text(json.dumps(data))
    """
)


def write_scan(root, names):
    for name in names:
        for board in range(1, 5):
            path = root / "RAW" / f"S{name.split('_')[0]}" / f"{name}_{board}.tiff"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def write_config(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """Run every segment through STUB, following the plan the test writes."""
    script = tmp_path / "stub.py"
    script.write_text(STUB, encoding="utf-8")
    plan = tmp_path / "plan.json"

    def command(self, *extra, output=None):
        segment = extra[list(extra).index("--segment") + 1]
        return [sys.executable, str(script), str(output or self.output), segment, str(plan)]

    monkeypatch.setattr(scans.Scan, "command", command)
    for name, value in (
        ("POLL_SECONDS", 0.05),
        ("PAUSE_SECONDS", 0),
        ("STATE_SECONDS", 0),
    ):
        monkeypatch.setattr(scans, name, value)
    return lambda steps: plan.write_text(json.dumps(steps))


def test_the_config_lists_scans_with_stripe_options(tmp_path):
    (tmp_path / "sub").mkdir()
    config = write_config(
        tmp_path / "sub" / "scans.ini",
        r"""
        ; Every scan shares these.
        [DEFAULT]
        reference = holes.tiff
        method = raw, contour harmonic
        edge-strips = yes
        no-reject = no

        [scan-A]
        root = D:\ACTIS\scan A
        output = D:\ACTIS\results\scan-A
        gpu = 0

        [scan-B]
        root = ..\scan-B
        output = out\scan-B
        gpu = 2
        method = all
        edge-strips = 64
        no-reject = yes
        stripe = 1 3
        """,
    )
    one, two = read_config(config)
    assert (one.name, one.gpu) == ("scan-A", 0)
    assert str(one.root) == r"D:\ACTIS\scan A" or one.root.as_posix().endswith("scan A")
    assert one.parsed.method == ["raw", "contour", "harmonic"]
    assert one.parsed.edge_strips == 32 and not one.parsed.no_reject
    # Relative paths are taken from the folder of the config file.
    assert one.parsed.reference == tmp_path / "sub" / "holes.tiff"
    assert two.root == tmp_path / "scan-B"
    assert two.output == tmp_path / "sub" / "out" / "scan-B"
    assert two.parsed.method == ["all"] and two.parsed.edge_strips == 64
    assert two.parsed.no_reject and two.parsed.stripe == [1, 3]
    command = two.command("--segment", "1_1")
    assert command[1:4] == ["-m", "actis.stripes", str(two.root)]
    assert command[-2:] == ["--segment", "1_1"] and "--gpu" in command

    for body, message in (
        ("[a]\nroot = x\ngpu = 0\nreference = r", "needs output"),
        ("[a]\nroot = x\noutput = y\ngpu = one\nreference = r", "GPU number"),
        ("[a b]\nroot = x\noutput = y\ngpu = 0\nreference = r", "letters, digits"),
        ("[a]\nroot = x\noutput = y\ngpu = 0\nreference = r\nsegment = 1_1", "cannot be set"),
        ("[a]\nroot = x\noutput = y\ngpu = 0\nreference = r\nfast = yes", "unrecognized"),
        ("[a]\nroot = x\noutput = y\ngpu = 0", "--reference"),
        ("[a]\nroot = x\noutput = y\ngpu = 0\nreference = r\n[b]\nroot = z\noutput = y\n"
         "gpu = 1\nreference = r", "both write to"),
        ("[DEFAULT]\nreference = r", "lists no scans"),
    ):
        with pytest.raises(ValueError, match=message):
            read_config(write_config(tmp_path / "bad.ini", body))
    with pytest.raises(ValueError, match="no config file"):
        read_config(tmp_path / "missing.ini")


def test_a_lock_is_held_by_one_process_at_a_time(tmp_path):
    first, second = Lock(tmp_path / "x.lock"), Lock(tmp_path / "x.lock")
    assert first.acquire()
    assert not second.acquire()
    assert scans._running(tmp_path, "x")
    first.release()
    assert not scans._running(tmp_path, "x")
    assert second.acquire()
    second.release()


def scan_of(tmp_path, names):
    root = tmp_path / "scan"
    write_scan(root, names)
    config = write_config(
        tmp_path / "scans.ini",
        f"""
        [one]
        root = {root}
        output = {tmp_path / "out"}
        gpu = 0
        reference = holes.tiff
        """,
    )
    return read_config(config)[0]


def test_a_supervisor_measures_every_segment_and_tries_failures_again(
    tmp_path, stub, capsys
):
    names = ["1_1", "1_2", "2_1", "3_1", "4_1"]
    scan = scan_of(tmp_path, names)
    logs = tmp_path / "logs"
    stub(
        {
            "1_1": ["complete"],
            "1_2": ["crash", "complete"],  # crashes once, then works
            "2_1": ["hang", "complete"],  # stalls once
            "3_1": ["failed"],  # the files are bad
            "4_1": ["complete"],
        }
    )
    assert supervise(scan, logs, stall_seconds=1.0) == 1
    state = json.loads((logs / "one.json").read_text(encoding="utf-8"))
    assert state["status"] == "finished" and state["total"] == 5
    assert state["reason"] == "4 of 5 segments complete"
    records = state["segments"]
    assert {name: r["status"] for name, r in records.items()} == {
        "1_1": "complete",
        "1_2": "complete",
        "2_1": "complete",
        "3_1": "failed",
        "4_1": "complete",
    }
    assert [records[n]["attempts"] for n in names] == [1, 2, 2, 2, 1]
    assert records["3_1"]["error"] == "3_1 cannot be read"
    assert "error" not in records["1_2"]  # the crash of the first attempt is gone
    log = capsys.readouterr().out
    assert "1_2: crashed after" in log and "exit code 3" in log
    assert "Traceback: something broke" in log
    assert "2_1: no progress for" in log and "2_1: stalled after" in log
    assert "Trying the 3 segments that did not complete again" in log
    assert not scans._running(logs, "one")

    # A second supervisor goes on with what is left.
    stub({"3_1": ["complete"]})
    assert supervise(scan, logs) == 0
    assert list(json.loads((logs / "one.json").read_text())["segments"]) == ["3_1"]

    lines = []
    status([scan], logs, out=lines.append)
    text = "\n".join(lines)
    assert "one: finished" in text and "5 of 5 segments complete" in text
    assert "Tiles in bins: raw 50" in text


def test_a_supervisor_stops_when_no_segment_gets_recorded(tmp_path, stub, monkeypatch):
    scan = scan_of(tmp_path, ["1_1", "2_1", "3_1", "4_1"])
    monkeypatch.setattr(scans, "STOP_AFTER", 2)
    stub({name: ["crash"] for name in ("1_1", "2_1", "3_1", "4_1")})
    assert supervise(scan, tmp_path / "logs", retries=0) == 1
    state = json.loads((tmp_path / "logs" / "one.json").read_text())
    assert state["status"] == "stopped" and "2 segment processes in a row" in state["reason"]
    assert list(state["segments"]) == ["1_1", "2_1"]
    lines = []
    status([scan], tmp_path / "logs", out=lines.append)
    text = "\n".join(lines)
    assert "one: stopped" in text and "0 of 4 segments complete" in text
    assert "1_1 did not complete: crashed" in text


def test_a_second_supervisor_does_not_start(tmp_path, stub):
    scan = scan_of(tmp_path, ["1_1"])
    (tmp_path / "logs").mkdir()
    held = Lock(tmp_path / "logs" / "one.lock")
    assert held.acquire()
    try:
        assert supervise(scan, tmp_path / "logs") == 3
        lines = []
        stop([scan], tmp_path / "logs", out=lines.append)
    finally:
        held.release()
    assert not (tmp_path / "out").exists()
    assert "left no state saying which" in lines[0]


def test_problems_and_warnings_hold_back_a_start(tmp_path, monkeypatch):
    scan = scan_of(tmp_path, ["1_1", "2_1"])
    logs = tmp_path / "logs"
    launched = []
    monkeypatch.setattr(scans, "_detach", lambda *args: launched.append(args))
    lines = []
    # Results of something else in the output folder stop the start.
    scan.output.mkdir()
    (scan.output / "notes.txt").touch()
    assert start(tmp_path / "scans.ini", [scan], logs, trial=False, out=lines.append) == 1
    assert any("holds other files" in line for line in lines)
    assert any(line.startswith("Nothing was started") for line in lines)
    (scan.output / "notes.txt").unlink()

    # Two scans on one GPU warn, and without --yes nothing starts.
    other = scan_of(tmp_path / "second", ["1_1"])
    other.name = "two"
    del lines[:]
    monkeypatch.setattr(sys, "stdin", None)
    assert start(tmp_path / "scans.ini", [scan, other], logs, trial=False, out=lines.append) == 1
    assert any("Warning, one and two: share GPU 0" in line for line in lines)
    assert any("Add --yes" in line for line in lines) and not launched


def trial_manifest(output, **changes):
    entry = {
        "stripe": 2,
        "vertical": 1,
        "status": "complete",
        "pages": 30,
        "accepted_tiles": 1200,
        "tested_tiles": 1400,
        "accepted_by_rows": {"within_pages": [900, 1000], "across_pages": [300, 400]},
        "seam_correlation_p90": [0.99, 0.98, None],
        "seconds": {"total": 15.0},
        "methods": {
            name: {"used_tiles": 0 if name == "pooled" else 1100, "failed_batches": 0}
            for name in METHODS
        },
    }
    entry.update(changes)
    manifest = {
        "format": "actis-stripes-2",
        "header": None,
        "pixel_source": "--pixel-nm",
        "gpu": "NVIDIA RTX 6000 Ada Generation",
        "tile_size": 476,
        "settings": {"pixel_um": [0.021, 0.021], "max_pages": 30, "methods": list(METHODS)},
        "reference_values": {name: [None if name == "pooled" else 3.5] for name in METHODS},
        "reference_errors": {},
        "segments": [entry],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / MANIFEST).write_text(json.dumps(manifest))
    return manifest


def test_a_trial_reports_what_it_found(tmp_path, monkeypatch):
    scan = scan_of(tmp_path, ["1_1", "2_1", "3_1"])
    plan = scans._plan(scan)
    folder = tmp_path / "trial"
    manifest = trial_manifest(folder)
    monkeypatch.setattr(scans, "_page_bytes", lambda segment: 2048 * 1376)
    trial = scans._Trial(plan.segments[1], 30, "", manifest, folder)
    lines, problems, warnings, need = scans._judge(scan, plan, trial)
    text = "\n".join(lines)
    assert problems == []
    assert "Pixels of 21 x 21 nm from --pixel-nm" in text
    assert "1,200 of 1,400 tiles accepted (90% within pages, 75% across pages)" in text
    assert "seams 0.99 0.98 n/a" in text
    assert "With values: raw, contour, model" in text and "0.50 s per merged page" in text
    assert warnings == [
        "pooled gave no value on any accepted tile, nor on the reference tile itself, "
        "so it will add nothing to this scan's results"
    ]
    assert need == 0  # the stub scan's files are empty

    # A method that fails, no accepted tile, and an output folder of another run.
    entry = manifest["segments"][0]
    entry["methods"]["model"].update(failed_batches=2, error="IndexError: x")
    entry["accepted_tiles"] = 0
    plan.earlier = {**manifest, "settings": {**manifest["settings"], "max_pages": None, "tile_um": 5}}
    _, problems, warnings, _ = scans._judge(scan, plan, trial)
    assert any("model failed in 2 batches of the trial" in w for w in warnings)
    assert any("No tile of the first 30 pages of 2_1 matched" in w for w in warnings)
    assert any("holds a stripe run with other settings" in p for p in problems)

    failed = trial_manifest(tmp_path / "failed", status="failed", error="bad header")
    trial = scans._Trial(plan.segments[1], 30, "", failed, folder)
    assert scans._judge(scan, plan, trial)[1] == ["The trial on 2_1 failed: bad header"]
    trial = scans._Trial(plan.segments[1], 30, "a\nACTIS stripes: no GPU 3\n", None, folder)
    assert "It ended with: a | ACTIS stripes: no GPU 3" in scans._judge(scan, plan, trial)[1][0]
