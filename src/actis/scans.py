"""Measure several scans unattended, each on its own GPU.

A config file lists the scans: the scan folder, the folder for its results,
the GPU that measures it, and any option of the stripe run. Starting measures
a few pages of every scan at once, each on its own GPU, and reports what that
trial found, so that a wrong path, a reference that does not fit or a method
that fails shows up while someone is there to fix it. Then every scan gets a
supervisor that runs in the background, also after the console that started
it is closed.

A supervisor measures one segment per process. A process that fails, crashes
or stops making progress costs only its own segment, and the next segment
starts with a fresh GPU. Segments that did not complete are tried once more at
the end. Starting again continues where the runs ended, also after the PC has
restarted, because the stripe run keeps every segment it has completed.

    python scripts/measure_scans.py scans.ini            check, then start
    python scripts/measure_scans.py scans.ini --status   what the runs have done
    python scripts/measure_scans.py scans.ini --stop     stop them
"""

import argparse
import configparser
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .stripes import MANIFEST, build_parser, find_segments, replace

# Stripe options that take several values, written with spaces between them.
LISTS = {"method", "stripe"}
# Options that hold a path, taken from the folder of the config file when relative.
PATHS = {"reference"}
# Set by the runner for every process it starts.
NOT_IN_CONFIG = {"segment", "device", "help"}
YES, NO = {"yes", "true", "on"}, {"no", "false", "off"}
SCAN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# A heartbeat of the stripe run: seconds since it began, then what it is doing.
HEARTBEAT = re.compile(r"\d+s \| (.*)")
TRIAL_PAGES = 30
TRIAL_TIMEOUT = 1800
# A segment process whose progress has not changed for this long is stopped.
STALL_MINUTES = 30
# After this many segments in a row that did not complete, wait this long
# before each next one, so that a disk that is away for a while costs few.
PAUSE_AFTER, PAUSE_SECONDS = 3, 600
# This many processes in a row that ended without recording their segment
# point to something outside the segments, such as the GPU or the header.
STOP_AFTER = 5
POLL_SECONDS = 5
STATE_SECONDS = 60
# Time to start a segment process, prepare the reference and save.
SEGMENT_OVERHEAD = 5
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class Scan:
    """One scan of the config file and the stripe run that measures it."""

    name: str
    root: Path
    output: Path
    gpu: int
    options: list  # stripe options other than the scan folder, --output and --gpu
    parsed: argparse.Namespace  # all options, as the stripe run reads them

    def command(self, *extra, output=None):
        return [
            sys.executable,
            "-m",
            "actis.stripes",
            str(self.root),
            "--output",
            str(output or self.output),
            "--gpu",
            str(self.gpu),
            *self.options,
            *extra,
        ]

    def segments(self):
        found, _ = find_segments(self.root, self.parsed.boards)
        if self.parsed.stripe is not None:
            found = [s for s in found if s.stripe in set(self.parsed.stripe)]
        return found


def read_config(path):
    """The scans listed in the config file at `path`."""
    path = Path(path).resolve()
    config = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8-sig") as handle:
            config.read_file(handle)
    except FileNotFoundError:
        raise ValueError(f"There is no config file {path}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not UTF-8 text. Save it as UTF-8") from None
    except configparser.Error as error:
        raise ValueError(f"{path}: {error}") from None

    def place(value):
        value = Path(value.strip().strip('"')).expanduser()
        return value if value.is_absolute() else (path.parent / value).resolve()

    scans, outputs = [], {}
    for name in config.sections():
        if not SCAN_NAME.fullmatch(name):
            raise ValueError(
                f"[{name}]: name a scan with letters, digits, '.', '-' and '_' only"
            )
        items = dict(config.items(name))
        missing = [key for key in ("root", "output", "gpu") if not items.get(key)]
        if missing:
            raise ValueError(f"[{name}] needs {' and '.join(missing)}")
        root, output = place(items.pop("root")), place(items.pop("output"))
        gpu = items.pop("gpu").strip()
        if not gpu.isdigit():
            raise ValueError(f"[{name}] gpu must be a GPU number such as 0, not {gpu}")
        options = []
        for key, value in items.items():
            if key in NOT_IN_CONFIG:
                raise ValueError(f"[{name}] {key} cannot be set in the config file")
            value = value.strip()
            if value.lower() in NO:
                continue
            options.append(f"--{key}")
            if value.lower() in YES:
                continue
            if key in LISTS:
                options.extend(value.replace(",", " ").split())
            else:
                options.append(str(place(value)) if key in PATHS else value)
        parser = build_parser()

        def refuse(message, name=name):
            raise ValueError(f"[{name}] {message}")

        parser.error = refuse
        parsed = parser.parse_args(
            [str(root), "--output", str(output), "--gpu", gpu, *options]
        )
        if output in outputs:
            raise ValueError(f"[{outputs[output]}] and [{name}] both write to {output}")
        outputs[output] = name
        scans.append(Scan(name, root, output, int(gpu), options, parsed))
    if not scans:
        raise ValueError(f"{path} lists no scans")
    return scans


class Lock:
    """An exclusive lock on a file, held until released or the process ends."""

    def __init__(self, path):
        self.path, self.handle = Path(path), None

    def acquire(self):
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self):
        if self.handle is not None:
            self.handle.close()  # which releases the lock
            self.handle = None


def _running(logs, name):
    """Whether a supervisor is measuring the scan `name`."""
    lock = Lock(logs / f"{name}.lock")
    if lock.acquire():
        lock.release()
        return False
    return True


def _read_json(path):
    for _ in range(50):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except PermissionError:  # being replaced, on Windows
            time.sleep(0.1)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    return None


def _write_json(path, data):
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
    replace(partial, path)


def _entries(output):
    """The segments of the stripe run in `output`, by name."""
    manifest = _read_json(Path(output) / MANIFEST) or {}
    return {f"{e['stripe']}_{e['vertical']}": e for e in manifest.get("segments", [])}


def _completed(output):
    """Names of the segments complete in `output`, with all their files."""
    return {
        name
        for name, entry in _entries(output).items()
        if entry["status"] == "complete"
        and all(
            (Path(output) / entry[kept]).is_file()
            for kept in ("result", "edges")
            if kept in entry
        )
    }


def _stamp(path):
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_mtime_ns, status.st_size


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _when(stamp):
    return datetime.fromisoformat(stamp).strftime("%a %d %b %H:%M")


def _gb(size):
    return f"{size / 1e9:,.0f} GB" if size >= 1e11 else f"{size / 1e9:.2g} GB"


def _hours(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _child_env():
    """The environment of a process the runner starts.

    It finds this copy of the code, and writes UTF-8 at once, whatever the
    console's code page.
    """
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (source, env.get("PYTHONPATH")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _keep_awake():
    """Keep Windows from sleeping while this process runs."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required
        )


def _kill_tree(pid):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=_NO_WINDOW,
        )
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class _Say:
    """Write a line with the time to standard output, from any thread."""

    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, line):
        with self.lock:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}", flush=True)


class _Watch:
    """What a segment process last said, and when its progress last changed."""

    def __init__(self):
        self.started = self.changed = time.monotonic()
        self.signature = self.progress = None
        self.tail = deque(maxlen=6)
        self.lock = threading.Lock()

    def see(self, line):
        beat = HEARTBEAT.fullmatch(line)
        signature = beat.group(1) if beat else line
        with self.lock:
            if beat:
                self.progress = signature
            elif line:
                self.tail.append(line)
            if signature != self.signature:
                self.signature, self.changed = signature, time.monotonic()

    def idle(self):
        with self.lock:
            return time.monotonic() - self.changed


def _run_segment(scan, name, say, state, save, stall_seconds):
    """Measure one segment in its own process and return how it went."""
    manifest = scan.output / MANIFEST
    before = _stamp(manifest)
    process = subprocess.Popen(
        scan.command("--segment", name),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_child_env(),
        creationflags=_NO_WINDOW,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    watch = _Watch()
    state["current"] = {"segment": name, "pid": process.pid, "since": _now()}
    save()

    def pump():
        for line in process.stdout:
            line = line.rstrip()
            say(f"  {line}")
            watch.see(line)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    stalled, saved = False, time.monotonic()
    while True:
        try:
            process.wait(timeout=POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        if watch.idle() > stall_seconds:
            stalled = True
            say(
                f"{name}: no progress for {stall_seconds / 60:g} minutes. "
                f"Stopping its process {process.pid}"
            )
            # The whole tree: in a virtual environment on Windows, python.exe
            # starts the interpreter that does the work as a process of its own.
            _kill_tree(process.pid)
            process.kill()
            process.wait()
            break
        if time.monotonic() - saved > STATE_SECONDS:
            state["current"]["progress"] = watch.progress
            save()
            saved = time.monotonic()
    reader.join(timeout=10)
    state["current"] = None
    seconds = round(time.monotonic() - watch.started)
    entry = _entries(scan.output).get(name) if _stamp(manifest) != before else None
    if entry is not None:
        if entry["status"] == "complete":
            return {"status": "complete", "seconds": seconds}
        return {"status": "failed", "error": entry.get("error"), "seconds": seconds}
    if stalled:
        how = f"stalled: no progress for {stall_seconds / 60:g} minutes"
    else:
        how = (
            f"crashed: its process ended with exit code {process.returncode} "
            "before recording the segment"
        )
    last = " | ".join(watch.tail)
    return {
        "status": "stalled" if stalled else "crashed",
        "error": how + (f". Last output: {last}" if last else ""),
        "seconds": seconds,
        "unrecorded": True,
    }


def supervise(scan, logs, stall_seconds=STALL_MINUTES * 60, retries=1):
    """Measure every segment of `scan` that is not complete, one process each.

    Writes what it does to standard output and its state to <logs>/<scan>.json.
    Returns 0 when every segment is complete.
    """
    say = _Say()
    logs.mkdir(parents=True, exist_ok=True)
    lock = Lock(logs / f"{scan.name}.lock")
    if not lock.acquire():
        say(f"Another supervisor is measuring {scan.name}, so this one stops")
        return 3
    _keep_awake()
    path = logs / f"{scan.name}.json"
    state = {
        "scan": scan.name,
        "pid": os.getpid(),
        "gpu": scan.gpu,
        "root": str(scan.root),
        "output": str(scan.output),
        "status": "running",
        "started": _now(),
        "finished": None,
        "reason": None,
        "total": None,
        "current": None,
        "segments": {},
    }

    def save():
        _write_json(path, state)

    save()
    try:
        names = [segment.name for segment in scan.segments()]
        state["total"] = len(names)
        save()
        say(
            f"Supervisor {os.getpid()}: {len(names)} segments of {scan.root} on GPU "
            f"{scan.gpu} into {scan.output}"
        )
        incomplete = unrecorded = 0
        for attempt in range(1 + retries):
            done = _completed(scan.output)
            todo = [name for name in names if name not in done]
            if attempt:
                todo = [name for name in todo if name in state["segments"]]
                if todo:
                    say(f"Trying the {len(todo)} segments that did not complete again")
            for number, name in enumerate(todo, 1):
                if incomplete >= PAUSE_AFTER:
                    say(
                        f"{incomplete} segments in a row did not complete. Waiting "
                        f"{PAUSE_SECONDS // 60} minutes before {name}"
                    )
                    time.sleep(PAUSE_SECONDS)
                say(f"{name}: starting, {number} of {len(todo)}")
                outcome = _run_segment(scan, name, say, state, save, stall_seconds)
                attempts = state["segments"].get(name, {}).get("attempts", 0) + 1
                state["segments"][name] = {**outcome, "attempts": attempts, "at": _now()}
                save()
                say(
                    f"{name}: {outcome['status']} after {_hours(outcome['seconds'])}"
                    + (f". {outcome['error']}" if outcome.get("error") else "")
                )
                incomplete = 0 if outcome["status"] == "complete" else incomplete + 1
                unrecorded = unrecorded + 1 if outcome.get("unrecorded") else 0
                if unrecorded >= STOP_AFTER:
                    state["status"] = "stopped"
                    state["reason"] = (
                        f"{STOP_AFTER} segment processes in a row ended without "
                        "recording their segment, so something outside the "
                        f"segments is wrong. The last one {outcome['error']}"
                    )
                    say(state["reason"])
                    return 1
        complete = len(_completed(scan.output) & set(names))
        state["status"] = "finished"
        state["reason"] = f"{complete} of {len(names)} segments complete"
        say(f"Finished: {state['reason']}")
        return 0 if complete == len(names) else 1
    except BaseException as error:
        state["status"] = "stopped"
        state["reason"] = f"The supervisor failed: {type(error).__name__}: {error}"
        say(state["reason"])
        say("".join(traceback.format_exception(type(error), error, error.__traceback__)))
        raise
    finally:
        state["current"], state["finished"] = None, _now()
        save()
        lock.release()


@dataclass
class _Plan:
    segments: list
    done: set
    earlier: dict  # the manifest already in the output folder, or None

    @property
    def remaining(self):
        return [s for s in self.segments if s.name not in self.done]


def _plan(scan):
    """What a start would measure. Raises ValueError for what stops it."""
    if not scan.root.is_dir():
        raise ValueError(f"The scan folder {scan.root} does not exist")
    segments = scan.segments()
    if not segments:
        raise ValueError(f"No segment of the chosen stripes under {scan.root}")
    earlier = None
    if (scan.output / MANIFEST).is_file():
        earlier = _read_json(scan.output / MANIFEST)
    elif scan.output.exists() and (
        not scan.output.is_dir() or any(scan.output.iterdir())
    ):
        raise ValueError(
            f"{scan.output} holds other files and no stripe run. Choose a new "
            "output folder"
        )
    done = _completed(scan.output) if earlier else set()
    return _Plan(segments, done & {s.name for s in segments}, earlier)


@dataclass
class _Trial:
    segment: object
    pages: int
    text: str
    manifest: dict
    folder: Path


def _trial(scan, plan, logs, pages):
    """Measure the first `pages` pages of a segment in the middle of the scan."""
    folder = logs / f"trial-{scan.name}"
    shutil.rmtree(folder, ignore_errors=True)
    segment = plan.segments[len(plan.segments) // 2]
    command = scan.command(
        "--segment", segment.name, "--max-pages", str(pages), output=folder
    )
    try:
        # On the console, so that Ctrl+C stops the trials too.
        done = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            timeout=TRIAL_TIMEOUT,
        )
        text = done.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as error:
        text = (error.stdout or b"").decode("utf-8", "replace")
        text += f"\nThe trial took more than {TRIAL_TIMEOUT // 60} minutes and was stopped"
    (logs / f"trial-{scan.name}.log").write_text(text, encoding="utf-8")
    return _Trial(segment, pages, text, _read_json(folder / MANIFEST), folder)


def _page_bytes(segment):
    """Bytes of one page of one board of `segment`."""
    try:
        import tifffile

        with tifffile.TiffFile(segment.files[0]) as tiff:
            page = tiff.pages[0]
            return int(page.shape[0]) * int(page.shape[1]) * page.dtype.itemsize
    except Exception:
        return None


def _judge(scan, plan, trial):
    """What a trial found.

    Returns report lines, problems that stop a start, warnings, and the bytes
    the scan's results will take, None if unknown.
    """
    lines, problems, warnings, need = [], [], [], None
    name = trial.segment.name
    entry = None
    if trial.manifest:
        entry = {f"{e['stripe']}_{e['vertical']}": e for e in trial.manifest["segments"]}
        entry = entry.get(name)
    if entry is None:
        tail = " | ".join(line for line in trial.text.strip().splitlines()[-4:])
        problems.append(f"The trial did not measure {name}. It ended with: {tail}")
        return lines, problems, warnings, need
    if entry["status"] != "complete":
        problems.append(f"The trial on {name} failed: {entry['error']}")
        return lines, problems, warnings, need
    manifest = trial.manifest
    settings = {**manifest["settings"], "max_pages": scan.parsed.max_pages}
    if plan.earlier is not None and (
        plan.earlier.get("format") != manifest["format"]
        or plan.earlier.get("settings") != settings
    ):
        problems.append(
            f"{scan.output} holds a stripe run with other settings or in another "
            "format. Choose a new output folder, or the settings of that run"
        )
    x, y = (1000 * size for size in manifest["settings"]["pixel_um"])
    lines.append(
        f"GPU {scan.gpu}: {manifest['gpu']}. Pixels of {x:.4g} x {y:.4g} nm from "
        f"{manifest['header'] or manifest['pixel_source']}, tiles of "
        f"{manifest['tile_size']} px."
    )
    shares = ", ".join(
        f"{accepted / tested:.0%} {where.replace('_', ' ')}"
        for where, (accepted, tested) in entry["accepted_by_rows"].items()
        if tested
    )
    seams = " ".join(
        "n/a" if s is None else f"{s:.2f}" for s in entry["seam_correlation_p90"]
    )
    lines.append(
        f"Trial on {name}, {entry['pages']} pages: {entry['accepted_tiles']:,} of "
        f"{entry['tested_tiles']:,} tiles accepted"
        + (f" ({shares})" if shares else "")
        + (f", seams {seams}" if seams else "")
        + "."
    )
    found, empty = [], []
    for method, row in entry["methods"].items():
        reference = manifest["reference_values"].get(method, [])
        if method in manifest.get("reference_errors", {}):
            warnings.append(
                f"{method} failed on the reference tile: "
                f"{manifest['reference_errors'][method]}"
            )
        if row.get("failed_batches"):
            warnings.append(
                f"{method} failed in {row['failed_batches']} batches of the trial, "
                f"first with {row['error']}. Its tiles there have no value"
            )
        if row["used_tiles"]:
            found.append(method)
        elif entry["accepted_tiles"]:
            empty.append(method)
            gave = [v for v in reference if v is not None]
            warnings.append(
                f"{method} gave no value on any accepted tile"
                + ("" if gave else ", nor on the reference tile itself")
                + ", so it will add nothing to this scan's results"
            )
    if found:
        lines.append(f"With values: {', '.join(found)}.")
    if not entry["accepted_tiles"]:
        warnings.append(
            f"No tile of the first {trial.pages} pages of {name} matched the "
            "reference. If those pages show the hole array, the reference does "
            "not fit this scan"
        )
    # How long the rest will take at the trial's pace.
    board_bytes = _page_bytes(trial.segment)
    remaining = plan.remaining
    if board_bytes and entry["pages"]:
        page_bytes = board_bytes * len(trial.segment.files)
        size = sum(path.stat().st_size for s in remaining for path in s.files)
        pages = size / page_bytes
        per_page = entry["seconds"]["total"] / entry["pages"]
        estimate = pages * per_page + len(remaining) * SEGMENT_OVERHEAD
        lines.append(
            f"{per_page:.2f} s per merged page in the trial. At that pace the "
            f"{len(remaining)} segments still to measure ({_gb(size)}, "
            f"about {pages:,.0f} pages) take about {_hours(estimate)}."
        )
        result = sum(p.stat().st_size for p in trial.folder.glob("*.npz"))
        need = pages * result / entry["pages"]
    return lines, problems, warnings, need


def _free_space(scans, needs):
    """Problems for output drives without room for what the runs will write."""
    by_drive, problems = {}, []
    for scan in scans:
        if scan.name not in needs:
            continue
        place = scan.output
        while not place.exists():
            place = place.parent
        drive = os.stat(place).st_dev
        need, names, _ = by_drive.get(drive, (0, [], place))
        by_drive[drive] = need + needs[scan.name], names + [scan.name], place
    for need, names, place in by_drive.values():
        free = shutil.disk_usage(place).free
        if free < 1.2 * need:
            problems.append(
                (
                    " and ".join(names),
                    f"The results need about {_gb(need)} and the drive of "
                    f"{place} has {_gb(free)} free",
                )
            )
    return problems


def _detach(command, log):
    """Start `command` so that it outlives this console, writing to `log`.

    Returns the process and whether it could leave the console's job object,
    without which closing the console may end it.
    """
    with open(log, "ab") as out:
        options = dict(
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            close_fds=True,
        )
        if os.name != "nt":
            return subprocess.Popen(command, start_new_session=True, **options), True
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                command,
                creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB,
                **options,
            )
            return process, True
        except OSError:
            return subprocess.Popen(command, creationflags=flags, **options), False


def start(
    config,
    scans,
    logs,
    *,
    trial_pages=TRIAL_PAGES,
    trial=True,
    assume_yes=False,
    stall_minutes=STALL_MINUTES,
    retries=1,
    out=print,
    ask=input,
):
    """Check every scan, then start a supervisor for each. Returns an exit code."""
    logs.mkdir(parents=True, exist_ok=True)
    problems, warnings, plans = [], [], {}
    for scan in scans:
        if _running(logs, scan.name):
            out(f"{scan.name}: already being measured. Left as it is.")
            continue
        try:
            plans[scan.name] = _plan(scan)
        except ValueError as error:
            problems.append((scan.name, str(error)))
    todo = [scan for scan in scans if scan.name in plans]
    for gpu, count in Counter(scan.gpu for scan in todo).items():
        if count > 1:
            names = " and ".join(s.name for s in todo if s.gpu == gpu)
            warnings.append((names, f"share GPU {gpu}"))
    trials = {}
    if trial and todo:
        out(
            f"Trial: the first {trial_pages} pages of one segment of every scan, "
            "all at once ..."
        )
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            running = {
                scan.name: pool.submit(_trial, scan, plans[scan.name], logs, trial_pages)
                for scan in todo
            }
            trials = {name: job.result() for name, job in running.items()}
    needs = {}
    for scan in scans:
        if scan.name not in plans:
            continue
        plan = plans[scan.name]
        out("")
        out(f"{scan.name}: {scan.root} -> {scan.output}")
        size = sum(path.stat().st_size for s in plan.segments for path in s.files)
        stripes = len({s.stripe for s in plan.segments})
        out(
            f"  {len(plan.segments)} segments in {stripes} stripes, {_gb(size)}. "
            + (
                f"Continues an earlier run: {len(plan.done)} segments complete."
                if plan.earlier
                else "New output folder."
            )
        )
        if scan.name not in trials:
            continue
        lines, stopping, warned, need = _judge(scan, plan, trials[scan.name])
        for line in lines:
            out(f"  {line}")
        if need is not None:
            needs[scan.name] = need
        problems += [(scan.name, text) for text in stopping]
        warnings += [(scan.name, text) for text in warned]
    problems += _free_space(todo, needs)
    out("")
    for name, text in problems:
        out(f"Problem, {name}: {text}.")
    for name, text in warnings:
        out(f"Warning, {name}: {text}.")
    if trials:
        out(f"The trials' output is in {logs}.")
    if problems:
        out("Nothing was started. Fix the problems above and run the command again.")
        return 1
    if not todo:
        out("Nothing to start.")
        return 0
    if warnings and not assume_yes:
        if not sys.stdin or not sys.stdin.isatty():
            out("Nothing was started. Add --yes to start despite the warnings.")
            return 1
        try:
            answer = ask("Start anyway? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            out("Nothing was started. Add --yes to start despite the warnings.")
            return 1
    started, contained = [], False
    # The supervisors' state carries their own process number, which in a
    # virtual environment on Windows is not that of the process started here.
    launched = datetime.now().replace(microsecond=0).isoformat()
    for scan in todo:
        command = [
            sys.executable,
            "-m",
            "actis.scans",
            str(config),
            "--supervise",
            scan.name,
            "--logs",
            str(logs),
            "--stall-minutes",
            str(stall_minutes),
            "--retries",
            str(retries),
        ]
        process, free = _detach(command, logs / f"{scan.name}.log")
        contained |= not free
        started.append((scan, process))
    out("")
    failed = 0
    deadline = time.monotonic() + 60
    for scan, process in started:
        while True:
            state = _read_json(logs / f"{scan.name}.json") or {}
            begun = state.get("started", "") >= launched
            ended = process.poll() is not None
            if (begun and (ended or _running(logs, scan.name))) or (
                ended or time.monotonic() > deadline
            ):
                break
            time.sleep(0.2)
        if begun:
            out(f"{scan.name}: started on GPU {scan.gpu}, supervisor {state['pid']}.")
        else:
            failed += 1
            out(f"{scan.name}: did not start. See {logs / (scan.name + '.log')}")
    quoted = f'"{config}"' if " " in str(config) else str(config)
    script = f"python scripts\\measure_scans.py {quoted}"
    out("")
    out("The runs go on after this console is closed. Do not sign out of Windows,")
    out("which ends them. Disconnecting a remote desktop session is fine.")
    if contained:
        out("Windows did not let the runs leave this console's job, so closing the")
        out("console may end them here. Leave it open, or start from a desktop session.")
    out(f"  What they have done:  {script} --status")
    out(f"  Stop them:            {script} --stop")
    out(f"  Logs:                 {logs}")
    return 1 if failed else 0


def _method_totals(entries):
    totals = {}
    for entry in entries:
        for method, row in entry.get("methods", {}).items():
            total = totals.setdefault(
                method, {"tiles": 0, "bins": 0, "failed": 0, "segments": 0, "error": None}
            )
            total["tiles"] += row.get("used_tiles", 0)
            total["bins"] += row.get("bins", 0)
            if row.get("failed_batches"):
                total["failed"] += row["failed_batches"]
                total["segments"] += 1
                total["error"] = total["error"] or row.get("error")
    return totals


def status(scans, logs, out=print):
    """Say what the runs of every scan have done."""
    for scan in scans:
        state = _read_json(logs / f"{scan.name}.json")
        entries = _entries(scan.output)
        running = _running(logs, scan.name)
        out("")
        if state is None and not entries:
            out(f"{scan.name}: not started.")
            continue
        state = state or {}
        total = state.get("total")
        complete = [n for n, e in entries.items() if e["status"] == "complete"]
        started = state.get("started")
        if running:
            elapsed = (datetime.now() - datetime.fromisoformat(started)).total_seconds()
            head = f"running on GPU {state['gpu']} since {_when(started)} ({_hours(elapsed)})"
        elif state.get("status") == "finished":
            elapsed = (
                datetime.fromisoformat(state["finished"]) - datetime.fromisoformat(started)
            ).total_seconds()
            head = f"finished {_when(state['finished'])} after {_hours(elapsed)}"
        elif state.get("status") == "stopped":
            head = f"stopped {_when(state['finished'])}. {state['reason']}"
        elif state:
            log = logs / f"{scan.name}.log"
            last = datetime.fromtimestamp(log.stat().st_mtime) if log.exists() else None
            head = (
                "not running, and it did not finish: its supervisor ended without "
                "a word"
                + (f" after {last:%a %d %b %H:%M}" if last else "")
                + ", as when the PC restarts or the user signs out. Start again to "
                "continue"
            )
        else:
            head = "measured without the runner"
        out(f"{scan.name}: {head}.")
        out(
            f"  {len(complete)} of {total if total is not None else '?'} segments "
            f"complete. Results in {scan.output}"
        )
        current = state.get("current") if running else None
        if current:
            out(
                f"  Now on {current['segment']} since {_when(current['since'])}"
                + (f": {current['progress']}" if current.get("progress") else "")
            )
        times = [
            r["seconds"] for r in state.get("segments", {}).values()
            if r.get("status") == "complete"
        ]
        if running and times and total:
            left = (total - len(complete)) * sum(times) / len(times)
            out(f"  About {_hours(left)} left at the pace so far.")
        problems = {
            n: e.get("error") for n, e in entries.items() if e["status"] == "failed"
        }
        for name, record in state.get("segments", {}).items():
            if name not in complete and record.get("status") in ("crashed", "stalled"):
                problems[name] = record.get("error")
        for name, error in sorted(problems.items())[:12]:
            out(f"  {name} did not complete: {(error or '').splitlines()[0]}")
        if len(problems) > 12:
            out(f"  ... and {len(problems) - 12} more, listed in stripes.json and the log")
        totals = _method_totals(entries[n] for n in complete)
        if totals:
            out(
                "  Tiles in bins: "
                + ", ".join(
                    f"{method} {row['tiles']:,}" if row["tiles"] else f"{method} none"
                    for method, row in totals.items()
                )
            )
        for method, row in totals.items():
            if row["failed"]:
                out(
                    f"  {method} failed in {row['failed']} batches of {row['segments']} "
                    f"segments, first with {row['error']}"
                )
        out(f"  Log: {logs / (scan.name + '.log')}")


def stop(scans, logs, out=print):
    """Stop the runs of the scans. Starting again continues them."""
    for scan in scans:
        if not _running(logs, scan.name):
            out(f"{scan.name}: not running.")
            continue
        path = logs / f"{scan.name}.json"
        state = _read_json(path) or {}
        if "pid" not in state:
            out(
                f"{scan.name}: a process holds {logs / (scan.name + '.lock')} but left "
                "no state saying which. Stop it in Task Manager."
            )
            continue
        _kill_tree(state["pid"])
        deadline = time.monotonic() + 15
        while _running(logs, scan.name) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _running(logs, scan.name):
            out(f"{scan.name}: could not stop supervisor {state['pid']}.")
            continue
        state.update(status="stopped", reason="Stopped with --stop", current=None)
        state["finished"] = _now()
        _write_json(path, state)
        out(f"{scan.name}: stopped. Starting again continues it.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Every key of a scan in the config file is an option of "
        "measure_stripes.py without its dashes. See scripts/scans.example.ini.",
    )
    parser.add_argument("config", type=Path, help="The file that lists the scans")
    what = parser.add_mutually_exclusive_group()
    what.add_argument(
        "--status", action="store_true", help="Say what the runs have done"
    )
    what.add_argument(
        "--stop", action="store_true", help="Stop the runs. Starting again continues"
    )
    what.add_argument("--supervise", metavar="SCAN", help=argparse.SUPPRESS)
    parser.add_argument(
        "--only", nargs="+", metavar="SCAN", help="Only these scans of the file"
    )
    parser.add_argument(
        "--logs",
        type=Path,
        help="Folder for logs and trials (default: <config name>-logs beside the "
        "config file)",
    )
    parser.add_argument(
        "--trial-pages",
        type=int,
        default=TRIAL_PAGES,
        help=f"Pages per scan in the trial before starting (default: {TRIAL_PAGES})",
    )
    parser.add_argument(
        "--no-trial", action="store_true", help="Start without a trial"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Start even if the checks warn"
    )
    parser.add_argument(
        "--stall-minutes",
        type=float,
        default=STALL_MINUTES,
        help="Stop a segment whose progress has not changed for this long, and go "
        f"on with the next (default: {STALL_MINUTES})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Times to try again, at the end, the segments that did not complete "
        "(default: 1)",
    )
    args = parser.parse_args(argv)
    if args.trial_pages < 1 or args.stall_minutes <= 0 or args.retries < 0:
        parser.error("Use a positive trial length and stall time, and retries >= 0")
    try:
        scans = read_config(args.config)
    except ValueError as error:
        parser.exit(2, f"ACTIS scans: {error}\n")
    config = args.config.resolve()
    logs = (args.logs or config.with_name(config.stem + "-logs")).resolve()
    named = {scan.name: scan for scan in scans}
    if args.supervise:
        if sys.stdout is not None:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        if args.supervise not in named:
            parser.exit(2, f"ACTIS scans: {config} has no scan {args.supervise}\n")
        return supervise(
            named[args.supervise], logs, args.stall_minutes * 60, args.retries
        )
    if args.only:
        unknown = [name for name in args.only if name not in named]
        if unknown:
            parser.exit(2, f"ACTIS scans: {config} has no scan {', '.join(unknown)}\n")
        scans = [named[name] for name in args.only]
    if args.status:
        status(scans, logs)
        return 0
    if args.stop:
        stop(scans, logs)
        return 0
    return start(
        config,
        scans,
        logs,
        trial_pages=args.trial_pages,
        trial=not args.no_trial,
        assume_yes=args.yes,
        stall_minutes=args.stall_minutes,
        retries=args.retries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
