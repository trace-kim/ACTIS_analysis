"""Windows CLI for the eight-position reference benchmark."""

import argparse
import json
import os
import platform
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import numpy as np
from PIL import Image, PngImagePlugin

from . import __version__
from .analysis import analyze
from .files import read_reference, save_json, write_table
from .report import make_report
from .scores import summarize
from .stress import run_stress
from .synthetic import render_repeats


def _patterns(study):
    patterns = []
    for r in study.regions:
        contours = [
            s
            for s in study.holes
            if s["method"] == "contour" and s["region"] == r.region_id and s["valid"]
        ]
        size = [s["cd_px"] for s in contours]
        cd = np.median(size) if size else 0.4 * r.pitch
        # Use observed apparent dimensions only to select a synthetic stress scale.
        cd = float(cd) * study.pixel_nm
        lattice = r.lattice if r.lattice in ("square", "staggered") else "oblique"
        shapes = (
            [r.shape] if r.shape != "ambiguous" else ["circle", "square", "ellipse"]
        )
        ratio = (
            np.median(
                [s["height_px"] / s["width_px"] for s in contours if s["width_px"] > 0]
            )
            if contours
            else 0.75
        )
        for shape in shapes:
            patterns.append(
                dict(
                    region=r.region_id,
                    shape=shape,
                    lattice=lattice,
                    pitch_px=r.pitch,
                    cd_nm=float(cd),
                    angle_deg=float(np.degrees(r.angle)),
                    ellipse_ratio=float(np.clip(ratio, 0.05, 1.0))
                    if shape == "ellipse"
                    else 1.0,
                    basis_xy=r.basis,
                )
            )
    return patterns


def _response(family, method, scenario):
    return next(
        (
            s
            for s in family
            if s["method"] == method
            and s["scenario"] == scenario
            and s["status"] == "ok"
            and s.get("shape_agreement")
        ),
        {},
    )


def _equivalent_nm(family, cv, method):
    rate = _response(family, method, "global_cd_tiles").get("fractional_slope", np.nan)
    return cv / (100 * rate) if rate > 0 else np.nan


def _comparison(study, scores, stress):
    result = []
    for row in scores:
        if (
            row["level"] != "Pilot tile"
            or row["metric"] not in ("CD", "Integrated signal")
            or row["method"] == "raw"
        ):
            continue
        region = next(r for r in study.regions if r.region_id == row["region"])
        family = [
            s
            for s in stress
            if s.get("region") == region.region_id
            and s["shape"] == region.shape
            and s.get("stress_group") == "source_geometry"
        ]
        raw = row.get("paired_raw_percent", np.nan)
        adjusted = _equivalent_nm(family, row["percent"], row["method"])
        raw_adjusted = _equivalent_nm(family, raw, "raw")
        local = _response(family, row["method"], "local_cd")
        signal = local.get("measured_change", np.nan)
        nuisance = [
            _response(family, row["method"], scenario).get("rmse", np.nan)
            for scenario in (
                "gain_only",
                "offset_only",
                "blur_only",
                "noise_only",
                "nuisance",
            )
        ]
        full_coverage = row["valid"] == row["possible"]
        if not row["valid"]:
            finding = "No common tile measurements"
        elif not np.isfinite([adjusted, raw_adjusted, signal]).all() or signal <= 0:
            finding = "Size sensitivity unresolved"
        elif not np.isfinite(nuisance).all() or max(nuisance) >= signal:
            finding = (
                "Nuisance response unresolved or exceeds simulated local CD signal"
            )
        elif not full_coverage:
            finding = "Partial tile coverage; inspect exclusions"
        elif row["percent"] < raw and adjusted < raw_adjusted:
            finding = (
                "Lower scatter on measured tiles after modeled response adjustment"
            )
        else:
            finding = "No consistent advantage over raw brightness"
        result.append(
            dict(
                region=region.region_id,
                method=row["method"],
                tiles=row["valid"],
                percent=row["percent"],
                eligible_tiles=row["possible"],
                raw_percent=raw,
                equivalent_nm=adjusted,
                raw_equivalent_nm=raw_adjusted,
                simulated_local_response=signal,
                max_nuisance_rmse=max(nuisance)
                if np.isfinite(nuisance).all()
                else np.nan,
                finding=finding,
            )
        )
    return result


def _hardware(backend):
    from .fit import array_backend

    xp = array_backend(backend)
    state = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "requested_backend": backend,
        "fit_backend": "cpu" if xp is np else "cuda",
        "package_versions": {
            p: version(p) for p in ("numpy", "scipy", "pillow", "matplotlib")
        },
    }
    if xp is not np:
        device = xp.cuda.Device()
        state.update(
            gpu=xp.cuda.runtime.getDeviceProperties(device.id)["name"].decode(),
            gpu_index=device.id,
            visible_devices=xp.cuda.runtime.getDeviceCount(),
            cupy=xp.__version__,
        )
        # Initialize CUDA kernels before recording fit times.
        xp.sum(xp.asarray([1.0, 2.0])).get()
    return state


def _read_speed(path):
    start = perf_counter()
    size = 0
    buf = bytearray(8 * 1024 * 1024)
    with path.open("rb") as src:
        while n := src.readinto(buf):
            size += n
    elapsed = perf_counter() - start
    return dict(
        file=path.name, seconds=elapsed, bytes=size, mib_s=size / 2**20 / elapsed
    )


def _pipeline(args, out):
    pipeline_start = perf_counter()
    simulated = args.command == "synthetic"
    synthetic_settings = vars_for_synthetic(args) if simulated else None
    hardware = _hardware(args.backend)
    if simulated:
        source = out / "synthetic-input"
        source.mkdir()
        print("Rendering eight independent synthetic acquisitions", flush=True)
        simulated_images, truth = render_repeats(
            shape=args.shape,
            lattice=args.lattice,
            image_shape=(args.height, args.width),
            pitch_px=args.pitch_px,
            cd_nm=args.cd_nm,
            cd_variation_nm=args.cd_variation_nm,
            pixel_nm=args.pixel_nm,
            angle_deg=args.angle_deg,
            seed=args.seed,
            supersample=5,
            sigma_xy_px=args.sigma_xy_px,
            n_repeats=8,
        )
        tag = PngImagePlugin.PngInfo()
        tag.add_text("actis-data-origin", "synthetic")
        for k, im in enumerate(simulated_images):
            Image.fromarray(im).save(source / f"synthetic-{k + 1:02}.png", pnginfo=tag)
        np.savez_compressed(out / "synthetic-truth.npz", **truth)
    else:
        source = args.input
    start = perf_counter()
    images, info = read_reference(source)
    input_seconds = perf_counter() - start
    simulated = simulated or any(row.get("synthetic") for row in info)
    start = perf_counter()
    study = analyze(
        images,
        args.pixel_nm,
        args.tile_um,
        backend=args.backend,
        progress=lambda m: print(m, flush=True),
        allow_unpatterned=True,
    )
    analysis_seconds = perf_counter() - start
    if len({r["sha256"] for r in info}) < len(info):
        study.notes.append(
            "Some inputs are byte-identical. Duplicated acquisitions reduce the apparent dispersion."
        )
    if not simulated and images[0].shape != (2048, 676):
        study.notes.append(
            f"Observed image shape {images[0].shape} differs from the described 2048 × 676 reference frames."
        )
    if study.tile_um >= 100 and not any(t["method"] == "raw" for t in study.tiles):
        study.notes.append("No complete 100 µm bins are available.")
    print(
        "Scoring matched sites and running the independent synthetic response study",
        flush=True,
    )
    start = perf_counter()
    summary, splits, spatial, aggregates = summarize(study)
    performance = {
        "input_seconds": input_seconds,
        "statistics_seconds": perf_counter() - start,
        "hardware": hardware,
        "stages": {},
    }
    total_pixels = sum(im.size for im in images)
    stages = {
        **study.timing,
        "all methods including association and reduction": analysis_seconds,
    }
    for name, seconds in stages.items():
        rate = total_pixels / (2048 * 1376) / max(seconds, 1e-12)
        performance["stages"][name] = dict(
            seconds=seconds, pages_per_s=rate, projected_seconds=1500 / rate
        )
    if args.io_sample:
        performance["disk"] = _read_speed(args.io_sample)
    start = perf_counter()
    stress = run_stress(_patterns(study), study.pixel_nm, args.backend)
    performance["stress_seconds"] = perf_counter() - start
    comparisons = _comparison(study, summary, stress)
    print("Writing offline figures, report and measurement tables", flush=True)
    start = perf_counter()
    regions = [
        dict(
            id=r.region_id,
            shape=r.shape,
            best_shape=r.best_shape,
            angle_rad=r.angle,
            confidence=r.confidence,
            lattice=r.lattice,
            basis_px=r.basis,
            bounds_px=r.bounds,
            pitch_px=r.pitch,
            sites=len(r.centers),
            diagnostics=r.diagnostics,
        )
        for r in study.regions
    ]
    save_json(
        out / "run.json",
        dict(
            version=__version__,
            synthetic=simulated,
            inputs=info,
            frames=study.frames,
            regions=regions,
            common_bounds_xyxy_px=study.bounds,
            registration=study.registration,
            notes=study.notes,
            hardware=hardware,
            pixel_nm=study.pixel_nm,
            tile_um=study.tile_um,
            historical_tile_um=100,
            sample_ddof=1,
            repeat_indices="Measurement table repeat indices and triplets are zero-based; displayed acquisition numbers begin at one.",
            source_folder=str(args.input.resolve())
            if args.command != "synthetic"
            else "synthetic-input",
            synthetic_settings=synthetic_settings,
        ),
    )
    write_table(out / "holes.csv.gz", study.holes)
    for filename, rows in [
        ("precision", summary),
        ("triplets", splits),
        ("tile_scores", spatial),
        ("tile_values", aggregates),
        ("native_tiles", study.tiles),
        ("stress", stress),
        ("comparison", comparisons),
    ]:
        write_table(out / (filename + ".csv"), rows)
    performance["export_seconds"] = perf_counter() - start
    performance["report_seconds"] = make_report(
        out,
        images,
        info,
        study,
        summary,
        splits,
        spatial,
        stress,
        comparisons,
        performance,
        simulated,
        synthetic_settings=synthetic_settings,
    )
    performance["total_seconds"] = perf_counter() - pipeline_start
    save_json(out / "performance.json", performance)


def vars_for_synthetic(args):
    return {
        k: getattr(args, k)
        for k in (
            "shape",
            "lattice",
            "height",
            "width",
            "pitch_px",
            "cd_nm",
            "cd_variation_nm",
            "sigma_xy_px",
            "angle_deg",
            "seed",
        )
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bench = commands.add_parser(
        "benchmark", help="Time resident tile kernels on the target workstation"
    )
    bench.add_argument("--input", type=Path, required=True)
    bench.add_argument(
        "--output", type=Path, required=True, help="New benchmark JSON file"
    )
    bench.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Minimum measured seconds per candidate",
    )
    bench.add_argument(
        "--workers", type=int, default=1, help="Number of concurrently processed frames"
    )
    bench.add_argument("--pixel-nm", type=float, default=21.0)
    bench.add_argument("--tile-um", type=float, default=5.0)
    bench.add_argument(
        "--io-sample",
        type=Path,
        help="Optional representative board file for disk timing",
    )
    real = commands.add_parser(
        "analyze-reference", help="Analyze eight remote-workstation PNGs"
    )
    real.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing exactly eight 16-bit grayscale PNGs",
    )
    synth = commands.add_parser(
        "synthetic", help="Generate, analyze and score eight simulated reference frames"
    )
    synth.add_argument(
        "--shape", choices=["square", "circle", "ellipse"], default="square"
    )
    synth.add_argument("--lattice", choices=["square", "staggered"], default="square")
    synth.add_argument("--height", type=int, default=192)
    synth.add_argument("--width", type=int, default=256)
    synth.add_argument("--pitch-px", type=float, default=6.77)
    synth.add_argument("--cd-nm", type=float, default=80.0)
    synth.add_argument(
        "--cd-variation-nm",
        type=float,
        default=7.0,
        help="Maximum fixed site CD deviation from nominal, in mask nm",
    )
    synth.add_argument(
        "--sigma-xy-px",
        nargs=2,
        type=float,
        default=(0.70, 0.76),
        metavar=("X", "Y"),
        help="Nominal simulated optical blur widths in native pixels",
    )
    synth.add_argument("--angle-deg", type=float, default=0.0)
    synth.add_argument("--seed", type=int, default=47)
    for command in (real, synth):
        command.add_argument(
            "--output", type=Path, required=True, help="A new or empty result folder"
        )
        command.add_argument(
            "--pixel-nm",
            type=float,
            default=21.0,
            help="Nominal mask nm per native pixel",
        )
        command.add_argument(
            "--tile-um",
            type=float,
            default=5.0,
            help="Pilot tile side in mask micrometres",
        )
        command.add_argument(
            "--backend", choices=["cpu", "cuda", "auto"], default="auto"
        )
        command.add_argument(
            "--io-sample",
            type=Path,
            help="Optional representative board file for a sequential read timing",
        )
    synth.set_defaults(tile_um=1.25)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dst = args.output.resolve()
    if args.command == "benchmark":
        from .throughput import benchmark

        if dst.exists():
            print(f"Choose a new benchmark output file: {dst}", file=sys.stderr)
            return 2
        try:
            result = benchmark(
                args.input,
                args.seconds,
                args.pixel_nm,
                args.tile_um,
                args.io_sample,
                args.workers,
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
            print(f"Throughput benchmark: {dst}")
            return 0
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"ACTIS benchmark: {exc}", file=sys.stderr)
            return 1
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        print(f"Output must be a new or empty directory: {dst}", file=sys.stderr)
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(prefix=".actis-", dir=dst.parent))
    try:
        _pipeline(args, stage)
        if dst.is_dir():
            dst.rmdir()  # succeeds only while still empty
        stage.rename(dst)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ACTIS benchmark: {exc}", file=sys.stderr)
        return 1
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Report: {dst / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
