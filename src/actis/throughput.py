"""Measure qualified reference kernels on resident, unmodified PNG acquisitions."""

import os
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, UnidentifiedImageError

from .analysis import common_bounds, rect_mean
from .discovery import discover, register
from .estimators import estimate_holes
from .files import native, read_reference
from .harmonics import harmonic_tiles
from .pooled import njit, pooled_tiles

BOARD_PIXELS = 2048 * 1376


def _disk(path):
    path = Path(path)
    buf = bytearray(4 * 2**20)
    passes = []
    for label in ("first_read_cache_unknown", "immediate_reread"):
        start = perf_counter()
        count = 0
        with path.open("rb") as src:
            while n := src.readinto(buf):
                count += n
        elapsed = perf_counter() - start
        passes.append(
            dict(
                label=label,
                seconds=elapsed,
                bytes=count,
                MB_per_s=count / 1e6 / elapsed,
            )
        )
    if not passes[0]["bytes"]:
        raise ValueError("I/O sample is empty")
    board_pages = None
    try:
        with Image.open(path) as sample:
            if sample.format == "TIFF":
                n = sample.n_frames
                for i in range(n):
                    sample.seek(i)
                    if sample.size != (1376, 2048) or sample.mode != "L":
                        break
                else:
                    board_pages = n
    except (UnidentifiedImageError, OSError, ValueError):
        pass
    for item in passes:
        item["board_pages_per_s"] = (
            board_pages / item["seconds"] if board_pages else None
        )
    return dict(
        file=path.name,
        passes=passes,
        board_pages=board_pages,
        cache_policy="No eviction attempted. Both reads may be served from the Windows file cache.",
        sample_gate="Uses on-disk bytes including compression, and page count only for verified 8-bit grayscale board-size TIFFs.",
    )


def _layout(images, pixel_nm, tile_um):
    regions = discover(images[0])
    if not regions:
        raise ValueError(
            "No periodic hole array was detected in the reference acquisition"
        )
    pitch = min(r.pitch for r in regions)
    regs = [register(images[0], im, pitch_px=pitch) for im in images[1:]]
    shifts = np.array([(0, 0), *(r.shift_xy for r in regs)])
    if not np.isfinite(shifts).all():
        raise ValueError("Registration could not establish the common footprint")
    x0, y0, x1, y1 = common_bounds(images[0].shape, shifts)
    side = tile_um * 1000 / pixel_nm
    if side < 4:
        raise ValueError(
            "Benchmark tiles must be at least four native pixels on a side"
        )
    nx, ny = np.maximum(0, np.floor([(x1 - x0) / side, (y1 - y0) / side]).astype(int))
    prepared = []
    for region in regions:
        left, top, right, bottom = np.asarray(region.bounds) - 0.5
        corners = []
        lookup = {}
        for j in range(ny):
            for i in range(nx):
                x, y = x0 + i * side, y0 + j * side
                if x >= left and y >= top and x + side <= right and y + side <= bottom:
                    lookup[i, j] = len(corners)
                    corners.append((x, y, x + side, y + side))
        xy = region.centers
        ids = np.floor((xy - [x0, y0]) / side).astype(int)
        tile_index = np.array([lookup.get(tuple(p), -1) for p in ids], dtype=int)
        eligible = ((xy >= [x0, y0]) & (xy < [x1, y1])).all(axis=1)
        prepared.append((region, xy[eligible], tile_index[eligible], corners))
    ambiguous = [False] + [bool(r.ambiguous) for r in regs]
    return shifts, prepared, (x0, y0), side, ambiguous


def _kernel(method, image, shift, prepared, origin, side, ambiguous):
    """Per-frame reductions match the reference metrology's chosen native observables."""
    attempted = valid = tile_count = 0
    useful_area = 0.0
    signature = []
    for region, centers, tile_id, corners in prepared:
        n_tiles = len(corners)
        tile_count += n_tiles
        if method == "raw":
            moved = np.tile(shift, 2)
            values = [rect_mean(image, tuple(np.array(box) + moved)) for box in corners]
            ok = np.isfinite(values)
            attempted += n_tiles
            valid += int(ok.sum())
            useful_area += ok.sum() * side**2
            signature.extend(values)
        elif method == "flux":
            fits = estimate_holes(
                image,
                centers + shift,
                pitch_px=region.pitch,
                shape=region.shape,
                angle_rad=region.angle,
                methods=("flux",),
            )["flux"]
            flags = np.array([r["valid"] for r in fits], dtype=bool)
            in_tile = tile_id >= 0
            keep = in_tile & flags
            attempted += int(in_tile.sum())
            valid += int(keep.sum())
            total = np.bincount(tile_id[in_tile], minlength=n_tiles)
            counts = np.bincount(tile_id[keep], minlength=n_tiles)
            # Each tile contributes proportionally to its valid-site fraction.
            useful_area += float(np.sum(counts / np.maximum(1, total)) * side**2)
            for field in ("flux_dn_px2", "x_px", "y_px"):
                measurements = np.asarray([r[field] for r in fits], dtype=float)
                sums = np.bincount(
                    tile_id[keep], weights=measurements[keep], minlength=n_tiles
                )
                signature.extend(np.divide(sums, np.maximum(counts, 1)))
        else:
            estimator = harmonic_tiles if method == "harmonic" else pooled_tiles
            rows = estimator(
                image, region, shift_xy=tuple(shift), tile_px=side, origin_xy=origin
            )
            include = {(round(b[0], 6), round(b[1], 6)) for b in corners}
            selected = [
                r for r in rows if (round(r["x0"], 6), round(r["y0"], 6)) in include
            ]
            ok = [
                r for r in selected if r["status"] == "ok" and np.isfinite(r["cd_px"])
            ]
            attempted += n_tiles
            valid += len(ok)
            useful_area += sum(r.get("valid_fraction", 1.0) for r in ok) * side**2
            signature.extend(
                [
                    r.get(field, np.nan)
                    for r in selected
                    for field in ("cd_px", "dx_px", "dy_px")
                ]
            )
    output = dict(
        attempted=attempted,
        valid=valid,
        tiles=tile_count,
        attempted_area=tile_count * side**2,
        useful_area=useful_area if not ambiguous else 0.0,
    )
    return output, np.asarray(signature, dtype=float)


def _timed(method, images, shifts, layout, origin, side, ambiguous, seconds, workers):
    def run(i):
        return _kernel(method, images[i], shifts[i], layout, origin, side, ambiguous[i])

    t0 = perf_counter()
    single = [run(i) for i in range(len(images))]
    warm = perf_counter() - t0
    timed, cycles, identical = 0.0, 0, True
    cycles_per_batch = (workers + len(images) - 1) // len(images)
    tasks = list(range(len(images))) * cycles_per_batch
    # Each call owns its work arrays, with shared read-only images and geometry.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while cycles == 0 or timed < seconds:
            start = perf_counter()
            result = (
                list(pool.map(run, tasks)) if workers > 1 else [run(i) for i in tasks]
            )
            timed += perf_counter() - start
            cycles += cycles_per_batch
            for i, (b, y) in zip(tasks, result):
                a, x = single[i]
                identical &= (
                    a == b
                    and x.shape == y.shape
                    and np.allclose(x, y, rtol=1e-7, atol=1e-7, equal_nan=True)
                )
    totals = {key: sum(item[0][key] for item in single) for key in single[0][0]}
    input_px = sum(im.size for im in images)
    pixel_rate = input_px * cycles / timed
    valid_rate = totals["useful_area"] * cycles / timed
    return dict(
        method=method,
        observable={
            "raw": "Mean native brightness (DN)",
            "flux": "Background-subtracted aperture signal (DN·px²), centroid",
            "harmonic": "Apparent tile CD (px), lattice phase placement",
            "pooled": "Apparent native aperture CD (px), mean center displacement",
        }[method],
        workers=workers,
        warm_serial_seconds=warm,
        timed_seconds=timed,
        timed_cycles=cycles,
        timed_images=len(images) * cycles,
        checksum_matches_serial=bool(identical),
        support_per_cycle=totals,
        coverage_fraction=totals["useful_area"] / input_px,
        input_Mpixels_per_s=pixel_rate / 1e6,
        input_board_equivalents_per_s=pixel_rate / BOARD_PIXELS,
        valid_area_board_equivalents_per_s=valid_rate / BOARD_PIXELS,
        projected_seconds_for_1500_valid_area_equivalents=1500
        * BOARD_PIXELS
        / valid_rate
        if valid_rate > 0
        else None,
    )


def benchmark(source_folder, seconds, pixel_nm, tile_um, io_sample=None, workers=1):
    """Cycle eight true native crops; seconds is a minimum per-kernel timed budget.

    Reported board equivalents scale area at the observed pattern density. They
    exclude discovery, alignment, disk, TIFF decode, and report generation.
    Read-only in-memory acquisitions are processed independently by frame workers.
    """
    if (
        not np.isfinite([seconds, pixel_nm, tile_um]).all()
        or seconds < 0
        or min(pixel_nm, tile_um) <= 0
    ):
        raise ValueError(
            "positive finite pitch and tile sizes and nonnegative timed seconds are required"
        )
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    start = perf_counter()
    images, info = read_reference(source_folder)
    loaded = perf_counter() - start
    workers_used = min(int(workers), os.cpu_count() or 1)
    start = perf_counter()
    shifts, layout, origin, side, ambiguous = _layout(images, pixel_nm, tile_um)
    setup = perf_counter() - start
    methods = [
        _timed(
            m, images, shifts, layout, origin, side, ambiguous, seconds, workers_used
        )
        for m in ("raw", "flux", "harmonic", "pooled")
    ]
    disk = _disk(io_sample) if io_sample else None
    # 2 GB in 30–60 s is stated in decimal bytes. Assumes uncompressed 8-bit input.
    targets = dict(
        stated_slow_uncompressed=2e9 / 60 / BOARD_PIXELS,
        stated_fast_uncompressed=2e9 / 30 / BOARD_PIXELS,
    )
    if disk and disk["board_pages"]:
        targets.update({p["label"]: p["board_pages_per_s"] for p in disk["passes"]})
    for result in methods:
        usable = result["checksum_matches_serial"] and result["coverage_fraction"] > 0
        result["compute_rate_meets_target"] = {
            label: bool(result["valid_area_board_equivalents_per_s"] >= rate)
            if usable
            else None
            for label, rate in targets.items()
        }
    return native(
        dict(
            description="Native kernel benchmark; 2048 × 1376 × uint8 board equivalents by useful image area.",
            provenance="synthetic"
            if all(i["synthetic"] for i in info)
            else "mixed"
            if any(i["synthetic"] for i in info)
            else "user-supplied reference",
            inputs=info,
            native_crop_shape_yx=images[0].shape,
            reference_dtype=str(images[0].dtype),
            preload_read_hash_decode_seconds=loaded,
            discovery_and_alignment_seconds=setup,
            source_pixel_nm=pixel_nm,
            tile_um=tile_um,
            tile_px=side,
            registration_ambiguous=ambiguous,
            regions=[
                dict(
                    shape=r.shape, pitch_px=r.pitch, sites=len(s), complete_tiles=len(c)
                )
                for r, s, _, c in layout
            ],
            requested_seconds_per_method=seconds,
            requested_workers=int(workers),
            active_workers=workers_used,
            cpu_count=os.cpu_count(),
            platform=platform.platform(),
            pooled_backend="numba" if njit else "numpy",
            thread_environment={
                k: os.getenv(k)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            board_pixels=BOARD_PIXELS,
            target_board_pages_per_s=targets,
            io_sample=disk,
            methods=methods,
            interpretation="Rates are warm CPU kernel measurements at observed pattern density. Useful area covers complete tiles and discounts rejected measurements or unresolved registration. Flux remains an intensity proxy. Gate values compare compute only; source HDD, TIFF decoding and new-field discovery require separate validation.",
        )
    )
