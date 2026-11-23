"""Score the contour lab's workflows on synthetic images with known geometry.

    python scripts/evaluate_contour_flows.py [--markdown]

Hole arrays (three cases, eight acquisitions each, with gain, offset,
illumination, blur, shift and noise varied): how many holes are found, how
repeatable their diameter is across the acquisitions, and how closely a size
difference between holes is followed. A noisy 8-bit version of the first case.
A page of noise alone, and a mixed page (two kinds of holes, lines, a pad with
faint holes beside it, an empty dark area and an empty bright one): what is
found, and what is found that is not there.

The images come from this repository's renderer, whose optics have no
calibration to the instrument: the numbers say how the workflows differ, not
what they will measure on a mask.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actis.contourlab import PRESETS, measure, preset, run_workflow
from actis.contourlab.modules import image_contours
from actis.synthetic import _aperture, _capture, render_repeats

PIXEL_NM = 21.0
CASES = {
    "A": dict(shape="square", lattice="square", pitch_px=6.77, cd_nm=80.0),
    "B": dict(shape="circle", lattice="staggered", pitch_px=12.0, cd_nm=100.0),
    "C": dict(
        shape="square",
        lattice="square",
        pitch_px=6.0,
        cd_nm=70.0,
        sigma_xy_px=(1.0, 1.15),
    ),
}


def contours_of(image, name):
    """(contours in the image's pixels or None, seconds) of a workflow to start from."""
    start = time.perf_counter()
    run = run_workflow(image, preset(name))
    seconds = time.perf_counter() - start
    if run.error is not None or run.final.contours is None:
        return None, seconds
    return image_contours(run.final), seconds


def matched(contours, sites, reach):
    """(diameter in pixels of the closed contour at every site, NaN where there
    is none within `reach`), and how many closed contours belong to no site."""
    found = np.full(len(sites), np.nan)
    if contours is None or not len(contours):
        return found, 0
    measures = measure(contours)
    whole = contours.closed & np.isfinite(measures["diameter"]) & (measures["area"] > 0)
    if not whole.any():
        return found, 0
    centres, diameters = measures["centre"][whole], measures["diameter"][whole]
    distance, nearest = cKDTree(centres).query(sites)
    found[distance <= reach] = diameters[nearest[distance <= reach]]
    stray = len(centres) - len(np.unique(nearest[distance <= reach]))
    return found, stray


def array_case(name, settings, eight_bit=False):
    """Found share, repeatability (nm) and size response of every workflow on one case."""
    frames, truth = render_repeats(
        image_shape=(224, 256), n_repeats=8, cd_variation_nm=7.0, seed=11, **settings
    )
    if eight_bit:  # a contrast of about ten noise sigma in 8 bits
        rng = np.random.default_rng(5)
        scaled = (frames.astype(np.float64) - 1000.0) / 24000.0 * 60.0 + 30.0
        frames = np.clip(
            np.rint(scaled + rng.normal(0, 5.5, frames.shape)), 0, 255
        ).astype(np.uint8)
    pitch = settings["pitch_px"]
    rows = {}
    for flow in PRESETS:
        sizes, seconds = [], 0.0
        for run in range(len(frames)):
            contours, spent = contours_of(frames[run], flow)
            seconds += spent
            sites = truth["centers_xy_px"] + truth["shift_xy_px"][run]
            inside = ((sites > pitch) & (sites < np.array([256, 224]) - 1 - pitch)).all(
                axis=1
            )
            found, _ = matched(contours, sites, 0.35 * pitch)
            sizes.append(np.where(inside, found, np.nan))
        sizes = np.array(sizes) * PIXEL_NM  # (acquisition, site), nm
        counted = np.isfinite(sizes).sum(axis=0)
        everywhere = counted == len(frames)
        inside_any = ~np.all(np.isnan(sizes), axis=0) | True
        asked = np.array([np.isfinite(s).sum() for s in sizes]).sum()
        possible = 0
        for run in range(len(frames)):
            sites = truth["centers_xy_px"] + truth["shift_xy_px"][run]
            possible += int(
                ((sites > pitch) & (sites < np.array([256, 224]) - 1 - pitch))
                .all(axis=1)
                .sum()
            )
        row = {"found": asked / max(possible, 1), "seconds": seconds / len(frames)}
        if everywhere.sum() > 10:
            per_site = sizes[:, everywhere]
            row["repeat_nm"] = float(np.median(per_site.std(axis=0, ddof=1)))
            true = truth["area_diameter_nm"][everywhere]
            row["response"] = float(np.polyfit(true, per_site.mean(axis=0), 1)[0])
            row["median_nm"] = float(np.median(per_site))
        rows[flow] = row
        del inside_any
    return rows


def mixed_page(seed=3, samples=4):
    """An 8-bit page of 1024 x 768 px that mixes patterns, and what is on it.

    Dense small holes; wide round holes; faint holes far apart; lines; a pad
    wider than two tiles of the tile threshold, with faint holes close beside
    it; a bright area that runs out of the image, with dark dots in it; an
    empty dark area. Uneven illumination, shot noise, line offsets.
    Returns (image uint8, what is where)."""
    h, w = 768, 1024
    rng = np.random.default_rng(seed)

    def lattice(x0, y0, x1, y1, pitch, offset=0.37):
        xs = np.arange(x0 + pitch / 2 + offset, x1 - pitch / 2, pitch)
        ys = np.arange(y0 + pitch / 2 - offset, y1 - pitch / 2, pitch)
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(), gy.ravel()])

    def boxes(centres, wide, high):
        widths = np.broadcast_to(np.asarray(wide, float), (len(centres),))
        heights = np.broadcast_to(np.asarray(high, float), (len(centres),))
        return _aperture((h, w), centres, widths, heights, True, 1.0, 0.0, samples)

    small = lattice(20, 20, 330, 250, 6.77)
    wide = lattice(360, 20, 660, 250, 16.0)
    sparse = lattice(690, 20, 1000, 250, 40.0)
    beside = lattice(690, 290, 800, 530, 8.0)
    dots = lattice(60, 620, 300, 740, 20.0)
    lines_x = np.arange(32.0, 320.0, 12.0)
    lines = np.column_stack([lines_x, np.full(len(lines_x), 380.0)])
    pad = np.array([[530.3, 410.2]])
    clear = np.array([[270.0, 700.0]])

    aperture = boxes(small, 80 / PIXEL_NM, 80 / PIXEL_NM)
    np.maximum(aperture, boxes(sparse, 60 / PIXEL_NM, 60 / PIXEL_NM), out=aperture)
    np.maximum(aperture, boxes(beside, 60 / PIXEL_NM, 60 / PIXEL_NM), out=aperture)
    np.maximum(aperture, boxes(lines, 5.0, 180.0), out=aperture)
    np.maximum(aperture, boxes(pad, 300.0, 260.0), out=aperture)
    open_area = boxes(clear, 500.0, 240.0)  # runs out of the image at its lower edge
    open_area *= 1 - boxes(dots, 6.0, 6.0)
    np.maximum(aperture, open_area, out=aperture)
    widths = np.full(len(wide), 160 / PIXEL_NM)
    np.maximum(
        aperture,
        _aperture((h, w), wide, widths, widths, False, 1.0, 0.0, samples),
        out=aperture,
    )
    signal = _capture(aperture, (0.8, 0.95), samples, h, w)
    xn, yn = np.linspace(-1, 1, w)[None, :], np.linspace(-1, 1, h)[:, None]
    signal = signal * (1 + 0.10 * xn - 0.06 * yn)  # uneven illumination
    counts = 18.0 + 190.0 * signal  # grey levels of an 8-bit page
    noisy = counts + rng.normal(0, 1, counts.shape) * np.sqrt(0.6 + 0.035 * counts)
    noisy += rng.normal(0, 0.5, (h, 1))  # line offsets of the scan
    image = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    truth = {
        "holes": {
            "Small holes": small,
            "Wide holes": wide,
            "Sparse faint holes": sparse,
            "Faint holes by the pad": beside,
        },
        "pitch": {
            "Small holes": 6.77,
            "Wide holes": 16.0,
            "Sparse faint holes": 40.0,
            "Faint holes by the pad": 8.0,
        },
        "dots": dots,
        "lines": [(x - 2.5, 290.0, x + 2.5, 470.0) for x in lines_x],
        "pads": [(380.3, 280.2, 680.3, 540.2)],
        # inside the pad, the open area beside its dots, and the dark area
        "empty": [(392, 292, 668, 528), (330, 600, 510, 760), (560, 580, 1000, 750)],
    }
    return image, truth


def mixed_case(image, truth, flow):
    """What a workflow finds on the mixed page."""
    contours, seconds = contours_of(image, flow)
    row = {"seconds": seconds}
    if contours is None:
        return dict(row, failed=True)
    measures = measure(contours)
    low = np.minimum.reduceat(contours.xy, contours.start[:-1], axis=0)
    high = np.maximum.reduceat(contours.xy, contours.start[:-1], axis=0)
    for name, sites in truth["holes"].items():
        found, _ = matched(contours, sites, min(0.35 * truth["pitch"][name], 4.0))
        row[name] = float(np.isfinite(found).mean())
    # a dark dot in a bright area is the boundary of a hole in a region: its area is negative
    inner = contours.closed & (np.nan_to_num(measures["area"]) < 0)
    if inner.any():
        distance, _ = cKDTree(measures["centre"][inner]).query(truth["dots"])
        row["dots"] = float((distance <= 3.0).mean())
    else:
        row["dots"] = 0.0

    def boxed(box, slack):
        x0, y0, x1, y1 = box
        near = (np.abs(low[:, 0] - x0) < slack) & (np.abs(low[:, 1] - y0) < slack)
        near &= (np.abs(high[:, 0] - x1) < slack) & (np.abs(high[:, 1] - y1) < slack)
        return bool((near & contours.closed).any())

    row["lines"] = float(np.mean([boxed(box, 2.5) for box in truth["lines"]]))
    row["pad"] = float(np.mean([boxed(box, 2.5) for box in truth["pads"]]))
    stray = 0
    for x0, y0, x1, y1 in truth["empty"]:
        inside = (
            (low[:, 0] > x0) & (low[:, 1] > y0) & (high[:, 0] < x1) & (high[:, 1] < y1)
        )
        stray += int(inside.sum())
    row["stray"] = stray
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--markdown", action="store_true", help="print the tables as Markdown"
    )
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    results = {name: array_case(name, settings) for name, settings in CASES.items()}
    noisy = array_case("A", CASES["A"], eight_bit=True)
    rng = np.random.default_rng(9)
    blank = np.clip(np.rint(rng.normal(40, 3, (384, 384))), 0, 255).astype(np.uint8)
    page, truth = mixed_page()
    mixed = {flow: mixed_case(page, truth, flow) for flow in PRESETS}
    empty = {}
    for flow in PRESETS:
        contours, _ = contours_of(blank, flow)
        empty[flow] = 0 if contours is None else len(contours)

    def three(key, digits, scale=1.0):
        return " / ".join(
            f"{scale * results[c][flow][key]:.{digits}f}"
            if key in results[c][flow]
            else "-"
            for c in CASES
        )

    bar = " | " if args.markdown else "  "
    edge = "| " if args.markdown else ""
    close = " |" if args.markdown else ""
    print("Hole arrays (cases A / B / C)")
    head = [
        "Workflow",
        "Holes found, %",
        "Repeatability, nm",
        "Size response",
        "8-bit noisy A: found %, nm",
    ]
    print(edge + bar.join(head) + close)
    if args.markdown:
        print("|" + "---|" * len(head))
    for flow in PRESETS:
        low = noisy[flow]
        tail = (
            f"{100 * low['found']:.0f}, {low['repeat_nm']:.1f}"
            if "repeat_nm" in low
            else f"{100 * low['found']:.0f}, -"
        )
        cells = [
            f"{flow:26s}",
            three("found", 1, 100.0),
            three("repeat_nm", 2),
            three("response", 2),
            tail,
        ]
        print(edge + bar.join(cells) + close)
    print()
    print("Mixed 8-bit page, and a page of noise alone")
    kinds = list(truth["holes"])
    head = (
        ["Workflow"]
        + [f"{kind}, %" for kind in kinds]
        + [
            "Lines, %",
            "Pad",
            "Dark dots, %",
            "Contours in empty areas",
            "On the noise page",
            "Seconds",
        ]
    )
    print(edge + bar.join(head) + close)
    if args.markdown:
        print("|" + "---|" * len(head))
    for flow in PRESETS:
        row = mixed[flow]
        if row.get("failed"):
            print(edge + bar.join([f"{flow:26s}", "failed"]) + close)
            continue
        cells = (
            [f"{flow:26s}"]
            + [f"{100 * row[kind]:.0f}" for kind in kinds]
            + [f"{100 * row['lines']:.0f}"]
        )
        cells += [
            "found" if row["pad"] else "missed",
            f"{100 * row['dots']:.0f}",
            str(row["stray"]),
            str(empty[flow]),
            f"{row['seconds']:.2f}",
        ]
        print(edge + bar.join(cells) + close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
