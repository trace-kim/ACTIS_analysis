"""The offline audit must reproduce the production CD and preserve native input."""

import base64
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from test_discovery import array_image

from actis.discovery import discover
from actis.harmonics import harmonic_tiles

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_harmonics.py"


def report(path, *options):
    output = path.with_suffix(".html")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "-o", str(output), *options],
        check=True,
    )
    return output.read_text(encoding="utf-8")


def payload(html, key):
    return re.search(
        rf'<script type="[^"]+" id="{key}">(.*?)</script>', html, re.DOTALL
    )[1]


def test_report_replays_signed_fit_and_detector_pixels(tmp_path):
    image = np.rint(1000 + 42000 * array_image((193, 269))).astype("uint16")
    path = tmp_path / "field.png"
    Image.fromarray(image).save(path)
    html = report(path, "--shape", "circle", "--tile-px", "96")
    meta = json.loads(payload(html, "study"))
    assert meta["pixel_nm"] is None
    assert meta["layers"]["raw"]["levels"] == [[269, 193], [135, 97]]
    raw = zlib.decompress(base64.b64decode(payload(html, "raw-0-1-0")))
    np.testing.assert_array_equal(
        np.frombuffer(raw, "<u2").reshape(193, 13), image[:, 256:]
    )
    (region,) = discover(image, shape="circle")
    production = harmonic_tiles(image, region, tile_px=96, origin_xy=(-0.5, -0.5))
    assert len(production) == len(meta["tiles"]) == 4
    for index, (expected, actual) in enumerate(zip(production, meta["tiles"])):
        assert actual["status"] == expected["status"] == "ok"
        np.testing.assert_allclose(actual["cd_px"], expected["cd_px"], rtol=1e-12)
        trace = json.loads(payload(html, f"tile-{index}"))
        fit = trace["fit"]
        predicted = fit["gain"] * np.asarray(fit["form"]) * fit["mtf"] * fit["pixel"]
        np.testing.assert_allclose(predicted, fit["predicted"])
        np.testing.assert_allclose(
            (predicted - fit["target"]) / fit["error"], fit["residual"], atol=1e-10
        )
        assert fit["winner"] == np.argmin([run["cost"] for run in fit["starts"]])
        assert actual["cd_px"] == max(fit["width"], fit["height"])
        a = np.asarray(trace["raw_re"]) + 1j * np.asarray(trace["raw_im"])
        freq = np.asarray(trace["freq"])
        rotation = (
            2j
            * np.pi
            * (freq @ (np.array(trace["anchor_local"]) + trace["displacement"]))
        )
        np.testing.assert_allclose(
            (a * np.exp(rotation)).real, trace["signed"], atol=1e-9
        )


def test_report_captures_rejection_and_blank_tiff_page(tmp_path):
    path = tmp_path / "pages.tif"
    image = Image.fromarray(np.rint(200 * array_image((96, 96))).astype("u1"))
    image.save(path, save_all=True, append_images=[Image.new("L", (96, 96))])
    html = report(path, "--shape", "circle", "--tile-px", "32")
    meta = json.loads(payload(html, "study"))
    assert len(meta["tiles"]) == 9
    assert all(
        t["status"] == "insufficient_pattern_coverage" and t["cd_px"] is None
        for t in meta["tiles"]
    )
    assert "fit" not in json.loads(payload(html, "tile-0"))
    blank = json.loads(payload(report(path, "--page", "1"), "study"))
    assert blank["page"] == 1 and not blank["tiles"] and not blank["regions"]
    assert blank["discovery"]["rejected"][0]["reason"] == "too_few_centers"
