import csv
import gzip
import json
from collections import Counter
from html.parser import HTMLParser

import numpy as np
import pytest
from PIL import Image

from actis.analysis import analyze, common_bounds, rect_mean
from actis.cli import main
from actis.files import read_reference
from actis.synthetic import render_repeats


def test_fractional_native_area_integral():
    image = np.array([[2, 10], [4, 22]], dtype=np.uint16)
    assert rect_mean(image, (-0.5, -0.5, 1.5, 1.5)) == pytest.approx(image.mean())
    assert rect_mean(image, (0.25, -0.5, 1.25, 0.5)) == pytest.approx(
        0.25 * 2 + 0.75 * 10
    )
    assert common_bounds((48, 128), [(0.0, 0.0)]) == (-0.5, -0.5, 127.5, 47.5)


def test_blank_images_retain_only_raw_page_precision():
    images = [np.full((64, 64), i + 1300, np.uint16) for i in range(8)]
    result = analyze(images, allow_unpatterned=True)
    assert (
        result.regions == []
        and result.holes == []
        and result.registration[0]["ambiguous"]
    )
    assert result.frames[-1]["mean_dn"] == 1307


def test_reference_contract(tmp_path):
    with pytest.raises(ValueError, match="eight"):
        read_reference(tmp_path)
    for i in [10, 2, 1, 3, 4, 5, 6, 7]:
        Image.fromarray(np.full((20, 14), 1700 + i, dtype=np.uint16)).save(
            tmp_path / f"frame{i}.png"
        )
    images, info = read_reference(tmp_path)
    assert [x["file"] for x in info][:2] == ["frame1.png", "frame2.png"]
    assert images[-1][0, 0] == 1710
    Image.fromarray(np.ones((20, 14), dtype=np.uint8)).save(tmp_path / "frame7.png")
    with pytest.raises(ValueError, match="16-bit grayscale"):
        read_reference(tmp_path)


def test_reference_entrypoint_measures_and_renders_eight_images(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    images, _ = render_repeats(
        image_shape=(88, 80), pitch_px=11, cd_nm=100, n_repeats=8, supersample=4
    )
    for i, im in enumerate(images):
        Image.fromarray(im).save(source / f"{i}.png")
    out = tmp_path / "report"
    assert (
        main(
            [
                "analyze-reference",
                "--input",
                str(source),
                "--output",
                str(out),
                "--backend",
                "cpu",
                "--tile-um",
                "1.25",
            ]
        )
        == 0
    )
    with (out / "precision.csv").open(encoding="utf-8") as f:
        scores = list(csv.DictReader(f))
    raw = next(
        r for r in scores if r["level"] == "Native page" and r["method"] == "raw"
    )
    values = np.mean(images.astype(np.float64), axis=(1, 2))
    assert float(raw["percent"]) == pytest.approx(
        300 * np.std(values, ddof=1) / np.mean(values)
    )
    with (out / "triplets.csv").open(encoding="utf-8") as f:
        assert sum(r["level"] == "Native page" for r in csv.DictReader(f)) == 56
    with (out / "run.json").open(encoding="utf-8") as f:
        run = json.load(f)
    assert len(run["inputs"]) == 8 and run["synthetic"] is False

    class ReportPage(HTMLParser):
        in_manifest = False
        manifest = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            self.in_manifest = tag == "script" and attrs.get("id") == "study"
            for key in ("href", "src"):
                ref = attrs.get(key)
                if not ref or ref.startswith("#"):
                    continue
                if tag == "a" and ref.startswith("https://"):
                    continue  # Published sources are external links.
                assert (out / ref).is_file(), ref

        def handle_data(self, data):
            if self.in_manifest:
                self.manifest = json.loads(data)

        def handle_endtag(self, tag):
            if tag == "script":
                self.in_manifest = False

    page = ReportPage()
    html = (out / "index.html").read_text(encoding="utf-8")
    page.feed(html)
    assert "phase" in html and "synthetic-truth.npz" not in html
    manifest = page.manifest
    assert (manifest["width"], manifest["height"]) == (80, 88)
    assert len(manifest["frames"]) == len(images)
    assert set(manifest["methods"]) == {
        "contour",
        "model",
        "differential",
        "flux",
        "raw",
        "harmonic",
        "pooled",
    }
    with gzip.open(out / "holes.csv.gz", "rt", encoding="utf-8") as f:
        hole_rows = list(csv.DictReader(f))
    expected = Counter((int(r["repeat"]), r["method"]) for r in hole_rows)
    site_keys = {(int(r["region"]), int(r["site"])) for r in hole_rows}
    assert site_keys and {tuple(s[:2]) for s in manifest["sites"]} == site_keys
    with (out / "native_tiles.csv").open(encoding="utf-8") as f:
        expected.update((int(r["repeat"]), r["method"]) for r in csv.DictReader(f))

    for repeat, (frame, original) in enumerate(zip(manifest["frames"], images)):
        assert frame["file"] == f"{repeat}.png"
        assert frame["mean"] == pytest.approx(values[repeat])
        with Image.open(out / frame["source"]) as saved:
            np.testing.assert_array_equal(np.asarray(saved), original)
            assert np.asarray(saved).dtype == np.uint16
        with Image.open(out / frame["image"]) as display:
            assert display.size == (80, 88) and display.mode == "L"
        assert (out / frame["thumb"]).is_file()
        assert frame["layers"].keys() == manifest["methods"].keys()
        for method, path in frame["layers"].items():
            script = (out / path).read_text(encoding="utf-8")
            prefix = f'window.ACTIS_LAYER({repeat},"{method}",'
            assert script.startswith(prefix) and script.endswith(");\n")
            layer = json.loads(script[len(prefix) : -3])
            assert len(layer["rows"]) == layer["total"] == expected[repeat, method]

    assert (
        main(["analyze-reference", "--input", str(source), "--output", str(out)]) == 2
    )
