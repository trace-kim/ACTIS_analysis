import csv
import json
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
    assert len(list((out / "images").glob("native-*.png"))) == 8
    assert len(list((out / "images").glob("overlay-contour-*.png"))) == 8
    with (out / "run.json").open(encoding="utf-8") as f:
        run = json.load(f)
    assert len(run["inputs"]) == 8 and run["synthetic"] is False
    assert "phase" in (out / "index.html").read_text(encoding="utf-8")

    class References(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, ref in attrs:
                if key in {"href", "src"} and not ref.startswith(("#", "https:")):
                    assert (out / ref).is_file(), ref

    References().feed((out / "index.html").read_text(encoding="utf-8"))
    assert (
        main(["analyze-reference", "--input", str(source), "--output", str(out)]) == 2
    )
