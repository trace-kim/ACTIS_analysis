import json

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from actis.synthetic import render_repeats
from actis.throughput import benchmark


@pytest.fixture
def reference(tmp_path):
    images, _ = render_repeats(
        image_shape=(78, 128),
        n_repeats=8,
        pitch_px=10,
        cd_nm=106,
        supersample=3,
        nuisance=True,
    )
    folder = tmp_path / "reference"
    folder.mkdir()
    tag = PngImagePlugin.PngInfo()
    tag.add_text("actis-data-origin", "synthetic")
    for i, frame in enumerate(images):
        Image.fromarray(frame).save(folder / f"repeat-{i}.png", pnginfo=tag)
    return folder


def test_benchmark_measures_actual_work_and_checks_concurrent_numerics(
    reference, tmp_path
):
    sample = tmp_path / "single-page.tiff"
    Image.fromarray(np.zeros((2048, 1376), dtype=np.uint8)).save(sample)
    result = benchmark(
        reference, seconds=0, pixel_nm=21, tile_um=1.134, io_sample=sample, workers=2
    )
    json.dumps(result, allow_nan=False)
    assert (
        result["native_crop_shape_yx"] == [78, 128]
        and result["provenance"] == "synthetic"
    )
    assert result["io_sample"]["board_pages"] == 1
    assert len(result["io_sample"]["passes"]) == 2
    assert result["io_sample"]["passes"][0]["bytes"] == sample.stat().st_size
    methods = {r["method"]: r for r in result["methods"]}
    assert set(methods) == {"raw", "flux", "harmonic", "pooled"}
    for item in methods.values():
        assert item["timed_images"] >= 8 and item["checksum_matches_serial"]
        assert item["timed_seconds"] > 0 and item["input_board_equivalents_per_s"] > 0
        assert (
            0
            <= item["valid_area_board_equivalents_per_s"]
            <= item["input_board_equivalents_per_s"]
        )
    assert methods["flux"]["support_per_cycle"]["valid"] > 0
    assert methods["raw"]["coverage_fraction"] > 0
    assert "first_read_cache_unknown" in methods["raw"]["compute_rate_meets_target"]


def test_empty_spatial_support_never_passes_a_speed_gate(reference):
    result = benchmark(reference, seconds=0, pixel_nm=21, tile_um=100)
    for row in result["methods"]:
        assert row["coverage_fraction"] == 0
        assert row["projected_seconds_for_1500_valid_area_equivalents"] is None
        assert all(v is None for v in row["compute_rate_meets_target"].values())


def test_invalid_worker_count_is_rejected():
    with pytest.raises(ValueError, match="workers"):
        benchmark("unused", seconds=0, pixel_nm=21, tile_um=5, workers=0)
