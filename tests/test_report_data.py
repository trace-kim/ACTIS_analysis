import base64
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from actis.report_data import _image_levels, write_viewer_data


def test_report_preserves_site_identity_native_contours_and_tile_support(tmp_path):
    image = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    vertices = np.array([[9.95, 8.48], [12.05, 8.42], [11.0, 10.1]])
    hole = {
        "repeat": 0,
        "region": 1,
        "site": 0,
        "method": "contour",
        "valid": True,
        "flag": "interpolated_rays",
        "ref_x_px": 10.0,
        "ref_y_px": 10.0,
        "x_px": 11.0,
        "y_px": 9.0,
        "cd_px": 3.5,
        "width_px": 3.5,
        "height_px": 3.2,
        "residual_x_nm": np.nan,
        "residual_y_nm": -0.2,
        "contour_xy": vertices,
    }
    blank = dict(
        hole,
        region=2,
        valid=False,
        flag="unmatched_center",
        ref_x_px=25.0,
        ref_y_px=20.0,
        contour_xy=None,
    )
    tile = {
        "repeat": 0,
        "region": 1,
        "method": "raw",
        "i": 0,
        "j": 0,
        "x_px": 5.0,
        "y_px": 6.0,
        "value": 248.2,
        "status": "ok",
    }
    study = SimpleNamespace(
        pixel_nm=21.0,
        tile_um=0.21,
        bounds=(0.0, 0.0, 38.0, 30.0),
        frames=[{"mean_dn": image.mean(), "min_dn": 0, "max_dn": 1279}],
        regions=[],
        registration=[{"shift_xy": (1.0, -1.0), "reason": "ok", "ambiguous": False}],
        holes=[hole, blank],
        tiles=[tile],
    )
    manifest = write_viewer_data(tmp_path, [image], [{"file": "input.png"}], study)
    frame = manifest["frames"][0]
    assert frame["levels"] == [] and manifest["overview"] == frame["image"]
    assert manifest["sites"] == [[1, 0, 10, 10], [2, 0, 25, 20]]
    assert manifest["width"] == 40 and manifest["height"] == 32
    with Image.open(tmp_path / frame["source"]) as saved:
        np.testing.assert_array_equal(saved, image)
    with Image.open(tmp_path / frame["image"]) as saved:
        assert saved.size == (40, 32) and saved.getextrema() == (6, 249)

    def load(method):
        source = (tmp_path / frame["layers"][method]).read_text(encoding="utf-8")
        prefix = f'window.ACTIS_LAYER(0,"{method}",'
        assert source.startswith(prefix) and source.endswith(");\n")
        return json.loads(source[len(prefix) : -3])

    contour = load("contour")
    assert (contour["valid"], contour["total"]) == (1, 2)
    assert contour["rows"][0][:5] == [0, 11.0, 9.0, 73.5, "interpolated_rays"]
    assert contour["rows"][0][-2:] == [None, -0.2]
    assert contour["rows"][1][:5] == [1, 26.0, 19.0, None, "unmatched_center"]
    index, xy = (base64.b64decode(s) for s in contour["outlines"])
    np.testing.assert_array_equal(np.frombuffer(index, dtype="<u4"), [0, 3, 3])
    delta = np.frombuffer(xy, dtype="<i2").reshape(-1, 2) / contour["outlineScale"]
    np.testing.assert_allclose(delta + [11.0, 9.0], vertices, atol=0.5 / 256)
    assert load("raw")["rows"] == [[1, 0, 0, 5.0, 6.0, 248.2, "ok"]]
    assert load("flux")["rows"] == []
    assert manifest["methods"]["flux"]["unit"] == "DN·px²"

    # Independently calibrated flux measurements carry a CD like the geometry methods.
    study.holes = [
        dict(hole, method="flux", flux_dn_px2=400.0, flag="reference_normalized_flux")
    ]
    calibrated = write_viewer_data(tmp_path, [image], [{"file": "input.png"}], study)
    assert calibrated["methods"]["flux"]["unit"] == "nm"
    assert load("flux")["rows"][0][3] == 73.5


def test_reduced_images_filter_subpixel_array_instead_of_aliasing(tmp_path):
    (tmp_path / "images").mkdir()
    y, x = np.indices((512, 384))
    fine_pattern = Image.fromarray(((x + y) % 2 * 255).astype(np.uint8))
    levels = _image_levels(tmp_path, fine_pattern, 0)
    assert [level["scale"] for level in levels] == [0.5, 0.25]
    with Image.open(tmp_path / levels[-1]["image"]) as image:
        assert image.size == (96, 128)
        np.testing.assert_allclose(np.asarray(image)[3:-3, 3:-3], 127.5, atol=1)
