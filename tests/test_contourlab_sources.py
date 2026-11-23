"""Image files of the contour lab: PNG, and TIFF by the page or stacked."""

import sys

import numpy as np
import pytest
from PIL import Image

from actis.contourlab import sources
from actis.contourlab.sources import ImageFile, display_range, load_image


def pages(count=5, shape=(12, 9), dtype=np.uint16):
    """Pages that tell which they are: page k holds 1000 k + its row number."""
    rows = np.arange(shape[0])[:, None] * np.ones(shape[1], int)
    return np.stack([1000 * k + rows for k in range(count)]).astype(dtype)


def write_tiff(path, stack):
    frames = [Image.fromarray(page) for page in stack]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    return path


@pytest.fixture(params=["tifffile", "pillow"])
def reader(request, monkeypatch):
    """Both ways a TIFF is read: with tifffile, and without it installed."""
    if request.param == "pillow":
        monkeypatch.setitem(sys.modules, "tifffile", None)
    else:
        pytest.importorskip("tifffile")
    return request.param


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_a_png_is_one_page_with_its_grey_levels_as_stored(tmp_path, dtype):
    rng = np.random.default_rng(0)
    top = np.iinfo(dtype).max
    pixels = rng.integers(0, top, (20, 31), endpoint=True).astype(dtype)
    Image.fromarray(pixels).save(tmp_path / "one.png")

    with ImageFile(tmp_path / "one.png") as file:
        assert file.pages == 1 and file.shape == (20, 31) and file.dtype == dtype
        assert file.name == "one.png" and file.notes == []
        np.testing.assert_array_equal(file.page(0), pixels)
        with pytest.raises(ValueError, match="has pages 1 to 1, not 2"):
            file.page(1)
    np.testing.assert_array_equal(load_image(tmp_path / "one.png"), pixels)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_a_tiff_is_read_page_by_page(tmp_path, reader, dtype):
    stack = pages(dtype=dtype) if dtype == np.uint16 else (pages() % 251).astype(dtype)
    path = write_tiff(tmp_path / "scan.tiff", stack)

    with ImageFile(path) as file:
        assert file.pages == 5 and file.shape == (12, 9) and file.dtype == dtype
        for k in (3, 0, 4, 1):  # in any order
            page = file.page(k)
            np.testing.assert_array_equal(page, stack[k])
            assert page.dtype == dtype and page.flags.c_contiguous
        with pytest.raises(ValueError, match="pages 1 to 5, not 6"):
            file.page(5)
        with pytest.raises(ValueError, match="pages 1 to 5, not 0"):
            file.page(-1)
    np.testing.assert_array_equal(load_image(path, 2), stack[2])


def test_pages_are_stacked_into_one_strip(tmp_path, reader):
    stack = pages()
    path = write_tiff(tmp_path / "scan.tif", stack)

    with ImageFile(path) as file:
        strip = file.stack(1, 3)
        np.testing.assert_array_equal(strip, np.concatenate(stack[1:4]))
        assert strip.shape == (36, 9) and strip.dtype == np.uint16
        whole = file.stack(0, 5)
        assert whole.shape == (60, 9) and whole[59, 0] == 4011
        # a stripe scanned upwards: its pages in reverse order, each upside down
        np.testing.assert_array_equal(
            file.stack(1, 3, reverse=True), np.concatenate(stack[3:0:-1])
        )
        np.testing.assert_array_equal(
            file.stack(1, 3, flip=True), np.concatenate(stack[1:4, ::-1])
        )
        both = file.stack(0, 5, reverse=True, flip=True)
        np.testing.assert_array_equal(both, whole[::-1])  # the strip upside down
        np.testing.assert_array_equal(file.stack(2, 1), stack[2])
        for first, count in ((0, 0), (-1, 2), (3, 3), (5, 1)):
            with pytest.raises(ValueError, match="cannot stack"):
                file.stack(first, count)


def test_a_strip_too_large_to_stack_is_refused(tmp_path, monkeypatch):
    path = write_tiff(tmp_path / "scan.tif", pages())
    monkeypatch.setattr(sources, "MAX_STACK_PIXELS", 12 * 9 * 3)

    with ImageFile(path) as file:
        assert file.stack(0, 3).shape == (36, 9)
        with pytest.raises(
            ValueError, match="too many to stack at once: stack 3 or fewer"
        ):
            file.stack(0, 4)


def test_pages_of_different_sizes_are_shown_but_not_stacked(tmp_path, reader):
    frames = [
        Image.fromarray(np.full(shape, 7, np.uint8)) for shape in ((8, 6), (5, 6))
    ]
    frames[0].save(tmp_path / "mixed.tif", save_all=True, append_images=frames[1:])

    with ImageFile(tmp_path / "mixed.tif") as file:
        assert file.page(0).shape == (8, 6) and file.page(1).shape == (5, 6)
        with pytest.raises(ValueError, match="page 2 of mixed.tif is 6 x 5 px"):
            file.stack(0, 2)


def test_a_tiff_stored_with_white_as_zero_is_read_brighter_is_higher(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pixels = np.arange(60, dtype=np.uint8).reshape(6, 10)
    tifffile.imwrite(tmp_path / "white.tif", pixels, photometric="miniswhite")

    with ImageFile(tmp_path / "white.tif") as file:
        np.testing.assert_array_equal(file.page(0), 255 - pixels)
        assert file.notes == ["stored with white as zero: shown inverted"]


def test_a_colour_image_is_read_as_its_grey_levels(tmp_path):
    rng = np.random.default_rng(1)
    colour = rng.integers(0, 255, (10, 12, 3)).astype(np.uint8)
    Image.fromarray(colour).save(tmp_path / "colour.png")
    Image.fromarray(colour).convert("RGBA").save(tmp_path / "alpha.png")
    Image.fromarray(colour[:, :, 0]).convert("P").save(tmp_path / "palette.png")
    Image.fromarray(colour[:, :, 0] > 128).save(tmp_path / "bits.png")

    grey = load_image(tmp_path / "colour.png")

    assert grey.shape == (10, 12) and grey.dtype == np.uint8
    np.testing.assert_allclose(grey, colour.mean(axis=2), atol=0.51)
    np.testing.assert_array_equal(load_image(tmp_path / "alpha.png"), grey)
    np.testing.assert_array_equal(load_image(tmp_path / "palette.png"), colour[:, :, 0])
    bits = load_image(tmp_path / "bits.png")
    assert bits.dtype == np.uint8 and set(np.unique(bits)) == {0, 255}


def test_a_float_tiff_keeps_its_values(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    values = np.linspace(-1, 2, 48, dtype=np.float32).reshape(6, 8)
    tifffile.imwrite(tmp_path / "float.tif", values)

    loaded = load_image(tmp_path / "float.tif")

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, values)


def test_a_file_that_is_no_image_says_so(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "broken.png").write_bytes(b"not a png")
    (tmp_path / "broken.tif").write_bytes(b"II*\x00garbage")

    with pytest.raises(FileNotFoundError, match="there is no file"):
        ImageFile(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="open a PNG or TIFF image"):
        ImageFile(tmp_path / "notes.txt")
    with pytest.raises(ValueError, match="broken.png: not a readable image"):
        ImageFile(tmp_path / "broken.png")
    with pytest.raises(ValueError, match="broken.tif: not a readable"):
        ImageFile(tmp_path / "broken.tif")
    (tmp_path / "broken.tif").unlink()  # the file was closed again: Windows lets go


def test_a_page_that_cannot_be_read_says_which_and_why(tmp_path, reader, monkeypatch):
    path = write_tiff(tmp_path / "scan.tif", pages())

    with ImageFile(path) as file:
        whole = file._read

        def damaged(index):
            if index == 3:
                raise KeyError("a compression nobody here knows")
            return whole(index)

        monkeypatch.setattr(file, "_read", damaged)
        assert file.page(2).shape == (12, 9)
        with pytest.raises(
            ValueError, match="page 4 of scan.tif cannot be read: .*nobody"
        ):
            file.page(3)
        with pytest.raises(ValueError, match="page 4 of scan.tif cannot be read"):
            file.stack(2, 3)
        assert file.page(4).shape == (12, 9)  # the file is still open for the others


def test_a_closed_file_lets_go_of_its_file(tmp_path, reader):
    path = write_tiff(tmp_path / "scan.tif", pages())

    file = ImageFile(path)
    file.close()
    file.close()  # twice is fine

    path.unlink()
    assert not path.exists()


def test_the_display_range_spans_the_images_grey_levels():
    rng = np.random.default_rng(2)
    image = rng.normal(1000, 50, (300, 400))
    image[0, 0] = 60000  # one hot pixel does not set the range

    black, white = display_range(image)

    assert 850 < black < 900 and 1100 < white < 1150
    assert display_range(np.full((5, 5), 3.0)) == (3.0, 4.0)
    sparse = np.zeros((40, 50))
    sparse[3, 4] = 9  # both percentiles are 0: the whole range instead
    assert display_range(sparse) == (0.0, 9.0)
