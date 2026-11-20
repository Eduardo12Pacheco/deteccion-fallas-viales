"""Still-image decoding, EXIF orientation and PNG encoding (S5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rdv.adapters.media.images import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, PillowImageSource
from rdv.errors import MediaError


def _write_jpeg(
    path: Path,
    width: int = 30,
    height: int = 20,
    exif_orientation: int | None = None,
) -> None:
    image = Image.new("RGB", (width, height), (10, 20, 30))
    exif = Image.Exif()
    if exif_orientation is not None:
        exif[0x0112] = exif_orientation
    image.save(path, format="JPEG", exif=exif)


def test_load_returns_rgb_uint8_with_the_stored_shape(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    _write_jpeg(path)

    image = PillowImageSource().load_image(path)

    assert image.shape == (20, 30, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == pytest.approx([10, 20, 30], abs=2)


def test_load_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    _write_jpeg(path, exif_orientation=6)

    image = PillowImageSource().load_image(path)

    assert image.shape == (30, 20, 3)


def test_missing_file_is_a_media_error(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="Image not found") as excinfo:
        PillowImageSource().load_image(tmp_path / "absent.jpg")

    assert excinfo.value.exit_code == 5


def test_corrupt_file_is_a_media_error_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"definitely not a jpeg")

    with pytest.raises(MediaError, match=r"corrupt\.jpg") as excinfo:
        PillowImageSource().load_image(path)

    assert excinfo.value.exit_code == 5


def test_save_png_round_trips_the_array(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.png"
    image = np.zeros((8, 6, 3), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 1] = 100

    PillowImageSource().save_png(path, image)
    loaded = PillowImageSource().load_image(path)

    assert loaded.shape == image.shape
    assert loaded.tolist() == image.tolist()


@pytest.mark.parametrize(
    "image",
    [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8)],
)
def test_save_png_rejects_invalid_arrays(tmp_path: Path, image: np.ndarray) -> None:
    with pytest.raises(MediaError):
        PillowImageSource().save_png(tmp_path / "bad.png", image)


def test_extension_sets_do_not_overlap() -> None:
    assert VIDEO_EXTENSIONS.isdisjoint(IMAGE_EXTENSIONS)
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".mp4" in VIDEO_EXTENSIONS
