"""Image decoding/encoding with Pillow, including EXIF orientation (S5).

Pillow is used instead of OpenCV for still images because it reads EXIF
orientation and preserves the intended display rotation; video frames (T7) go
through OpenCV. Only this adapter may depend on Pillow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from rdv.errors import MediaError

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


class PillowImageSource:
    """Media source for still images (``MediaSource`` port)."""

    def load_image(self, path: Path) -> NDArray[np.uint8]:
        """Decode ``path`` to an ``HxWx3`` RGB ``uint8`` array.

        EXIF orientation is applied so a phone photo is analyzed the way it is
        displayed, not the way it is stored.

        Raises:
            MediaError: the file is missing or cannot be decoded as an image.
        """
        path = Path(path)
        if not path.is_file():
            raise MediaError(f"Image not found: {path}")
        try:
            with Image.open(path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                rgb = oriented.convert("RGB")
                return np.asarray(rgb, dtype=np.uint8)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise MediaError(f"Cannot decode image {path}: {exc}") from exc

    def save_png(self, path: Path, image: NDArray[np.uint8]) -> None:
        """Write an RGB array as PNG.

        Raises:
            MediaError: the array is invalid or the file cannot be written.
        """
        path = Path(path)
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise MediaError(
                f"Cannot encode image with shape {array.shape} and dtype {array.dtype}"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).convert("RGB").save(path, format="PNG")
        except OSError as exc:
            raise MediaError(f"Cannot write image {path}: {exc}") from exc
