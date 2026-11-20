"""Image preprocessing for the RT-DETR ONNX graph (decision D8).

The checkpoint's ``preprocessor_config.json`` rescales to ``[0, 1]`` without
mean/std normalization and stretches the image to a fixed 640x640 square (no
letterboxing). Keeping this in one function guarantees the CLI, the web app,
and the benchmark feed the network exactly the same tensor.

The portable desktop bundle omits OpenCV (its codec payload would exceed the
S15 size budget), so :func:`resize_linear` mirrors ``cv2.INTER_LINEAR`` with
numpy and is unit-tested against OpenCV within one 8-bit level. Installations
of the CLI use OpenCV as before (decision D8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rdv.application.ports import RgbImage

try:
    import cv2
except ImportError:  # pragma: no cover - exercised by the desktop bundle smoke test
    cv2 = None  # type: ignore[assignment]

INPUT_SIZE = 640
"""Side of the square network input, in pixels (contract S4)."""


def preprocess_image(
    image: RgbImage,
    size: int = INPUT_SIZE,
) -> NDArray[np.float32]:
    """Convert an RGB image into the ``(1, 3, size, size)`` network tensor.

    Args:
        image: ``HxWx3`` ``uint8`` RGB array.
        size: square output side in pixels.

    Returns:
        Float32 NCHW tensor with values in ``[0, 1]``.

    Raises:
        ValueError: the array is not an ``HxWx3`` RGB image or ``size`` is
            not positive.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype {image.dtype}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")

    resized = (
        cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        if cv2 is not None
        else resize_linear(image, size)
    )
    scaled = resized.astype(np.float32) / np.float32(255.0)
    tensor = np.transpose(scaled, (2, 0, 1))
    return np.ascontiguousarray(tensor[np.newaxis, ...], dtype=np.float32)


def resize_linear(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Stretch an RGB image to ``size x size`` with bilinear interpolation.

    This is a numpy fallback for the desktop bundle, where OpenCV is not
    collected. The geometry matches ``cv2.INTER_LINEAR`` (half-pixel centers,
    edge clamping); only the fixed-point rounding of OpenCV is replaced by
    float arithmetic, so results differ by at most one 8-bit level.

    Args:
        image: ``HxWx3`` ``uint8`` RGB array.
        size: square output side in pixels.

    Returns:
        ``size x size x 3`` ``uint8`` RGB array.
    """
    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        raise ValueError("Cannot resize an empty image")

    ys = (np.arange(size, dtype=np.float64) + 0.5) * (source_height / size) - 0.5
    xs = (np.arange(size, dtype=np.float64) + 0.5) * (source_width / size) - 0.5

    y0 = np.clip(np.floor(ys), 0, source_height - 1).astype(np.intp)
    x0 = np.clip(np.floor(xs), 0, source_width - 1).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)

    wy = np.clip(ys - y0, 0.0, 1.0)[:, np.newaxis, np.newaxis]
    wx = np.clip(xs - x0, 0.0, 1.0)[np.newaxis, :, np.newaxis]

    top = image[y0][:, x0].astype(np.float64) * (1.0 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0].astype(np.float64) * (1.0 - wx) + image[y1][:, x1] * wx
    resized = top * (1.0 - wy) + bottom * wy
    rounded: NDArray[np.uint8] = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    return rounded
