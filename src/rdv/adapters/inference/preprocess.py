"""Image preprocessing for the RT-DETR ONNX graph (decision D8).

The checkpoint's ``preprocessor_config.json`` rescales to ``[0, 1]`` without
mean/std normalization and stretches the image to a fixed 640x640 square (no
letterboxing). Keeping this in one function guarantees the CLI, the web app,
and the benchmark feed the network exactly the same tensor.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from rdv.application.ports import RgbImage

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

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / np.float32(255.0)
    tensor = np.transpose(scaled, (2, 0, 1))
    return np.ascontiguousarray(tensor[np.newaxis, ...], dtype=np.float32)
