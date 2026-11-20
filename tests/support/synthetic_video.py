"""Deterministic synthetic MP4 clips for video tests (decision S6).

The generator draws a bright rectangle moving across a seeded asphalt-like
background. Encoding uses the same ``mp4v`` codec as the production sink, so
tests exercise the real decode/encode path without network or GPU access.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_moving_rectangle_video(
    path: Path,
    *,
    frames: int = 40,
    width: int = 320,
    height: int = 240,
    fps: float = 10.0,
    seed: int = 42,
) -> Path:
    """Encode a deterministic clip and return its path.

    Args:
        path: destination ``.mp4`` file (parent directories are created).
        frames: number of frames to encode.
        width: frame width in pixels (even).
        height: frame height in pixels (even).
        fps: frame rate stored in the container.
        seed: seed of the background texture.

    Raises:
        RuntimeError: OpenCV cannot open the MP4 writer in this environment.
    """
    if frames < 1:
        raise ValueError("frames must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4v writer")
    rng = np.random.default_rng(seed)
    background = rng.integers(55, 115, size=(height, width, 3), dtype=np.uint8)
    box_width = max(16, width // 4)
    box_height = max(16, height // 3)
    top = (height - box_height) // 2
    travel = max(1, width - box_width - 1)
    try:
        for index in range(frames):
            frame = background.copy()
            left = round(travel * index / max(frames - 1, 1))
            cv2.rectangle(
                frame,
                (left, top),
                (left + box_width, top + box_height),
                (225, 225, 225),
                -1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_truncated_video(path: Path, *, keep_bytes: int = 1024, **kwargs: object) -> Path:
    """Write a valid clip and keep only its first ``keep_bytes`` bytes."""
    source = write_moving_rectangle_video(path.with_suffix(".full.mp4"), **kwargs)  # type: ignore[arg-type]
    data = source.read_bytes()[:keep_bytes]
    source.unlink()
    path = Path(path)
    path.write_bytes(data)
    return path
