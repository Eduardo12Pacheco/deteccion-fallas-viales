"""OpenCV video decoding and MP4 encoding (S6, decision D20).

Frames are always handled as RGB ``uint8`` arrays, matching the inference and
rendering paths; OpenCV's BGR order stays inside this adapter. The sink encodes
to ``<name>.part`` and only renames after a complete, successful stream, so a
corrupt or truncated input never leaves an artifact under its final name.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from rdv.application.ports import RgbImage
from rdv.errors import MediaError

MP4V_FOURCC = "mp4v"
_VIDEO_WRITER_FOURCC = getattr(cv2, "VideoWriter_fourcc")  # noqa: B009
_ORIENTATION_PROPERTY = getattr(cv2, "CAP_PROP_ORIENTATION_META", 37)
_ORIENTATION_AUTO_PROPERTY = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", 48)


def rotate_frame(frame: RgbImage, orientation: int) -> RgbImage:
    """Apply container rotation metadata (clockwise degrees) to one frame.

    Args:
        frame: ``HxWx3`` RGB array.
        orientation: clockwise rotation in degrees; multiples of 90 only.

    Raises:
        MediaError: the orientation is not a multiple of 90 degrees.
    """
    normalized = orientation % 360
    if normalized == 0:
        return frame
    if normalized == 90:
        return np.ascontiguousarray(np.rot90(frame, k=-1))
    if normalized == 180:
        return np.ascontiguousarray(np.rot90(frame, k=2))
    if normalized == 270:
        return np.ascontiguousarray(np.rot90(frame, k=1))
    raise MediaError(f"Unsupported video orientation metadata: {orientation} degrees")


class OpenCvVideoSource:
    """``VideoSource`` adapter backed by ``cv2.VideoCapture`` (S6)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise MediaError(f"Video not found: {self._path}")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise MediaError(f"Cannot decode video {self._path}: unsupported or corrupt file")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0.0:
            capture.release()
            raise MediaError(f"Cannot decode video {self._path}: invalid frame rate {fps!r}")
        if width <= 0 or height <= 0:
            capture.release()
            raise MediaError(
                f"Cannot decode video {self._path}: invalid frame size {width}x{height}"
            )
        # Disable OpenCV's automatic rotation so the metadata is applied once,
        # explicitly, with a deterministic frame size.
        capture.set(_ORIENTATION_AUTO_PROPERTY, 0)
        try:
            orientation = round(float(capture.get(_ORIENTATION_PROPERTY)))
        except (TypeError, ValueError):  # pragma: no cover - defensive for odd builds
            orientation = 0
        orientation %= 360
        if orientation not in {0, 90, 180, 270}:
            capture.release()
            raise MediaError(
                f"Cannot decode video {self._path}: unsupported rotation {orientation}"
            )
        if orientation in {90, 270}:
            width, height = height, width
        self._capture: cv2.VideoCapture | None = capture
        self._fps = fps
        self._frame_count = frame_count if frame_count > 0 else None
        self._width = width
        self._height = height
        self._orientation = orientation

    @property
    def path(self) -> Path:
        """Input file path."""
        return self._path

    @property
    def fps(self) -> float:
        """Container frame rate."""
        return self._fps

    @property
    def frame_count(self) -> int | None:
        """Declared frame count, or ``None`` when unavailable."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Display width after rotation metadata."""
        return self._width

    @property
    def height(self) -> int:
        """Display height after rotation metadata."""
        return self._height

    def frames(self) -> Iterator[RgbImage]:
        """Yield decoded frames in display order, releasing the capture at the end."""
        capture = self._capture
        if capture is None:
            raise MediaError(f"Video source is closed: {self._path}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                yield rotate_frame(rgb, self._orientation)
        finally:
            self.close()

    def close(self) -> None:
        """Release the capture handle; idempotent."""
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class OpenCvVideoSink:
    """``VideoSink`` adapter writing ``mp4v`` files atomically (S6)."""

    def write_video(
        self,
        path: Path,
        frames: Iterable[RgbImage],
        *,
        fps: float,
    ) -> None:
        """Encode ``frames`` to ``path``; a failure removes the temporary file.

        Raises:
            MediaError: the writer cannot be opened, a frame has the wrong
                shape/dtype, or the stream is empty.
        """
        path = Path(path)
        if not math.isfinite(fps) or fps <= 0.0:
            raise MediaError(f"Invalid output frame rate: {fps!r}")
        part = path.with_name(f"{path.stem}.part{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        writer: cv2.VideoWriter | None = None
        size: tuple[int, int] | None = None
        written = 0
        try:
            for frame in frames:
                array: NDArray[np.uint8] = np.asarray(frame)
                if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                    raise MediaError(
                        "Cannot encode video frame with shape "
                        f"{array.shape} and dtype {array.dtype}"
                    )
                height, width = int(array.shape[0]), int(array.shape[1])
                if size is None:
                    size = (width, height)
                    writer = cv2.VideoWriter(
                        str(part),
                        _VIDEO_WRITER_FOURCC(*MP4V_FOURCC),
                        float(fps),
                        size,
                    )
                    if not writer.isOpened():
                        raise MediaError(f"Cannot open the MP4 writer for {path}")
                elif size != (width, height):
                    raise MediaError(
                        f"Video frame size changed from {size[0]}x{size[1]} "
                        f"to {width}x{height}: {path}"
                    )
                assert writer is not None
                writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                written += 1
            if writer is None or written == 0:
                raise MediaError(f"Cannot encode video {path}: no frames were produced")
        except BaseException:
            if writer is not None:
                writer.release()
                writer = None
            part.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.release()
        os.replace(part, path)
