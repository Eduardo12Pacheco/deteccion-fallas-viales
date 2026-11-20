"""OpenCV video source/sink adapter (S6, D20)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from rdv.adapters.media.video import OpenCvVideoSink, OpenCvVideoSource, rotate_frame
from rdv.application.ports import RgbImage
from rdv.errors import MediaError
from tests.support.synthetic_video import write_moving_rectangle_video, write_truncated_video


def _clip(tmp_path: Path, **kwargs: object) -> Path:
    return write_moving_rectangle_video(tmp_path / "clip.mp4", **kwargs)  # type: ignore[arg-type]


def _source(tmp_path: Path) -> OpenCvVideoSource:
    return OpenCvVideoSource(_clip(tmp_path))


def _rgb_frames(count: int, width: int = 32, height: int = 24) -> list[RgbImage]:
    return [np.full((height, width, 3), index % 255, dtype=np.uint8) for index in range(count)]


def test_source_reports_container_metadata(tmp_path: Path) -> None:
    source = _source(tmp_path)

    try:
        assert source.path == tmp_path / "clip.mp4"
        assert source.fps == pytest.approx(10.0)
        assert source.frame_count == 40
        assert (source.width, source.height) == (320, 240)
    finally:
        source.close()


def test_source_missing_file_is_a_media_error(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="Video not found") as excinfo:
        OpenCvVideoSource(tmp_path / "missing.mp4")

    assert excinfo.value.exit_code == 5


def test_source_rejects_non_video_bytes(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")

    with pytest.raises(MediaError, match="Cannot decode video") as excinfo:
        OpenCvVideoSource(broken)

    assert excinfo.value.exit_code == 5


def test_source_rejects_a_truncated_mp4(tmp_path: Path) -> None:
    truncated = write_truncated_video(tmp_path / "broken.mp4")

    with pytest.raises(MediaError) as excinfo:
        OpenCvVideoSource(truncated)

    assert excinfo.value.exit_code == 5


def test_source_yields_exactly_the_encoded_frames_as_rgb(tmp_path: Path) -> None:
    source = OpenCvVideoSource(_clip(tmp_path, frames=12, width=64, height=48))

    try:
        frames = list(source.frames())
    finally:
        source.close()

    assert len(frames) == 12
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8
    assert np.array_equal(frames[0], frames[0])


def test_source_close_is_idempotent_and_releases_the_capture(tmp_path: Path) -> None:
    source = _source(tmp_path)
    next(source.frames())

    source.close()
    source.close()


@pytest.mark.parametrize(
    ("orientation", "expected_shape"),
    [
        (0, (24, 32, 3)),
        (90, (32, 24, 3)),
        (180, (24, 32, 3)),
        (270, (32, 24, 3)),
        (360, (24, 32, 3)),
        (-90, (32, 24, 3)),
    ],
)
def test_rotate_frame_handles_container_orientation(
    orientation: int, expected_shape: tuple[int, int, int]
) -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)

    rotated = rotate_frame(frame, orientation)

    assert rotated.shape == expected_shape


def test_rotate_frame_rejects_unsupported_angles() -> None:
    with pytest.raises(MediaError, match="orientation"):
        rotate_frame(np.zeros((24, 32, 3), dtype=np.uint8), 45)


def test_sink_writes_an_mp4_with_the_same_frame_count(tmp_path: Path) -> None:
    output = tmp_path / "out" / "clip.annotated.mp4"
    frames = _rgb_frames(9)

    OpenCvVideoSink().write_video(output, frames, fps=7.5)

    assert output.is_file()
    assert not list(output.parent.glob("*.part*"))
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    assert decoded == 9


def test_sink_removes_the_part_file_when_the_frame_stream_fails(tmp_path: Path) -> None:
    output = tmp_path / "clip.annotated.mp4"

    def failing() -> Iterator[RgbImage]:
        yield from _rgb_frames(2)
        raise RuntimeError("decoder exploded")

    with pytest.raises(RuntimeError, match="decoder exploded"):
        OpenCvVideoSink().write_video(output, failing(), fps=10.0)

    assert not output.exists()
    assert not list(output.parent.glob("*.part*"))


def test_sink_rejects_an_empty_stream(tmp_path: Path) -> None:
    output = tmp_path / "empty.mp4"

    with pytest.raises(MediaError, match="no frames"):
        OpenCvVideoSink().write_video(output, [], fps=10.0)

    assert not output.exists()


def test_sink_rejects_frames_of_inconsistent_size(tmp_path: Path) -> None:
    output = tmp_path / "clip.mp4"
    frames: Sequence[RgbImage] = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((30, 32, 3), dtype=np.uint8),
    ]

    with pytest.raises(MediaError, match="size"):
        OpenCvVideoSink().write_video(output, frames, fps=10.0)

    assert not output.exists()
    assert not list(output.parent.glob("*.part*"))
