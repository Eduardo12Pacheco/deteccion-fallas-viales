"""Video detection end to end through the real OpenCV and ONNX adapters (S6)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from jsonschema import Draft202012Validator

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.adapters.media.renderer import PillowRenderer
from rdv.adapters.media.video import OpenCvVideoSink, OpenCvVideoSource
from rdv.adapters.reporting.detections_json import DetectionsJsonWriter
from rdv.adapters.reporting.summary_md import SummaryWriter
from rdv.application.detect_video import detect_video
from rdv.domain.classes import DamageClass
from rdv.errors import MediaError
from tests.support.synthetic_video import write_moving_rectangle_video, write_truncated_video

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
SCHEMA = REPO_ROOT / "docs" / "schemas" / "detections.v1.schema.json"


def _detector() -> OnnxDetector:
    return OnnxDetector(TINY_MODEL, providers=["CPUExecutionProvider"])


def _detect(
    source: OpenCvVideoSource,
    output_dir: Path,
    **kwargs: object,
):
    return detect_video(
        source,
        detector=_detector(),
        renderer=PillowRenderer(),
        sink=OpenCvVideoSink(),
        detections_writer=DetectionsJsonWriter(),
        summary_writer=SummaryWriter(),
        output_dir=output_dir,
        **kwargs,  # type: ignore[arg-type]
    )


def _decode_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def test_detect_video_writes_artifacts_with_the_expected_semantics(tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    out = tmp_path / "out"
    source = OpenCvVideoSource(clip)

    result = _detect(source, out, analysis_fps=5.0)

    assert (out / "clip.annotated.mp4").is_file()
    assert not list(out.glob("*.part*"))
    assert _decode_count(out / "clip.annotated.mp4") == 40
    assert result.frames_processed == 40
    assert result.frames_analyzed == 20  # ceil(40 / stride 2)
    assert result.frames_with_damage == 20
    assert result.unique_instances == {
        DamageClass.D00: 1,
        DamageClass.D10: 0,
        DamageClass.D20: 0,
        DamageClass.D40: 1,
    }

    document = json.loads((out / "clip.detections.json").read_text(encoding="utf-8"))
    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(document)
    assert document["source"]["kind"] == "video"
    assert document["source"]["fps"] == pytest.approx(10.0)
    assert document["source"]["frames"] == 40
    assert document["summary"]["frames_analyzed"] == 20
    assert document["summary"]["unique_instances"] == {
        "D00": 1,
        "D10": 0,
        "D20": 0,
        "D40": 1,
    }
    assert [frame["index"] for frame in document["frames"]] == list(range(0, 40, 2))
    assert {entry["track_id"] for frame in document["frames"] for entry in frame["detections"]} == {
        1,
        2,
    }
    assert {entry["class"] for frame in document["frames"] for entry in frame["detections"]} == {
        "D00",
        "D40",
    }

    summary = json.loads((out / "clip.summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["frames_analyzed"] == 20
    assert summary["totals"]["truncated"] is False
    assert "Frames analyzed" in (out / "clip.summary.md").read_text(encoding="utf-8")


def test_max_frames_stops_early_and_the_summary_says_so(tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    out = tmp_path / "out"

    result = _detect(OpenCvVideoSource(clip), out, analysis_fps=5.0, max_frames=10)

    assert result.frames_processed == 10
    assert result.frames_analyzed == 5
    assert result.truncated is True
    assert _decode_count(out / "clip.annotated.mp4") == 10
    document = json.loads((out / "clip.detections.json").read_text(encoding="utf-8"))
    assert document["params"]["max_frames"] == 10
    summary = json.loads((out / "clip.summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["truncated"] is True
    assert "Max frames: 10" in (out / "clip.summary.md").read_text(encoding="utf-8")


def test_truncated_stream_fails_without_final_artifacts(tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=10)
    out = tmp_path / "out"

    class ShortSource:
        path = clip
        fps = 10.0
        frame_count = 10
        width = 320
        height = 240

        def frames(self):
            capture = cv2.VideoCapture(str(clip))
            for _ in range(3):
                ok, frame = capture.read()
                assert ok
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            capture.release()

        def close(self) -> None:
            pass

    with pytest.raises(MediaError, match="ended after"):
        _detect(ShortSource(), out)  # type: ignore[arg-type]

    assert not (out / "clip.annotated.mp4").exists()
    assert not (out / "clip.detections.json").exists()
    assert not list(out.glob("*.part"))


def test_corrupt_video_bytes_are_a_media_error(tmp_path: Path) -> None:
    corrupt = write_truncated_video(tmp_path / "corrupt.mp4")

    with pytest.raises(MediaError):
        OpenCvVideoSource(corrupt)
