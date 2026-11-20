"""Image detection use case over fake ports (S5 view-model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rdv.application.detect_image import (
    SCHEMA,
    SUMMARY_SCHEMA,
    ModelInfo,
    build_detections_document,
    detect_images,
)
from rdv.application.events import ProgressEvent
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.errors import MediaError

INFO = ModelInfo(
    id="fake-model",
    input_size=640,
    score_threshold=0.25,
    device="cpu",
    sha256="a" * 64,
)


class FakeDetector:
    def __init__(self, batches: list[list[Detection]]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def detect(self, images):  # type: ignore[no-untyped-def]
        self.calls += len(images)
        return [list(self._batches.pop(0)) for _ in images]

    @property
    def info(self) -> ModelInfo:
        return INFO


class FakeMedia:
    def __init__(self) -> None:
        self.saved: dict[Path, np.ndarray] = {}

    def load_image(self, path: Path) -> np.ndarray:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[..., 0] = 7
        return image

    def save_png(self, path: Path, image: np.ndarray) -> None:
        self.saved[path] = image


class FakeRenderer:
    def render(self, image, detections):  # type: ignore[no-untyped-def]
        return image.copy()


class RecordingWriter:
    def __init__(self) -> None:
        self.documents: list[tuple[Path, dict]] = []

    def write_detections(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.documents.append((path, dict(document)))


class RecordingSummaryWriter:
    def __init__(self) -> None:
        self.json: list[tuple[Path, dict]] = []
        self.markdown: list[tuple[Path, dict]] = []

    def write_summary_json(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.json.append((path, dict(document)))

    def write_summary_markdown(self, path: Path, document):  # type: ignore[no-untyped-def]
        self.markdown.append((path, dict(document)))


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _detection(code: DamageClass, x1: float, score: float) -> Detection:
    return Detection(code, score, BBox(x1, 5.0, x1 + 20.0, 25.0))


def test_detect_images_writes_every_artifact_and_a_consistent_summary(tmp_path: Path) -> None:
    detector = FakeDetector(
        [
            [_detection(DamageClass.D00, 5.0, 0.9)],
            [_detection(DamageClass.D20, 10.0, 0.8), _detection(DamageClass.D40, 30.0, 0.7)],
        ]
    )
    media = FakeMedia()
    writer = RecordingWriter()
    summary_writer = RecordingSummaryWriter()
    progress = RecordingProgress()

    result = detect_images(
        [tmp_path / "first.jpg", tmp_path / "second.jpg"],
        detector=detector,
        media=media,
        renderer=FakeRenderer(),
        detections_writer=writer,
        summary_writer=summary_writer,
        output_dir=tmp_path / "out",
        progress=progress,
    )

    assert len(result.images) == 2
    assert result.total_detections == 3
    assert result.summary_json == tmp_path / "out" / "summary.json"
    assert result.summary_markdown == tmp_path / "out" / "summary.md"
    assert (tmp_path / "out" / "first.annotated.png") in media.saved
    assert (tmp_path / "out" / "second.annotated.png") in media.saved
    assert [path.name for path, _ in writer.documents] == [
        "first.detections.json",
        "second.detections.json",
    ]
    first_document = writer.documents[0][1]
    assert first_document["schema"] == SCHEMA
    assert first_document["source"]["path"] == str(tmp_path / "first.jpg")
    assert first_document["summary"]["counts"] == {"D00": 1, "D10": 0, "D20": 0, "D40": 0}
    assert first_document["summary"]["frames_with_damage"] == 1
    assert first_document["frames"][0]["detections"][0]["name"] == "Longitudinal crack"
    assert summary_writer.json[0][1]["schema"] == SUMMARY_SCHEMA
    assert summary_writer.json[0][1]["totals"] == {
        "images": 2,
        "detections": 3,
        "images_with_damage": 2,
        "max_score": 0.9,
        "mean_coverage_ratio": pytest.approx(0.25),
    }
    assert summary_writer.json[0][1]["classes"]["D40"]["count"] == 1
    assert [event.current for event in progress.events] == [1, 2]
    assert progress.events[-1].is_complete
    assert progress.events[0].fraction == pytest.approx(0.5)


def test_zero_detections_still_writes_every_file(tmp_path: Path) -> None:
    media = FakeMedia()
    writer = RecordingWriter()
    summary_writer = RecordingSummaryWriter()

    result = detect_images(
        [tmp_path / "empty.jpg"],
        detector=FakeDetector([[]]),
        media=media,
        renderer=FakeRenderer(),
        detections_writer=writer,
        summary_writer=summary_writer,
        output_dir=tmp_path / "out",
    )

    document = writer.documents[0][1]
    assert document["frames"][0]["detections"] == []
    assert document["summary"]["frames_with_damage"] == 0
    assert document["summary"]["max_score"] == 0.0
    assert summary_writer.json[0][1]["totals"]["images_with_damage"] == 0
    assert len(media.saved) == 1
    assert result.total_detections == 0


def test_no_inputs_is_a_media_error(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="No input images") as excinfo:
        detect_images(
            [],
            detector=FakeDetector([]),
            media=FakeMedia(),
            renderer=FakeRenderer(),
            detections_writer=RecordingWriter(),
            summary_writer=RecordingSummaryWriter(),
            output_dir=tmp_path / "out",
        )

    assert excinfo.value.exit_code == 5


def test_detections_document_rounds_numbers_for_stable_output() -> None:
    detection = Detection(DamageClass.D00, 0.123456789, BBox(1.23456, 2.34567, 3.45678, 4.56789))

    document = build_detections_document(
        source=Path("image.jpg"),
        image_size=(100, 80),
        detections=[detection],
        model=INFO,
        score_threshold=0.333333333,
    )

    payload = document["frames"][0]["detections"][0]
    assert payload["score"] == 0.123457
    assert payload["box"] == [1.23, 2.35, 3.46, 4.57]
    assert document["params"]["score_threshold"] == 0.333333
    assert document["summary"]["coverage_ratio"] == 0.000617
