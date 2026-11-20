"""Detect damage in a batch of images and write the S5 artifacts.

The use case owns orchestration only: media decoding, rendering, inference and
file writing are ports, and the documents are plain JSON-ready dictionaries so
the same view-model feeds the CLI, the web API (T11) and the HTML report (T8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdv import __version__
from rdv.application.events import ProgressEvent
from rdv.application.ports import (
    DetectionsWriter,
    Detector,
    MediaSource,
    ModelInfo,
    ProgressSink,
    Renderer,
    RgbImage,
    SummaryWriter,
)
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.summary import DamageSummary
from rdv.errors import MediaError

SCHEMA = "rdv.detections/v1"
SUMMARY_SCHEMA = "rdv.summary/v1"
TOOL_NAME = "road-damage-vision"
_CLASS_ORDER = tuple(DamageClass)


@dataclass(frozen=True, slots=True)
class ImageOutcome:
    """One processed image and the files written for it."""

    source: Path
    width: int
    height: int
    detections: tuple[Detection, ...]
    summary: DamageSummary
    annotated: Path
    detections_json: Path


@dataclass(frozen=True, slots=True)
class DetectImagesResult:
    """Everything written by one ``rdv detect`` image run."""

    output_dir: Path
    images: tuple[ImageOutcome, ...]
    summary_json: Path
    summary_markdown: Path

    @property
    def total_detections(self) -> int:
        """Sum of detections over every input image."""
        return sum(len(image.detections) for image in self.images)


def round_value(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def class_counts(detections: Sequence[Detection]) -> dict[str, int]:
    counts = dict.fromkeys((damage_class.code for damage_class in _CLASS_ORDER), 0)
    for detection in detections:
        counts[detection.damage_class.code] += 1
    return counts


def detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "class": detection.damage_class.code,
        "name": detection.damage_class.display_name,
        "score": round_value(detection.score),
        "box": [round_value(value, 2) for value in detection.bbox.to_xyxy()],
        "track_id": None,
    }


def build_detections_document(
    *,
    source: Path,
    image_size: tuple[int, int],
    detections: Sequence[Detection],
    model: ModelInfo,
    score_threshold: float,
) -> dict[str, Any]:
    """Build the ``rdv.detections/v1`` document for one image (S5 contract)."""
    width, height = image_size
    summary = DamageSummary.from_detections(detections, image_size)
    return {
        "schema": SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "model": {
            "id": model.id,
            "sha256": model.sha256,
            "input_size": model.input_size,
        },
        "source": {
            "path": str(source),
            "kind": "image",
            "width": width,
            "height": height,
        },
        "params": {
            "score_threshold": round_value(score_threshold),
            "device": model.device,
        },
        "frames": [
            {
                "index": 0,
                "t": None,
                "detections": [detection_payload(detection) for detection in detections],
            }
        ],
        "summary": {
            "counts": class_counts(detections),
            "unique_instances": None,
            "frames_analyzed": 1,
            "frames_with_damage": 1 if detections else 0,
            "coverage_ratio": round_value(summary.coverage_ratio),
            "max_score": round_value(summary.max_score),
        },
    }


def build_summary_document(
    *,
    images: Sequence[ImageOutcome],
    model: ModelInfo,
    score_threshold: float,
) -> dict[str, Any]:
    """Build the aggregate ``rdv.summary/v1`` view-model for one run."""
    counts = dict.fromkeys((damage_class.code for damage_class in _CLASS_ORDER), 0)
    for image in images:
        for code, count in class_counts(image.detections).items():
            counts[code] += count
    total_detections = sum(len(image.detections) for image in images)
    max_score = max(
        (detection.score for image in images for detection in image.detections),
        default=0.0,
    )
    mean_coverage = (
        sum(image.summary.coverage_ratio for image in images) / len(images) if images else 0.0
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "tool": {"name": TOOL_NAME, "version": __version__},
        "model": {
            "id": model.id,
            "sha256": model.sha256,
            "input_size": model.input_size,
        },
        "params": {
            "score_threshold": round_value(score_threshold),
            "device": model.device,
        },
        "totals": {
            "images": len(images),
            "detections": total_detections,
            "images_with_damage": sum(1 for image in images if image.detections),
            "max_score": round_value(max_score),
            "mean_coverage_ratio": round_value(mean_coverage),
        },
        "classes": {
            code: {"name": DamageClass[code].display_name, "count": count}
            for code, count in counts.items()
        },
        "images": [
            {
                "path": str(image.source),
                "width": image.width,
                "height": image.height,
                "detections": len(image.detections),
                "max_score": round_value(image.summary.max_score),
                "coverage_ratio": round_value(image.summary.coverage_ratio),
                "annotated": image.annotated.name,
                "detections_json": image.detections_json.name,
            }
            for image in images
        ],
    }


def detect_images(
    paths: Sequence[Path],
    *,
    detector: Detector,
    media: MediaSource,
    renderer: Renderer,
    detections_writer: DetectionsWriter,
    summary_writer: SummaryWriter,
    output_dir: Path,
    progress: ProgressSink | None = None,
) -> DetectImagesResult:
    """Detect damage in each image and persist the S5 artifact set.

    Args:
        paths: input image files, processed in order.
        detector: inference port.
        media: image decoder/PNG encoder.
        renderer: overlay renderer.
        detections_writer: per-image JSON writer.
        summary_writer: aggregate JSON/Markdown writer.
        output_dir: directory receiving every artifact (created if needed).
        progress: optional observer of per-image progress.

    Returns:
        Paths and per-image outcomes of the run.

    Raises:
        MediaError: no inputs were given, or an input cannot be decoded or
            written. The original error carries the file path.
    """
    if not paths:
        raise MediaError("No input images were provided")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[ImageOutcome] = []
    total = len(paths)
    for index, raw_path in enumerate(paths, start=1):
        source = Path(raw_path)
        image: RgbImage = media.load_image(source)
        height, width = int(image.shape[0]), int(image.shape[1])
        detections = tuple(detector.detect([image])[0])
        rendered = renderer.render(image, detections)
        annotated = output_dir / f"{source.stem}.annotated.png"
        media.save_png(annotated, rendered)
        detections_json = output_dir / f"{source.stem}.detections.json"
        detections_writer.write_detections(
            detections_json,
            build_detections_document(
                source=source,
                image_size=(width, height),
                detections=detections,
                model=detector.info,
                score_threshold=detector.info.score_threshold,
            ),
        )
        summary = DamageSummary.from_detections(detections, (width, height))
        outcomes.append(
            ImageOutcome(
                source=source,
                width=width,
                height=height,
                detections=detections,
                summary=summary,
                annotated=annotated,
                detections_json=detections_json,
            )
        )
        if progress is not None:
            progress.emit(
                ProgressEvent(
                    stage="detect",
                    current=index,
                    total=total,
                    message=f"{source.name}: {len(detections)} detection(s)",
                )
            )

    summary_document = build_summary_document(
        images=outcomes,
        model=detector.info,
        score_threshold=detector.info.score_threshold,
    )
    summary_json = output_dir / "summary.json"
    summary_markdown = output_dir / "summary.md"
    summary_writer.write_summary_json(summary_json, summary_document)
    summary_writer.write_summary_markdown(summary_markdown, summary_document)
    return DetectImagesResult(
        output_dir=output_dir,
        images=tuple(outcomes),
        summary_json=summary_json,
        summary_markdown=summary_markdown,
    )
