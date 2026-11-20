"""Unit tests for the evaluation use case orchestration (T8/S7)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from rdv.application.evaluate_model import (
    EvaluationOutcome,
    evaluate_model,
    operating_point,
    select_gallery_cases,
    select_operating_threshold,
)
from rdv.application.ports import (
    EvaluationArtifacts,
    EvaluationReport,
    GalleryCase,
    RgbImage,
)
from rdv.domain.annotation import AnnotatedImage
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.error_analysis import ErrorCase, ErrorCategory
from tests.support.fakes import (
    EmptyDetector,
    GroundTruthEchoDetector,
    InMemoryDatasetRepository,
    annotation,
    solid_image,
)


class CapturingWriter:
    """Capture the report bundle instead of rendering files."""

    def __init__(self) -> None:
        self.reports: list[EvaluationReport] = []

    def write_report(
        self,
        output_dir: Path,
        report: EvaluationReport,
        *,
        load_image,
    ) -> EvaluationArtifacts:
        self.reports.append(report)
        output_dir.mkdir(parents=True, exist_ok=True)
        return EvaluationArtifacts(
            output_dir=output_dir,
            metrics_json=output_dir / "metrics.json",
            report_markdown=output_dir / "report.md",
            report_html=output_dir / "report.html",
            pr_curves_png=output_dir / "pr_curves.png",
            pr_curve_csvs={},
            confusion_png=output_dir / "confusion_matrix.png",
            confusion_csv=output_dir / "confusion_matrix.csv",
            error_analysis_json=output_dir / "error_analysis.json",
            gallery_index=output_dir / "gallery" / "index.html",
        )


def _repository(
    images: Sequence[AnnotatedImage],
    pixels: Mapping[str, RgbImage],
    *,
    name: str,
    events: list[tuple[str, str]] | None = None,
) -> InMemoryDatasetRepository:
    def record(repo: str, event: str) -> None:
        if events is not None:
            events.append((repo, event))

    return InMemoryDatasetRepository(images, pixels, name=name, on_event=record)


def _image(image_id: str, annotations: tuple = ()) -> AnnotatedImage:
    return AnnotatedImage(image_id=image_id, width=32, height=32, annotations=annotations)


def _pixels(ids: Sequence[str]) -> dict[str, RgbImage]:
    return {image_id: solid_image(32, 32, index + 1) for index, image_id in enumerate(ids)}


def test_select_operating_threshold_uses_iou_matching() -> None:
    ground_truth = (_image("Japan_000001", (annotation(DamageClass.D00, (0, 0, 10, 10)),)),)
    predictions = {
        "Japan_000001": [
            Detection(DamageClass.D00, 0.9, BBox(0, 0, 10, 10)),
            Detection(DamageClass.D00, 0.4, BBox(20, 20, 30, 30)),
        ]
    }

    selection = select_operating_threshold(ground_truth, predictions)

    assert selection.threshold == pytest.approx(0.9)
    assert selection.true_positives == 1
    assert selection.false_positives == 0
    assert selection.f1 == pytest.approx(1.0)


def test_operating_point_at_an_explicit_threshold_filters_predictions() -> None:
    ground_truth = (_image("Japan_000001", (annotation(DamageClass.D00, (0, 0, 10, 10)),)),)
    predictions = {
        "Japan_000001": [
            Detection(DamageClass.D00, 0.9, BBox(0, 0, 10, 10)),
            Detection(DamageClass.D00, 0.4, BBox(20, 20, 30, 30)),
        ]
    }

    point = operating_point(ground_truth, predictions, 0.5)

    assert point.threshold == pytest.approx(0.5)
    assert point.true_positives == 1
    assert point.false_positives == 0
    assert (point.precision, point.recall, point.f1) == pytest.approx((1.0, 1.0, 1.0))


def test_gallery_selection_is_deterministic_balanced_and_capped() -> None:
    cases = []
    for index in range(10):
        for category in (
            ErrorCategory.CLASSIFICATION,
            ErrorCategory.LOCALIZATION,
            ErrorCategory.DUPLICATE,
            ErrorCategory.BACKGROUND,
        ):
            cases.append(
                ErrorCase(
                    image_id=f"Japan_{index:06d}",
                    category=category,
                    damage_class=DamageClass.D00,
                    bbox=BBox(1, 1, 5, 5),
                    score=index / 20.0,
                )
            )
        cases.append(
            ErrorCase(
                image_id=f"Japan_{index:06d}",
                category=ErrorCategory.MISSED,
                damage_class=DamageClass.D40,
                bbox=BBox(2, 2, 20, 20),
            )
        )
    images = {case.image_id: _image(case.image_id) for case in cases}

    first = select_gallery_cases(cases, images, {})
    second = select_gallery_cases(cases, images, {})

    signature = [(case.category, case.image_id, case.detection_bbox) for case in first]
    assert signature == [(case.category, case.image_id, case.detection_bbox) for case in second]
    assert len(first) == 24
    per_category = {
        category: sum(1 for case in first if case.category is category)
        for category in ErrorCategory
    }
    for category in ErrorCategory:
        assert 1 <= per_category[category] <= 6
    backgrounds = [case for case in first if case.category is ErrorCategory.BACKGROUND]
    assert backgrounds[0].score == pytest.approx(max(case.score for case in backgrounds))


def test_empty_detector_gives_zero_map(tmp_path: Path) -> None:
    images = (
        _image("Japan_000001", (annotation(DamageClass.D00, (2, 2, 12, 12)),)),
        _image("Japan_000002", (annotation(DamageClass.D40, (4, 4, 20, 20)),)),
    )
    repository = _repository(images, _pixels([image.image_id for image in images]), name="test")
    writer = CapturingWriter()

    outcome = evaluate_model(
        repository,
        detector=EmptyDetector(),
        writer=writer,
        output_dir=tmp_path,
        split="test",
        operating_threshold=0.0,
    )

    assert isinstance(outcome, EvaluationOutcome)
    document = writer.reports[0].document
    assert document["metrics"]["map_50"] == 0.0
    assert document["metrics"]["map_50_95"] == 0.0
    assert document["dataset"]["images"] == 2
    assert document["params"]["operating_source"] == "explicit"


def test_echo_detector_scores_one_map_at_jitter_zero(tmp_path: Path) -> None:
    images = (
        _image("Japan_000001", (annotation(DamageClass.D00, (2, 2, 12, 12)),)),
        _image("Japan_000002", (annotation(DamageClass.D40, (4, 4, 20, 20)),)),
    )
    pixels = _pixels([image.image_id for image in images])
    repository = _repository(images, pixels, name="test")
    detector = GroundTruthEchoDetector(images, image_loader=pixels.__getitem__)
    writer = CapturingWriter()

    outcome = evaluate_model(
        repository,
        detector=detector,
        writer=writer,
        output_dir=tmp_path,
        split="test",
        operating_threshold=0.0,
    )

    assert outcome.metrics.mAP50 == pytest.approx(1.0)
    assert outcome.metrics.mAP50_95 == pytest.approx(1.0)


def test_threshold_selection_reads_only_the_validation_split(tmp_path: Path) -> None:
    val_images = (
        _image("Japan_000001", (annotation(DamageClass.D00, (0, 0, 10, 10)),)),
        _image("Japan_000002", (annotation(DamageClass.D40, (4, 4, 20, 20)),)),
    )
    test_images = (_image("Japan_100001", (annotation(DamageClass.D00, (1, 1, 9, 9)),)),)
    all_images = val_images + test_images
    pixels = _pixels([image.image_id for image in all_images])
    events: list[tuple[str, str]] = []
    val_repository = _repository(val_images, pixels, name="val", events=events)
    test_repository = _repository(test_images, pixels, name="test", events=events)
    detector = GroundTruthEchoDetector(all_images, image_loader=pixels.__getitem__)
    writer = CapturingWriter()

    evaluate_model(
        test_repository,
        selection_dataset=val_repository,
        detector=detector,
        writer=writer,
        output_dir=tmp_path,
        split="test",
    )

    names = [name for name, _ in events]
    assert names.count("val") >= 1
    last_val = max(index for index, name in enumerate(names) if name == "val")
    first_test = min(index for index, name in enumerate(names) if name == "test")
    assert first_test > last_val
    assert writer.reports[0].document["params"]["operating_source"] == "val"


def test_explicit_threshold_does_not_touch_the_validation_split(tmp_path: Path) -> None:
    test_images = (_image("Japan_100001", (annotation(DamageClass.D00, (1, 1, 9, 9)),)),)
    val_images = (_image("Japan_000001", (annotation(DamageClass.D00, (0, 0, 10, 10)),)),)
    pixels = _pixels([image.image_id for image in (*test_images, *val_images)])
    events: list[tuple[str, str]] = []
    val_repository = _repository(val_images, pixels, name="val", events=events)
    test_repository = _repository(test_images, pixels, name="test", events=events)
    detector = GroundTruthEchoDetector((*test_images, *val_images), image_loader=pixels.__getitem__)

    evaluate_model(
        test_repository,
        selection_dataset=val_repository,
        detector=detector,
        writer=CapturingWriter(),
        output_dir=tmp_path,
        split="test",
        operating_threshold=0.5,
    )

    assert all(name != "val" for name, _ in events)


def test_outcome_reports_artifacts_and_evaluated_count(tmp_path: Path) -> None:
    images = (_image("Japan_000001", (annotation(DamageClass.D00, (2, 2, 12, 12)),)),)
    repository = _repository(images, _pixels([image.image_id for image in images]), name="test")
    writer = CapturingWriter()

    outcome = evaluate_model(
        repository,
        detector=GroundTruthEchoDetector(images, image_loader=repository.load_image),
        writer=writer,
        output_dir=tmp_path,
        split="test",
        operating_threshold=0.0,
    )

    assert outcome.images_evaluated == 1
    assert outcome.artifacts.metrics_json == tmp_path / "metrics.json"
    assert outcome.operating_threshold == pytest.approx(0.0)
    gallery: tuple[GalleryCase, ...] = writer.reports[0].gallery
    assert gallery == ()
