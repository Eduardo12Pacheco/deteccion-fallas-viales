"""Integration tests for evaluation on the real mini fixture (S7/T8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdv.adapters.datasets.rdd2022 import Rdd2022VocRepository, VocDirectoryRepository
from rdv.adapters.reporting.eval_report import EvaluationReportWriter
from rdv.application.evaluate_model import evaluate_model
from tests.support.fakes import GroundTruthEchoDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
COUNTRIES = ("Japan", "India", "Czech")


def _repository() -> Rdd2022VocRepository:
    return Rdd2022VocRepository(MINI, COUNTRIES, split="all")


def test_echo_detector_gives_perfect_map_and_identity_confusion(tmp_path: Path) -> None:
    repository = _repository()
    detector = GroundTruthEchoDetector.from_repository(repository, jitter=0)

    outcome = evaluate_model(
        repository,
        detector=detector,
        writer=EvaluationReportWriter(),
        output_dir=tmp_path,
        split="all",
        operating_threshold=0.0,
    )

    assert outcome.metrics.mAP50 == pytest.approx(1.0)
    assert outcome.metrics.mAP50_95 == pytest.approx(1.0)
    document = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    matrix = document["confusion_matrix"]["values"]
    assert [matrix[index][index] for index in range(4)] == [3, 3, 4, 4]
    for row in range(5):
        for column in range(5):
            if row != column:
                assert matrix[row][column] == 0
    assert document["errors"]["counts"] == {
        "classification": 0,
        "localization": 0,
        "duplicate": 0,
        "background": 0,
        "missed": 0,
    }


def test_jittered_echo_detector_keeps_map50_but_loses_high_iou(tmp_path: Path) -> None:
    repository = _repository()
    detector = GroundTruthEchoDetector.from_repository(repository, jitter=4)

    outcome = evaluate_model(
        repository,
        detector=detector,
        writer=EvaluationReportWriter(),
        output_dir=tmp_path,
        split="all",
        operating_threshold=0.0,
    )

    assert outcome.metrics.mAP50 == pytest.approx(1.0)
    assert 0.5 < outcome.metrics.mAP50_95 < 1.0


def test_evaluation_writes_every_s7_artifact_and_ood_metrics(tmp_path: Path) -> None:
    repository = _repository()
    detector = GroundTruthEchoDetector.from_repository(repository, jitter=0)

    outcome = evaluate_model(
        repository,
        detector=detector,
        writer=EvaluationReportWriter(),
        output_dir=tmp_path,
        ood_dataset=VocDirectoryRepository(MINI),
        split="all",
        operating_threshold=0.0,
    )

    artifacts = outcome.artifacts
    for path in (
        artifacts.metrics_json,
        artifacts.report_markdown,
        artifacts.report_html,
        artifacts.pr_curves_png,
        artifacts.confusion_png,
        artifacts.confusion_csv,
        artifacts.error_analysis_json,
        artifacts.gallery_index,
    ):
        assert path.is_file(), path
    assert set(artifacts.pr_curve_csvs) == set(outcome.metrics.per_class)
    document = json.loads(artifacts.metrics_json.read_text(encoding="utf-8"))
    assert document["schema"] == "rdv.metrics/v1"
    assert document["ood"]["images"] == 16
    assert document["metrics"]["map_50"] == pytest.approx(1.0)
    html = artifacts.report_html.read_text(encoding="utf-8")
    assert "data:image/" in html
    assert "Evaluation report" in html
    assert "Out of domain" in html
