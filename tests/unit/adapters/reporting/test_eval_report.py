"""Unit tests for the evaluation report adapters (S7)."""

from __future__ import annotations

import base64
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from rdv.adapters.reporting import plots
from rdv.adapters.reporting.eval_report import render_report_markdown
from rdv.adapters.reporting.gallery import GalleryItem, write_gallery, write_gallery_index
from rdv.adapters.reporting.html import render_report_html
from rdv.application.ports import GalleryCase
from rdv.domain.annotation import AnnotatedImage, Annotation
from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.metrics.confusion import confusion_matrix
from rdv.domain.metrics.error_analysis import ErrorCategory, error_breakdown
from rdv.domain.metrics.evaluator import evaluate
from tests.support.fakes import solid_image


def _document(model_id: str = "tiny") -> dict:
    return {
        "schema": "rdv.metrics/v1",
        "tool": {"name": "road-damage-vision", "version": "0.1.0"},
        "model": {"id": model_id, "sha256": "a" * 64, "input_size": 640},
        "dataset": {
            "split": "val",
            "images": 2,
            "ground_truth_boxes": 2,
            "classes": {"D00": 1, "D10": 0, "D20": 0, "D40": 1},
            "countries": {"Japan": 2},
            "score_threshold": 0.001,
        },
        "params": {
            "split": "val",
            "score_threshold": 0.001,
            "operating_threshold": 0.5,
            "operating_source": "val",
            "device": "cpu",
            "max_images": None,
        },
        "metrics": {
            "map_50": 1.0,
            "map_50_95": 0.8,
            "iou_thresholds": [0.5],
            "per_class": {
                "D00": {"num_gt": 1, "ap_50": 1.0, "ap_50_95": 0.8},
                "D10": {"num_gt": 0, "ap_50": None, "ap_50_95": None},
                "D20": {"num_gt": 0, "ap_50": None, "ap_50_95": None},
                "D40": {"num_gt": 1, "ap_50": 1.0, "ap_50_95": 0.8},
            },
        },
        "operating_point": {
            "threshold": 0.5,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 2,
            "false_positives": 0,
        },
        "confusion_matrix": {
            "labels": ["D00", "D10", "D20", "D40", "background"],
            "values": [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        "errors": {
            "counts": {
                "classification": 0,
                "localization": 0,
                "duplicate": 0,
                "background": 1,
                "missed": 0,
            },
            "total_false_positives": 1,
            "total_missed": 0,
        },
        "per_country": {
            "Japan": {
                "images": 2,
                "ground_truth_boxes": 2,
                "map_50": 1.0,
                "map_50_95": 0.8,
                "per_class": {},
            }
        },
        "ood": None,
        "gallery": {"total_cases": 1, "per_category": {}},
    }


def _case(image_id: str = "Japan_000001") -> GalleryCase:
    return GalleryCase(
        image_id=image_id,
        category=ErrorCategory.BACKGROUND,
        damage_class=DamageClass.D00,
        ground_truth=(),
        detections=(),
        detection_bbox=BBox(1, 1, 10, 10),
        score=0.9,
    )


def _gallery_item(tmp_path: Path) -> GalleryItem:
    image_path = tmp_path / "gallery" / "background" / "01_x.jpg"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(solid_image(24, 16, 90)).save(image_path, format="JPEG")
    return GalleryItem(
        case=_case(),
        path=image_path,
        relative_path="gallery/background/01_x.jpg",
    )


def test_markdown_report_has_every_section(tmp_path: Path) -> None:
    text = render_report_markdown(_document(), (_gallery_item(tmp_path),))

    for heading in (
        "# Evaluation report",
        "## Metrics per class",
        "## Operating point",
        "## Per country",
        "## Error taxonomy",
        "## Confusion matrix",
        "## Artifacts",
        "## Error gallery",
    ):
        assert heading in text
    assert "| D00 | Longitudinal crack | 1 | 1.0000 | 0.8000 |" in text
    assert "gallery/background/01_x.jpg" in text


def test_html_report_is_self_contained_and_escapes_content(tmp_path: Path) -> None:
    item = _gallery_item(tmp_path)
    html = render_report_html(
        _document(model_id="<script>alert(1)</script>"),
        pr_curves_png=None,
        confusion_png=None,
        gallery_items=(item,),
    )

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    encoded = base64.b64encode(item.path.read_bytes()).decode("ascii")
    assert "data:image/jpeg;base64," in html
    assert encoded
    assert "http://" not in html and "https://" not in html


def test_plots_write_pngs_and_csvs(tmp_path: Path) -> None:
    ground_truth = (
        AnnotatedImage(
            image_id="Japan_000001",
            width=32,
            height=32,
            annotations=(
                Annotation(DamageClass.D00, BBox(0, 0, 10, 10)),
                Annotation(DamageClass.D40, BBox(10, 10, 20, 20)),
            ),
        ),
    )
    predictions = {
        "Japan_000001": [
            Detection(DamageClass.D00, 0.9, BBox(0, 0, 10, 10)),
            Detection(DamageClass.D40, 0.8, BBox(10, 10, 20, 20)),
        ]
    }
    result = evaluate(ground_truth, predictions)
    curves = {
        damage_class: result.per_class[damage_class].pr_curve
        for damage_class in DamageClass
        if result.per_class[damage_class].pr_curve is not None
    }
    confusion = confusion_matrix(ground_truth, predictions, score_threshold=0.5)

    csvs = plots.write_pr_curve_csvs(tmp_path, curves)
    plots.write_pr_curves_png(tmp_path / "pr_curves.png", curves)
    plots.write_confusion_csv(tmp_path / "confusion_matrix.csv", confusion)
    plots.write_confusion_png(tmp_path / "confusion_matrix.png", confusion)

    assert set(csvs) == {DamageClass.D00, DamageClass.D40}
    for path in csvs.values():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["recall", "precision", "score"]
        assert len(rows) > 1
    with (tmp_path / "confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["truth\\prediction", "D00", "D10", "D20", "D40", "background"]
    assert rows[4][4] == "1"
    assert rows[5][5] == "0"
    assert (tmp_path / "pr_curves.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / "confusion_matrix.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    image = np.asarray(Image.open(tmp_path / "confusion_matrix.png"))
    assert image.ndim == 3


def test_gallery_index_references_only_written_files(tmp_path: Path) -> None:
    cases = (
        _case("Japan_000001"),
        GalleryCase(
            image_id="Japan_000002",
            category=ErrorCategory.MISSED,
            damage_class=DamageClass.D40,
            ground_truth=(),
            detections=(),
            detection_bbox=None,
            ground_truth_bbox=BBox(2, 2, 20, 20),
        ),
    )
    pixels = {
        "Japan_000001": solid_image(24, 16, 10),
        "Japan_000002": solid_image(24, 16, 20),
    }
    items = write_gallery(tmp_path, cases, pixels.__getitem__)
    index = write_gallery_index(tmp_path, items)

    assert index == tmp_path / "gallery" / "index.html"
    html = index.read_text(encoding="utf-8")
    for item in items:
        reference = item.relative_path.removeprefix("gallery/")
        assert reference in html
        assert (tmp_path / item.relative_path).is_file()
    assert "background" in html and "missed" in html
    assert html.count("<img ") == 2


def test_error_breakdown_fixture_has_background_cases() -> None:
    ground_truth = (
        AnnotatedImage("Japan_000001", 32, 32, (Annotation(DamageClass.D00, BBox(0, 0, 8, 8)),)),
    )
    predictions = {
        "Japan_000001": [
            Detection(DamageClass.D00, 0.9, BBox(20, 20, 30, 30)),
        ]
    }
    breakdown = error_breakdown(ground_truth, predictions, score_threshold=0.5)
    assert breakdown.counts[ErrorCategory.BACKGROUND] == 1
    assert breakdown.cases[0].category is ErrorCategory.BACKGROUND
