"""Evaluation report writer: JSON, Markdown, HTML, plots and gallery (S7).

This adapter implements ``EvaluationReportWriter``. The application layer
hands it two JSON-ready documents plus the typed curve/confusion/gallery
values; everything rendering-related (Jinja2, matplotlib, Pillow) stays here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdv.adapters.reporting.detections_json import dumps_document
from rdv.adapters.reporting.gallery import GalleryItem, write_gallery, write_gallery_index
from rdv.adapters.reporting.html import render_report_html
from rdv.adapters.reporting.plots import (
    write_confusion_csv,
    write_confusion_png,
    write_pr_curve_csvs,
    write_pr_curves_png,
)
from rdv.application.ports import (
    EvaluationArtifacts,
    EvaluationReport,
    RgbImage,
)
from rdv.application.ports import (
    EvaluationReportWriter as EvaluationReportWriterPort,
)
from rdv.domain.classes import DamageClass
from rdv.errors import MediaError

_CATEGORY_ORDER = (
    "classification",
    "localization",
    "duplicate",
    "background",
    "missed",
)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_report_markdown(
    document: Mapping[str, Any],
    gallery_items: Sequence[GalleryItem] = (),
) -> str:
    """Render the human-readable Markdown report from the metrics document."""
    metrics = document["metrics"]
    params = document["params"]
    dataset = document["dataset"]
    operating = document["operating_point"]
    lines = [
        "# Evaluation report",
        "",
        f"- Model: `{_cell(document['model']['id'])}` (device `{_cell(params['device'])}`)",
        f"- Split: `{_cell(params['split'])}`",
        f"- Images: {_cell(dataset['images'])}",
        f"- Ground-truth boxes: {_cell(dataset['ground_truth_boxes'])}",
        f"- Operating threshold: {_format(operating['threshold'])} "
        f"({_cell(params['operating_source'])})",
        f"- mAP@0.5: {_format(metrics['map_50'])}",
        f"- mAP@0.5:0.95: {_format(metrics['map_50_95'])}",
        "",
        "## Metrics per class",
        "",
        "| Class | Name | Boxes | AP@0.5 | AP@0.5:0.95 |",
        "|-------|------|-------|--------|-------------|",
    ]
    for damage_class in DamageClass:
        values = metrics["per_class"][damage_class.code]
        lines.append(
            f"| {damage_class.code} | {_cell(damage_class.display_name)} | "
            f"{_cell(values['num_gt'])} | {_format(values['ap_50'])} | "
            f"{_format(values['ap_50_95'])} |"
        )
    lines.extend(
        [
            "",
            "## Operating point",
            "",
            "| Threshold | Precision | Recall | F1 | TP | FP |",
            "|-----------|-----------|--------|----|----|----|",
            f"| {_format(operating['threshold'])} | {_format(operating['precision'])} | "
            f"{_format(operating['recall'])} | {_format(operating['f1'])} | "
            f"{_cell(operating['true_positives'])} | {_cell(operating['false_positives'])} |",
            "",
            "## Per country",
            "",
            "| Country | Images | Boxes | mAP@0.5 | mAP@0.5:0.95 |",
            "|---------|--------|-------|---------|--------------|",
        ]
    )
    for country, values in document["per_country"].items():
        lines.append(
            f"| {_cell(country)} | {_cell(values['images'])} | "
            f"{_cell(values['ground_truth_boxes'])} | {_format(values['map_50'])} | "
            f"{_format(values['map_50_95'])} |"
        )
    ood = document.get("ood")
    if ood is not None:
        lines.extend(
            [
                "",
                "## Out of domain",
                "",
                "| Images | Boxes | mAP@0.5 | mAP@0.5:0.95 |",
                "|--------|-------|---------|--------------|",
                f"| {_cell(ood['images'])} | {_cell(ood['ground_truth_boxes'])} | "
                f"{_format(ood['map_50'])} | {_format(ood['map_50_95'])} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Error taxonomy",
            "",
            "| Category | Count |",
            "|----------|-------|",
        ]
    )
    for category in _CATEGORY_ORDER:
        lines.append(f"| {category} | {_cell(document['errors']['counts'].get(category, 0))} |")
    lines.extend(
        [
            f"| **false positives** | {_cell(document['errors']['total_false_positives'])} |",
            f"| **missed** | {_cell(document['errors']['total_missed'])} |",
            "",
            "## Confusion matrix",
            "",
        ]
    )
    labels = document["confusion_matrix"]["labels"]
    lines.append("| Truth \\ prediction | " + " | ".join(_cell(label) for label in labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for label, values in zip(labels, document["confusion_matrix"]["values"], strict=True):
        lines.append(f"| {_cell(label)} | " + " | ".join(_cell(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- [Precision-recall curves](pr_curves.png)",
            "- [Confusion matrix](confusion_matrix.png)",
            "- [Error analysis](error_analysis.json)",
            "- [Error gallery](gallery/index.html)",
            "",
        ]
    )
    if gallery_items:
        lines.extend(
            [
                "## Error gallery",
                "",
                "| Category | Image | Class | Score | IoU |",
                "|----------|-------|-------|-------|-----|",
            ]
        )
        for item in gallery_items:
            case = item.case
            lines.append(
                f"| {_cell(case.category.value)} | {_cell(item.relative_path)} | "
                f"{_cell(case.damage_class.code)} | {_format(case.score, 3)} | "
                f"{_format(case.iou, 3)} |"
            )
        lines.append("")
    return "\n".join(lines)


class EvaluationReportWriter(EvaluationReportWriterPort):
    """Render every S7 artifact for one evaluation run."""

    def write_report(
        self,
        output_dir: Path,
        report: EvaluationReport,
        *,
        load_image: Callable[[str], RgbImage],
    ) -> EvaluationArtifacts:
        """Write metrics, Markdown, HTML, plots, CSVs, errors and gallery."""
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MediaError(f"Cannot create report directory {output_dir}: {exc}") from exc

        metrics_json = output_dir / "metrics.json"
        error_json = output_dir / "error_analysis.json"
        pr_curves_png = output_dir / "pr_curves.png"
        confusion_png = output_dir / "confusion_matrix.png"
        confusion_csv = output_dir / "confusion_matrix.csv"
        report_markdown = output_dir / "report.md"
        report_html = output_dir / "report.html"

        try:
            metrics_json.write_text(dumps_document(report.document), encoding="utf-8")
            error_json.write_text(dumps_document(report.error_document), encoding="utf-8")
        except OSError as exc:
            raise MediaError(f"Cannot write report JSON under {output_dir}: {exc}") from exc

        pr_curve_csvs = write_pr_curve_csvs(output_dir, report.pr_curves)
        write_pr_curves_png(pr_curves_png, report.pr_curves)
        write_confusion_csv(confusion_csv, report.confusion)
        write_confusion_png(confusion_png, report.confusion)
        gallery_items = write_gallery(output_dir, report.gallery, load_image)
        gallery_index = write_gallery_index(output_dir, gallery_items)

        try:
            report_markdown.write_text(
                render_report_markdown(report.document, gallery_items), encoding="utf-8"
            )
            report_html.write_text(
                render_report_html(
                    report.document,
                    pr_curves_png=pr_curves_png,
                    confusion_png=confusion_png,
                    gallery_items=gallery_items,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise MediaError(f"Cannot write report under {output_dir}: {exc}") from exc

        return EvaluationArtifacts(
            output_dir=output_dir,
            metrics_json=metrics_json,
            report_markdown=report_markdown,
            report_html=report_html,
            pr_curves_png=pr_curves_png,
            pr_curve_csvs=pr_curve_csvs,
            confusion_png=confusion_png,
            confusion_csv=confusion_csv,
            error_analysis_json=error_json,
            gallery_index=gallery_index,
        )
