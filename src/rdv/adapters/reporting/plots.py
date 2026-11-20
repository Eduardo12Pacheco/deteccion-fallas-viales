"""Matplotlib/CSV artifacts for the evaluation report (S7).

The palette is the Okabe-Ito set from decision D11, so the PR curves stay
colorblind-safe. Figures use the non-interactive Agg backend and deterministic
titles/labels; the CLI never opens a window.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rdv.domain.classes import DamageClass
from rdv.domain.metrics.confusion import BACKGROUND_LABEL, ConfusionMatrix
from rdv.domain.metrics.curves import PRCurve
from rdv.errors import MediaError

PALETTE: Mapping[DamageClass, str] = {
    DamageClass.D00: "#0072B2",
    DamageClass.D10: "#E69F00",
    DamageClass.D20: "#CC79A7",
    DamageClass.D40: "#D55E00",
}
MEAN_COLOR = "#333333"
_DIGITS = 6


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_pr_curve_csvs(
    output_dir: Path, curves: Mapping[DamageClass, PRCurve]
) -> dict[DamageClass, Path]:
    """Write one raw precision/recall trace per class at IoU 0.5."""
    paths: dict[DamageClass, Path] = {}
    output_dir = Path(output_dir)
    for damage_class in DamageClass:
        curve = curves.get(damage_class)
        if curve is None:
            continue
        path = output_dir / f"pr_curve_{damage_class.code}.csv"
        rows = [
            [round(recall, _DIGITS), round(precision, _DIGITS), round(score, _DIGITS)]
            for recall, precision, score in zip(
                curve.recall, curve.precision, curve.scores, strict=True
            )
        ]
        _write_csv(path, ["recall", "precision", "score"], rows)
        paths[damage_class] = path
    return paths


def write_pr_curves_png(path: Path, curves: Mapping[DamageClass, PRCurve]) -> None:
    """Plot the 101-point interpolated PR curve of each class plus the mean."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6.4, 4.8))
    try:
        available: list[np.ndarray] = []
        for damage_class in DamageClass:
            curve = curves.get(damage_class)
            if curve is None:
                continue
            color = PALETTE[damage_class]
            axes.plot(
                curve.recall_101,
                curve.precision_101,
                color=color,
                linewidth=2,
                label=f"{damage_class.code} (AP {curve.ap:.3f})",
            )
            available.append(np.asarray(curve.precision_101, dtype=np.float64))
        if available:
            mean_precision = np.mean(np.vstack(available), axis=0)
            recall = np.asarray(next(iter(curves.values())).recall_101, dtype=np.float64)
            axes.plot(
                recall,
                mean_precision,
                color=MEAN_COLOR,
                linewidth=1.5,
                linestyle="--",
                label="mean",
            )
        axes.set_title("Precision-recall curves (IoU 0.5)")
        axes.set_xlabel("Recall")
        axes.set_ylabel("Precision (101-point interpolated)")
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.05)
        axes.grid(True, color="#dddddd", linewidth=0.6)
        axes.legend(loc="lower left", frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def write_confusion_csv(path: Path, confusion: ConfusionMatrix) -> None:
    """Write the 5x5 confusion matrix with a ``truth\\prediction`` header."""
    labels = list(confusion.labels)
    rows = [
        [labels[row], *[int(value) for value in confusion.matrix[row]]]
        for row in range(len(labels))
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(path, ["truth\\prediction", *labels], rows)


def write_confusion_png(path: Path, confusion: ConfusionMatrix) -> None:
    """Plot the confusion matrix as an annotated heatmap."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(confusion.labels)
    matrix = np.asarray(confusion.matrix, dtype=np.int64)
    figure, axes = plt.subplots(figsize=(6.0, 5.0))
    try:
        image = axes.imshow(matrix, cmap="Blues", interpolation="nearest")
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
        axes.set_title("Confusion matrix (IoU 0.5, operating threshold)")
        axes.set_xlabel("Predicted class")
        axes.set_ylabel("Ground-truth class")
        axes.set_xticks(range(len(labels)), labels=labels)
        axes.set_yticks(range(len(labels)), labels=labels)
        threshold = matrix.max() / 2.0 if matrix.size else 0.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                axes.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "#222222",
                    fontsize=9,
                )
        axes.text(
            0.0,
            -0.14,
            f"background label: {BACKGROUND_LABEL}",
            transform=axes.transAxes,
            fontsize=8,
            color="#555555",
        )
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def ensure_output_dir(path: Path) -> Path:
    """Create the report directory (kept here for adapter-side file errors)."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"Cannot create report directory {path}: {exc}") from exc
    return path
