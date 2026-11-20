"""Post-release smoke: released wheel + real model + packaged samples (S18).

The release workflow installs the published wheel, pulls the real model (whose
sha256 ``rdv models pull`` verifies), and then runs this script. It detects each
packaged sample with the released code and compares the observed classes with
``reports/v1/sample_expectations.json``: every expected class must appear with
a score at or above the pinned threshold. A missing class exits 1.

Usage:
    uv run python scripts/smoke_release.py \\
        --model "$(rdv models path)" \\
        --samples src/rdv/assets/samples \\
        --expectations reports/v1/sample_expectations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.adapters.media.images import PillowImageSource
from rdv.application.ports import DEFAULT_CLASS_CODES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "src" / "rdv" / "assets" / "samples"
DEFAULT_EXPECTATIONS = ROOT / "reports" / "v1" / "sample_expectations.json"
EXPECTATIONS_SCHEMA = "rdv.sample-expectations/v1"


@dataclass(frozen=True, slots=True)
class SampleExpectation:
    """One packaged sample and the classes the released model must report."""

    image: str
    expected_classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SampleStandout:
    """Observed best score per expected class for one sample."""

    image: str
    observed: dict[str, float]


def load_expectations(path: Path) -> tuple[float, list[SampleExpectation]]:
    """Read and validate the committed sample expectations.

    Raises:
        ValueError: the file is missing, malformed or internally inconsistent.
    """
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read expectations {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTATIONS_SCHEMA:
        raise ValueError(f"{path} must declare schema {EXPECTATIONS_SCHEMA!r}")
    threshold = payload.get("threshold")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError(f"{path} 'threshold' must be a number")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{path} 'threshold' must be within [0, 1]")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"{path} 'samples' must be a non-empty list")
    samples: list[SampleExpectation] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, dict):
            raise ValueError(f"{path} sample {index} must be an object")
        image = raw.get("image")
        classes = raw.get("expected_classes")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{path} sample {index} is missing 'image'")
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"{path} sample {image!r} must list expected_classes")
        unknown = [code for code in classes if code not in DEFAULT_CLASS_CODES]
        if unknown:
            raise ValueError(f"{path} sample {image!r} has unknown classes {unknown!r}")
        samples.append(SampleExpectation(image=image, expected_classes=tuple(classes)))
    return threshold, samples


def best_scores(
    model: Path,
    samples: Path,
    expectations: list[SampleExpectation],
    threshold: float,
) -> list[SampleStandout]:
    """Detect every expected sample and return its best score per class."""
    detector = OnnxDetector(model, score_threshold=threshold)
    source = PillowImageSource()
    results: list[SampleStandout] = []
    for expectation in expectations:
        image_path = samples / expectation.image
        if not image_path.is_file():
            raise ValueError(f"sample image not found: {image_path}")
        detections = detector.detect([source.load_image(image_path)])[0]
        observed: dict[str, float] = {}
        for detection in detections:
            code = detection.damage_class.code
            observed[code] = max(observed.get(code, 0.0), detection.score)
        results.append(SampleStandout(image=expectation.image, observed=observed))
    return results


def missing_classes(
    expectations: list[SampleExpectation],
    standouts: list[SampleStandout],
) -> list[str]:
    """Return one human-readable failure line per sample with missing classes."""
    failures: list[str] = []
    for expectation, standout in zip(expectations, standouts, strict=True):
        missing = [code for code in expectation.expected_classes if code not in standout.observed]
        if missing:
            observed = ", ".join(
                f"{code} {score:.4f}" for code, score in sorted(standout.observed.items())
            )
            failures.append(
                f"{expectation.image}: missing expected classes {missing!r} "
                f"(observed: {observed or 'none'})"
            )
    return failures


def run_smoke(
    model: Path,
    samples: Path,
    expectations_path: Path,
) -> list[str]:
    """Run the whole smoke and return the failure lines (empty on success)."""
    if not model.is_file():
        return [f"model file not found: {model}"]
    if not samples.is_dir():
        return [f"samples directory not found: {samples}"]
    try:
        threshold, expectations = load_expectations(expectations_path)
        standouts = best_scores(model, samples, expectations, threshold)
    except ValueError as exc:
        return [str(exc)]
    return missing_classes(expectations, standouts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test released detections against the pinned sample expectations."
    )
    parser.add_argument("--model", type=Path, required=True, help="Released ONNX model file.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    args = parser.parse_args(argv)

    failures = run_smoke(args.model, args.samples, args.expectations)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"smoke_release: {len(failures)} sample(s) failed", file=sys.stderr)
        return 1

    threshold, expectations = load_expectations(args.expectations)
    print(
        f"OK: {len(expectations)}/{len(expectations)} samples match {args.expectations} "
        f"at threshold {threshold:g} with {args.model.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
