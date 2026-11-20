#!/usr/bin/env python3
"""Render the honest-metrics block of the READMEs from committed reports (D17).

The README numbers must never be edited by hand: this script reads
``reports/v1/metrics.json`` (plus the benchmark report when present) and writes
the Markdown between ``<!-- metrics:start -->`` and ``<!-- metrics:end -->``.
The S10 minimum bar is evaluated here as well, so a "below target" run is
rendered as such instead of being silently rounded up.

Usage:
    python scripts/render_readme_metrics.py                # rewrite README blocks
    python scripts/render_readme_metrics.py --check        # fail if blocks differ
    python scripts/render_readme_metrics.py --status       # PASS / BELOW TARGET
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

START_MARKER = "<!-- metrics:start -->"
END_MARKER = "<!-- metrics:end -->"
DEFAULT_README = ("README.md", "README.es.md")
DEFAULT_REPORTS = Path("reports/v1")

# S10 minimum bar: mAP@0.5, mAP@0.5:0.95 and every class AP@0.5.
MIN_MAP50 = 0.35
MIN_MAP50_95 = 0.15
MIN_CLASS_AP50 = 0.10
CLASS_CODES = ("D00", "D10", "D20", "D40")


class ReportError(Exception):
    """A required report is missing or does not follow the committed schema."""


def load_json(path: Path) -> dict[str, Any]:
    """Read a committed JSON report, mapping failures to a clear message."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReportError(f"Missing report: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportError(f"Report {path} must contain a JSON object")
    return payload


def load_reports(reports_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load the required metrics report and the optional benchmark report."""
    metrics = load_json(reports_dir / "metrics.json")
    if metrics.get("schema") != "rdv.metrics/v1":
        raise ReportError(f"{reports_dir / 'metrics.json'} is not an rdv.metrics/v1 report")
    benchmark_path = reports_dir / "benchmark.json"
    benchmark = load_json(benchmark_path) if benchmark_path.is_file() else None
    return metrics, benchmark


def _number(payload: dict[str, Any], key: str, *, source: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ReportError(f"{source} is missing a numeric {key!r}")
    return float(value)


def status_lines(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return ``(passes_min_bar, report_lines)`` for the S10 thresholds."""
    values = metrics.get("metrics")
    if not isinstance(values, dict):
        raise ReportError("metrics.json is missing the 'metrics' object")
    map50 = _number(values, "map_50", source="metrics.map_50")
    map50_95 = _number(values, "map_50_95", source="metrics.map_50_95")
    per_class = values.get("per_class")
    if not isinstance(per_class, dict):
        raise ReportError("metrics.json is missing 'metrics.per_class'")

    class_ap: list[tuple[str, float]] = []
    for code in CLASS_CODES:
        entry = per_class.get(code)
        if not isinstance(entry, dict) or not isinstance(entry.get("ap_50"), (int, float)):
            raise ReportError(f"metrics per_class.{code}.ap_50 is missing")
        class_ap.append((code, float(entry["ap_50"])))

    failures: list[str] = []
    if map50 < MIN_MAP50:
        failures.append(f"mAP@0.5 {map50:.4f} < {MIN_MAP50:.2f}")
    if map50_95 < MIN_MAP50_95:
        failures.append(f"mAP@0.5:0.95 {map50_95:.4f} < {MIN_MAP50_95:.2f}")
    for code, ap in class_ap:
        if ap < MIN_CLASS_AP50:
            failures.append(f"{code} AP@0.5 {ap:.4f} < {MIN_CLASS_AP50:.2f}")

    class_text = ", ".join(f"{code} {ap:.4f}" for code, ap in class_ap)
    if failures:
        lines = [
            "BELOW TARGET: " + "; ".join(failures),
            f"mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}, per-class AP@0.5: {class_text}",
        ]
        return False, lines
    line = (
        f"PASS (>= min bar) mAP@0.5={map50:.4f} (min {MIN_MAP50:.2f}), "
        f"mAP@0.5:0.95={map50_95:.4f} (min {MIN_MAP50_95:.2f}), "
        f"per-class AP@0.5: {class_text} (min {MIN_CLASS_AP50:.2f})"
    )
    return True, [line]


def render_block(metrics: dict[str, Any], benchmark: dict[str, Any] | None) -> str:
    """Build the Markdown block shared by the English and Spanish READMEs."""
    values = metrics["metrics"]
    passes, status = status_lines(metrics)
    rows: list[tuple[str, str]] = [
        ("mAP@0.5", f"{_number(values, 'map_50', source='metrics.map_50'):.4f}"),
        ("mAP@0.5:0.95", f"{_number(values, 'map_50_95', source='metrics.map_50_95'):.4f}"),
    ]
    for code in CLASS_CODES:
        entry = values["per_class"][code]
        ap = entry.get("ap_50")
        rows.append(
            (f"AP@0.5 {code}", f"{float(ap):.4f}" if isinstance(ap, (int, float)) else "n/a")
        )
    if benchmark is not None:
        cpu = benchmark.get("cpu") or {}
        gpu = benchmark.get("gpu") or {}
        if isinstance(cpu.get("p50_ms"), (int, float)):
            rows.append(("CPU p50 (ms/image)", f"{float(cpu['p50_ms']):.1f}"))
        if isinstance(gpu.get("p50_ms"), (int, float)):
            rows.append(("GPU p50 (ms/image)", f"{float(gpu['p50_ms']):.1f}"))
    ood = metrics.get("ood")
    if isinstance(ood, dict):
        if isinstance(ood.get("map_50"), (int, float)):
            rows.append(("OOD Norway mAP@0.5", f"{float(ood['map_50']):.4f}"))
        if isinstance(ood.get("map_50_95"), (int, float)):
            rows.append(("OOD Norway mAP@0.5:0.95", f"{float(ood['map_50_95']):.4f}"))
    rows.append(("S10 status", "PASS (>= min bar)" if passes else "BELOW TARGET"))

    lines = [START_MARKER, "| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    if len(status) > 1:
        lines.append("")
        lines.append(status[1])
    lines.append(END_MARKER)
    return "\n".join(lines)


def _replace_block(text: str, block: str, *, path: Path) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ReportError(f"{path} has no {START_MARKER} ... {END_MARKER} block")
    return text[:start] + block + text[end + len(END_MARKER) :]


def update_readme(path: Path, block: str, *, check: bool) -> None:
    """Rewrite one README block, or verify it when ``check`` is true."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if check:
            return
        raise
    if START_MARKER not in text:
        if check:
            return
        raise ReportError(f"{path} has no {START_MARKER} marker")
    updated = _replace_block(text, block, path=path)
    if check:
        if updated != text:
            raise ReportError(f"{path} metrics block is stale; run the renderer without --check")
        return
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the renderer; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--readme", type=Path, action="append", dest="readmes")
    parser.add_argument("--status", action="store_true", help="Print the S10 verdict only.")
    parser.add_argument("--check", action="store_true", help="Fail when README blocks are stale.")
    args = parser.parse_args(argv)

    try:
        metrics, benchmark = load_reports(args.reports)
        if args.status:
            passes, lines = status_lines(metrics)
            for line in lines:
                print(line)
            return 0 if passes else 1
        block = render_block(metrics, benchmark)
        readmes = args.readmes if args.readmes else [Path(name) for name in DEFAULT_README]
        found = False
        for readme in readmes:
            if not readme.is_file():
                # README.es.md is created by T16; until then the English file
                # is the only one that exists on a fresh checkout.
                continue
            update_readme(readme, block, check=args.check)
            found = True
        if not found:
            raise ReportError("No README file with a metrics block was found")
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
