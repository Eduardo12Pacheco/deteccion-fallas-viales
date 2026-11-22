# 0010 - Honest metrics: generate the numbers, never hand-edit them

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D16, D17, S10, S20)
- **Task:** T10 (first real run), T16 (README blocks)

## Context

The project's whole claim is reproducibility. A hand-written mAP number in a
README can drift from the committed report, can be rounded up, or can hide a
run that missed the documented minimum bar. The plan fixes the S10 bar
(mAP@0.5 >= 0.35, mAP@0.5:0.95 >= 0.15, every class AP@0.5 >= 0.10) and the
D16 retry policy, so the presentation must be derived mechanically from
`reports/v1/metrics.json`.

## Decision

- `reports/v1/` is the single source of truth: `metrics.json` comes from
  `rdv evaluate`, `benchmark.json` from `rdv benchmark`, and `run.json` from
  the trainer. These files are committed; the weights, checkpoints and ONNX
  artifact are not.
- `scripts/render_readme_metrics.py` renders the Markdown between
  `<!-- metrics:start -->` and `<!-- metrics:end -->` in `README.md` and
  `README.en.md` from those reports. It also evaluates S10 and prints
  `PASS (>= min bar)` or `BELOW TARGET` with the failing values.
- `--check` exits non-zero when a README block differs from what the reports
  would render, so CI can reject an edited number.
- A run below the bar is published unchanged with the `BELOW TARGET` label.
  One bounded D16 retry (12 extra epochs at lr x 0.3) is allowed; both runs are
  kept in `runs/`, the better one is evaluated, and neither is silently
  "fixed". If a per-class AP@0.5 stays below 0.10 the task is surfaced as a
  decision item instead of relaxed.
- The committed `metrics.json` is the artifact referenced by the model
  manifest (`metrics_ref`) and by the model card; the same values feed the
  sample expectations used by the post-release smoke.
- `--status` is the command used in the task Log, so the reported status and
  the README status cannot disagree.

## Consequences

- README numbers can be regenerated from the repository without a GPU.
- A stale README block is a test failure, not a documentation preference.
- Metrics below the bar remain visible; the repository documents what the
  model actually achieved instead of claiming the target.
- Cross-machine float noise is handled at the reproduction layer
  (`docs/TRAINING.md`, D23), not by loosening the renderer.
