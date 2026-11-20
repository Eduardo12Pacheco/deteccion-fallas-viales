# 0005 - Own metrics engine with pycocotools parity

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D10, S7)
- **Task:** T3

## Context

The evaluation report is the central evidence of this portfolio: mAP@0.5,
mAP@0.5:0.95, per-class AP, precision/recall curves, a confusion matrix and a
TIDE-style error taxonomy. The obvious implementation is to call
`pycocotools`, but it is a dev-only C extension that drags a build toolchain
into every supported platform and hides the evaluation semantics behind an
opaque `COCOeval` object. The report also needs per-country breakdowns, an
operating threshold chosen on validation data and error cases that
`pycocotools` does not expose.

The plan requires the metrics engine to live in the pure domain (stdlib plus
NumPy only, decision D9) and to match COCO semantics within `1e-4`
(decision D10, spec S7).

## Decision

Implement the metrics engine in `rdv.domain.metrics` with NumPy and verify it
against `pycocotools` in a `parity`-marked integration test:

- `matching.py` reproduces COCO's per-image, per-class greedy matching:
  detections ranked by descending score (stable), at most `maxDets=100` per
  image and class, each detection taking the last still-unmatched ground-truth
  box whose IoU is at least the current best. Matching exactly at the
  threshold counts; at IoU 1.0 the reference caps the threshold at
  `1 - 1e-10`.
- `average_precision.py` computes the 101-point interpolated AP with COCO's
  monotone precision envelope and the `np.spacing(1)` denominator guard.
  Categories without ground truth are excluded from the mean, not scored 0.
- `curves.py` exposes the raw trace, the 101-point trace and score arrays, and
  selects the operating threshold by maximum F1 at IoU 0.5 (first maximum,
  highest score wins).
- `confusion.py` builds a 5x5 class-agnostic matrix (rows = ground truth,
  columns = prediction, background last) with greedy IoU 0.5 matching by
  score.
- `error_analysis.py` classifies false positives as classification
  (other-class IoU >= 0.5), localization (same-class `0.1 <= IoU < 0.5`),
  duplicate (same-class already matched, IoU >= 0.5) or background, in that
  order. The D10 list puts classification first, so it wins when a box could
  be both a classification and a duplicate error.
- `evaluator.py` is the public use case: `evaluate(gt, predictions,
  iou_thresholds, max_dets=100)` returns per-class metrics, mAP@0.5 and
  mAP@0.5:0.95, plus a PR curve at IoU 0.5 always (even when the caller's
  thresholds omit 0.5). `by_prefix` groups image ids such as
  `Japan_000001` for per-country metrics.

`pycocotools` stays a dev-only dependency used by the parity test; no runtime
code imports it.

## Consequences

- The report layer consumes typed, immutable value objects and can serialise
  any field without parsing `COCOeval` internals.
- Evaluation is deterministic and testable on any OS with a base install; no
  C build is needed at runtime.
- The engine is ~400 lines plus tests; parity with COCO is proven only for the
  supported subset (bbox, no crowd/ignore flags, no area ranges, `maxDets`
  configurable but a single value). That subset is exactly what spec S7 needs;
  extending it requires extending the parity test first.
- Numeric differences from COCO are bounded by IoU rounding and the
  `np.spacing(1)` guard; the parity suite asserts `|delta| <= 1e-4` for mAP,
  mAP@0.5 and per-class AP@0.5.
- The 16-image mini fixture is created in T5. The T3 parity test builds a
  compact synthetic fixture with the same shape (3 boxes per class, 4
  negatives) and asserts AP parity plus an identity confusion matrix; T8
  evaluates the real fixture end to end.
