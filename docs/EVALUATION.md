# Evaluation

`rdv evaluate` runs a detector over a labeled split, computes COCO-style
object-detection metrics with the pure-domain engine, and writes one report
directory. It never uses the test split to tune anything.

## Command

```bash
rdv evaluate \
  --data data/rdd2022 \
  --splits data/rdd2022/splits.json \
  --split test \
  --model rdv-rtdetrv2-r18-rdd2022 \
  --out reports/v1
```

Options:

| Option | Meaning |
|--------|---------|
| `--data DIR` | Dataset root containing `<Country>/train/{images,annotations/xmls}`. |
| `--splits FILE` | Split manifest; defaults to `<data>/splits.json`. |
| `--split {val,test,all}` | Labeled split to evaluate (the CLI contract in S7). |
| `--model ID\|PATH` | Manifest model id or a local single-file ONNX model. |
| `--out DIR` | Output directory (default `reports/v1`). |
| `--op-threshold auto\|F` | `auto` selects the F1 optimum on `val`; a number is used as-is. |
| `--ood DIR` | Optional out-of-domain VOC directory (Norway sample). |
| `--max-images N` | Evaluate only the first N ids after sorting. |
| `--device auto\|cpu\|cuda` | Execution device. |

Detections are collected at a score cutoff of `0.001` so the metrics engine
sees the full ranked distribution. The **operating threshold** is separate: it
is used only for the confusion matrix, the error taxonomy and the gallery.

## Threshold policy

The operating threshold is the score that maximises global F1 at IoU 0.5,
computed by pooling every class and every image of the **validation** split
(decision D10). The test split is never read during selection; a repository
spy test enforces that the validation repository is fully consumed before the
evaluated split is opened. With `--op-threshold F` the validation split is not
opened at all. Reports record `operating_source` as `val` or `explicit`.

## Metric semantics

- mAP@0.5 and mAP@0.5:0.95 use COCO's 10 IoU thresholds and the 101-point
  interpolated precision envelope, max 100 detections per image/class.
- Per-class AP is reported for D00, D10, D20 and D40. A class with no
  ground-truth boxes in the evaluated split contributes `null` and is excluded
  from the mean, matching `pycocotools`.
- Per-country metrics group image ids by their leading token (the country name
  in RDD2022 ids), so they reuse the same engine unchanged.
- When `--ood` is given, the same metrics are computed on that independent set
  and stored under `ood`.
- Metrics are rounded to six decimals; the JSON is written with sorted keys so
  two identical runs are byte-identical.

## Error taxonomy (D10)

Matching is class-agnostic and greedy by descending score at IoU 0.5. Each
unmatched prediction with a score at or above the operating threshold becomes:

| Category | Rule (checked in this order) |
|----------|------------------------------|
| `classification` | IoU >= 0.5 with a ground-truth box of another class. |
| `localization` | `0.1 <= IoU < 0.5` with a same-class ground-truth box. |
| `duplicate` | IoU >= 0.5 with a same-class box already matched by a higher score. |
| `background` | Anything else. |

Unmatched ground-truth boxes are counted as `missed`. `error_analysis.json`
stores the counts and every case with its image id, class, score, box and the
matched ground-truth box when applicable.

## Artifacts

| File | Content |
|------|---------|
| `metrics.json` | `rdv.metrics/v1`: model identity, dataset counts, mAP/per-class, operating point, confusion matrix, taxonomy counts, per-country and OOD metrics. |
| `report.md` | Human-readable report with every table and links to the other artifacts. |
| `report.html` | Self-contained report: inline CSS, base64 plots and gallery thumbnails, Jinja2 autoescaping. |
| `pr_curves.png` | Precision-recall curves at IoU 0.5 for the four classes plus the mean. |
| `pr_curve_<CLS>.csv` | Raw `recall,precision,score` trace per class. |
| `confusion_matrix.png` / `.csv` | 5x5 matrix with background, at the operating threshold. |
| `error_analysis.json` | `rdv.error-analysis/v1` taxonomy plus every case. |
| `gallery/<category>/*.jpg` | Selected worst cases with ground-truth and prediction boxes. |
| `gallery/index.html` | Gallery index; links only to the images written in the same run. |

## Gallery selection

False positives are ranked by score, misses by ground-truth area. At most six
cases per category are kept and the final list is capped at 24. A round-robin
pass guarantees that every non-empty category is represented before any
category receives its sixth case, so the gallery never hides an entire error
type behind the cap. Selection is deterministic: ties break by image id,
class id and box coordinates.

## Determinism and limitations

- No timestamps or absolute paths are written to `metrics.json`.
- The metrics engine is verified against `pycocotools` in the `parity` test
  suite; see `docs/adr/0005-own-metrics-engine.md`.
- RDD2022 test annotations are not public, so the project evaluates on a seeded
  block split of the labeled train folders (decision D3). Residual near-dup
  leakage between blocks is possible and is stated in the dataset docs.
- Norway is out-of-domain only: it is never trained on, and its images come
  from different cameras and resolutions.
- These metrics are research numbers, not a safety-critical certification.
