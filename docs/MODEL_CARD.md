# Model card — `rdv-rtdetrv2-r18-rdd2022` v1

- **Status:** published artifact, first real fine-tune (T10)
- **Artifact:** `rdv-rtdetrv2-r18-rdd2022-v1.onnx`, single-file ONNX, opset 17,
  sha256 `806be021998a850eb074892b0025e7b6361b771fe773378191039af513b9a7de`,
  80,760,832 bytes
- **Metrics reference:** [`reports/v1/metrics.json`](../reports/v1/metrics.json)
- **Code license:** MIT (this repository). **Base weights:** Apache-2.0.
  **Dataset derivatives:** CC BY-SA 4.0 stance (see
  [DATA.md](DATA.md)); the license status of the fine-tuned weights themselves
  is discussed under [Ethical and legal notes](#ethical-and-legal-notes).

## Model details

| Field | Value |
|---|---|
| Architecture | RT-DETRv2 R18vd (`RTDetrV2ForObjectDetection`, Hugging Face Transformers 4.57.6) |
| Base checkpoint | `PekingU/rtdetr_v2_r18vd`, revision `5650961749fa93567c0d46fc7f43ea4f9e914107`, `model.safetensors` sha256 `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115` |
| Base license | Apache-2.0 (Hugging Face model card and API metadata) |
| Input | RGB, stretched to 640x640, float32 `pixel_values [N,3,640,640]`, range 0-1, no mean/std normalization |
| Outputs | `logits [N,300,4]` (pre-sigmoid) and `pred_boxes [N,300,4]` (cxcywh normalized to the stretched input) |
| Classes | `D00` longitudinal crack, `D10` transverse crack, `D20` alligator crack, `D40` pothole (ids 0..3) |
| Parameters | about 20 M (R18 backbone + RT-DETR encoder/decoder) |
| Runtime | ONNX Runtime CPU by default; optional PyTorch/CUDA inference inside the training environment |

The ONNX graph was exported from the fine-tuned EMA checkpoint with
`torch.onnx.export` (legacy TorchScript path, opset 17) and re-saved as a single
file. A torch-vs-ONNX Runtime parity check on eight validation images measured a
maximum matched row difference of `3.2e-5` for logits and boxes, below the
`1e-3` contract. On one of those eight images the exact-score TopK tie ordering
differs between eager and traced execution; the outputs are a set of 300
queries, and the bijective row matching confirms the values agree.

## Intended use

- Local, offline demonstration of road-damage detection in still images and
  video, including the `rdv serve` browser UI and the portable desktop bundle.
- Research/portfolio use, reproducible evaluation, and a starting point for a
  domain-specific detector.
- Interactive exploration: a confidence slider and per-class toggles let a user
  trade precision for recall without re-running inference.

## Out of scope

- Safety-critical, legal, insurance or maintenance-priority decisions. The
  model is not validated for those uses and must not drive them.
- Pavement condition indices (PCI/IRI), severity scoring, segmentation,
  depth/3D reconstruction, GPS mapping or lane-level localisation.
- Production/vehicle deployment, real-time guarantees, or claims of
  state-of-the-art accuracy.
- Imagery from the excluded RDD2022 subsets (United States Google Street View,
  China Drone/MotorBike) and non-road-damage photography.

## Training data

RDD2022 (CRDDC'2022), Figshare article `21431547`, DOI
`10.6084/m9.figshare.21431547.v1`. Only the labeled `train/` folders of Japan,
India and Czech were downloaded (about 1.9 GB via ranged nested-ZIP reads) and
split with the seeded block rule from D3:

| Split | Images | Japan | India | Czech |
|---|---:|---:|---:|---:|
| train | 16,741 | 8,406 | 6,106 | 2,229 |
| val | 2,200 | 1,100 | 800 | 300 |
| test | 2,100 | 1,000 | 800 | 300 |
| OOD (Norway, never trained) | 1,000 | — | — | — |

- Seed 42, block size 100, ratios 0.8/0.1/0.1 per country (total 21,041).
- Images with no objects are kept as negatives; labels outside
  `{D00,D10,D20,D40}` are dropped and counted (`docs/DATA.md`).
- The official RDD2022 test annotations are not public; the test split above is
  this project's own held-out split, not the CRDDC challenge test set.
- The Norway sample is an out-of-domain set only: 3,650x2,044 imagery stretched
  to 640x640, never used for training or threshold selection.
- United States data is excluded (Google Street View terms); China
  Drone/MotorBike are excluded (different sensors/viewpoints).

The committed mini fixture (`tests/fixtures/rdd2022-mini`, 16 images) and the
six packaged samples are redistributed under the CC BY-SA 4.0 stance documented
in [DATA.md](DATA.md).

## Training procedure

| Setting | Value |
|---|---|
| Epochs | 24 (D16 budget rule; measured `26.01 img/s` in `--benchmark-only`) |
| Effective batch | 24 = micro-batch 12 x gradient accumulation 2 (`ceil(16/12)`) |
| Optimizer | AdamW, head lr `1e-4`, backbone lr `1e-5`, weight decay `1e-4` (no decay on norm/bias), grad-clip `0.1` |
| Schedule | 500-step linear warmup + cosine decay to 1% |
| Precision | bf16 autocast on CUDA |
| EMA | `0.9999`, warmup `min(0.9999, (1+t)/(10+t))` |
| Augmentation | stretch resize 640x640, hflip 0.5, photometric distort 0.5, affine p=0.5 (translate 0.1, scale 0.8-1.2), `SanitizeBoundingBoxes(min_size=2)` |
| Hardware | NVIDIA GeForce RTX 4060 8 GB, driver 620.70.06, CUDA 13.0 runtime, 12-thread CPU |
| Peak allocated VRAM | 5.82 GB (auto micro-batch probe cap 7.2 GB; batch 16 did not fit) |
| Wall clock | 14,660 s (4.07 h) for 24 epochs, including validation |
| Seeds | 42 (Python, NumPy, torch, DataLoader workers) |

The run record (git sha, config hash, versions, wall clock, peak memory) is
committed as [`reports/v1/run.json`](../reports/v1/run.json); the per-epoch log
is [`reports/v1/epochs.csv`](../reports/v1/epochs.csv). The final artifact is the
EMA of the last epoch, which scored slightly better on validation than the
best-by-raw-weights checkpoint (mAP@0.5:0.95 0.1690 vs 0.1661); both were
measured on the validation split and the better EMA was published.

## Evaluation

Held-out in-domain test split (Japan + India + Czech, 2,100 images, 2,469
ground-truth boxes). COCO-style metrics from the project's numpy engine, 101
recall points, maxDets 100. The confusion matrix and error taxonomy use the
operating threshold selected on the validation split, never on test.

| Metric | Value |
|---|---:|
| mAP@0.5 | **0.3794** |
| mAP@0.5:0.95 | **0.1696** |
| AP@0.5 D00 longitudinal crack | 0.3383 |
| AP@0.5 D10 transverse crack | 0.3089 |
| AP@0.5 D20 alligator crack | 0.4955 |
| AP@0.5 D40 pothole | 0.3749 |
| F1-optimal operating threshold (val) | 0.0216 |
| Precision / recall at that threshold | 0.425 / 0.448 |

S10 minimum bar (mAP@0.5 >= 0.35, mAP@0.5:0.95 >= 0.15, every class AP@0.5
>= 0.10): **met**; `scripts/render_readme_metrics.py --status` prints
`PASS (>= min bar)`.

### Confidence calibration

The v1 head is under-confident: the score distribution tops out around 0.06 on
the packaged samples and the F1-optimal validation threshold is 0.0216. A
"0.25 by convention" default would therefore show no detections at all on the
demo samples. The manifest's `recommended_score_threshold` is **0.01**, the
lowest point at which all six packaged samples still produce at least one
detection; users can move the UI slider up for fewer, more confident boxes.
No packaged sample reached the T10 margin of 0.15 above the threshold, and
`reports/v1/sample_expectations.json` records that fact explicitly rather than
dropping every sample. This is a calibration limitation, not a metrics edit:
the reported mAP values are ranking-based and unchanged.

### Per-country test results

| Country | Images | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|
| Japan | 1,000 | 0.4261 | 0.1994 |
| India | 800 | 0.2666 | 0.1189 |
| Czech | 300 | 0.2780 | 0.1014 |

India's `D10` AP@0.5 collapses to `0.0002` (Japan carries the global 0.3089),
and the Czech subset is small; the global number hides this domain gap. The
full per-class breakdown is in
[`reports/v1/per_country.json`](../reports/v1/per_country.json).

### Out-of-domain (Norway)

| Metric | Value |
|---|---:|
| Images / boxes | 1,000 / 1,193 |
| mAP@0.5 | 0.0588 |
| mAP@0.5:0.95 | 0.0229 |

The large drop is expected: Norway uses a different camera, resolution and
road appearance, and was never trained on. The model is not presented as
geographically general.

### Error profile at the operating threshold

| Category | Count |
|---|---:|
| Missed (FN) | 1,441 |
| Localization (FP) | 378 |
| Background (FP) | 760 |
| Duplicate (FP) | 121 |
| Classification (FP) | 62 |

Twenty-four worst cases (six per category, capped at 24) are committed under
[`reports/v1/gallery/`](../reports/v1/gallery/) with the taxonomy in
[`reports/v1/error_analysis.json`](../reports/v1/error_analysis.json).

## Inference performance

Measured by `rdv benchmark` on the artifact (batch 1, 640x640 input, 20 warmup +
100 timed runs): CPU p50 **156.3 ms** (p95 210.1 ms, 12th Gen Intel i5-12400T,
ONNX Runtime) and GPU p50 **14.1 ms** (p95 16.1 ms, RTX 4060, torch bf16).
Synthetic 320x240 video throughput on CPU: **6.09 analyzed frames/s**. Both S11
sanity gates pass (CPU <= 1500 ms, GPU <= 100 ms). Full hardware and run
configuration: [`reports/v1/benchmark.json`](../reports/v1/benchmark.json).

## Limitations

- **Not safety-critical.** Accuracy is far from a production inspection tool;
  missed detections are the dominant error.
- **Domain shift.** Norway mAP@0.5 is 0.0588; Indian `D10` is near zero on the
  test split. Do not extrapolate to unseen countries, cameras or resolutions.
- **Own split, not the official challenge test set.** The official RDD2022 test
  annotations were never released.
- **Block-split caveat.** Blocks of consecutive ids reduce but do not eliminate
  near-duplicate frame leakage between splits.
- **Crack-type confusion.** Longitudinal and transverse cracks can look alike
  at 640x640, and the model was not trained on the dropped RDD2022 labels
  (D43/D44/D50/D01/D11/D0w0).
- **Video counts are approximate.** Unique instances come from a greedy IoU
  tracker without ego-motion compensation (D20/ADR 0011).
- **Coverage ratio is descriptive**, not a severity or PCI score (D21).
- **Small-model trade-off.** RT-DETRv2 R18 was chosen for an 8 GB GPU and
  permissive licenses, not for maximum accuracy.
- **Under-confident scores.** Absolute scores are low (sample maxima <= 0.062),
  so the recommended threshold is 0.01 rather than a conventional 0.25. Use
  ranking/relative confidence or recalibrate before any downstream rule that
  assumes calibrated probabilities.

## Ethical and legal notes

- Road imagery can contain faces, licence plates and location cues. The
  packaged samples and fixture images are real road photos; they are included
  for demonstration only (CC BY-SA 4.0 stance, attribution in
  [DATA.md](DATA.md)).
- Predictions must not be used to deny claims, rank repairs or make budget
  decisions without human inspection.
- The dataset record says CC BY 4.0 while the paper and some mirrors say
  CC BY-SA 4.0; this project follows the conservative reading (attribute the
  authors, redistribute derivative images under CC BY-SA 4.0) and documents the
  discrepancy instead of resolving it silently.
- The legal status of generative/model weights derived from CC-licensed images
  is unsettled. The base weights are Apache-2.0; the fine-tuned weights are
  distributed for portfolio/demo use with this caveat, not with a claim of
  clean title.

## Citation

If you use the data or the base architecture, cite:

- Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y. (2022).
  RDD2022: A multi-national image dataset for automatic road damage detection.
  arXiv:2209.08538.
- Lyu, W. et al. (2024). RT-DETRv2: Improved Real-time Detection Transformer.
  arXiv:2407.17140.
