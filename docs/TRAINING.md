# Training

This document describes the reproducible fine-tuning pipeline and the measured
run published in `reports/v1`. The base install is torch-free; training needs
the `train` extra plus one of the mutually exclusive PyTorch extras.

## Environment

```bash
uv sync --extra train --extra cuda    # the real run (RTX 4060)
uv sync --extra train --extra cpu     # CPU tests, smoke runs and dry runs
```

`rdv train` and `rdv export` raise `MissingExtraError` (exit code 6) with the
same hint when the stack is not installed. Measured on the owner's machine:
torch `2.14.1+cu130`, torchvision `0.29.1+cu130`, transformers `4.57.6`,
CUDA runtime 13.0, NVIDIA driver 620.70.06, RTX 4060 8 GB.

## Data

Download and split RDD2022 first (see [DATA.md](DATA.md)):

```bash
export RDV_DATA_DIR=data          # keeps ~2 GB out of git and ~/.local
uv run rdv data download --countries Japan India Czech
uv run rdv data download --ood-norway 1000
uv run rdv data verify
uv run rdv data prepare
```

The trainer reads `<RDV_DATA_DIR>/rdd2022/splits.json` and the country train
folders; validation uses the same root's `val` split. The data directory is
never derived from a training-config path. Final split sizes: train 16,741
(Japan 8,406 / India 6,106 / Czech 2,229), val 2,200, test 2,100, Norway OOD
1,000.

## Configuration

`configs/train/rdd2022-rtdetrv2-r18.yaml` is the final D16 configuration. It is
fully typed (`rdv.adapters.training.config.TrainConfig`, Pydantic with
`extra="forbid"`) and supports `${ENV_VAR}` expansion in any string value.

| Field | Meaning |
|-------|---------|
| `model.hf_id`, `model.revision`, `model.weights_sha256` | Pinned base checkpoint. The loader downloads `config.json`, `preprocessor_config.json` and `model.safetensors` and verifies the weight digest before instantiating the model. A local directory is accepted for offline/test runs. |
| `train.epochs`, `train.effective_batch` | Run length and effective batch target (gradient accumulation keeps at least this many samples per step). |
| `train.lr`, `train.backbone_lr` | Head and backbone learning rates (D16). |
| `train.warmup_steps`, `train.lr_min_factor` | Linear warmup followed by a cosine decay. |
| `train.precision` | `bf16` autocast on CUDA; CPU always uses float32. |
| `train.ema` | `0.9999` decay with the `min(decay, (1+t)/(10+t))` warmup ramp. |
| `train.eval_every` | Epoch interval for val mAP@0.5:0.95 and `best/` checkpoints. |
| `train.max_images_per_epoch` | Budget-rule override (null for this run). |
| `augment.*` | torchvision v2 pipeline: stretch resize, hflip, photometric distort, affine, optional vflip/rot90/zoom-out, then `SanitizeBoundingBoxes(min_size=2)`. |
| `budget.vram_cap_gb`, `budget.max_hours` | Auto micro-batch cap (7.2 GB) and wall-clock stop rule (10 h). |

## Commands

```bash
# Validate config, dataset and model instantiation without training.
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml --dry-run

# Measure img/s, chosen micro-batch and peak VRAM (100 steps, D16 budget rule).
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml \
  --run-dir runs --benchmark-only

# Fine-tune, resuming automatically from runs/v1/last.pt when present.
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml \
  --run-dir runs/v1 --resume auto

# Export the published EMA checkpoint to the S4 single-file ONNX graph.
uv run rdv export --checkpoint runs/v1/selected \
  --out artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx

# Evaluate the test split and the out-of-domain Norway sample.
uv run rdv evaluate --data data/rdd2022 --splits data/rdd2022/splits.json \
  --split test --model artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx \
  --op-threshold auto --ood data/rdd2022-ood-norway --out reports/v1

# Real-model checks (never in CI).
RDV_REAL_MODEL=artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx \
RDV_REAL_CHECKPOINT=runs/v1/selected RDV_DATA_DIR=data \
  uv run pytest -m realmodel -q
```

`make reproduce` runs the Data steps and the `rdv evaluate` command above in that
order, with `RDV_DATA_DIR=data`. It does not train or export: the ONNX file under
`artifacts/` must already exist (written by `rdv export`). It downloads the
country archives (about 2 GB) and overwrites the tracked `reports/v1`, so review
the result with `git diff` before committing anything.

## Measured run (v1)

| Field | Value |
|---|---|
| Benchmark | micro-batch 12, 26.0115 img/s, peak 5.6607 GB (100 steps) |
| Chosen micro-batch | 12 (16 did not fit); accumulation 2; effective batch 24 |
| Epochs | 24 (24 x 643.6 s = 15,446 s <= 36,000 s, so no epoch reduction and no `max_images_per_epoch` cap) |
| Wall clock | 14,660 s = 4.07 h, including validation every 2 epochs |
| Peak allocated VRAM | 5.8182 GB (cap 7.2 GB) |
| Best raw-weight validation metric | mAP@0.5:0.95 = 0.1684 at epoch 22 |
| Published checkpoint | EMA at epoch 24, val mAP@0.5 = 0.3788 / mAP@0.5:0.95 = 0.1690 |
| Test result | mAP@0.5 = 0.3794, mAP@0.5:0.95 = 0.1696 |
| Artifact | 80,760,832 B, sha256 `806be021998a850eb074892b0025e7b6361b771fe773378191039af513b9a7de` |

The run record is `runs/v1/run.json` (committed unchanged as
[`reports/v1/run.json`](../reports/v1/run.json)); `runs/v1/epochs.csv` is
[`reports/v1/epochs.csv`](../reports/v1/epochs.csv). `git_sha` in the record is
the commit the run started from (`4b3866d`).

The validation metric is computed on raw weights during the run, so `best/`
tracks the best raw epoch while saving its EMA counterpart. After training,
both the saved `best/` EMA and the epoch-24 EMA were scored on the validation
split; the epoch-24 EMA was slightly better (0.3788/0.1690 vs
0.3735/0.1661 mAP@0.5 / mAP@0.5:0.95) and was exported as
`runs/v1/selected`. This comparison is recorded here instead of silently
choosing one; both numbers come from `rdv evaluate`'s metric engine.

## Training semantics

- **Model.** `RTDetrV2ForObjectDetection.from_pretrained` with four labels
  (`D00`, `D10`, `D20`, `D40`); the 80-class COCO classification head is
  re-initialised with `ignore_mismatched_sizes=True`, everything else is
  loaded from the pinned checkpoint.
- **Config is mandatory.** Only downloading `model.safetensors` without
  `config.json` would make `transformers` fall back to the default (larger)
  config and silently re-initialise parts of the backbone; the loader therefore
  requires the full pinned snapshot.
- **Dataset.** `VocTorchDataset` wraps the VOC repository lazily, keeps images
  without boxes as negatives, and converts boxes to the Hugging Face contract:
  `class_labels` plus normalized `cxcywh` in the 640x640 stretched canvas.
- **Optimizer.** AdamW with separate backbone/head learning rates and no weight
  decay on biases or normalization parameters.
- **Accumulation.** The micro-batch comes from `[16, 12, 8, 6, 4]`; on CUDA the
  ladder is probed with 20 real steps capped at `budget.vram_cap_gb`, using a
  dataset large enough for those 20 steps. Gradient accumulation uses
  `ceil(effective_batch / micro_batch)`, so the effective batch is at least 16
  (12 -> 24 here). On CPU the micro-batch is capped at 4 for smoke tests.
- **Checkpoints.** `last.pt` is written atomically at every epoch boundary and
  stores model, optimizer, EMA, DataLoader generator state and every RNG, so a
  resumed run reproduces the next step exactly (asserted to 1e-5). `best/`
  contains `save_pretrained` weights of the EMA whenever val mAP@0.5:0.95
  improves. Weights are never committed to git.
- **Logs.** `metrics.jsonl` (one object per optimizer step), `epochs.csv` (one
  row per epoch) and `run.json` (git sha, config hash, seed, versions, GPU,
  wall clock and peak memory, rewritten every epoch). `benchmark.json` and
  `dry_run.json` follow the same as-run-directory layout.
- **Seeds.** Python, NumPy and torch are seeded from `seed` (42 by default);
  DataLoader workers receive `seed + worker_id`.

## Report assembly

`rdv evaluate` writes `metrics.json`, `report.md`, `report.html`,
`pr_curves.png`, `pr_curve_<CLS>.csv`, `confusion_matrix.png/.csv`,
`error_analysis.json` and the gallery. The published `reports/v1` additionally
contains:

- `per_country.json` and `ood_norway.json`: the `per_country` and `ood`
  sections of `metrics.json`, lifted into standalone documents;
- `run.json` and `epochs.csv`: copied from `runs/v1/`;
- `benchmark.json`: the `rdv benchmark` output for the published ONNX file;
- `sample_expectations.json`: classes and maximum scores on the six packaged
  samples, used by the post-release smoke test.

To keep the committed report set below the 6 MB budget, the gallery JPEGs were
re-encoded to a 480 px long side at quality 80 and the raw PR CSV traces were
thinned to at most 4,000 evenly spaced points (both ends kept). `metrics.json`,
the plots and the error taxonomy are unchanged; the PNG curves still show the
101-point interpolation. No number used by the README or model card depends on
the thinned rows.

## CPU smoke tests

The `train`-marked tests instantiate a reduced RT-DETRv2 (R18 depths `[1,1,1,1]`,
widths 16/32/64/128, `d_model=32`, 20 queries) and verify, on CPU:

- finite loss and loss decrease on a four-image overfit;
- exact checkpoint-resume reproducibility of the next-step loss;
- all-negative batches do not crash;
- augmentation keeps boxes inside the canvas and drops degenerates;
- EMA closed form and the OOM auto-batch ladder;
- ONNX export passes `onnx.checker`, exposes `pixel_values [N,3,640,640]` and
  `logits`/`pred_boxes` `[N,300,4]`, and matches torch within `1e-3`.

Run them with `uv run pytest -m train -q`.

## Export contract

`rdv export` loads the checkpoint, wraps the model so the query axis is always
300 (the tiny test model pads from 20; the real model is already 300), and tries
`torch.onnx.export` at opset 18, opset 17, then the legacy TorchScript exporter.
The real model selected the legacy path at opset 17. The graph is re-saved as a
single file, `onnx.checker` is run, and torch-vs-ONNX Runtime parity must stay
below `1e-3` on logits and boxes before the CLI reports success (measured on
random input: logits `1.96e-5`, boxes `1.44e-5`). The `realmodel` test repeats
the comparison on eight validation images (matched-row maximum `3.2e-5`).

## Benchmark and GPU inference

`rdv benchmark --model <onnx> --checkpoint runs/v1/selected` adds the torch bf16
GPU row; the published numbers are in
[`reports/v1/benchmark.json`](../reports/v1/benchmark.json): CPU p50 156.3 ms
(p95 210.1 ms), GPU p50 14.1 ms (p95 16.1 ms), CPU video 6.09 analyzed
frames/s. Both S11 gates pass. The local `.onnx` path is matched to the
manifest by sha256, so the benchmark records the published `model.id` and the
recommended threshold. `TorchDetector`
loads a checkpoint directory, uses bf16 autocast on CUDA and float32 on CPU,
and shares the pure-numpy decode with the ONNX detector. The adapter imports
torch through `importlib`, because the architecture contract allows a static
torch import only inside `rdv.adapters.training`; the file still lives in
`rdv.adapters.inference` as the T9 architecture specifies.

## Budget rule

After `--benchmark-only` reports `img_per_s` and peak memory:

```
epoch_s = n_train / img_per_s                     # 16741 / 26.0115 = 643.6 s
epochs  = 24            if 24 * epoch_s <= 36000  # 15446 s -> 24
        = max(12, floor(36000 / epoch_s)) otherwise
if 12 * epoch_s > 36000: max_images_per_epoch = max(8000, floor(3000 * img_per_s))
```

Neither fallback applied: the committed YAML keeps `epochs: 24` and
`max_images_per_epoch: null`. The 10 h wall-clock guard in `budget.max_hours`
stopped nothing (the run took 4.07 h). One bounded retry of 12 extra epochs at
`lr x 0.3` is allowed when the S10 bar is missed; the v1 run met the bar, so no
retry was executed.

## Limitations

- The published metrics come from this project's own seeded split, not the
  official RDD2022 test set (annotations not released).
- Domain shift is real: Norway mAP@0.5 is 0.0588 and Indian `D10` is near zero;
  see `docs/MODEL_CARD.md`.
- Data augmentation is a starting recipe, not an ablation; D16 values are
  published as starting points.
- No ego-motion compensation is used for video (see ADR 0011).
- The pinned base weights are Apache-2.0; RDD2022 derivatives follow the
  CC BY-SA 4.0 stance documented in [DATA.md](DATA.md).
