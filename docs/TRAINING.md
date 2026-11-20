# Training

This document describes the reproducible fine-tuning pipeline introduced in
T9. The real GPU run, its published metrics and the downloadable ONNX artifact
are produced by T10 on the owner's RTX 4060; everything documented here is the
recipe and the CPU-verified contract.

## Environment

The base install is torch-free (S2). Training needs the `train` extra plus one
of the mutually exclusive PyTorch extras:

```bash
uv sync --extra train --extra cpu     # CPU, tests and dry runs
uv sync --extra train --extra cuda    # CUDA (the real run)
```

`rdv train` and `rdv export` raise `MissingExtraError` (exit code 6) with this
exact hint when the stack is not installed.

## Data

Download and split RDD2022 first (see [DATA.md](DATA.md)):

```bash
uv run rdv data download --countries Japan India Czech
uv run rdv data prepare
```

The trainer reads `<RDV_DATA_DIR>/rdd2022/splits.json` and the country train
folders; validation uses the same root's `val` split. The data directory is
never derived from a training-config path.

## Configuration

`configs/train/rdd2022-rtdetrv2-r18.yaml` is the final D16 configuration. It is
fully typed (`rdv.adapters.training.config.TrainConfig`, Pydantic with
`extra="forbid"`) and supports `${ENV_VAR}` expansion in any string value.
Important fields:

| Field | Meaning |
|-------|---------|
| `model.hf_id`, `model.revision`, `model.weights_sha256` | Pinned base checkpoint; the `model.safetensors` digest is verified before the model is instantiated. A local directory is accepted for offline/test runs. |
| `train.epochs`, `train.effective_batch` | Run length and effective batch target (gradient accumulation keeps at least this many samples per step). |
| `train.lr`, `train.backbone_lr` | Head and backbone learning rates (D16 starting values). |
| `train.warmup_steps`, `train.lr_min_factor` | Linear warmup followed by a cosine decay. |
| `train.precision` | `bf16` autocast on CUDA; CPU always uses float32. |
| `train.ema` | `0.9999` decay with the `min(decay, (1+t)/(10+t))` warmup ramp. |
| `train.eval_every` | Epoch interval for val mAP@0.5:0.95 and `best/` checkpoints. |
| `train.max_images_per_epoch` | Budget-rule override applied after `--benchmark-only` (T10). |
| `augment.*` | torchvision v2 pipeline: stretch resize, hflip, photometric distort, affine, optional vflip/rot90/zoom-out, then `SanitizeBoundingBoxes(min_size=2)`. |
| `budget.vram_cap_gb`, `budget.max_hours` | Auto micro-batch cap and wall-clock stop rule. |

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

# Export the EMA-best checkpoint to the S4 single-file ONNX graph.
uv run rdv export --checkpoint runs/v1/best \
  --out artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx
```

## Training semantics

- **Model.** `RTDetrV2ForObjectDetection.from_pretrained` with four labels
  (`D00`, `D10`, `D20`, `D40`); an 80-class COCO head is re-initialised with
  `ignore_mismatched_sizes=True`.
- **Dataset.** `VocTorchDataset` wraps the VOC repository lazily, keeps images
  without boxes as negatives, and converts boxes to the Hugging Face contract:
  `class_labels` plus normalized `cxcywh` in the 640×640 stretched canvas.
- **Optimizer.** AdamW with separate backbone/head learning rates and no weight
  decay on biases or normalization parameters.
- **Accumulation.** The micro-batch comes from `[16, 12, 8, 6, 4]`; on CUDA the
  ladder is probed with 20 real steps capped at `budget.vram_cap_gb`. Gradient
  accumulation uses `ceil(effective_batch / micro_batch)`, so the effective
  batch is at least 16. On CPU the micro-batch is capped at 4 for smoke tests.
- **Checkpoints.** `last.pt` is written atomically at every epoch boundary and
  stores model, optimizer, EMA, DataLoader generator state and every RNG, so a
  resumed run reproduces the next step exactly (asserted to 1e-5). `best/`
  contains `save_pretrained` weights of the EMA whenever val mAP@0.5:0.95
  improves. Weights are never committed to git.
- **Logs.** `metrics.jsonl` (one object per optimizer step) and `epochs.csv`
  (one row per epoch). `benchmark.json` and `dry_run.json` follow the same
  as-run-directory layout.
- **Seeds.** Python, NumPy and torch are seeded from `seed` (42 by default);
  DataLoader workers receive `seed + worker_id`.

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

Run them with:

```bash
uv run pytest -m train -q
```

## Export contract

`rdv export` loads the checkpoint, wraps the model so the query axis is always
300 (the tiny test model pads from 20; the real model is already 300), and
tries `torch.onnx.export` at opset 18, opset 17, then the legacy TorchScript
exporter (the pinned `transformers<5` path). The graph is re-saved as a single
file, `onnx.checker` is run, and torch-vs-ONNX Runtime parity must stay below
`1e-3` on logits and boxes before the CLI reports success. The legacy exporter
does not infer the constant query/class dimensions, so the exporter writes the
true `[N,300,4]` shape into the output metadata after checking the computed
graph produces that shape.

## Benchmark and GPU inference

`rdv benchmark --model <onnx> --checkpoint runs/v1/best` adds the torch bf16
GPU row. `TorchDetector` loads a checkpoint directory, uses bf16 autocast on
CUDA and float32 on CPU, and shares the pure-numpy decode with the ONNX
detector. This adapter imports torch through `importlib`, because the
architecture contract allows a static torch import only inside
`rdv.adapters.training`; the file still lives in `rdv.adapters.inference` as
the T9 architecture specifies.

## Budget rule (applied in T10)

After `--benchmark-only` reports `img_per_s` and peak memory:

```
epoch_s = n_train / img_per_s
epochs  = 24                      if 24 * epoch_s <= 36000 s
        = max(12, floor(36000 / epoch_s)) otherwise
if 12 * epoch_s > 36000: max_images_per_epoch = max(8000, floor(3000 * img_per_s))
```

The chosen values are written back into the YAML config before the overnight
run. One bounded retry of 12 extra epochs at `lr × 0.3` is allowed when the
S10 bar is missed; the better run is published unchanged, including a
"below target" label when applicable.

## Limitations

- The real checkpoint download and the CUDA run are T10 work; T9 verifies the
  pipeline on CPU with a reduced model.
- No ego-motion compensation is used for video (see ADR 0011).
- Data augmentation is a starting recipe, not an ablation; D16 values are
  published as starting points, and only T10 measured numbers are quotable.
- The pinned base weights are Apache-2.0; RDD2022 derivatives follow the
  CC BY-SA 4.0 stance documented in [DATA.md](DATA.md).
