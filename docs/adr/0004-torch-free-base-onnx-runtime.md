# 0004 - Torch-free base install and ONNX Runtime as the default engine

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D7, D8, D9, S2, S4)
- **Task:** T6

## Context

The repository must install and infer on Linux, Windows and macOS with a
single `uv tool install` (spec S1) while the training stack (PyTorch,
torchvision, Transformers, SciPy) is large, platform-specific and only needed
to reproduce the fine-tune. Shipping those packages in the base install would
inflate the wheel environment to several GB, make the desktop bundle
impossible to keep under the size limit and pull CUDA wheels into machines
that only need CPU inference. Spec S2 states the hard requirement: after
importing the CLI, `torch`, `torchvision`, `transformers` and `scipy` must not
be loaded; importing the inference adapter must not load `matplotlib`.

The detector must still be deployable with a well-defined tensor contract
(spec S4) so the same file can be used by the CLI, the web app, the desktop
bundle and the benchmark.

## Decision

Keep the base dependency set torch-free and expose the training stack as
opt-in extras:

- `train` adds `transformers`, `scipy`, `pyyaml` and `huggingface_hub`.
- `cpu`/`cuda` add `torch` and `torchvision` from the explicit PyTorch
  indexes; the two extras are mutually exclusive (`[tool.uv] conflicts`).
- Base install contains ONNX Runtime, NumPy, Pillow, OpenCV-headless, FastAPI
  and the reporting stack.
- Inference defaults to `OnnxDetector`: one single-file ONNX graph whose input
  is `pixel_values` `float32 [N,3,640,640]` in `[0,1]` and whose outputs are
  `logits [N,300,4]` and `pred_boxes [N,300,4]` (normalized `cx,cy,w,h`).
  Preprocessing is one function (`rdv.adapters.inference.preprocess`) and
  decoding stays in the pure domain (`rdv.domain.decoding`).
- Device selection is a Strategy (`rdv.adapters.inference.providers`):
  `RDV_DEVICE=auto` picks CUDA only when ONNX Runtime exposes
  `CUDAExecutionProvider`, otherwise CPU; an explicit `cuda` request without
  the provider raises `ModelError` (exit 4) instead of silently running on
  CPU. `onnxruntime-gpu` is documented as optional because it conflicts with
  the base `onnxruntime` wheel.
- The import-linter contract forbids `torch`, `torchvision`, `transformers`
  and `scipy` everywhere except `rdv.adapters.training`, so the boundary is
  enforced by a test and not by convention.

## Consequences

- `uv tool install` produces a usable CLI on a clean machine without a GPU,
  CUDA toolkit or multi-GB downloads.
- The CLI and inference adapter can be imported in tests and in the desktop
  bundle without dragging in the training stack; spec S2 is asserted by an
  integration test that imports the CLI and checks `sys.modules`.
- The torch backend (`TorchDetector`, T9) is an alternative adapter behind the
  same `Detector` port; the rest of the application does not change.
- ONNX Runtime GPU support is opt-in and not exercised by CI. The benchmark
  reports CPU numbers by default and GPU numbers only when the torch CUDA
  extra is installed.
