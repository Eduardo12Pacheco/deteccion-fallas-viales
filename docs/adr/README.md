# Architecture decision records

Each significant decision is recorded as a short document with **Status**,
**Context**, **Decision** and **Consequences**. The records are append-only:
superseded decisions keep their file and say so in the status line.

| ADR | Decision | Status | Task | Plan |
|-----|----------|--------|------|------|
| [0001](0001-detector-choice.md) | RT-DETRv2 R18vd (Apache-2.0) as the detector; AGPL-3.0 Ultralytics and weights with unclear terms rejected | Accepted | T6, T16 | D1 |
| [0002](0002-dataset-subset-licensing.md) | Train on RDD2022 Japan/India/Czech; conservative CC BY-SA 4.0 stance for derivative images; own seeded block splits | Accepted | T5 | D2, D3, D4 |
| [0003](0003-hexagonal-architecture.md) | Hexagonal ports and adapters with `entrypoints > composition > adapters > application > domain`, enforced by import-linter and an AST test | Accepted | T4 | D5, D9 |
| [0004](0004-torch-free-base-onnx-runtime.md) | Torch-free base install, mutually exclusive `cpu`/`cuda` extras, ONNX Runtime CPU as the default engine | Accepted | T6, T9 | D7, D8 |
| [0005](0005-own-metrics-engine.md) | NumPy COCO mAP engine in the domain, verified against `pycocotools` within `1e-4` | Accepted | T3 | D10 |
| [0006](0006-local-web-ui.md) | FastAPI + vanilla ES modules, polling jobs, loopback-only security and the 16080 default port | Accepted | T11 | D11, D19 |
| [0007](0007-desktop-bundles.md) | PyInstaller onedir bundles with an embedded model, OpenCV excluded, 250 MiB size gate | Accepted | T13 | D12 |
| [0008](0008-model-distribution.md) | Pinned `models.json` manifest, resumable sha256-verified store and a `models-v1` release asset | Accepted | T6, T10 | D13, D22 |
| [0009](0009-ranged-nested-zip.md) | HTTP-range access to the nested Figshare ZIP with presigned-URL refresh and CRC32 pins | Accepted | T5 | D14 |
| [0010](0010-honest-metrics-policy.md) | README/model-card numbers generated from `reports/v1`, never hand-edited; below-target stays visible | Accepted | T10, T16 | D16, D17 |
| [0011](0011-video-semantics.md) | Analysis stride, greedy IoU tracker limits, approximate unique counts and atomic video outputs | Accepted | T7 | D20, D21 |

## Conventions

- Filename: `NNNN-kebab-case-title.md`, four-digit, monotonically increasing.
- Status is one of `Proposed`, `Accepted`, `Superseded by NNNN` or `Deprecated`.
- Every record names the plan decision(s) it materializes and the task that
  implemented it.
- The binding source is the implementation plan `new-projects/road-damage-vision.md`;
  an ADR explains **why**, the code and docs describe **what**.
