# 0001 - Detector choice: RT-DETRv2 R18vd and the AGPL avoidance

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decision D1; specs S3, S4, S9 and S10)
- **Task:** T6 (inference and export), T16 (record)

## Context

The project needs a pre-trained road-damage detector that can be fine-tuned on
an 8 GB consumer GPU, exported to a single ONNX file and redistributed as part
of an MIT-licensed portfolio repository. The base weights must carry an
explicit permissive license, because the repository publishes the fine-tuned
weights as a release asset and distributes them inside desktop bundles.

The obvious choice, Ultralytics YOLO (YOLOv5/YOLOv8/YOLO11 and its RT-DETR
wrapper), is **AGPL-3.0** according to both the GitHub and PyPI metadata. A
network service or a redistributed derivative under AGPL-3.0 would force the
whole project to adopt that license, contradicting the MIT requirement in S3.
That option was therefore rejected before writing the inference path.

A detector also has to satisfy the deployment contract in S4: a single ONNX
graph at 640x640 with `logits [N,300,4]` and `pred_boxes [N,300,4]`, no NMS,
and an export path maintained by a widely used library.

## Decision

Use **RT-DETRv2 R18vd through Hugging Face Transformers**:

- checkpoint `PekingU/rtdetr_v2_r18vd`, revision
  `5650961749fa93567c0d46fc7f43ea4f9e914107`;
- `model.safetensors` 80,904,640 bytes, sha256
  `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`,
  pinned in `configs/train/rdd2022-rtdetrv2-r18.yaml` and verified before
  training;
- Apache-2.0 for code and weights (Hugging Face API `cardData.license` and the
  model card front matter), COCO-pretrained;
- about 20 M parameters, end-to-end (no NMS), fixed 640x640 input, exported by
  T9/T10 with the D8 fallback ladder (opset 18, opset 17, legacy TorchScript);
- ONNX Runtime CPU as the default inference engine; PyTorch/CUDA bf16 only
  inside the training environment (ADR 0004).

If the v2 export had failed after the D8 attempts, the configured fallback was
the v1 family `PekingU/rtdetr_r18vd` (also Apache-2.0), selected only through
`model.hf_id`; the export succeeded, so no switch was needed.

### Rejected alternatives

| Option | License evidence | Why it was rejected |
|---|---|---|
| Ultralytics YOLOv5/v8/YOLO11 and its RT-DETR wrapper | AGPL-3.0 (GitHub API and PyPI metadata) | Copyleft incompatible with the MIT code and redistributed weights |
| `torchvision` detection models | BSD-3-Clause code, no license statement for the pre-trained weights (README asks users to determine permission) | Weights license ambiguity makes redistribution unsafe |
| YOLOX | Apache-2.0, but the last PyPI release is 0.3.0 from 2022-04 | Stale packaging and unmaintained export tooling |
| RF-DETR | Apache-2.0 repository, heavier dependency tree, per-checkpoint licenses not verified | Extra dependencies and unverified weight terms |
| Segmentation or pavement-index models | — | Out of scope: the project detects boxes, not masks or severity |

## Consequences

- The fine-tune stays within the 8 GB VRAM budget: measured peak 5.8182 GB on
  the RTX 4060 with micro-batch 12, 24 epochs in 4.07 h (see
  [MODEL_CARD.md](../MODEL_CARD.md)).
- The base install is torch-free; inference depends only on ONNX Runtime and
  numpy, and the desktop bundle stays under the 250 MiB gate (ADR 0007).
- The COCO classification head is re-initialized for the four RDD2022 classes
  (`ignore_mismatched_sizes=True`); the exported graph has a 300-query fixed
  output, which the pure-numpy decoder in `rdv.domain.decoding` consumes.
- Accuracy is honest but modest (test mAP@0.5 0.3794, mAP@0.5:0.95 0.1696), and
  the v1 head is under-confident (recommended threshold 0.01). The model card
  and README publish those numbers unchanged.
- License hygiene is enforced by `tests/unit/test_license_guard.py` (no
  AGPL/GPL runtime dependency) and by a source scan that only allows the
  Ultralytics name in the ADRs and `docs/DATA.md`.
- The Ultralytics name remains in this decision record and in the data
  provenance table, exactly as S3 permits: the rejected option is documented
  instead of hidden.

## References

- Hugging Face model card and API: <https://huggingface.co/PekingU/rtdetr_v2_r18vd>
- RT-DETRv2 paper: Lyu, W. et al. (2024), arXiv:2407.17140
- Ultralytics license: <https://github.com/ultralytics/ultralytics>
- Torchvision pre-trained model license note: <https://github.com/pytorch/vision>
- Dataset and training decisions: [DATA.md](../DATA.md), [TRAINING.md](../TRAINING.md)
