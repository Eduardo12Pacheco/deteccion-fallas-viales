"""Deterministic tiny RT-DETRv2 checkpoint for CPU training/export tests.

The real checkpoint is 80 MB and needs the network; T9 only needs the same
forward/backward/export contract at a size a CI runner can train for a few
steps. The architecture (backbone stages, encoder/decoder, 20 queries, four
labels) is the pinned one, only the widths and depths are reduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection
from transformers.models.rt_detr.configuration_rt_detr import RTDetrResNetConfig

from rdv.hashing import sha256_file


def tiny_model_config(*, num_queries: int = 20, num_labels: int = 4) -> RTDetrV2Config:
    """Build the minimal RT-DETRv2 R18 configuration used by the tests."""
    backbone = RTDetrResNetConfig(
        depths=[1, 1, 1, 1],
        hidden_sizes=[16, 32, 64, 128],
        out_features=["stage2", "stage3", "stage4"],
    )
    return RTDetrV2Config(
        backbone_config=backbone,
        d_model=32,
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        decoder_in_channels=[32, 32, 32],
        decoder_layers=1,
        encoder_layers=1,
        decoder_attention_heads=2,
        encoder_attention_heads=2,
        num_queries=num_queries,
        num_labels=num_labels,
        num_denoising=10,
        num_feature_levels=3,
        anchor_image_size=None,
        use_timm_backbone=False,
        use_pretrained_backbone=False,
    )


def build_tiny_model(*, num_queries: int = 20, seed: int = 0) -> RTDetrV2ForObjectDetection:
    """Instantiate the tiny model with deterministic weights."""
    torch.manual_seed(seed)
    return RTDetrV2ForObjectDetection(tiny_model_config(num_queries=num_queries))


def save_tiny_checkpoint(
    directory: Path,
    *,
    num_queries: int = 20,
    seed: int = 0,
) -> tuple[Path, str]:
    """Save a tiny checkpoint and return its directory and weight digest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = build_tiny_model(num_queries=num_queries, seed=seed)
    model.save_pretrained(directory)
    weights = directory / "model.safetensors"
    return directory, sha256_file(weights)


def tiny_train_payload(
    checkpoint: Path,
    weights_sha256: str,
    *,
    epochs: int = 2,
    effective_batch: int = 4,
    eval_every: int = 1000,
    backbone_lr: float = 1.0e-4,
    lr: float = 1.0e-3,
    augmentation: bool = False,
    num_queries: int = 20,
) -> dict[str, Any]:
    """A complete ``TrainConfig`` mapping pointing at a tiny checkpoint."""
    return {
        "seed": 42,
        "model": {
            "hf_id": str(checkpoint),
            "revision": "local",
            "weights_sha256": weights_sha256,
            "input_size": 640,
            "num_labels": 4,
        },
        "data": {
            "countries": ["Japan"],
            "splits": {"seed": 42, "block_size": 100, "ratios": [0.8, 0.1, 0.1]},
            "ood": {},
        },
        "train": {
            "epochs": epochs,
            "effective_batch": effective_batch,
            "lr": lr,
            "backbone_lr": backbone_lr,
            "weight_decay": 1.0e-4,
            "warmup_steps": 2,
            "lr_min_factor": 0.01,
            "grad_clip": 1.0,
            "precision": "fp32",
            "ema": {"decay": 0.9999, "warmup": True},
            "eval_every": eval_every,
            "num_workers": 0,
            "max_images_per_epoch": None,
        },
        "augment": {
            "resize": [640, 640],
            "hflip": 0.5 if augmentation else 0.0,
            "photometric_distort": 0.5 if augmentation else 0.0,
            "affine": {
                "p": 0.5 if augmentation else 0.0,
                "translate": 0.1,
                "scale": [0.8, 1.2],
            },
            "vflip": False,
            "rot90": False,
            "zoom_out": False,
        },
        "budget": {"max_hours": 10, "min_epochs": 1, "vram_cap_gb": 7.2},
    }
