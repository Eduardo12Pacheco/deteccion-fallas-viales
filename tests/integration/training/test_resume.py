"""Checkpoint resume reproduces the next optimizer step (S9, D16)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from rdv.adapters.datasets.rdd2022 import Rdd2022VocRepository
from rdv.adapters.training.config import TrainConfig
from rdv.adapters.training.trainer import RtDetrTrainer
from rdv.application.train_model import TrainingRequest
from tests.support.synthetic import write_synthetic_dataset
from tests.support.tiny_model import (
    save_tiny_checkpoint,
    tiny_train_payload,
)

pytestmark = pytest.mark.train


def test_resume_reproduces_the_next_step_loss(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "rdd2022"
    write_synthetic_dataset(
        data_root,
        countries=("Japan",),
        images_per_country=4,
        width=96,
        height=64,
        block_size=4,
        ratios=(1.0, 0.0, 0.0),
    )
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    config = TrainConfig.from_mapping(
        tiny_train_payload(checkpoint, digest, epochs=6, effective_batch=4)
    )
    repository = Rdd2022VocRepository(data_root, ("Japan",), split="all")

    def trainer() -> RtDetrTrainer:
        return RtDetrTrainer(
            config,
            repository,
            device="cpu",
            micro_batch=4,
            num_workers=0,
        )

    baseline = trainer().train(TrainingRequest(run_dir=tmp_path / "baseline", max_steps=4))
    partial = trainer().train(TrainingRequest(run_dir=tmp_path / "resumed", max_steps=3))
    resumed = trainer().train(TrainingRequest(run_dir=tmp_path / "resumed", max_steps=4))

    assert partial.steps == 3
    assert partial.epochs_completed == 3
    assert resumed.steps == 1
    assert resumed.epochs_completed == 4
    assert baseline.steps == 4
    assert resumed.losses == pytest.approx([baseline.losses[3]], abs=1e-5)
    last = tmp_path / "resumed" / "last.pt"
    assert last.is_file()
