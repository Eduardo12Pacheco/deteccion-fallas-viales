"""CPU training smoke: finite loss, overfit decrease and empty batches (S9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from rdv.adapters.datasets.rdd2022 import Rdd2022VocRepository
from rdv.adapters.training.config import TrainConfig
from rdv.adapters.training.trainer import RtDetrTrainer
from rdv.application.train_model import TrainingRequest
from rdv.domain.annotation import AnnotatedImage
from tests.support.fakes import InMemoryDatasetRepository
from tests.support.synthetic import write_synthetic_dataset
from tests.support.tiny_model import (
    save_tiny_checkpoint,
    tiny_train_payload,
)

pytestmark = pytest.mark.train


def _setup(tmp_path: Path):  # type: ignore[no-untyped-def]
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
    payload = tiny_train_payload(checkpoint, digest, epochs=15, effective_batch=4)
    config = TrainConfig.from_mapping(payload)
    repository = Rdd2022VocRepository(data_root, ("Japan",), split="all")
    trainer = RtDetrTrainer(
        config,
        repository,
        device="cpu",
        micro_batch=4,
        num_workers=0,
    )
    return trainer


def test_loss_is_finite_and_decreases_on_a_four_image_overfit(tmp_path: Path) -> None:
    trainer = _setup(tmp_path)

    result = trainer.train(TrainingRequest(run_dir=tmp_path / "run", max_steps=15))

    assert result.steps == 15
    assert result.epochs_completed == 15
    assert len(result.losses) == 15
    assert all(np.isfinite(value) for value in result.losses)
    assert float(np.mean(result.losses[-3:])) < float(np.mean(result.losses[:3]))
    assert result.last_loss is not None and np.isfinite(result.last_loss)


def test_batch_with_only_negative_images_does_not_crash(tmp_path: Path) -> None:
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    payload = tiny_train_payload(checkpoint, digest, epochs=2, effective_batch=2)
    config = TrainConfig.from_mapping(payload)
    images = [
        AnnotatedImage(image_id=f"Japan/train/images/{index}.jpg", width=64, height=64)
        for index in range(2)
    ]
    pixels = {image.image_id: np.zeros((64, 64, 3), dtype=np.uint8) for image in images}
    repository = InMemoryDatasetRepository(images, pixels)
    trainer = RtDetrTrainer(config, repository, device="cpu", micro_batch=2, num_workers=0)

    result = trainer.train(TrainingRequest(run_dir=tmp_path / "run", max_steps=2))

    assert result.steps == 2
    assert result.losses
    assert all(np.isfinite(value) for value in result.losses)


def test_dry_run_validates_without_writing_a_checkpoint(tmp_path: Path) -> None:
    trainer = _setup(tmp_path)
    run_dir = tmp_path / "dry"

    result = trainer.train(TrainingRequest(run_dir=run_dir, dry_run=True))

    assert result.dry_run is True
    assert result.steps == 0
    assert (run_dir / "dry_run.json").is_file()
    assert not (run_dir / "last.pt").exists()


def test_benchmark_only_writes_throughput_and_memory(tmp_path: Path) -> None:
    trainer = _setup(tmp_path)
    trainer._benchmark_steps = 2

    result = trainer.train(TrainingRequest(run_dir=tmp_path / "bench", benchmark_only=True))

    assert result.benchmark is not None
    assert result.benchmark["schema"] == "rdv.train.benchmark/v1"
    assert result.benchmark["steps"] == 2
    assert result.benchmark["img_per_s"] > 0.0
    assert (tmp_path / "bench" / "benchmark.json").is_file()


def test_validation_saves_the_best_ema_checkpoint(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    data_root = tmp_path / "data" / "rdd2022"
    write_synthetic_dataset(
        data_root,
        countries=("Japan",),
        images_per_country=10,
        width=96,
        height=64,
        block_size=2,
        ratios=(0.6, 0.2, 0.2),
    )
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    config = TrainConfig.from_mapping(
        tiny_train_payload(
            checkpoint,
            digest,
            epochs=1,
            effective_batch=2,
            eval_every=1,
        )
    )
    train_repository = Rdd2022VocRepository(data_root, ("Japan",), split="train")
    val_repository = Rdd2022VocRepository(data_root, ("Japan",), split="val")
    trainer = RtDetrTrainer(
        config,
        train_repository,
        val_repository=val_repository,
        device="cpu",
        micro_batch=2,
        num_workers=0,
    )
    monkeypatch.setattr(trainer, "_validate", lambda model, dataset, device: 0.5)

    result = trainer.train(TrainingRequest(run_dir=tmp_path / "run"))

    assert result.best_metric == pytest.approx(0.5)
    assert (tmp_path / "run" / "best" / "model.safetensors").is_file()
    metrics = (tmp_path / "run" / "best" / "metrics.json").read_text(encoding="utf-8")
    assert '"mAP50_95": 0.5' in metrics


def test_validate_returns_a_bounded_map(tmp_path: Path) -> None:
    from rdv.adapters.training.trainer import load_training_model

    data_root = tmp_path / "data" / "rdd2022"
    write_synthetic_dataset(
        data_root,
        countries=("Japan",),
        images_per_country=10,
        width=96,
        height=64,
        block_size=2,
        ratios=(0.6, 0.2, 0.2),
    )
    checkpoint, digest = save_tiny_checkpoint(tmp_path / "tiny")
    config = TrainConfig.from_mapping(tiny_train_payload(checkpoint, digest, epochs=1))
    val_repository = Rdd2022VocRepository(data_root, ("Japan",), split="val")
    trainer = RtDetrTrainer(config, val_repository, device="cpu", num_workers=0)
    model = load_training_model(config, device="cpu")
    dataset = trainer._make_dataset(val_repository, training=False)
    assert dataset is not None

    score = trainer._validate(model, dataset, "cpu")

    assert 0.0 <= score <= 1.0
