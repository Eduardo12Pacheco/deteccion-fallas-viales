"""RT-DETRv2 fine-tuning adapter (decision D16, spec S9).

The trainer owns the PyTorch/Hugging Face details: pinned base-weight
verification, parameter groups, warmup + cosine schedule, bf16 autocast on
CUDA, gradient accumulation, EMA, resumable checkpoints and the step/epoch
logs. It is deliberately resumable at epoch boundaries: ``last.pt`` stores the
model, optimizer, EMA, DataLoader generator and every RNG that affects the
next epoch, so an interrupted overnight run continues deterministically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from rdv.adapters.training.autobatch import GIB, AutoBatchResult, choose_micro_batch
from rdv.adapters.training.config import TrainConfig, resolve_local_checkpoint
from rdv.adapters.training.dataset import TorchSample, VocTorchDataset, collate_samples, seed_worker
from rdv.adapters.training.ema import ModelEMA
from rdv.application.events import ProgressEvent
from rdv.application.ports import DatasetRepository, ProgressSink
from rdv.application.train_model import TrainingRequest, TrainingResult
from rdv.domain.classes import DamageClass
from rdv.domain.decoding import decode_detr_outputs
from rdv.domain.metrics.evaluator import evaluate
from rdv.errors import ConfigError, DatasetError, ModelError
from rdv.hashing import sha256_file

CHECKPOINT_SCHEMA = "rdv.train.checkpoint/v1"
BENCHMARK_SCHEMA = "rdv.train.benchmark/v1"
LOG_SCHEMA = "rdv.train.log/v1"
DEFAULT_RUN_DIR = Path("runs/train")
CPU_MICRO_BATCH_CAP = 4


@dataclass(frozen=True, slots=True)
class _TrainingState:
    epoch: int
    optimizer_steps: int
    best_metric: float


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_factor: float) -> float:
    """Warmup + cosine factor for one optimizer step (D16)."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def resolve_device(device: str) -> str:
    """Resolve ``auto`` and reject an unavailable explicit CUDA request."""
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ModelError("CUDA training was requested but torch.cuda.is_available() is false")
    if normalized not in {"cpu", "cuda"}:
        raise ConfigError(f"Unsupported device {device!r}; expected auto, cpu or cuda")
    return normalized


def verify_weights(path: Path, expected_sha256: str) -> None:
    """Verify the pinned ``model.safetensors`` digest before training (S9)."""
    if not path.is_file():
        raise ModelError(f"Base weights not found: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ModelError(
            f"Base weights sha256 mismatch for {path}: expected {expected_sha256}, got {digest}"
        )


def resolve_base_checkpoint(config: TrainConfig, *, offline: bool = False) -> Path:
    """Return the local checkpoint directory, downloading it when needed.

    Raises:
        ModelError: offline mode without a local checkpoint, a download
            failure, or a missing/mismatched ``model.safetensors``.
    """
    local = resolve_local_checkpoint(config.model.hf_id)
    if local is not None:
        weights = local / "model.safetensors"
        verify_weights(weights, config.model.weights_sha256)
        return local
    if offline:
        raise ModelError(
            f"RDV_OFFLINE is set and base checkpoint {config.model.hf_id!r} is not local"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - the train extra provides it
        raise ModelError("huggingface_hub is required to download the base checkpoint") from exc
    try:
        weights_path = Path(
            hf_hub_download(
                repo_id=config.model.hf_id,
                filename="model.safetensors",
                revision=config.model.revision,
            )
        )
    except Exception as exc:  # network and hub errors share one user-facing message
        raise ModelError(f"Cannot download base checkpoint {config.model.hf_id!r}: {exc}") from exc
    verify_weights(weights_path, config.model.weights_sha256)
    return weights_path.parent


def load_training_model(
    config: TrainConfig,
    *,
    device: str,
    offline: bool = False,
) -> torch.nn.Module:
    """Load the pinned RT-DETRv2 checkpoint with the four damage classes."""
    from transformers import RTDetrV2ForObjectDetection

    checkpoint = resolve_base_checkpoint(config, offline=offline)
    id2label = {int(damage_class): damage_class.display_name for damage_class in DamageClass}
    label2id = {name: index for index, name in id2label.items()}
    try:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            str(checkpoint),
            num_labels=config.model.num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        raise ModelError(f"Cannot load base checkpoint {checkpoint}: {exc}") from exc
    return cast(torch.nn.Module, model.to(device))


def build_parameter_groups(
    model: torch.nn.Module,
    *,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split parameters into backbone/head and decay/no-decay groups (D16)."""
    decay_backbone: list[torch.nn.Parameter] = []
    decay_head: list[torch.nn.Parameter] = []
    no_decay_backbone: list[torch.nn.Parameter] = []
    no_decay_head: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = ".backbone." in f".{name}" or name.startswith("backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        if is_backbone:
            (no_decay_backbone if no_decay else decay_backbone).append(parameter)
        else:
            (no_decay_head if no_decay else decay_head).append(parameter)
    groups: list[dict[str, Any]] = []
    for parameters, group_lr, group_decay in (
        (decay_backbone, backbone_lr, weight_decay),
        (no_decay_backbone, backbone_lr, 0.0),
        (decay_head, lr, weight_decay),
        (no_decay_head, lr, 0.0),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": group_lr,
                    "weight_decay": group_decay,
                }
            )
    if not groups:
        raise ConfigError("The model has no trainable parameters")
    return groups


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class RtDetrTrainer:
    """Concrete ``Trainer`` implementing the T9 training contract."""

    def __init__(
        self,
        config: TrainConfig,
        train_repository: DatasetRepository,
        *,
        val_repository: DatasetRepository | None = None,
        device: str = "auto",
        micro_batch: int | None = None,
        num_workers: int | None = None,
        offline: bool = False,
        progress: ProgressSink | None = None,
        benchmark_steps: int = 100,
    ) -> None:
        self.config = config
        self._train_repository = train_repository
        self._val_repository = val_repository
        self._device_name = device
        self._micro_batch = micro_batch
        self._num_workers_override = num_workers
        self._offline = offline
        self._progress = progress
        self._benchmark_steps = benchmark_steps
        if micro_batch is not None and micro_batch < 1:
            raise ValueError("micro_batch must be positive or None")
        if benchmark_steps < 1:
            raise ValueError("benchmark_steps must be positive")

    @property
    def config_hash(self) -> str:
        """Stable digest of the validated config (recorded in checkpoints)."""
        payload = self.config.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Run the requested mode (dry run, benchmark or fine-tune)."""
        if request.dry_run:
            return self._dry_run(request)
        _seed_all(self.config.seed)
        device = resolve_device(self._device_name)
        model = load_training_model(self.config, device=device, offline=self._offline)
        if request.benchmark_only:
            return self._benchmark(request, model, device)
        return self._fit(request, model, device)

    def _dry_run(self, request: TrainingRequest) -> TrainingResult:
        device = resolve_device(self._device_name)
        model = load_training_model(self.config, device=device, offline=self._offline)
        train_dataset = self._make_dataset(
            self._train_repository,
            training=True,
            max_images=self.config.train.max_images_per_epoch,
        )
        if train_dataset is None:
            raise DatasetError("Training split is empty")
        val_dataset = self._make_dataset(self._val_repository, training=False)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if self._progress is not None:
            self._progress.emit(
                ProgressEvent(
                    stage="train",
                    current=0,
                    total=self.config.train.epochs,
                    message=(
                        f"dry run: {len(train_dataset)} train images, "
                        f"{len(val_dataset) if val_dataset is not None else 0} val images, "
                        f"{parameters} parameters on {device}"
                    ),
                )
            )
        _atomic_json(
            request.run_dir / "dry_run.json",
            {
                "schema": "rdv.train.dry-run/v1",
                "device": device,
                "parameters": parameters,
                "train_images": len(train_dataset),
                "val_images": len(val_dataset) if val_dataset is not None else 0,
                "config_hash": self.config_hash,
                "epochs": self.config.train.epochs,
                "effective_batch": self.config.train.effective_batch,
            },
        )
        return TrainingResult(
            run_dir=request.run_dir,
            epochs_completed=0,
            steps=0,
            dry_run=True,
        )

    def _make_dataset(
        self,
        repository: DatasetRepository | None,
        *,
        training: bool,
        max_images: int | None = None,
    ) -> VocTorchDataset | None:
        if repository is None:
            return None
        return VocTorchDataset.from_repository(
            repository,
            augment=self.config.augment,
            training=training,
            max_images=max_images,
        )

    def _make_loader(
        self,
        dataset: VocTorchDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        generator: torch.Generator,
        num_workers: int,
    ) -> DataLoader[TorchSample]:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_samples,
            generator=generator,
            worker_init_fn=lambda worker_id: seed_worker(worker_id, seed=seed),
            pin_memory=False,
            drop_last=False,
        )

    def _micro_batch_for(self, device: str) -> int:
        if self._micro_batch is not None:
            return self._micro_batch
        if device == "cuda":
            return self._probe_micro_batch(device).micro_batch
        return min(self.config.train.effective_batch, CPU_MICRO_BATCH_CAP)

    def _probe_micro_batch(self, device: str) -> AutoBatchResult:
        device_type = "cuda" if device == "cuda" else "cpu"

        def probe(size: int) -> int:
            dataset = self._make_dataset(self._train_repository, training=True, max_images=size * 2)
            if dataset is None:
                raise DatasetError("Training split is empty")
            generator = torch.Generator().manual_seed(self.config.seed)
            loader = self._make_loader(
                dataset,
                batch_size=size,
                seed=self.config.seed,
                shuffle=False,
                generator=generator,
                num_workers=0,
            )
            model = load_training_model(self.config, device=device, offline=self._offline)
            optimizer = torch.optim.AdamW(
                build_parameter_groups(
                    model,
                    lr=self.config.train.lr,
                    backbone_lr=self.config.train.backbone_lr,
                    weight_decay=self.config.train.weight_decay,
                )
            )
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model.train()
            completed = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {key: value.to(device) for key, value in sample.items()}
                    for sample in batch["labels"]
                ]
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda" and self.config.train.precision == "bf16",
                ):
                    loss = model(pixel_values=pixel_values, labels=labels).loss
                loss.backward()
                optimizer.step()
                completed += 1
                if completed >= 20:
                    break
            if completed < 20:
                raise DatasetError(f"Micro-batch probe needs 20 steps, dataset yields {completed}")
            if device_type == "cuda":
                return int(torch.cuda.max_memory_allocated())
            return 0

        return choose_micro_batch(
            probe,
            cap_bytes=int(self.config.budget.vram_cap_gb * GIB),
        )

    def _fit(
        self,
        request: TrainingRequest,
        model: torch.nn.Module,
        device: str,
    ) -> TrainingResult:
        run_dir = request.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        micro_batch = self._micro_batch_for(device)
        accum_steps = max(1, math.ceil(self.config.train.effective_batch / micro_batch))
        train_dataset = self._make_dataset(
            self._train_repository,
            training=True,
            max_images=self.config.train.max_images_per_epoch,
        )
        if train_dataset is None or len(train_dataset) == 0:
            raise DatasetError("Training split is empty")
        val_dataset = self._make_dataset(self._val_repository, training=False)
        generator = torch.Generator().manual_seed(self.config.seed)
        num_workers = (
            self._num_workers_override
            if self._num_workers_override is not None
            else self.config.train.num_workers
        )
        loader = self._make_loader(
            train_dataset,
            batch_size=micro_batch,
            seed=self.config.seed,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
        )
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / micro_batch))
        total_steps = steps_per_epoch * self.config.train.epochs
        optimizer = torch.optim.AdamW(
            build_parameter_groups(
                model,
                lr=self.config.train.lr,
                backbone_lr=self.config.train.backbone_lr,
                weight_decay=self.config.train.weight_decay,
            )
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        ema = ModelEMA(
            model,
            decay=self.config.train.ema.decay,
            warmup=self.config.train.ema.warmup,
        )
        state = self._load_checkpoint(request, model, optimizer, ema, generator)
        start_epoch = state.epoch
        optimizer_steps = state.optimizer_steps
        best_metric = state.best_metric
        losses: list[float] = []
        log_path = run_dir / "metrics.jsonl"
        csv_path = run_dir / "epochs.csv"
        if start_epoch == 0:
            log_path.write_text("", encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["epoch", "optimizer_steps", "mean_loss", "train_images", "elapsed_s"]
                )
        device_type = "cuda" if device == "cuda" else "cpu"
        started = time.perf_counter()
        epochs_completed = start_epoch
        steps_this_run = 0
        stop = False
        for epoch in range(start_epoch, self.config.train.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses: list[float] = []
            for batch_index, batch in enumerate(loader):
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {key: value.to(device) for key, value in sample.items()}
                    for sample in batch["labels"]
                ]
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda" and self.config.train.precision == "bf16",
                ):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise ConfigError(f"Non-finite training loss at step {optimizer_steps + 1}")
                scaled = loss / accum_steps
                scaled.backward()
                raw_loss = float(loss.detach().to("cpu"))
                epoch_losses.append(raw_loss)
                losses.append(raw_loss)
                should_step = (batch_index + 1) % accum_steps == 0 or batch_index + 1 == len(loader)
                if should_step:
                    if self.config.train.grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.config.train.grad_clip
                        )
                    factor = lr_factor(
                        optimizer_steps,
                        total_steps,
                        self.config.train.warmup_steps,
                        self.config.train.lr_min_factor,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = float(group.get("initial_lr", group["lr"])) * factor
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model, step=optimizer_steps)
                    optimizer_steps += 1
                    steps_this_run += 1
                    self._log_step(
                        log_path,
                        epoch=epoch,
                        step=optimizer_steps,
                        loss=raw_loss,
                        lr=float(optimizer.param_groups[0]["lr"]),
                    )
                    if self._progress is not None:
                        self._progress.emit(
                            ProgressEvent(
                                stage="train",
                                current=optimizer_steps,
                                total=total_steps,
                                message=f"epoch {epoch + 1}: loss {raw_loss:.4f}",
                            )
                        )
            if not epoch_losses:
                raise DatasetError("Training split produced no batches")
            epochs_completed = epoch + 1
            self._append_epoch(
                csv_path,
                epoch=epochs_completed,
                optimizer_steps=optimizer_steps,
                mean_loss=float(np.mean(epoch_losses)),
                train_images=len(train_dataset),
                elapsed=time.perf_counter() - started,
            )
            should_evaluate = (
                val_dataset is not None
                and len(val_dataset) > 0
                and epochs_completed % self.config.train.eval_every == 0
            )
            if should_evaluate and val_dataset is not None:
                metric = self._validate(model, val_dataset, device)
                if metric > best_metric:
                    best_metric = metric
                    self._save_best(model, ema, run_dir / "best")
                    _atomic_json(
                        run_dir / "best" / "metrics.json",
                        {"mAP50_95": metric, "epoch": epochs_completed},
                    )
            self._save_checkpoint(
                run_dir / "last.pt",
                epoch=epochs_completed,
                optimizer_steps=optimizer_steps,
                model=model,
                optimizer=optimizer,
                ema=ema,
                generator=generator,
                best_metric=best_metric,
            )
            if request.max_steps is not None and optimizer_steps >= request.max_steps:
                stop = True
            if time.perf_counter() - started > self.config.budget.max_hours * 3600.0:
                stop = True
            if stop:
                break
        return TrainingResult(
            run_dir=run_dir,
            epochs_completed=epochs_completed,
            steps=steps_this_run,
            last_loss=losses[-1] if losses else None,
            best_metric=best_metric if best_metric > 0.0 else None,
            losses=tuple(losses),
        )

    def _benchmark(
        self,
        request: TrainingRequest,
        model: torch.nn.Module,
        device: str,
    ) -> TrainingResult:
        run_dir = request.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        micro_batch = self._micro_batch_for(device)
        accum_steps = max(1, math.ceil(self.config.train.effective_batch / micro_batch))
        dataset = self._make_dataset(
            self._train_repository,
            training=True,
            max_images=self.config.train.max_images_per_epoch,
        )
        if dataset is None or len(dataset) == 0:
            raise DatasetError("Training split is empty")
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = self._make_loader(
            dataset,
            batch_size=micro_batch,
            seed=self.config.seed,
            shuffle=True,
            generator=generator,
            num_workers=self._num_workers_override or 0,
        )
        optimizer = torch.optim.AdamW(
            build_parameter_groups(
                model,
                lr=self.config.train.lr,
                backbone_lr=self.config.train.backbone_lr,
                weight_decay=self.config.train.weight_decay,
            )
        )
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        device_type = "cuda" if device == "cuda" else "cpu"
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = 0
        images = 0
        started = time.perf_counter()
        while steps < self._benchmark_steps:
            progressed = False
            steps_before_epoch = steps
            for batch_index, batch in enumerate(loader):
                progressed = True
                pixel_values = batch["pixel_values"].to(device)
                labels = [
                    {key: value.to(device) for key, value in sample.items()}
                    for sample in batch["labels"]
                ]
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda" and self.config.train.precision == "bf16",
                ):
                    loss = model(pixel_values=pixel_values, labels=labels).loss
                (loss / accum_steps).backward()
                images += int(pixel_values.shape[0])
                if (batch_index + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    if steps >= self._benchmark_steps:
                        break
            if not progressed:
                raise DatasetError("Training split produced no batches")
            if steps == steps_before_epoch:
                raise DatasetError(
                    "Benchmark dataset has fewer batches than the accumulation window"
                )
        elapsed = max(time.perf_counter() - started, 1e-9)
        peak = int(torch.cuda.max_memory_allocated()) if device_type == "cuda" else 0
        benchmark = {
            "schema": BENCHMARK_SCHEMA,
            "device": device,
            "micro_batch": micro_batch,
            "accumulation_steps": accum_steps,
            "effective_batch": micro_batch * accum_steps,
            "steps": steps,
            "planned_steps": self._benchmark_steps,
            "images": images,
            "img_per_s": round(images / elapsed, 4),
            "peak_memory_gb": round(peak / GIB, 4),
            "vram_cap_gb": self.config.budget.vram_cap_gb,
            "config_hash": self.config_hash,
        }
        _atomic_json(run_dir / "benchmark.json", benchmark)
        return TrainingResult(
            run_dir=run_dir,
            epochs_completed=0,
            steps=steps,
            benchmark=benchmark,
        )

    def _resume_path(self, request: TrainingRequest) -> Path | None:
        resume = request.resume.strip()
        if resume == "none":
            return None
        if resume == "auto":
            candidate = request.run_dir / "last.pt"
            return candidate if candidate.is_file() else None
        candidate = Path(resume).expanduser()
        if not candidate.is_file():
            raise ConfigError(f"Checkpoint not found: {candidate}")
        return candidate

    def _load_checkpoint(
        self,
        request: TrainingRequest,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: ModelEMA,
        generator: torch.Generator,
    ) -> _TrainingState:
        path = self._resume_path(request)
        if path is None:
            return _TrainingState(epoch=0, optimizer_steps=0, best_metric=0.0)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ConfigError(f"Cannot read checkpoint {path}: {exc}") from exc
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ConfigError(f"Unsupported checkpoint schema in {path}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema_state = ModelEMA.from_state_dict(model, payload["ema"])
        ema.shadow = ema_state.shadow
        ema.step = ema_state.step
        if "generator" in payload:
            generator.set_state(payload["generator"])
        rng = payload.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        return _TrainingState(
            epoch=int(payload.get("epoch", 0)),
            optimizer_steps=int(payload.get("optimizer_steps", 0)),
            best_metric=float(payload.get("best_metric", 0.0)),
        )

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        optimizer_steps: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ema: ModelEMA,
        generator: torch.Generator,
        best_metric: float,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "best_metric": best_metric,
            "config_hash": self.config_hash,
            "seed": self.config.seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "generator": generator.get_state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        _atomic_torch_save(path, payload)

    def _save_best(self, model: torch.nn.Module, ema: ModelEMA, directory: Path) -> None:
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ema.copy_to(model)
        directory.mkdir(parents=True, exist_ok=True)
        cast(Any, model).save_pretrained(directory)
        model.load_state_dict(original)

    def _validate(
        self,
        model: torch.nn.Module,
        dataset: VocTorchDataset,
        device: str,
    ) -> float:
        predictions = {}
        model.eval()
        with torch.no_grad():
            for index, annotation in enumerate(dataset.annotation):
                sample = dataset[index]
                outputs = model(pixel_values=sample.pixel_values.unsqueeze(0).to(device))
                logits = outputs.logits[0].float().to("cpu").numpy()
                boxes = outputs.pred_boxes[0].float().to("cpu").numpy()
                predictions[annotation.image_id] = decode_detr_outputs(
                    logits,
                    boxes,
                    (annotation.width, annotation.height),
                    score_threshold=0.001,
                )
        return evaluate(list(dataset.annotation), predictions).mAP50_95

    def _log_step(
        self,
        path: Path,
        *,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
    ) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema": LOG_SCHEMA,
                        "epoch": epoch,
                        "step": step,
                        "loss": round(loss, 6),
                        "lr": lr,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    @staticmethod
    def _append_epoch(
        path: Path,
        *,
        epoch: int,
        optimizer_steps: int,
        mean_loss: float,
        train_images: int,
        elapsed: float,
    ) -> None:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [epoch, optimizer_steps, round(mean_loss, 6), train_images, round(elapsed, 2)]
            )
