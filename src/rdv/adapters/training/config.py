"""Typed training configuration (decision D16).

The YAML file is the single source of truth for a run: seeds, the pinned base
checkpoint, optimizer hyper-parameters, augmentation and the time/VRAM budget.
``${ENV_VAR}`` references are expanded before validation so secrets-free local
paths and model ids can stay out of git.

Only the training extra imports this module (PyYAML), so the base CLI can show
``rdv train --help`` on a torch-free install.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rdv.errors import ConfigError

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HEX = re.compile(r"^[0-9a-f]{64}$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Model):
    """Base checkpoint and network contract (D1, S4)."""

    hf_id: str
    revision: str
    weights_sha256: str
    input_size: int = Field(default=640, gt=0)
    num_labels: int = Field(default=4, gt=0)

    @field_validator("hf_id", "revision")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("weights_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _HEX.match(normalized):
            raise ValueError("must be a lowercase 64-character hex sha256")
        return normalized


class SplitConfig(_Model):
    """Seeded block split inherited from the data pipeline (D3)."""

    seed: int = 42
    block_size: int = Field(default=100, gt=0)
    ratios: tuple[float, ...]

    @field_validator("ratios")
    @classmethod
    def _ratios(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != 3 or any(ratio <= 0.0 for ratio in value):
            raise ValueError("must contain three positive ratios")
        if not math.isclose(sum(value), 1.0, abs_tol=1e-9):
            raise ValueError("must sum to 1.0")
        return tuple(float(ratio) for ratio in value)


class DataConfig(_Model):
    """Countries used for training and the out-of-domain sample size (D2)."""

    countries: tuple[str, ...]
    splits: SplitConfig
    ood: dict[str, int] = Field(default_factory=dict)

    @field_validator("countries")
    @classmethod
    def _countries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not country.strip() for country in value):
            raise ValueError("must be a non-empty list of country names")
        return tuple(country.strip() for country in value)

    @field_validator("ood")
    @classmethod
    def _ood(cls, value: dict[str, int]) -> dict[str, int]:
        for country, count in value.items():
            if not country.strip() or count <= 0:
                raise ValueError("out-of-domain entries need a name and a positive count")
        return dict(value)


class EmaConfig(_Model):
    """Exponential moving average of the trainable weights (D16)."""

    decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    warmup: bool = True


class TrainParams(_Model):
    """Optimizer, schedule, precision and logging knobs (D16)."""

    epochs: int = Field(default=24, gt=0)
    effective_batch: int = Field(default=16, gt=0)
    lr: float = Field(default=1.0e-4, gt=0.0)
    backbone_lr: float = Field(default=1.0e-5, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    warmup_steps: int = Field(default=500, ge=0)
    lr_min_factor: float = Field(default=0.01, ge=0.0, le=1.0)
    grad_clip: float = Field(default=0.1, ge=0.0)
    precision: Literal["bf16", "fp32"] = "bf16"
    ema: EmaConfig = Field(default_factory=EmaConfig)
    eval_every: int = Field(default=2, gt=0)
    num_workers: int = Field(default=8, ge=0)
    max_images_per_epoch: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _max_images(self) -> TrainParams:
        if self.max_images_per_epoch is not None and self.max_images_per_epoch < 1:
            raise ValueError("max_images_per_epoch must be positive or null")
        return self


class AffineConfig(_Model):
    """Random affine augmentation (D16)."""

    p: float = Field(default=0.5, ge=0.0, le=1.0)
    translate: float = Field(default=0.1, ge=0.0, le=1.0)
    scale: tuple[float, float] = (0.8, 1.2)

    @field_validator("scale")
    @classmethod
    def _scale(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low <= 0.0 or high < low:
            raise ValueError("scale must be a positive increasing pair")
        return (float(low), float(high))


class AugmentConfig(_Model):
    """torchvision v2 augmentation pipeline (D16)."""

    resize: tuple[int, int] = (640, 640)
    hflip: float = Field(default=0.5, ge=0.0, le=1.0)
    photometric_distort: float = Field(default=0.5, ge=0.0, le=1.0)
    affine: AffineConfig = Field(default_factory=AffineConfig)
    vflip: bool = False
    rot90: bool = False
    zoom_out: bool = False

    @field_validator("resize")
    @classmethod
    def _resize(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(side <= 0 for side in value):
            raise ValueError("resize must be two positive integers")
        return (int(value[0]), int(value[1]))


class BudgetConfig(_Model):
    """Wall-clock and VRAM rules applied before the long run (D16)."""

    max_hours: float = Field(default=10.0, gt=0.0)
    min_epochs: int = Field(default=12, gt=0)
    vram_cap_gb: float = Field(default=7.2, gt=0.0)


class TrainConfig(_Model):
    """Complete, validated training configuration."""

    seed: int = 42
    model: ModelConfig
    data: DataConfig
    train: TrainParams
    augment: AugmentConfig
    budget: BudgetConfig

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        """Read a YAML file, expand ``${ENV}`` references and validate it.

        Raises:
            ConfigError: the file is missing, is not valid YAML, references an
                unset environment variable, or violates the schema.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Cannot read training config {path}: {exc}") from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Malformed YAML training config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError(f"Training config {path} must contain a YAML mapping")
        expanded = _expand_env(payload, path)
        return cls.from_mapping(expanded)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TrainConfig:
        """Validate an already expanded mapping (used by tests)."""
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            details: list[str] = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                details.append(f"{location or '<root>'}: {error['msg']}")
            raise ConfigError("Invalid training config: " + "; ".join(details)) from exc


def _expand_env(value: Any, source: Path) -> Any:
    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                missing.append(name)
                return match.group(0)
            return resolved

        expanded = _ENV_REFERENCE.sub(replace, value)
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ConfigError(f"Unset environment variable(s) in training config {source}: {names}")
        return expanded
    if isinstance(value, dict):
        return {key: _expand_env(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, source) for item in value]
    return value


def resolve_local_checkpoint(hf_id: str) -> Path | None:
    """Return the local checkpoint directory when ``hf_id`` is a path.

    A train config may reference a local directory (tests and offline runs) or
    a Hugging Face repository id (the pinned real checkpoint).
    """
    candidate = Path(hf_id).expanduser()
    return candidate if candidate.is_dir() else None
