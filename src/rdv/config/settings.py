"""Typed application settings read from ``RDV_*`` environment variables.

All runtime knobs live here so adapters never read the environment directly
(decision D19). Validation happens once at the composition root; invalid
values surface as :class:`rdv.errors.ConfigError` (exit code 6) with the
offending variable in the message.

The default web port is ``16080`` (inside the shared 16000-16999 range required
by the portfolio environment) instead of the historical ``8765``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from platformdirs import user_data_path
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rdv.errors import ConfigError

DeviceName = Literal["auto", "cpu", "cuda"]
LogFormat = Literal["console", "json"]

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _default_model_dir() -> Path:
    return user_data_path("rdv") / "models"


def _default_data_dir() -> Path:
    return user_data_path("rdv") / "data"


class Settings(BaseSettings):
    """Validated configuration for the library, CLI, and local web app.

    Every field maps to the uppercase environment variable ``RDV_<FIELD>``
    (for example ``RDV_PORT`` or ``RDV_DEVICE``). Explicit keyword arguments
    take precedence over the environment, which keeps tests hermetic.
    """

    model_config = SettingsConfigDict(
        env_prefix="RDV_",
        env_file=None,
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    model: str = "rdv-rtdetrv2-r18-rdd2022"
    """Model id from the packaged manifest or a path to a local ONNX file."""

    model_dir: Path = Field(default_factory=_default_model_dir)
    """Directory of the model cache (``rdv models pull`` writes here)."""

    data_dir: Path = Field(default_factory=_default_data_dir)
    """Root of downloaded datasets and generated splits."""

    device: DeviceName = "auto"
    """Execution provider preference; ``auto`` prefers CUDA then CPU."""

    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    """Detection threshold; ``None`` takes the manifest's recommended value."""

    log_level: str = "INFO"
    """Minimum log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``)."""

    log_format: LogFormat = "console"
    """``console`` for humans, ``json`` for machine-parseable log lines."""

    offline: bool = False
    """When true, every network attempt fails fast before opening a socket."""

    host: str = "127.0.0.1"
    """Bind address of the local web app; loopback only by default."""

    port: int = Field(default=16080, ge=1, le=65535)
    """Bind port of the local web app (shared 16000-16999 range)."""

    max_upload_mb: int = Field(default=512, gt=0)
    """Largest accepted upload for the web app."""

    max_video_seconds: int = Field(default=900, gt=0)
    """Videos longer than this are rejected by the web app."""

    analysis_ttl_min: int = Field(default=30, gt=0)
    """Minutes an uploaded analysis is kept before cleanup."""

    @field_validator("model", "host")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in LOG_LEVELS:
            expected = ", ".join(sorted(LOG_LEVELS))
            raise ValueError(f"must be one of {expected}")
        return normalized

    @field_validator("model_dir", "data_dir")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        expanded = Path(value).expanduser()
        if not str(expanded).strip():
            raise ValueError("must not be blank")
        return expanded

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from the environment plus optional explicit overrides.

        Raises:
            ConfigError: one or more values are missing or invalid. The message
                names the offending ``RDV_*`` variables and exits with code 6.
        """
        try:
            return cls(**overrides)
        except ValidationError as exc:
            details: list[str] = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                variable = f"RDV_{location.upper()}" if location else "RDV_"
                details.append(f"{variable}: {error['msg']}")
            raise ConfigError("Invalid configuration: " + "; ".join(details)) from exc
