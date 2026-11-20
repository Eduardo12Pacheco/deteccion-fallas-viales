"""Typed settings and their environment contract (decision D19)."""

from __future__ import annotations

import pytest

from rdv.config.settings import Settings
from rdv.errors import ConfigError

RDV_VARIABLES = (
    "RDV_MODEL",
    "RDV_MODEL_DIR",
    "RDV_DATA_DIR",
    "RDV_DEVICE",
    "RDV_SCORE_THRESHOLD",
    "RDV_LOG_LEVEL",
    "RDV_LOG_FORMAT",
    "RDV_OFFLINE",
    "RDV_HOST",
    "RDV_PORT",
    "RDV_MAX_UPLOAD_MB",
    "RDV_MAX_VIDEO_SECONDS",
    "RDV_ANALYSIS_TTL_MIN",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in RDV_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_follow_the_documented_contract() -> None:
    settings = Settings.from_env()

    assert settings.model == "rdv-rtdetrv2-r18-rdd2022"
    assert settings.device == "auto"
    assert settings.score_threshold is None
    assert settings.log_level == "INFO"
    assert settings.log_format == "console"
    assert settings.offline is False
    assert settings.host == "127.0.0.1"
    # L5: local servers bind a port inside the shared 16000-16999 range.
    assert 16000 <= settings.port <= 16999
    assert settings.port == 16080
    assert settings.max_upload_mb == 512
    assert settings.max_video_seconds == 900
    assert settings.analysis_ttl_min == 30
    assert settings.model_dir.name == "models"
    assert settings.data_dir.name == "data"


def test_environment_overrides_every_field(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RDV_PORT", "9000")
    monkeypatch.setenv("RDV_DEVICE", "cpu")
    monkeypatch.setenv("RDV_SCORE_THRESHOLD", "0.42")
    monkeypatch.setenv("RDV_LOG_FORMAT", "json")
    monkeypatch.setenv("RDV_OFFLINE", "true")
    monkeypatch.setenv("RDV_DATA_DIR", str(tmp_path / "dataset"))
    monkeypatch.setenv("RDV_MODEL_DIR", str(tmp_path / "weights"))

    settings = Settings.from_env()

    assert settings.port == 9000
    assert settings.device == "cpu"
    assert settings.score_threshold == pytest.approx(0.42)
    assert settings.log_format == "json"
    assert settings.offline is True
    assert settings.data_dir == tmp_path / "dataset"
    assert settings.model_dir == tmp_path / "weights"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RDV_PORT", "not-a-port"),
        ("RDV_PORT", "0"),
        ("RDV_PORT", "70000"),
        ("RDV_DEVICE", "tpu"),
        ("RDV_LOG_FORMAT", "xml"),
        ("RDV_LOG_LEVEL", "LOUD"),
        ("RDV_SCORE_THRESHOLD", "1.5"),
        ("RDV_MAX_UPLOAD_MB", "0"),
        ("RDV_MAX_VIDEO_SECONDS", "-1"),
        ("RDV_ANALYSIS_TTL_MIN", "0"),
        ("RDV_OFFLINE", "maybe"),
        ("RDV_HOST", "  "),
    ],
)
def test_invalid_values_raise_config_error(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError) as raised:
        Settings.from_env()

    assert raised.value.exit_code == 6
    assert name.lower() in raised.value.message.lower()


def test_log_level_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDV_LOG_LEVEL", "debug")

    assert Settings.from_env().log_level == "DEBUG"


def test_explicit_overrides_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDV_PORT", "9000")

    assert Settings.from_env(port=16081).port == 16081


def test_paths_expand_the_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDV_DATA_DIR", "~/rdv-data")

    settings = Settings.from_env()

    assert "~" not in str(settings.data_dir)
    assert settings.data_dir.is_absolute()
