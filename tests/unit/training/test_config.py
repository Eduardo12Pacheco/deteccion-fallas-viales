"""Typed training config: defaults, env expansion and validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from rdv.adapters.training.config import TrainConfig
from rdv.errors import ConfigError

pytestmark = pytest.mark.train

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs" / "train" / "rdd2022-rtdetrv2-r18.yaml"

MINIMAL = {
    "seed": 42,
    "model": {
        "hf_id": "PekingU/rtdetr_v2_r18vd",
        "revision": "5650961749fa93567c0d46fc7f43ea4f9e914107",
        "weights_sha256": ("d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115"),
    },
    "data": {
        "countries": ["Japan"],
        "splits": {"ratios": [0.8, 0.1, 0.1]},
    },
    "train": {"epochs": 1},
    "augment": {},
    "budget": {},
}


def test_committed_config_matches_decision_d16() -> None:
    config = TrainConfig.load(CONFIG)

    assert config.seed == 42
    assert config.model.hf_id == "PekingU/rtdetr_v2_r18vd"
    assert config.model.revision == "5650961749fa93567c0d46fc7f43ea4f9e914107"
    assert config.model.weights_sha256.startswith("d18309d0")
    assert config.model.input_size == 640
    assert config.model.num_labels == 4
    assert config.data.countries == ("Japan", "India", "Czech")
    assert config.data.splits == config.data.splits.__class__(
        seed=42, block_size=100, ratios=(0.8, 0.1, 0.1)
    )
    assert config.data.ood == {"Norway": 1000}
    assert config.train.epochs == 24
    assert config.train.effective_batch == 16
    assert config.train.precision == "bf16"
    assert config.train.ema.decay == pytest.approx(0.9999)
    assert config.train.ema.warmup is True
    assert config.train.max_images_per_epoch is None
    assert config.augment.resize == (640, 640)
    assert config.augment.affine.scale == (0.8, 1.2)
    assert config.budget.vram_cap_gb == pytest.approx(7.2)


def test_env_references_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RDV_TEST_CHECKPOINT", "local/checkpoint")
    path = tmp_path / "train.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "PekingU/rtdetr_v2_r18vd", "${RDV_TEST_CHECKPOINT}"
        ),
        encoding="utf-8",
    )

    config = TrainConfig.load(path)

    assert config.model.hf_id == "local/checkpoint"


def test_unset_env_reference_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RDV_TEST_UNSET", raising=False)
    path = tmp_path / "train.yaml"
    path.write_text(
        CONFIG.read_text(encoding="utf-8").replace("PekingU/rtdetr_v2_r18vd", "${RDV_TEST_UNSET}"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="RDV_TEST_UNSET") as excinfo:
        TrainConfig.load(path)

    assert excinfo.value.exit_code == 6


def test_malformed_and_missing_files_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read training config") as excinfo:
        TrainConfig.load(tmp_path / "missing.yaml")
    assert excinfo.value.exit_code == 6

    broken = tmp_path / "broken.yaml"
    broken.write_text("model: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError, match="Malformed YAML") as excinfo:
        TrainConfig.load(broken)
    assert excinfo.value.exit_code == 6


def test_unknown_keys_are_rejected() -> None:
    payload = {**MINIMAL, "model": {**MINIMAL["model"], "typo": True}}

    with pytest.raises(ConfigError, match="typo") as excinfo:
        TrainConfig.from_mapping(payload)

    assert excinfo.value.exit_code == 6


def test_invalid_ratios_and_hash_are_rejected() -> None:
    payload = {
        **MINIMAL,
        "data": {
            **MINIMAL["data"],
            "splits": {"ratios": [0.8, 0.2, 0.2]},
        },
    }
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        TrainConfig.from_mapping(payload)

    payload = {**MINIMAL, "model": {**MINIMAL["model"], "weights_sha256": "abc"}}
    with pytest.raises(ConfigError, match="sha256"):
        TrainConfig.from_mapping(payload)
