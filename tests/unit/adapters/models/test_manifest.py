"""The packaged model manifest validates the pinned artifact contract (D13)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rdv.adapters.models.manifest import ModelManifest, default_manifest_path
from rdv.errors import ModelError


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "rdv-test",
        "version": "v1",
        "url": "https://example.invalid/rdv-test.onnx",
        "sha256": "a" * 64,
        "size": 1024,
        "input_size": 640,
        "recommended_score_threshold": 0.35,
        "classes": ["D00", "D10", "D20", "D40"],
        "license": "Apache-2.0",
        "metrics_ref": "reports/v1/metrics.json",
    }
    entry.update(overrides)
    return entry


def _payload(*entries: dict[str, object]) -> dict[str, object]:
    return {"schema": "rdv.models/v1", "models": list(entries)}


def test_the_packaged_manifest_is_valid_and_empty_until_the_first_model() -> None:
    manifest = ModelManifest.load()

    assert default_manifest_path().name == "models.json"
    assert manifest.models == ()


def test_valid_entry_is_parsed(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_payload(_entry())), encoding="utf-8")

    manifest = ModelManifest.load(path)

    entry = manifest.get("rdv-test")
    assert entry.version == "v1"
    assert entry.size == 1024
    assert entry.input_size == 640
    assert entry.score_threshold == pytest.approx(0.35)
    assert entry.classes == ("D00", "D10", "D20", "D40")
    assert entry.license == "Apache-2.0"
    assert entry.metrics_ref == "reports/v1/metrics.json"
    assert manifest.find("missing") is None


def test_unknown_id_lists_available_models_and_the_local_path_hint() -> None:
    manifest = ModelManifest.from_dict(_payload(_entry()))

    with pytest.raises(ModelError, match="Pass --model PATH") as excinfo:
        manifest.get("nope")

    assert excinfo.value.exit_code == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "wrong/v1", "models": []},
        {"models": []},
        {"schema": "rdv.models/v1", "models": {}},
        {"schema": "rdv.models/v1", "models": ["not-an-object"]},
    ],
)
def test_invalid_manifest_documents_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ModelError):
        ModelManifest.from_dict(payload)


def test_missing_entry_fields_are_rejected() -> None:
    entry = _entry()
    del entry["sha256"]

    with pytest.raises(ModelError, match="missing"):
        ModelManifest.from_dict(_payload(entry))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sha256": "xyz"}, "sha256"),
        ({"size": 0}, "size"),
        ({"recommended_score_threshold": 1.5}, "score_threshold"),
        ({"classes": ["D00"]}, "classes"),
        ({"url": "ftp://example.invalid/model.onnx"}, "HTTP"),
        ({"id": ""}, "id"),
    ],
)
def test_invalid_entry_values_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ModelError, match=message):
        ModelManifest.from_dict(_payload(_entry(**overrides)))


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ModelError, match="Duplicate model id"):
        ModelManifest.from_dict(_payload(_entry(), _entry()))


def test_missing_manifest_file_is_a_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        ModelManifest.load(tmp_path / "absent.json")


def test_malformed_json_is_a_model_error(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelError, match="Cannot read model manifest"):
        ModelManifest.load(path)


def test_entry_sha256_matches_the_bytes_it_describes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"fake onnx payload")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = ModelManifest.from_dict(
        _payload(_entry(sha256=digest, size=artifact.stat().st_size))
    )

    assert manifest.get("rdv-test").sha256 == digest
