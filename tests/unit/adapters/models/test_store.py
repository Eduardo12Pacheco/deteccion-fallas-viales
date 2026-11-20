"""Model store: resume, integrity checks and offline behavior (D13)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from rdv.adapters.models.manifest import ModelManifest
from rdv.adapters.models.store import CachedModelStore
from rdv.application.ports import DEFAULT_CLASS_CODES, ModelEntry
from rdv.errors import ModelError

CONTENT = b"fake onnx graph payload " * 100


def _entry(content: bytes = CONTENT, **overrides: object) -> ModelEntry:
    values: dict[str, object] = {
        "id": "demo",
        "version": "v1",
        "url": "https://example.invalid/demo.onnx",
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "input_size": 640,
        "score_threshold": 0.3,
        "classes": DEFAULT_CLASS_CODES,
        "license": "Apache-2.0",
    }
    values.update(overrides)
    return ModelEntry(**values)  # type: ignore[arg-type]


def _store(
    tmp_path: Path, *, entry: ModelEntry | None = None, **kwargs: object
) -> CachedModelStore:
    manifest = ModelManifest((entry or _entry(),))
    return CachedModelStore(manifest, tmp_path / "models", **kwargs)  # type: ignore[arg-type]


def test_pull_downloads_verifies_and_installs(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=CONTENT)

    store = _store(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))

    path = store.pull("demo")

    assert path == tmp_path / "models" / "demo.onnx"
    assert path.read_bytes() == CONTENT
    assert not path.with_name("demo.onnx.part").exists()
    assert len(requests) == 1
    assert requests[0].headers.get("Range") is None


def test_pull_resumes_a_partial_file_with_a_range_request(tmp_path: Path) -> None:
    partial = tmp_path / "models" / "demo.onnx.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(CONTENT[:100])
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers.get("Range")
        return httpx.Response(206, content=CONTENT[100:])

    store = _store(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))

    path = store.pull("demo")

    assert seen["range"] == "bytes=100-"
    assert path.read_bytes() == CONTENT


def test_pull_restarts_when_the_server_ignores_the_range(tmp_path: Path) -> None:
    partial = tmp_path / "models" / "demo.onnx.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(CONTENT[:100])
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["range"] = request.headers.get("Range")
        return httpx.Response(200, content=CONTENT)

    store = _store(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))

    path = store.pull("demo")

    assert seen["range"] == "bytes=100-"
    assert path.read_bytes() == CONTENT


def test_pull_is_a_no_op_when_the_cached_file_already_matches(tmp_path: Path) -> None:
    cached = tmp_path / "models" / "demo.onnx"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(CONTENT)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=CONTENT)

    store = _store(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))

    assert store.pull("demo") == cached
    assert calls == 0


def test_sha_mismatch_leaves_no_final_file_and_removes_the_part(tmp_path: Path) -> None:
    entry = _entry(sha256="0" * 64)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CONTENT)

    store = _store(
        tmp_path, entry=entry, client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ModelError, match="sha256 mismatch") as excinfo:
        store.pull("demo")

    assert excinfo.value.exit_code == 4
    assert not (tmp_path / "models" / "demo.onnx").exists()
    assert not (tmp_path / "models" / "demo.onnx.part").exists()


def test_size_mismatch_keeps_the_part_for_a_later_resume(tmp_path: Path) -> None:
    entry = _entry(size=len(CONTENT) + 10)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CONTENT)

    store = _store(
        tmp_path, entry=entry, client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ModelError, match="Size mismatch") as excinfo:
        store.pull("demo")

    assert excinfo.value.exit_code == 4
    assert (tmp_path / "models" / "demo.onnx.part").is_file()


def test_http_failure_keeps_the_part_and_is_a_model_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    store = _store(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(ModelError, match="Download failed") as excinfo:
        store.pull("demo")

    assert excinfo.value.exit_code == 4


def test_offline_mode_fails_before_any_request(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("offline mode must not open a request")

    store = _store(
        tmp_path,
        offline=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ModelError, match="RDV_OFFLINE") as excinfo:
        store.pull("demo")

    assert excinfo.value.exit_code == 4


def test_resolve_requires_a_downloaded_model(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ModelError, match=r"rdv models pull demo"):
        store.resolve("demo")


def test_resolve_accepts_an_explicit_path(tmp_path: Path) -> None:
    model = tmp_path / "local.onnx"
    model.write_bytes(CONTENT)
    store = _store(tmp_path)

    assert store.resolve(str(model)) == model


def test_resolve_rejects_a_missing_explicit_path(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ModelError, match="Model file not found"):
        store.resolve(str(tmp_path / "absent.onnx"))


def test_verify_selects_the_matching_entry(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(CONTENT)
    store = _store(tmp_path)

    entry = store.verify(model)

    assert entry.id == "demo"


def test_verify_rejects_a_different_file(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"different bytes")
    store = _store(tmp_path)

    with pytest.raises(ModelError, match="does not match any model") as excinfo:
        store.verify(model)

    assert excinfo.value.exit_code == 4


def test_verify_by_id_reports_the_expected_digest(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"different bytes")
    store = _store(tmp_path)

    with pytest.raises(ModelError, match="sha256 mismatch"):
        store.verify(model, "demo")


def test_pull_with_an_unknown_id_lists_the_manifest(tmp_path: Path) -> None:
    manifest = ModelManifest(())
    store = CachedModelStore(manifest, tmp_path / "models")
    with pytest.raises(ModelError, match="not in the manifest"):
        store.pull("demo")
