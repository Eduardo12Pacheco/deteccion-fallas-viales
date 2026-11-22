"""Integration tests for the local-server security middleware (S14)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.config.settings import Settings
from rdv.entrypoints.web.app import create_app

# TestClient runs an anyio event loop; the Windows proactor loop opens a socketpair
# that pytest-socket would block. The app itself never leaves loopback in these tests.
pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
IMAGE = sorted((MINI / "Japan" / "train" / "images").glob("*.jpg"))[0]
BASE_URL = "http://127.0.0.1:16080"
MAX_UPLOAD_MB = 1


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model=str(TINY_MODEL),
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        max_upload_mb=MAX_UPLOAD_MB,
        analysis_ttl_min=1,
    )


def _app(settings: Settings) -> Any:
    detector = OnnxDetector(TINY_MODEL, providers=["CPUExecutionProvider"])
    return create_app(settings, lambda: detector)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(_app(_settings(tmp_path)), base_url=BASE_URL) as test_client:
        yield test_client


def _image_files() -> dict[str, tuple[str, bytes, str]]:
    return {"file": (IMAGE.name, IMAGE.read_bytes(), "image/jpeg")}


def test_host_header_must_be_local(client: TestClient) -> None:
    response = client.get("/api/health", headers={"host": "evil.example"})
    assert response.status_code == 421
    assert response.json()["detail"]


def test_cross_origin_post_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/analyses",
        files=_image_files(),
        headers={"origin": "http://evil.example"},
    )
    assert response.status_code == 403


def test_same_origin_post_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/api/analyses",
        files=_image_files(),
        headers={"origin": BASE_URL},
    )
    assert response.status_code == 202
    client.delete(f"/api/analyses/{response.json()['id']}")


def test_oversized_upload_is_rejected_from_content_length(client: TestClient) -> None:
    payload = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1024)
    response = client.post(
        "/api/analyses",
        files={"file": ("big.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 413


def test_oversized_chunked_upload_is_rejected_while_streaming(client: TestClient) -> None:
    boundary = "----rdv-limit"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    def chunks() -> Iterator[bytes]:
        yield prefix
        for _ in range(12):
            yield b"x" * (128 * 1024)
        yield suffix

    response = client.post(
        "/api/analyses",
        content=chunks(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    "path",
    [
        "/api/analyses/../../etc/passwd",
        "/api/analyses/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "/api/analyses/not-a-job-id",
    ],
)
def test_path_like_analysis_ids_are_not_found(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_responses_never_add_cors_headers(client: TestClient) -> None:
    for response in (
        client.get("/api/health"),
        client.post("/api/analyses", files=_image_files()),
    ):
        names = {name.lower() for name in response.headers}
        assert not any(name.startswith("access-control-") for name in names)
