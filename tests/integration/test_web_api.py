"""Integration tests for the local web API (S12/S14)."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from PIL import Image

from rdv.adapters.inference.onnx_detector import OnnxDetector
from rdv.config.settings import Settings
from rdv.entrypoints.web.app import create_app
from rdv.errors import ModelError
from tests.support.synthetic_video import write_moving_rectangle_video

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
IMAGE = sorted((MINI / "Japan" / "train" / "images").glob("*.jpg"))[0]
SCHEMA = json.loads(
    (REPO_ROOT / "docs" / "schemas" / "detections.v1.schema.json").read_text(encoding="utf-8")
)
BASE_URL = "http://127.0.0.1:16080"
MAX_UPLOAD_MB = 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "model": str(TINY_MODEL),
        "model_dir": tmp_path / "models",
        "data_dir": tmp_path / "data",
        "max_upload_mb": MAX_UPLOAD_MB,
        "analysis_ttl_min": 1,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _app(settings: Settings, clock: FakeClock | None = None):
    detector = OnnxDetector(TINY_MODEL, providers=["CPUExecutionProvider"])

    def factory() -> OnnxDetector:
        return detector

    return create_app(settings, factory, clock=clock or FakeClock())


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = _app(_settings(tmp_path))
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


def _image_files() -> dict[str, tuple[str, bytes, str]]:
    return {"file": (IMAGE.name, IMAGE.read_bytes(), "image/jpeg")}


def _wait_for(
    client: TestClient, job_id: str, timeout: float = 45.0
) -> tuple[dict[str, object], list[float | None]]:
    deadline = time.monotonic() + timeout
    fractions: list[float | None] = []
    while time.monotonic() < deadline:
        response = client.get(f"/api/analyses/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        fractions.append(body["progress"]["fraction"])
        if body["status"] in {"done", "error"}:
            return body, fractions
        time.sleep(0.01)
    raise AssertionError("analysis did not finish in time")


def test_health_and_model_info(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    model = client.get("/api/model")
    assert model.status_code == 200
    body = model.json()
    assert body["id"] == TINY_MODEL.stem
    assert body["input_size"] == 640
    assert body["classes"] == ["D00", "D10", "D20", "D40"]
    assert "path" not in body


def test_image_analysis_returns_inline_results_and_downloadables(client: TestClient) -> None:
    response = client.post("/api/analyses", files=_image_files())
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["kind"] == "image"
    assert created["status"] in {"queued", "running"}

    body, fractions = _wait_for(client, created["id"])
    assert body["status"] == "done", body
    measured = [value for value in fractions if value is not None]
    assert measured == sorted(measured)
    result = body["result"]
    assert result is not None
    document = result["document"]
    Draft202012Validator(SCHEMA).validate(document)
    assert document["source"]["kind"] == "image"
    assert document["source"]["path"] == IMAGE.name
    assert document["summary"]["counts"] == {"D00": 1, "D10": 0, "D20": 0, "D40": 1}

    report = client.get(f"/api/analyses/{created['id']}/report.json")
    assert report.status_code == 200
    Draft202012Validator(SCHEMA).validate(report.json())
    assert report.json()["source"]["path"] == IMAGE.name

    html = client.get(f"/api/analyses/{created['id']}/report.html")
    assert html.status_code == 200
    assert "D00" in html.text and "D40" in html.text
    assert "Longitudinal crack" in html.text

    annotated = client.get(f"/api/analyses/{created['id']}/annotated.png")
    assert annotated.status_code == 200
    assert annotated.headers["content-type"] == "image/png"
    assert annotated.content.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(annotated.content)) as image:
        image.load()
        assert image.width > 0 and image.height > 0

    assert client.get("/api/docs").status_code == 200
    index = client.get("/")
    assert index.status_code == 200
    assert "Road Damage Vision" in index.text


def test_delete_removes_artifacts_and_the_job(client: TestClient) -> None:
    response = client.post("/api/analyses", files=_image_files())
    job_id = response.json()["id"]
    _wait_for(client, job_id)
    directory = client.app.state.job_manager.work_dir / job_id
    assert directory.is_dir()

    assert client.delete(f"/api/analyses/{job_id}").status_code == 204
    assert not directory.exists()
    assert client.get(f"/api/analyses/{job_id}").status_code == 404
    assert client.get(f"/api/analyses/{job_id}/report.json").status_code == 404
    assert client.delete(f"/api/analyses/{job_id}").status_code == 404


def test_video_analysis_reports_monotone_progress(client: TestClient, tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")},
        data={"analysis_fps": "5"},
    )
    assert response.status_code == 202, response.text
    created = response.json()
    assert created["kind"] == "video"

    body, fractions = _wait_for(client, created["id"])
    assert body["status"] == "done", body
    measured = [value for value in fractions if value is not None]
    assert measured == sorted(measured)
    result = body["result"]
    assert result is not None
    document = result["document"]
    Draft202012Validator(SCHEMA).validate(document)
    assert document["source"]["kind"] == "video"
    assert document["summary"]["frames_analyzed"] == 20
    assert document["summary"]["unique_instances"]["D00"] == 1

    html = client.get(f"/api/analyses/{created['id']}/report.html")
    assert html.status_code == 200
    assert "Unique instances" in html.text
    assert client.get(f"/api/analyses/{created['id']}/annotated.png").status_code == 404


def test_ttl_sweep_removes_expired_analyses(tmp_path: Path) -> None:
    clock = FakeClock()
    app = _app(_settings(tmp_path), clock=clock)
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.post("/api/analyses", files=_image_files())
        job_id = response.json()["id"]
        _wait_for(client, job_id)
        directory = app.state.job_manager.work_dir / job_id
        assert directory.is_dir()

        clock.advance(61)
        assert app.state.job_manager.sweep_expired() == 1
        assert not directory.exists()
        assert client.get(f"/api/analyses/{job_id}").status_code == 404


def test_text_renamed_as_jpeg_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/analyses",
        files={"file": ("photo.jpg", b"this is not an image", "image/jpeg")},
    )
    assert response.status_code == 422


def test_missing_file_and_invalid_analysis_fps_are_rejected(client: TestClient) -> None:
    assert client.post("/api/analyses", files={}).status_code == 422
    response = client.post(
        "/api/analyses",
        files=_image_files(),
        data={"analysis_fps": "0"},
    )
    assert response.status_code == 422


def test_missing_model_reports_503_with_the_pull_hint(tmp_path: Path) -> None:
    def failing() -> OnnxDetector:
        raise ModelError("Model not found; run `rdv models pull`")

    app = create_app(_settings(tmp_path), failing)
    with TestClient(app, base_url=BASE_URL) as client:
        response = client.get("/api/model")
        assert response.status_code == 503
        assert "rdv models pull" in response.json()["detail"]
