"""Unit tests for the web analysis job manager (S12/S14)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from rdv.application.events import ProgressEvent
from rdv.config.settings import Settings
from rdv.entrypoints.web.jobs import AnalysisJob, JobManager
from rdv.errors import MediaError


class FakeClock:
    """Deterministic monotonic clock for TTL tests."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Detector:
    class info:
        id = "fake"
        input_size = 640
        score_threshold = 0.25
        device = "cpu"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        model="fake",
        model_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        analysis_ttl_min=1,
    )


def _manager(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    processor: Any = None,
    work_dir: Path | None = None,
) -> JobManager:
    return JobManager(
        _settings(tmp_path),
        lambda: _Detector(),  # type: ignore[arg-type,return-value]
        work_dir=work_dir if work_dir is not None else tmp_path / "web",
        clock=clock or FakeClock(),
        processor=processor,
    )


def _wait_status(manager: JobManager, job_id: str, wanted: set[str], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.status in wanted:
            return
        time.sleep(0.005)
    raise AssertionError(f"job never reached {wanted}: {manager.get(job_id)}")


def test_job_progress_is_reported_and_result_is_exposed(tmp_path: Path) -> None:
    gate = threading.Event()

    def processor(job: AnalysisJob, detector: Any, progress: Any) -> dict[str, Any]:
        progress.emit(ProgressEvent(stage="detect", current=1, total=2, message="half"))
        gate.wait(timeout=5.0)
        progress.emit(ProgressEvent(stage="detect", current=2, total=2, message="done"))
        return {"document": {"schema": "rdv.detections/v1"}}

    manager = _manager(tmp_path, processor=processor)
    try:
        job = manager.create_job("image", "photo.jpg")
        manager.enqueue(job)

        _wait_status(manager, job.id, {"running"})
        running = manager.get(job.id)
        assert running is not None
        assert running.progress["fraction"] == pytest.approx(0.5)
        assert running.progress["message"] == "half"

        gate.set()
        _wait_status(manager, job.id, {"done"})
        done = manager.get(job.id)
        assert done is not None
        payload = done.to_payload()
        assert payload["status"] == "done"
        assert payload["result"] == {"document": {"schema": "rdv.detections/v1"}}
        assert payload["progress"]["fraction"] == pytest.approx(1.0)
    finally:
        manager.close()


def test_processor_failure_marks_the_job_as_error_and_hides_the_temp_path(tmp_path: Path) -> None:
    def processor(job: AnalysisJob, detector: Any, progress: Any) -> dict[str, Any]:
        raise MediaError(f"cannot read {job.directory / 'source'}")

    manager = _manager(tmp_path, processor=processor)
    try:
        job = manager.create_job("image", "photo.jpg")
        manager.enqueue(job)
        _wait_status(manager, job.id, {"error"})
        failed = manager.get(job.id)
        assert failed is not None
        assert failed.error is not None
        assert "<upload>/source" in failed.error
        assert str(manager.work_dir) not in failed.error
    finally:
        manager.close()


def test_delete_removes_the_job_directory_and_lookup(tmp_path: Path) -> None:
    work = tmp_path / "web"
    manager = _manager(tmp_path, work_dir=work, processor=lambda *_: {"ok": True})
    try:
        job = manager.create_job("image", "photo.jpg")
        manager.enqueue(job)
        _wait_status(manager, job.id, {"done"})
        directory = job.directory
        assert directory.is_dir()

        assert manager.delete(job.id) is True
        assert manager.get(job.id) is None
        assert not directory.exists()
        assert manager.delete(job.id) is False
    finally:
        manager.close()


def test_ttl_sweep_removes_expired_jobs_only(tmp_path: Path) -> None:
    clock = FakeClock()
    manager = _manager(tmp_path, clock=clock, processor=lambda *_: {"ok": True})
    try:
        old = manager.create_job("image", "old.jpg")
        manager.enqueue(old)
        _wait_status(manager, old.id, {"done"})
        clock.advance(30)
        fresh = manager.create_job("image", "fresh.jpg")
        manager.enqueue(fresh)
        _wait_status(manager, fresh.id, {"done"})

        clock.advance(31)
        assert manager.sweep_expired() == 1
        assert manager.get(old.id) is None
        assert not old.directory.exists()
        assert manager.get(fresh.id) is not None
    finally:
        manager.close()


def test_close_removes_the_process_temp_directory(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    root = manager.work_dir
    job = manager.create_job("image", "photo.jpg")
    assert job.directory.is_dir()
    assert root.is_dir()

    manager.close()
    manager.close()

    assert not root.exists()
