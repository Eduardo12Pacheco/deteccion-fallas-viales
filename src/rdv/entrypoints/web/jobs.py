"""Background analysis jobs for the local web app (S12/S14).

The web UI never runs inference on the request thread. A single worker thread
consumes a queue so a laptop GPU/CPU is not oversubscribed, updates an
in-memory job record from :class:`ProgressEvent` values (Observer pattern), and
lets the browser poll for status. Every job owns one subdirectory of a
per-process temporary root that is removed at shutdown, on deletion, or when
its time-to-live expires.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rdv import composition
from rdv.application.detect_image import detect_images
from rdv.application.detect_video import DEFAULT_ANALYSIS_FPS, detect_video
from rdv.application.events import ProgressEvent
from rdv.application.ports import Detector, ProgressSink
from rdv.config.settings import Settings
from rdv.errors import RdvError

logger = logging.getLogger(__name__)

JobKind = Literal["image", "video"]
JobStatus = Literal["queued", "running", "done", "error"]
Processor = Callable[["AnalysisJob", Detector, ProgressSink], dict[str, Any]]
DetectorFactory = Callable[[], Detector]

_ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})
_EMPTY_PROGRESS: dict[str, Any] = {
    "stage": "queued",
    "current": 0,
    "total": None,
    "fraction": None,
    "message": "",
}


def initial_progress() -> dict[str, Any]:
    """Return a fresh progress record for a queued analysis."""
    return dict(_EMPTY_PROGRESS)


@dataclass(slots=True)
class AnalysisJob:
    """One upload being analyzed by the web worker.

    Attributes:
        id: opaque hex identifier used in API paths.
        kind: ``image`` or ``video`` after content sniffing.
        filename: sanitized display name supplied by the client; never used
            to build a filesystem path.
        directory: private working directory owned by the job manager.
        analysis_fps: requested video sampling rate, if any.
        status: ``queued``, ``running``, ``done`` or ``error``.
        progress: last progress view-model reported by the use case.
        result: inline result payload once the job is done.
        error: user-facing failure message when the job failed.
        created_at: manager clock value at creation time.
        updated_at: manager clock value of the last state change.
    """

    id: str
    kind: JobKind
    filename: str
    directory: Path
    analysis_fps: float | None = None
    status: JobStatus = "queued"
    progress: dict[str, Any] = field(default_factory=initial_progress)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def source_path(self) -> Path:
        """Fixed upload path; a name chosen by the client is never used."""
        return self.directory / "source"

    @property
    def is_active(self) -> bool:
        """True while the job is queued or running."""
        return self.status in _ACTIVE_STATUSES

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-ready job document served by the API."""
        return {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "status": self.status,
            "progress": dict(self.progress),
            "result": self.result if self.status == "done" else None,
            "error": self.error,
        }


class _JobProgress(ProgressSink):
    """Progress sink that records the latest monotone progress on one job."""

    def __init__(self, manager: JobManager, job: AnalysisJob) -> None:
        self._manager = manager
        self._job = job

    def emit(self, event: ProgressEvent) -> None:
        """Store the event if it advances the completion counter."""
        with self._manager._lock:
            if self._job.status != "running":
                return
            previous = int(self._job.progress.get("current") or 0)
            if event.current < previous:
                return
            self._job.progress = {
                "stage": event.stage,
                "current": event.current,
                "total": event.total,
                "fraction": event.fraction,
                "message": event.message,
            }
            self._job.updated_at = self._manager._clock()


class JobManager:
    """Single-worker queue, in-memory job registry and temporary-file owner."""

    def __init__(
        self,
        settings: Settings,
        detector_factory: DetectorFactory,
        *,
        work_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        processor: Processor | None = None,
    ) -> None:
        self._settings = settings
        self._detector_factory = detector_factory
        self._clock = clock
        self._processor: Processor = processor if processor is not None else process_analysis
        self._work_dir = (
            Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="rdv-web-"))
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._detector_lock = threading.Lock()
        self._jobs: dict[str, AnalysisJob] = {}
        self._queue: queue.Queue[AnalysisJob | None] = queue.Queue()
        self._stop = threading.Event()
        self._closed = False
        self._detector: Detector | None = None
        self._worker = threading.Thread(target=self._run_worker, name="rdv-web-worker", daemon=True)
        self._worker.start()
        interval = max(1.0, self._settings.analysis_ttl_min * 60.0 / 4.0)
        self._sweeper = threading.Thread(
            target=self._run_sweeper, args=(interval,), name="rdv-web-sweeper", daemon=True
        )
        self._sweeper.start()

    @property
    def work_dir(self) -> Path:
        """Per-process temporary root containing one directory per job."""
        return self._work_dir

    @property
    def ttl_seconds(self) -> float:
        """Configured analysis lifetime in seconds."""
        return float(self._settings.analysis_ttl_min) * 60.0

    def detector(self) -> Detector:
        """Build the detector once and reuse it for every job."""
        with self._detector_lock:
            if self._detector is None:
                self._detector = self._detector_factory()
            return self._detector

    def create_job(self, kind: JobKind, filename: str) -> AnalysisJob:
        """Allocate a private directory and register a queued job."""
        job_id = uuid.uuid4().hex
        directory = self._work_dir / job_id
        directory.mkdir(parents=True, exist_ok=False)
        now = self._clock()
        job = AnalysisJob(
            id=job_id,
            kind=kind,
            filename=filename,
            directory=directory,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def enqueue(self, job: AnalysisJob, *, analysis_fps: float | None = None) -> None:
        """Mark the job queued and hand it to the single worker thread."""
        job.analysis_fps = analysis_fps
        job.status = "queued"
        with self._lock:
            job.updated_at = self._clock()
        self._queue.put(job)

    def get(self, job_id: str) -> AnalysisJob | None:
        """Return a job by id, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_payload(self, job_id: str) -> dict[str, Any] | None:
        """Return the JSON-ready document for a job, or ``None``."""
        job = self.get(job_id)
        return None if job is None else job.to_payload()

    def delete(self, job_id: str) -> bool:
        """Forget a finished job and delete its directory."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.directory, ignore_errors=True)
        return True

    def sweep_expired(self) -> int:
        """Delete finished jobs older than the configured TTL."""
        now = self._clock()
        with self._lock:
            expired = [
                job.id
                for job in self._jobs.values()
                if not job.is_active and now - job.created_at > self.ttl_seconds
            ]
        for job_id in expired:
            self.delete(job_id)
        return len(expired)

    def close(self) -> None:
        """Stop both threads and remove the whole temporary root.

        Safe to call more than once. A job that is still running is stopped by
        the worker sentinel only after the current processor returns; shutdown
        never leaves the process bound to a deleted directory.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=10.0)
        self._sweeper.join(timeout=5.0)
        shutil.rmtree(self._work_dir, ignore_errors=True)

    def _run_worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: AnalysisJob) -> None:
        with self._lock:
            job.status = "running"
            job.progress = {
                "stage": "starting",
                "current": 0,
                "total": None,
                "fraction": None,
                "message": "Preparing analysis",
            }
            job.updated_at = self._clock()
        try:
            detector = self.detector()
            result = self._processor(job, detector, _JobProgress(self, job))
        except RdvError as exc:
            self._fail(job, exc.message)
        except Exception:  # pragma: no cover - defensive boundary
            logger.exception("Unexpected web analysis failure")
            self._fail(job, "Unexpected error while analyzing the upload")
        else:
            with self._lock:
                job.status = "done"
                job.result = result
                job.error = None
                total = job.progress.get("total")
                job.progress = {
                    "stage": "done",
                    "current": int(total) if isinstance(total, int) else 1,
                    "total": int(total) if isinstance(total, int) else None,
                    "fraction": 1.0,
                    "message": "Analysis complete",
                }
                job.updated_at = self._clock()

    def _fail(self, job: AnalysisJob, message: str) -> None:
        sanitized = message.replace(str(job.directory), "<upload>")
        with self._lock:
            job.status = "error"
            job.result = None
            job.error = sanitized
            job.progress = {
                **job.progress,
                "stage": "error",
                "fraction": job.progress.get("fraction"),
                "message": sanitized,
            }
            job.updated_at = self._clock()

    def _run_sweeper(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.sweep_expired()


def _load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _sanitize_summary_paths(summary: dict[str, Any], filename: str) -> dict[str, Any]:
    """Replace private server paths in the inline summary with the upload name."""
    for key in ("images", "videos"):
        for entry in summary.get(key) or []:
            if isinstance(entry, dict) and "path" in entry:
                entry["path"] = filename
    return summary


def process_analysis(
    job: AnalysisJob, detector: Detector, progress: ProgressSink
) -> dict[str, Any]:
    """Run the real image or video use case for one uploaded job."""
    if job.kind == "image":
        result = detect_images(
            [job.source_path],
            detector=detector,
            media=composition.build_media_source(),
            renderer=composition.build_renderer(),
            detections_writer=composition.build_detections_writer(),
            summary_writer=composition.build_summary_writer(),
            output_dir=job.directory,
            progress=progress,
        )
        outcome = result.images[0]
        writer = composition.build_detections_writer()
        document = _load_json(outcome.detections_json)
        document["source"]["path"] = job.filename
        writer.write_detections(outcome.detections_json, document)
        summary = _sanitize_summary_paths(_load_json(result.summary_json), job.filename)
        return {
            "document": document,
            "summary": summary,
            "report_json": outcome.detections_json.name,
            "annotated": outcome.annotated.name,
            "annotated_media": "image",
        }

    source = composition.build_video_source(job.source_path)
    try:
        video_outcome = detect_video(
            source,
            detector=detector,
            renderer=composition.build_renderer(),
            sink=composition.build_video_sink(),
            detections_writer=composition.build_detections_writer(),
            summary_writer=composition.build_summary_writer(),
            output_dir=job.directory,
            analysis_fps=job.analysis_fps if job.analysis_fps is not None else DEFAULT_ANALYSIS_FPS,
            progress=progress,
        )
    finally:
        source.close()
    writer = composition.build_detections_writer()
    document = _load_json(video_outcome.detections_json)
    document["source"]["path"] = job.filename
    writer.write_detections(video_outcome.detections_json, document)
    summary = _sanitize_summary_paths(_load_json(video_outcome.summary_json), job.filename)
    return {
        "document": document,
        "summary": summary,
        "report_json": video_outcome.detections_json.name,
        "annotated": video_outcome.annotated.name,
        "annotated_media": "video",
    }
