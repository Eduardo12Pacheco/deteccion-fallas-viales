"""HTTP API for the local web app (S12/S14).

Endpoints follow the plan's web contract: health and model metadata, an
asynchronous analysis resource with polling, downloadable detections JSON,
self-contained HTML and (images only) the annotated PNG, plus deletion.
Uploads are content-sniffed before a job is created so a text file renamed
``.jpg`` fails synchronously with 422 instead of failing inside the worker.
"""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse

from rdv import composition
from rdv.config.settings import Settings
from rdv.entrypoints.web.jobs import AnalysisJob, JobKind, JobManager
from rdv.entrypoints.web.report import render_detections_report_html
from rdv.errors import MediaError, RdvError

router = APIRouter()

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_FILENAME = 200
_ACTIVE = frozenset({"queued", "running"})


def get_manager(request: Request) -> JobManager:
    """FastAPI dependency returning the app-wide job manager."""
    manager: JobManager = request.app.state.job_manager
    return manager


ManagerDep = Annotated[JobManager, Depends(get_manager)]


def sanitize_filename(raw: str | None) -> str:
    """Reduce a client filename to a safe display label (never a path).

    Directory components of POSIX and Windows paths are stripped, control
    characters are removed and the result is bounded. The returned value is
    only used in JSON/HTML output; uploads are stored under a fixed name.
    """
    candidate = (raw or "").replace("\\", "/")
    name = PurePosixPath(candidate).name.strip()
    cleaned = "".join(character for character in name if character.isprintable())
    return cleaned[:_MAX_FILENAME] or "upload"


def _session_max_bytes(settings: Settings) -> int:
    return settings.max_upload_mb * 1024 * 1024


def sniff_media_kind(path: Path) -> JobKind:
    """Return ``image`` or ``video`` by decoding the uploaded bytes.

    Raises:
        HTTPException: the content is neither a decodable image nor a
            decodable video (422), matching S14's MIME/decode validation.
    """
    try:
        composition.build_media_source().load_image(path)
    except MediaError:
        pass
    else:
        return "image"
    try:
        source = composition.build_video_source(path)
    except MediaError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unsupported or corrupt media file; upload an image or a video.",
        ) from exc
    try:
        return "video"
    finally:
        source.close()


def _validate_video_duration(path: Path, settings: Settings) -> None:
    source = composition.build_video_source(path)
    try:
        if source.frame_count is not None and source.fps > 0:
            duration = source.frame_count / source.fps
            if duration > settings.max_video_seconds:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=(
                        f"Video is longer than the {settings.max_video_seconds} s analysis limit."
                    ),
                )
    finally:
        source.close()


async def _save_upload(upload: UploadFile, path: Path) -> int:
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    return written


def _job_or_404(manager: JobManager, analysis_id: str) -> AnalysisJob:
    if not _JOB_ID.fullmatch(analysis_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    job = manager.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return job


def _finished_job(manager: JobManager, analysis_id: str) -> AnalysisJob:
    job = _job_or_404(manager, analysis_id)
    if job.status == "error":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=job.error or "Analysis failed",
        )
    if job.status != "done" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis is not finished yet",
        )
    return job


def _artifact(job: AnalysisJob, key: str) -> Path:
    name = job.result.get(key) if job.result else None
    if not isinstance(name, str):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    root = job.directory.resolve()
    path = (root / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return path


@router.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness probe; never requires the model to be downloaded."""
    from rdv import __version__

    return {"status": "ok", "service": "road-damage-vision", "version": __version__}


@router.get("/api/model")
def model_info(manager: ManagerDep) -> dict[str, Any]:
    """Identity and inference parameters of the model the worker will use."""
    try:
        info = manager.detector().info
    except RdvError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message
        ) from exc
    return {
        "id": info.id,
        "input_size": info.input_size,
        "score_threshold": info.score_threshold,
        "device": info.device,
        "sha256": info.sha256,
        "version": info.version,
        "license": info.license,
        "classes": list(info.classes),
    }


@router.post("/api/analyses", status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(
    request: Request,
    manager: ManagerDep,
    file: Annotated[UploadFile, File(description="Image or video to analyze.")],
    analysis_fps: Annotated[float | None, Form(description="Video sampling rate.")] = None,
) -> dict[str, Any]:
    """Accept one upload, validate it synchronously and queue the analysis."""
    settings: Settings = request.app.state.settings
    if analysis_fps is not None and (not math.isfinite(analysis_fps) or analysis_fps <= 0):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="analysis_fps must be positive and finite",
        )

    job = manager.create_job("image", sanitize_filename(file.filename))
    try:
        written = await _save_upload(file, job.source_path)
        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The uploaded file is empty",
            )
        if written > _session_max_bytes(settings):
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Upload exceeds the size limit",
            )
        kind = sniff_media_kind(job.source_path)
        if kind == "video":
            _validate_video_duration(job.source_path, settings)
    except HTTPException:
        manager.delete(job.id)
        raise
    except OSError as exc:
        manager.delete(job.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not store the upload",
        ) from exc
    job.kind = kind
    manager.enqueue(job, analysis_fps=analysis_fps)
    return {
        "id": job.id,
        "kind": job.kind,
        "filename": job.filename,
        "status": job.status,
    }


@router.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str, manager: ManagerDep) -> dict[str, Any]:
    """Return status, monotone progress and (once done) inline results."""
    return _job_or_404(manager, analysis_id).to_payload()


@router.get("/api/analyses/{analysis_id}/report.json")
def get_report_json(analysis_id: str, manager: ManagerDep) -> FileResponse:
    """Download the ``rdv.detections/v1`` document (S5 contract)."""
    job = _finished_job(manager, analysis_id)
    return FileResponse(_artifact(job, "report_json"), media_type="application/json")


@router.get("/api/analyses/{analysis_id}/report.html")
def get_report_html(analysis_id: str, manager: ManagerDep) -> HTMLResponse:
    """Download a self-contained HTML report."""
    job = _finished_job(manager, analysis_id)
    assert job.result is not None
    annotated: Path | None = None
    if job.kind == "image":
        try:
            annotated = _artifact(job, "annotated")
        except HTTPException:
            annotated = None
    markup = render_detections_report_html(job.result["document"], annotated_png=annotated)
    return HTMLResponse(markup)


@router.get("/api/analyses/{analysis_id}/annotated.png")
def get_annotated_png(analysis_id: str, manager: ManagerDep) -> FileResponse:
    """Download the rendered overlay for an image analysis."""
    job = _finished_job(manager, analysis_id)
    if job.kind != "image":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return FileResponse(_artifact(job, "annotated"), media_type="image/png")


@router.delete("/api/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: str, manager: ManagerDep) -> None:
    """Delete a finished analysis and every file it owns."""
    job = _job_or_404(manager, analysis_id)
    if job.status in _ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis is still running",
        )
    manager.delete(job.id)
