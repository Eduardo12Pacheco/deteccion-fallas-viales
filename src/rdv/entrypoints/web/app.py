"""FastAPI application factory for the local web UI (S12/S14).

``rdv serve`` and the tests build the app through :func:`create_app` so the
detector and job manager are injected (no import-time model loading) and the
lifespan can always delete the per-process temporary directory, even when the
process receives SIGINT/SIGTERM and uvicorn shuts down gracefully.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rdv import __version__
from rdv.config.settings import Settings
from rdv.entrypoints.web.api import router
from rdv.entrypoints.web.jobs import DetectorFactory, JobManager, Processor
from rdv.entrypoints.web.security import LocalAccessMiddleware, UploadSizeLimitMiddleware

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings,
    detector_factory: DetectorFactory,
    *,
    clock: Callable[[], float] = time.monotonic,
    processor: Processor | None = None,
    work_dir: Path | None = None,
) -> FastAPI:
    """Build the local web application.

    Args:
        settings: validated runtime configuration (bind address, limits, TTL).
        detector_factory: builds the Detector on first use; may raise
            :class:`rdv.errors.ModelError`, reported by ``/api/model`` as 503.
        clock: injectable monotonic clock used for TTL sweeps.
        processor: injectable job body for unit tests; defaults to the real
            image/video use cases.
        work_dir: explicit per-process temp root (tests); defaults to a
            ``mkdtemp`` directory under the system temporary directory.
    """
    manager = JobManager(
        settings,
        detector_factory,
        work_dir=work_dir,
        clock=clock,
        processor=processor,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            manager.close()

    application = FastAPI(
        title="Road Damage Vision",
        description="Local road-damage detection API.",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.job_manager = manager
    application.state.detector_factory = detector_factory

    # Last added is outermost: Host/Origin checks run before the body is read,
    # then the size limit cuts oversized uploads off before multipart parsing.
    application.add_middleware(
        UploadSizeLimitMiddleware,
        max_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    application.add_middleware(LocalAccessMiddleware)
    application.include_router(router)

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the placeholder page replaced by the full UI in T12."""
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


__all__ = ["create_app"]
