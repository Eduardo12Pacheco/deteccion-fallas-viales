"""Fixtures for the Playwright UI tests (S12/S13).

A real ``rdv serve`` process bound to an ephemeral port in the project's
16000-16999 range is started once per session, using the tiny ONNX fixture
model so no GPU or model download is needed.
"""

from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
IMAGE = sorted((MINI / "Japan" / "train" / "images").glob("*.jpg"))[0]
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


def _free_local_port() -> int:
    for port in range(16000, 17000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 16000-16999")


@pytest.fixture(scope="session")
def app_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("ui-server")
    port = _free_local_port()
    log = root / "server.log"
    env = dict(os.environ)
    env["TMPDIR"] = str(root / "tmp")
    env["RDV_LOG_LEVEL"] = "WARNING"
    (root / "tmp").mkdir()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rdv",
                "serve",
                "--no-browser",
                "--port",
                str(port),
                "--model",
                str(TINY_MODEL),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    response = httpx.get(f"{base}/api/health", timeout=0.5)
                except httpx.HTTPError:
                    time.sleep(0.1)
                    continue
                if response.status_code == 200:
                    break
            else:
                raise RuntimeError(f"server did not start:\n{log.read_text(encoding='utf-8')}")
            yield base
        finally:
            process.terminate()
            try:
                process.communicate(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()
                process.communicate(timeout=5)


def open_app(page: Page, app_url: str) -> None:
    """Open the viewer and wait until the model chip is populated."""
    page.goto(f"{app_url}/", wait_until="networkidle")
    expect(page.locator("#model-name")).not_to_have_text("loading model…")


def drop_file(page: Page, path: Path) -> None:
    """Dispatch a real ``drop`` event with a File built in the browser."""
    mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    page.evaluate(
        """({ name, mime, payload }) => {
            const bytes = Uint8Array.from(atob(payload), (char) => char.charCodeAt(0));
            const file = new File([bytes], name, { type: mime });
            const transfer = new DataTransfer();
            transfer.items.add(file);
            const zone = document.querySelector('[data-testid="dropzone"]');
            zone.dispatchEvent(
              new DragEvent('drop', {
                bubbles: true,
                cancelable: true,
                dataTransfer: transfer,
              }),
            );
        }""",
        {"name": path.name, "mime": mime, "payload": payload},
    )


def wait_for_result(page: Page, timeout: float = 45_000) -> None:
    """Wait until an analysis finished and the result view is visible."""
    expect(page.locator('[data-testid="result-view"]')).to_be_visible(timeout=timeout)
    expect(page.locator('[data-testid="progress-bar"]')).to_have_attribute(
        "aria-valuenow", "100", timeout=timeout
    )


def overlay_alpha_count(page: Page) -> int:
    """Count non-transparent pixels in the overlay canvas."""
    return page.evaluate(
        """() => {
            const canvas = document.querySelector('[data-testid="overlay-canvas"]');
            if (!canvas || canvas.width === 0) return 0;
            const context = canvas.getContext('2d');
            const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
            let painted = 0;
            for (let index = 3; index < data.length; index += 4) {
                if (data[index] > 0) painted += 1;
            }
            return painted;
        }"""
    )


@pytest.fixture
def fixture_image() -> Path:
    return IMAGE
