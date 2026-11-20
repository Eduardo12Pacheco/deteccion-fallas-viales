"""Capture the README screenshots and demo GIF from the real local app (S13).

The script starts ``rdv serve`` on a free port in the project's 16000-16999
range, drives Chromium through Playwright exactly like a user (theme toggle,
file upload, video timeline), and writes PNG/WebP screenshots plus a compact
GIF. It never commits anything by itself; ``--out`` decides where files go.

Example:
    uv run python scripts/capture_media.py \
        --out /tmp/rdv-media --model tests/fixtures/models/tiny-detector.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "src" / "rdv" / "assets" / "samples"
DEFAULT_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
DESKTOP = {"width": 1360, "height": 900}
MOBILE = {"width": 390, "height": 844}
GIF_WIDTH = 760
GIF_FRAMES = 7


def _free_port() -> int:
    for port in range(16000, 17000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 16000-16999")


def _wait_for_server(base_url: str, process: subprocess.Popen[bytes], log: Path) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline and process.poll() is None:
        try:
            response = httpx.get(f"{base_url}/api/health", timeout=0.5)
        except httpx.HTTPError:
            time.sleep(0.1)
            continue
        if response.status_code == 200:
            return
    raise RuntimeError(f"server did not become healthy; see {log}")


def _start_server(model: Path | None, tmp_root: Path) -> tuple[subprocess.Popen[bytes], str, Path]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log = tmp_root / "serve.log"
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_root / "tmp")
    (tmp_root / "tmp").mkdir()
    command = [
        sys.executable,
        "-m",
        "rdv",
        "serve",
        "--no-browser",
        "--port",
        str(port),
    ]
    if model is not None:
        command += ["--model", str(model)]
    handle = log.open("wb")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    process._rdv_log = handle  # type: ignore[attr-defined]
    _wait_for_server(base_url, process, log)
    return process, base_url, log


def _make_clip(directory: Path) -> Path:
    """Write a short moving-rectangle clip encoded as WebM VP9 when possible."""
    mp4 = directory / "capture.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the mp4 writer")
    try:
        for index in range(72):
            frame = np.full((360, 640, 3), (75, 82, 90), dtype=np.uint8)
            left = 40 + index * 6
            cv2.rectangle(frame, (left, 90), (left + 150, 290), (215, 220, 225), -1)
            cv2.line(frame, (left + 70, 90), (left + 95, 290), (120, 125, 130), 5)
            cv2.putText(
                frame,
                "RDD2022 sample clip",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (235, 238, 240),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        webm = directory / "capture.webm"
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-c:v",
                "libvpx-vp9",
                "-crf",
                "38",
                str(webm),
            ],
            check=False,
        )
        if result.returncode == 0 and webm.is_file():
            return webm
    return mp4


def _screenshot(page: Page, path: Path, *, full_page: bool = True) -> None:
    page.screenshot(path=str(path), full_page=full_page)


def _assert_painted(page: Page) -> None:
    painted = page.evaluate(
        """() => {
            const canvas = document.querySelector('[data-testid="overlay-canvas"]');
            if (!canvas || canvas.width === 0) return 0;
            const context = canvas.getContext('2d');
            const data = context.getImageData(0, 0, canvas.width, canvas.height).data;
            let count = 0;
            for (let index = 3; index < data.length; index += 4) {
                if (data[index] > 0) count += 1;
            }
            return count;
        }"""
    )
    if painted <= 0:
        raise RuntimeError("the overlay canvas is empty; the capture would be misleading")


def _upload(page: Page, file: Path) -> None:
    page.set_input_files("#file-input", str(file))
    page.wait_for_selector('[data-testid="result-view"]:not([hidden])', timeout=45_000)
    page.wait_for_selector('[data-testid="progress-bar"][aria-valuenow="100"]', timeout=45_000)
    page.wait_for_timeout(300)


def _gif_frame(page: Page) -> Image.Image:
    png = page.screenshot(full_page=False)
    image = Image.open(BytesIO(png)).convert("RGB")
    ratio = GIF_WIDTH / image.width
    return image.resize((GIF_WIDTH, round(image.height * ratio)), Image.Resampling.LANCZOS)


def _write_gif(frames: list[Image.Image], destination: Path) -> None:
    palette = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames]
    palette[0].save(
        destination,
        format="GIF",
        save_all=True,
        append_images=palette[1:],
        duration=900,
        loop=0,
        optimize=True,
        disposal=2,
    )


def capture(out_dir: Path, model: Path | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="rdv-media-"))
    process, base_url, log = _start_server(model, tmp_root)
    samples = sorted(SAMPLES.glob("*.jpg")) or sorted(SAMPLES.glob("*.png"))
    if not samples:
        raise RuntimeError("no packaged samples found under src/rdv/assets/samples")
    sample = samples[0]
    clip = _make_clip(tmp_root)
    written: dict[str, Any] = {"base_url": base_url, "sample": sample.name, "files": []}
    frames: list[Image.Image] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(
                viewport=DESKTOP,
                color_scheme="light",
                reduced_motion="reduce",
            )
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"{base_url}/", wait_until="networkidle")
            page.wait_for_function(
                "() => document.querySelector('#model-name')?.textContent !== 'loading model…'",
                timeout=30_000,
            )
            page.wait_for_timeout(300)

            _screenshot(page, out_dir / "home-light.png")
            frames.append(_gif_frame(page))

            page.locator('[data-testid="theme-toggle"]').click()
            page.wait_for_timeout(200)
            _screenshot(page, out_dir / "home-dark.png")
            page.locator('[data-testid="theme-toggle"]').click()
            page.wait_for_timeout(200)

            _upload(page, sample)
            _assert_painted(page)
            _screenshot(page, out_dir / "image-result.png")
            frames.append(_gif_frame(page))

            page.locator('[data-testid="theme-toggle"]').click()
            page.wait_for_timeout(200)
            frames.append(_gif_frame(page))
            page.locator('[data-testid="legend-D00"]').click()
            page.wait_for_timeout(200)
            frames.append(_gif_frame(page))
            page.locator('[data-testid="legend-D00"]').click()
            page.locator('[data-testid="theme-toggle"]').click()
            page.wait_for_timeout(200)

            _upload(page, clip)
            try:
                page.wait_for_function(
                    "() => document.querySelector('#video-viewer')?.readyState >= 2",
                    timeout=8_000,
                )
                page.evaluate(
                    """() => {
                        const video = document.querySelector('#video-viewer');
                        if (video && Number.isFinite(video.duration)) {
                            video.currentTime = Math.min(video.duration - 0.1, 1.0);
                        }
                    }"""
                )
                page.wait_for_timeout(400)
            except Exception:  # pragma: no cover - codec-dependent cosmetic path
                pass
            frames.append(_gif_frame(page))
            _screenshot(page, out_dir / "video-result.png")

            mobile_context = browser.new_context(
                viewport=MOBILE,
                color_scheme="light",
                reduced_motion="reduce",
            )
            mobile = mobile_context.new_page()
            mobile.goto(f"{base_url}/", wait_until="networkidle")
            mobile.wait_for_function(
                "() => document.querySelector('#model-name')?.textContent !== 'loading model…'",
                timeout=30_000,
            )
            _upload(mobile, sample)
            _screenshot(mobile, out_dir / "mobile.png")
            frames.append(_gif_frame(mobile))
            mobile_context.close()

            while len(frames) < GIF_FRAMES:
                frames.append(frames[-1])
            gif_path = out_dir / "demo.gif"
            _write_gif(frames[:GIF_FRAMES], gif_path)
            if gif_path.stat().st_size > 8 * 1024 * 1024:
                _write_gif(frames[:4], gif_path)
            if errors:
                raise RuntimeError(f"browser page errors during capture: {errors}")
            context.close()
            browser.close()

        written["files"] = [
            str(path.relative_to(out_dir)) for path in sorted(out_dir.glob("*")) if path.is_file()
        ]
        written["gif_bytes"] = (out_dir / "demo.gif").stat().st_size
        return written
    finally:
        process.terminate()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            process.communicate(timeout=5)
        handle = getattr(process, "_rdv_log", None)
        if handle is not None:
            handle.close()
        shutil.rmtree(tmp_root, ignore_errors=True)
        if log.exists() and log.stat().st_size == 0:
            log.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture README screenshots and the demo GIF from the local web app.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory for PNG/GIF files.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL if DEFAULT_MODEL.is_file() else None,
        help="ONNX model path passed to rdv serve (default: tiny fixture model).",
    )
    args = parser.parse_args()
    result = capture(args.out.resolve(), args.model.resolve() if args.model else None)
    print(f"server: {result['base_url']}")
    print(f"sample: {result['sample']}")
    for name in result["files"]:
        print(f"wrote {args.out.resolve() / name}")
    print(f"gif bytes: {result['gif_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
