"""Smoke-test a built RoadDamageVision desktop bundle (S15, decision D12).

The script validates the bundle layout (no torch/transformers/scipy/matplotlib
directories, under 250 MB with the tiny model), starts the bundle with
``--no-browser`` on a free port in the shared 16000-16999 range, waits for
``/api/health``, uploads a packaged sample, asserts the JSON analysis result,
and asks the launcher to shut down with Ctrl+C (SIGINT / CTRL_BREAK).

Usage:
    uv run python scripts/smoke_bundle.py dist/desktop/RoadDamageVision
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from rdv.entrypoints import desktop

FORBIDDEN_DIRECTORIES = frozenset({"torch", "torchvision", "transformers", "scipy", "matplotlib"})
MAX_BUNDLE_BYTES = 250 * 1024 * 1024


def _fail(message: str) -> None:
    raise SystemExit(f"smoke_bundle: {message}")


def scan_bundle(bundle: Path) -> int:
    """Assert D12's size/exclusion budget and return the bundle size in bytes."""
    if not bundle.is_dir():
        _fail(f"bundle directory not found: {bundle}")
    forbidden = sorted(
        {
            str(path.relative_to(bundle))
            for path in bundle.rglob("*")
            if path.is_dir() and path.name in FORBIDDEN_DIRECTORIES
        }
    )
    if forbidden:
        _fail("forbidden packages in bundle: " + ", ".join(forbidden))

    total = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    if total >= MAX_BUNDLE_BYTES:
        _fail(f"bundle is {total / (1024 * 1024):.1f} MiB, budget is < 250 MiB")

    if not any(path.name == "models.json" for path in bundle.rglob("models.json")):
        _fail("bundle does not contain the packaged models.json manifest")
    return total


def find_packaged_sample(bundle: Path) -> Path:
    """Return a sample JPEG extracted inside the bundle (proves datas shipped)."""
    samples = sorted(
        path for path in bundle.rglob("*.jpg") if path.parent.name == "samples" and path.is_file()
    )
    if not samples:
        _fail("bundle does not contain the packaged sample images")
    return samples[0]


def _executable(bundle: Path) -> Path:
    name = "RoadDamageVision.exe" if sys.platform == "win32" else "RoadDamageVision"
    executable = bundle / name
    if not executable.is_file():
        _fail(f"missing executable: {executable}")
    return executable


def _spawn(
    executable: Path,
    port: int,
    log_path: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    log = log_path.open("wb")
    try:
        return subprocess.Popen(
            [str(executable), "--no-browser", "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
    finally:
        log.close()


def _stop(process: subprocess.Popen[bytes], timeout_s: float) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        if sys.platform == "win32":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
    except OSError:  # pragma: no cover - process raced with shutdown
        process.terminate()
    try:
        return int(process.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        _fail("bundle did not shut down after Ctrl+C")
    return 1  # pragma: no cover - _fail always raises


def _tail(path: Path, lines: int = 30) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _poll_analysis(base_url: str, analysis_id: str, timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = httpx.get(f"{base_url}/api/analyses/{analysis_id}", timeout=5.0, trust_env=False)
        response.raise_for_status()
        last = response.json()
        status = last.get("status")
        if status == "done":
            return last
        if status == "error":
            _fail(f"analysis failed: {last.get('error')}")
        time.sleep(0.2)
    _fail(f"analysis {analysis_id} did not finish in {timeout_s:.0f} s")


def run_smoke(bundle: Path, *, timeout_s: float = 90.0, analysis_timeout_s: float = 120.0) -> None:
    """Run the complete bundle smoke test, raising ``SystemExit`` on failure."""
    bundle = bundle.resolve()
    size = scan_bundle(bundle)
    sample = find_packaged_sample(bundle)
    executable = _executable(bundle)
    port = desktop.find_free_port(desktop.DEFAULT_PORT, "127.0.0.1")
    base_url = f"http://127.0.0.1:{port}"

    with tempfile.TemporaryDirectory(prefix="rdv-bundle-smoke-") as temp:
        temp_root = Path(temp)
        log_path = temp_root / "bundle.log"
        env = dict(os.environ)
        for variable in ("TMPDIR", "TMP", "TEMP"):
            env[variable] = str(temp_root)
        env.pop("RDV_MODEL", None)
        bucket = temp_root / "models"
        env["RDV_MODEL_DIR"] = str(bucket)

        process = _spawn(executable, port, log_path, env)
        try:
            healthy = desktop.wait_for_health(
                f"{base_url}/api/health",
                timeout_s=timeout_s,
                probe=desktop.http_health_probe,
            )
            if not healthy:
                _fail("bundle never became healthy\n" + _tail(log_path))

            response = httpx.post(
                f"{base_url}/api/analyses",
                files={"file": (sample.name, sample.read_bytes(), "image/jpeg")},
                timeout=30.0,
                trust_env=False,
            )
            if response.status_code != 202:
                _fail(f"upload rejected with {response.status_code}: {response.text}")
            analysis_id = str(response.json()["id"])
            completed = _poll_analysis(base_url, analysis_id, analysis_timeout_s)

            result = completed.get("result")
            if not isinstance(result, dict):
                _fail("analysis finished without a JSON result")
            document = result.get("document")
            if not isinstance(document, dict) or document.get("schema") != "rdv.detections/v1":
                _fail("analysis result is not an rdv.detections/v1 document")
            if not document.get("frames"):
                _fail("analysis result has no frames")
        finally:
            code = _stop(process, 30.0)

        if code != 0:
            _fail(f"bundle exited with code {code} after Ctrl+C\n" + _tail(log_path))
        leftovers = sorted(temp_root.glob("rdv-web-*"))
        if leftovers:
            _fail("bundle did not remove its temporary analysis directories")

    print(
        f"OK: {bundle.name} ({size / (1024 * 1024):.1f} MiB) served "
        f"{sample.name}, analyzed it and shut down cleanly on {base_url}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a built desktop bundle.")
    parser.add_argument("bundle", type=Path, help="Path to the RoadDamageVision onedir bundle.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Health timeout in seconds.")
    parser.add_argument(
        "--analysis-timeout",
        type=float,
        default=120.0,
        help="Analysis timeout in seconds.",
    )
    args = parser.parse_args(argv)
    run_smoke(
        args.bundle,
        timeout_s=args.timeout,
        analysis_timeout_s=args.analysis_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
