"""Build the portable PyInstaller desktop bundle (S15, decision D12).

Usage:
    uv sync --group build
    uv run python scripts/build_desktop.py --model tests/fixtures/models/tiny-detector.onnx \\
        --out dist/desktop

The script runs PyInstaller in a temporary work directory, verifies that the
onedir bundle was produced, and archives it with the D15 platform naming
convention (``RoadDamageVision-<version>-<platform>``). The bundle directory is
kept next to the archive so ``scripts/smoke_bundle.py`` can test exactly what
was built.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from rdv import __version__

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "rdv.spec"
BUNDLE_NAME = "RoadDamageVision"


def platform_tag() -> str:
    """Return the D15 platform label (linux-x64, windows-x64, macos-arm64)."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    os_name = {"linux": "linux", "windows": "windows", "darwin": "macos"}.get(system, system)
    arch = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    return f"{os_name}-{arch}"


def archive_bundle(bundle: Path, destination: Path) -> Path:
    """Archive an onedir bundle as ``.zip`` on Windows and ``.tar.gz`` elsewhere."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".zip":
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(bundle.rglob("*")):
                archive.write(path, path.relative_to(bundle.parent))
    else:
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(bundle, arcname=bundle.name, recursive=True)
    return destination


def build_bundle(model: Path, out: Path) -> Path:
    """Run PyInstaller for ``model`` and return the onedir output directory."""
    model = model.resolve()
    if not model.is_file() or model.suffix != ".onnx":
        raise SystemExit(f"Model must be an existing ONNX file: {model}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        import PyInstaller  # noqa: F401
    except ImportError as exc:  # pragma: no cover - checked by acceptance command
        raise SystemExit(
            "PyInstaller is not installed; run `uv sync --group build` first."
        ) from exc

    work_dir = Path(tempfile.mkdtemp(prefix="rdv-desktop-build-"))
    env = dict(os.environ)
    env["RDV_DESKTOP_MODEL"] = str(model)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--distpath",
        str(out),
        "--workpath",
        str(work_dir),
        "--noconfirm",
        "--clean",
    ]
    upx = os.environ.get("RDV_UPX_DIR") or shutil.which("upx")
    if upx:
        upx_path = Path(upx)
        command.extend(["--upx-dir", str(upx_path if upx_path.is_dir() else upx_path.parent)])
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    bundle = out / BUNDLE_NAME
    executable = bundle / (f"{BUNDLE_NAME}.exe" if sys.platform == "win32" else BUNDLE_NAME)
    if not executable.is_file():
        raise SystemExit(f"PyInstaller did not produce the expected executable: {executable}")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Road Damage Vision desktop bundle.")
    parser.add_argument("--model", type=Path, required=True, help="ONNX model to embed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dist" / "desktop",
        help="Output directory (default: dist/desktop).",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Keep only the onedir bundle, without the .tar.gz/.zip archive.",
    )
    args = parser.parse_args(argv)

    bundle = build_bundle(args.model, args.out)
    print(f"Built {bundle}")
    if not args.no_archive:
        suffix = ".zip" if sys.platform == "win32" else ".tar.gz"
        archive = args.out.resolve() / f"{BUNDLE_NAME}-{__version__}-{platform_tag()}{suffix}"
        archive_bundle(bundle, archive)
        print(f"Archived {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
