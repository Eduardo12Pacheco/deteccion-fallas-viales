"""PyInstaller onedir spec for RoadDamageVision (S15, decision D12).

The build script passes the model to embed through ``RDV_DESKTOP_MODEL`` and
points ``--distpath`` at the requested output directory. Only the torch-free
base stack is collected: evaluation-only and training-only packages are
excluded explicitly, which keeps the tiny-model bundle well under 250 MB.
"""

import importlib.util
import os
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"
ASSETS = SRC / "rdv" / "assets"
STATIC = SRC / "rdv" / "entrypoints" / "web" / "static"

_model_env = os.environ.get("RDV_DESKTOP_MODEL")
if not _model_env:
    raise SystemExit("RDV_DESKTOP_MODEL must point to the ONNX model to embed")
MODEL = Path(_model_env).resolve()
if not MODEL.is_file() or MODEL.suffix != ".onnx":
    raise SystemExit(f"RDV_DESKTOP_MODEL is not an ONNX file: {MODEL}")

datas = [
    (str(ASSETS / "models.json"), "rdv/assets"),
    (str(ASSETS / "samples"), "rdv/assets/samples"),
    (str(ASSETS / "schemas"), "rdv/assets/schemas"),
    (str(STATIC), "rdv/entrypoints/web/static"),
    (str(MODEL), "rdv/assets/models"),
]

_dynamic_imports = [
    # uvicorn resolves its loop, HTTP and lifespan implementations at runtime.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # FastAPI imports the multipart parser lazily for form uploads.
    "python_multipart",
    "python_multipart.multipart",
    "anyio._backends._asyncio",
]
hiddenimports = [name for name in _dynamic_imports if importlib.util.find_spec(name) is not None]

excludes = [
    "torch",
    "torchvision",
    "transformers",
    "scipy",
    "matplotlib",
    "tkinter",
    "IPython",
    "pytest",
    "hypothesis",
    "pycocotools",
    "onnx",
    "cv2",
    "pip",
    "setuptools",
]

analysis = Analysis(
    [str(ROOT / "packaging" / "rdv_desktop.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="RoadDamageVision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=True,
    upx=True,
    name="RoadDamageVision",
)
