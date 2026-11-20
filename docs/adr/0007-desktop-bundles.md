# 0007 - Desktop bundles: PyInstaller onedir launcher with an embedded model

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D12 and S15, amendment L5)
- **Task:** T13

## Context

S15 asks for portable desktop builds on GitHub Releases that start the same
local app and open the browser, without a second toolchain (Tauri) and without
the user installing Python. PyInstaller onedir was chosen in D12 because it
reuses the Python stack and avoids Linux webview dependencies. The bundle must
work offline, embed the ONNX model passed to `build_desktop.py`, keep the
torch/scipy/matplotlib training stack out, and stay under the 250 MB budget
when built with the tiny CI model.

Two environment constraints shaped the launcher: L5 requires every default
local port to live in 16000-16999 (so the historical 8765 became 16080), and
the uvicorn server must run on a worker thread while Ctrl+C is handled by the
launcher. PyInstaller 6.22 disables UPX on Linux, and the current
`opencv-python-headless` 5.x wheel contributes about 165 MB uncompressed
(79 MB `cv2.abi3.so` plus its codec libraries), which alone would exceed the
size budget together with Python, ONNX Runtime and numpy.

## Decision

- **Entry point and layout.** `packaging/rdv_desktop.py` calls
  `rdv.entrypoints.desktop.main`. `packaging/rdv.spec` builds an onedir
  `RoadDamageVision` executable with `console=True`, the web static assets,
  `models.json`, the samples, the JSON schemas and exactly one ONNX model
  under `rdv/assets/models/`. `scripts/build_desktop.py` passes the model via
  `RDV_DESKTOP_MODEL`, runs PyInstaller in a temporary work directory, and
  archives the result as `RoadDamageVision-<version>-<platform>.tar.gz`
  (`.zip` on Windows). `scripts/smoke_bundle.py` launches the built directory,
  verifies `/api/health`, uploads a sample extracted from the bundle, polls the
  analysis to `done`, then sends Ctrl+C and requires exit code 0.
- **Launcher.** `rdv.entrypoints.desktop` binds the first free port at or
  after `16080` (up to 50 attempts), starts uvicorn on a daemon thread, polls
  `/api/health` for up to 60 s, prints the URL and opens the browser unless
  `--no-browser` is passed. It also accepts `--host`, `--port` and `--model`.
  The model defaults to the embedded `assets/models/*.onnx`; `RDV_*` settings
  still apply, and a bundled model path bypasses any download.
- **Graceful shutdown.** The server thread is wrapped in a `ServerRunner`
  whose completion is signalled by a `threading.Event`. This is deliberate:
  on CPython, a `KeyboardInterrupt` raised inside `threading.Thread.join` can
  leave `is_alive()`/`join()` lying (the thread is reported dead while it is
  still running), which previously let the process exit before uvicorn ran the
  FastAPI lifespan shutdown and deleted its temporary analysis directory. The
  launcher now sets `server.should_exit`, waits on the event, and the smoke
  test asserts no `rdv-web-*` directory survives.
- **No OpenCV in the bundle.** The bundle excludes `cv2` to meet the hard
  250 MB gate; including it produced a 298 MB stripped bundle. Image analysis
  is unaffected (Pillow decodes, and `preprocess.resize_linear` mirrors
  `cv2.INTER_LINEAR` in numpy, unit-tested against OpenCV within one 8-bit
  level). Video uploads in the bundle return a 422 that explains the
  limitation instead of calling the file corrupt. The L1 CLI install still
  ships OpenCV and keeps full video support (S6/S12). This is a documented
  packaging trade-off, not a change to the web app or the CLI.
- **Excludes and size.** D12's excludes (`torch`, `torchvision`,
  `transformers`, `scipy`, `matplotlib`, `tkinter`) are enforced in the spec,
  plus development tools. The smoke test fails if any forbidden directory is
  collected or if the summed file size reaches 250 MiB; the tiny-model bundle
  is about 165 MiB by that measure.
- **Platforms and trust.** T15 builds Linux (`ubuntu-24.04`), Windows x64 and
  macOS arm64. The binaries are unsigned: macOS Gatekeeper and Windows
  SmartScreen require the user to allow the app, and the Linux build needs
  glibc 2.39+ because it is built on Ubuntu 24.04. These limits are stated in
  the release documentation rather than silently implied.

## Consequences

- The desktop app is a first-class image-analysis product with the real model
  embedded; video remains a CLI feature supported by the base install.
- Future OpenCV versions might be small enough to re-enable video in the
  bundle; the size check in `smoke_bundle.py` makes that a measured decision.
- `build_desktop.py` needs the pinned model file and the `build` dependency
  group (`uv sync --group build`); it never needs the train/cuda extras.
- A port race between `find_free_port` and uvicorn's bind is possible in
  principle; the launcher then fails its health wait with a clear message
  instead of hanging.
- The bundle writes only to the system temporary directory (analysis jobs and
  the uvicorn logs printed to stdout); it does not modify its install dir and
  works offline because no model download is attempted.
