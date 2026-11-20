# 0006 - Local web UI: FastAPI, polling jobs and loopback-only security

- **Status:** Accepted
- **Date:** 2026-11-20
- **Deciders:** project owner (plan decisions D11, D19, S12 and S14)
- **Task:** T11

## Context

`rdv serve` must expose the detector through a finished local browser app
without asking the user to install Node, a database or a cloud service. The
same server ships inside the desktop bundles (D12), so it must start in a few
hundred milliseconds, never load the torch stack, and leave no files behind.
The tool also runs on a shared laptop where several portfolio projects start
local servers in parallel, which is why L5 restricts default ports to the
16000-16999 range (`16080` by default) and tests to that range or port 0.

Inference is CPU/GPU-bound and must not run on the request thread. A browser
polling loop is simpler and more robust than SSE/WebSockets for a single-user
local app, and the detections document is already plain JSON.

## Decision

- **Stack.** FastAPI + uvicorn serving JSON plus a static ES-module page. The
  plan's server-side contract is implemented in `entrypoints/web`: `app.py`
  (factory + lifespan), `api.py` (routes), `jobs.py` (worker, queue, TTL) and
  `security.py` (ASGI middlewares). No Node build, no frontend framework, no
  database.
- **Job model.** `POST /api/analyses` validates the upload synchronously, then
  creates a job in a per-process `mkdtemp` root and enqueues it. One daemon
  worker thread per process consumes the queue, so concurrent requests cannot
  oversubscribe the GPU/CPU. Progress is an Observer over the existing
  `ProgressSink` events and is stored as `{stage, current, total, fraction,
  message}`; counters never move backwards. `GET /api/analyses/{id}` returns
  status, progress and (once done) the inline `rdv.detections/v1` document and
  summary, which the browser renders without another request.
- **Lifecycle.** The TTL comes from `RDV_ANALYSIS_TTL_MIN` (default 30). A
  background sweeper removes finished jobs, `DELETE` removes one job
  immediately, and the lifespan shutdown removes the whole temporary root.
  Jobs that are queued or running are not swept.
- **Uploads.** Files are stored under a fixed `source` name inside the job
  directory; the client filename is sanitized and only used as a display
  label. Content is sniffed by decoding with the real adapters: images via
  Pillow and videos via OpenCV. Anything that decodes as neither is 422.
- **Security (S14).** A `LocalAccessMiddleware` rejects any `Host` other than
  `localhost`, `127.0.0.1` or `[::1]` with 421 (DNS-rebinding protection) and
  any state-changing request whose `Origin` does not match the `Host` with
  403. `UploadSizeLimitMiddleware` enforces `RDV_MAX_UPLOAD_MB` from
  `Content-Length` and, for chunked requests, while streaming the body to a
  spooled temporary file; oversized requests get 413 before the multipart
  parser sees the data. No CORS middleware or headers are added. Path
  parameters are validated against a hex job-id pattern, so request paths
  never reach the filesystem.
- **Downloads.** `report.json` is the stored detections document with the
  upload name substituted for the private server path. `report.html` is
  self-contained (inline CSS and a base64 annotated still for images).
  `annotated.png` exists for image jobs; the video overlay is drawn client-side
  from the JSON in the frontend so the server never transcodes.
- **CLI.** `rdv serve [--host] [--port] [--model] [--no-browser]` builds the
  detector before binding (a missing model fails fast with exit 4 and the
  `rdv models pull` hint), then runs uvicorn and opens the browser unless
  `--no-browser` is passed. The default is `127.0.0.1:16080`.

## Consequences

- A slow analysis keeps its HTTP requests short; the UI polls at 500 ms.
- The single worker serializes CPU/GPU use, which is deliberate for a laptop;
  a queue of analyses waits instead of contesting memory.
- Spooling chunked uploads costs one extra copy on disk but bounds memory and
  guarantees 413 even when `Content-Length` is absent.
- Only loopback access works by default. Binding another address is possible
  with `--host`, but S14's Host check still accepts only loopback names, so
  remote access is intentionally unsupported.
- The static page shipped in T11 is a placeholder; T12 replaces it with the
  full viewer and Playwright-tested UI states. The API contract is frozen
  here so the frontend can build against it.
