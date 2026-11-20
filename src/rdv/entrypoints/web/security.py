"""Security middleware for the local server (S14).

The web app is a loopback-only tool, not a multi-tenant service. Two ASGI
middlewares enforce the contract before FastAPI sees a request:

* :class:`LocalAccessMiddleware` rejects foreign ``Host`` headers (DNS
  rebinding protection) with 421 and cross-origin state-changing requests with
  403. No CORS headers are ever added.
* :class:`UploadSizeLimitMiddleware` enforces the upload limit from the
  ``Content-Length`` header and, for chunked uploads, while the body is being
  streamed, returning 413 before the multipart parser reads the whole file.
"""

from __future__ import annotations

import tempfile
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOCAL_HOST_NAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
STATE_CHANGING_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class BodyTooLarge(Exception):
    """Internal signal raised from ``receive`` when the upload exceeds its cap."""


def parse_host_header(value: str) -> tuple[str, int | None] | None:
    """Split a ``Host`` header into ``(hostname, port)``.

    Handles bracketed IPv6 literals and optional ports. Returns ``None`` for
    malformed values so callers can reject the request.
    """
    candidate = value.strip().lower()
    if not candidate:
        return None
    if candidate.startswith("["):
        end = candidate.find("]")
        if end < 0:
            return None
        host = candidate[1:end]
        rest = candidate[end + 1 :]
        if not host:
            return None
        if not rest:
            return host, None
        if not rest.startswith(":"):
            return None
        port_text = rest[1:]
        if not port_text.isdigit():
            return None
        return host, int(port_text)
    if candidate.count(":") >= 2:
        # Unbracketed IPv6 literal (technically invalid in Host, but harmless
        # to recognize as the loopback address).
        return candidate, None
    host, separator, port_text = candidate.rpartition(":")
    if separator and port_text.isdigit():
        if not host:
            return None
        return host, int(port_text)
    if not candidate:
        return None
    return candidate, None


def is_local_host(value: str) -> bool:
    """True when the ``Host`` header names a loopback address."""
    parsed = parse_host_header(value)
    return parsed is not None and parsed[0] in LOCAL_HOST_NAMES


def origin_matches_host(origin: str, host_header: str) -> bool:
    """True when an ``Origin`` header refers to the same host and port."""
    parsed_origin = urlsplit(origin)
    if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.hostname:
        return False
    parsed_host = parse_host_header(host_header)
    if parsed_host is None:
        return False
    host_name, host_port = parsed_host
    try:
        origin_port = parsed_origin.port
    except ValueError:
        return False
    default_port = 443 if parsed_origin.scheme == "https" else 80
    return parsed_origin.hostname.lower() == host_name and (
        origin_port if origin_port is not None else default_port
    ) == (host_port if host_port is not None else default_port)


async def _reject(
    scope: Scope, receive: Receive, send: Send, status_code: int, detail: str
) -> None:
    response = JSONResponse({"detail": detail}, status_code=status_code)
    await response(scope, receive, send)


class LocalAccessMiddleware:
    """Reject non-local ``Host`` headers and cross-origin writes (S14)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "")
        if not is_local_host(host):
            await _reject(
                scope,
                receive,
                send,
                421,
                "Host header must be localhost, 127.0.0.1 or [::1]",
            )
            return
        method = str(scope.get("method", "GET")).upper()
        origin = headers.get("origin")
        if (
            method in STATE_CHANGING_METHODS
            and origin is not None
            and not origin_matches_host(origin, host)
        ):
            await _reject(scope, receive, send, 403, "Cross-origin request rejected")
            return
        await self.app(scope, receive, send)


class UploadSizeLimitMiddleware:
    """Enforce ``RDV_MAX_UPLOAD_MB`` before the multipart parser runs (S14).

    The body is spooled to disk while it is counted, so a chunked upload is
    cut off at the limit without buffering hundreds of megabytes in RAM. The
    inner app then receives a replay of the accepted bytes; FastAPI still sees
    a normal body and can report its own 422 for malformed multipart data.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int, path: str = "/api/analyses") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") != self.path
            or str(scope.get("method", "")).upper() != "POST"
        ):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await _reject(scope, receive, send, 413, "Upload exceeds the size limit")
                    return
            except ValueError:
                pass

        with tempfile.SpooledTemporaryFile(max_size=64 * 1024, mode="w+b") as spool:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body = message.get("body", b"")
                total += len(body)
                if total > self.max_bytes:
                    await _reject(scope, receive, send, 413, "Upload exceeds the size limit")
                    return
                spool.write(body)
                if not message.get("more_body", False):
                    break
            spool.seek(0)

            async def replay() -> Message:
                chunk = spool.read(64 * 1024)
                if chunk:
                    return {"type": "http.request", "body": chunk, "more_body": True}
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.app(scope, replay, send)
