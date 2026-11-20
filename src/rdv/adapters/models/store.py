"""Cached model store: resolve, resumable download, sha256 verify (D13).

Downloads write to ``<name>.part`` next to the final file and are renamed
atomically only after the size and sha256 pinned in the manifest match. A
partial file survives network failures and a later ``rdv models pull`` resumes
it with an HTTP ``Range`` request. ``RDV_OFFLINE=1`` fails before any request.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlparse

import httpx

from rdv.adapters.models.manifest import ModelManifest
from rdv.application.ports import ModelEntry
from rdv.errors import ModelError
from rdv.hashing import sha256_file

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_TIMEOUT = 60.0
_GITHUB_HOSTS = {"github.com", "www.github.com", "api.github.com"}


def request_headers(url: str, headers: Mapping[str, str]) -> dict[str, str]:
    """Add an optional GitHub token so private release assets can be pulled.

    The project needs zero mandatory keys: anonymous downloads work for public
    releases. For a private repository the user may export ``GITHUB_TOKEN`` or
    ``GH_TOKEN`` (for example ``gh auth token``) and the token is only sent to
    GitHub hosts, never to a third-party mirror.
    """
    merged = dict(headers)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and urlparse(url).hostname in _GITHUB_HOSTS:
        merged["Authorization"] = f"Bearer {token}"
    return merged


def _file_name(entry: ModelEntry) -> str:
    name = Path(urlparse(entry.url).path).name
    return name or f"{entry.id}.onnx"


class CachedModelStore:
    """Model store backed by a local cache directory and the packaged manifest.

    Args:
        manifest: validated model manifest.
        model_dir: directory where verified artifacts are cached.
        offline: when true, any download attempt fails immediately.
        client: optional preconfigured HTTP client (tests inject a mock
            transport; production creates one per download).
        timeout: HTTP timeout in seconds for the internally created client.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        model_dir: Path,
        *,
        offline: bool = False,
        client: httpx.Client | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._manifest = manifest
        self._model_dir = Path(model_dir)
        self._offline = offline
        self._client = client
        self._timeout = timeout

    def entries(self) -> tuple[ModelEntry, ...]:
        """Models published in the manifest, in file order."""
        return self._manifest.models

    def entry(self, model_id: str) -> ModelEntry | None:
        """Return the manifest entry for ``model_id``, or ``None``."""
        return self._manifest.find(model_id)

    def resolve(self, model: str) -> Path:
        """Resolve a manifest id or an explicit ONNX path to a local file.

        Raises:
            ModelError: the file is missing or the id is not published.
        """
        if model.endswith(".onnx") or Path(model).is_file():
            path = Path(model).expanduser()
            if not path.is_file():
                raise ModelError(
                    f"Model file not found: {path}. Run `rdv models pull` to download a model."
                )
            return path
        entry = self._manifest.get(model)
        path = self._model_dir / _file_name(entry)
        if not path.is_file():
            raise ModelError(
                f"Model {model!r} is not downloaded at {path}. Run `rdv models pull {model}`."
            )
        return path

    def pull(
        self,
        model_id: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download a manifest model, resuming a partial file when possible.

        Args:
            model_id: id published in the manifest.
            progress: optional ``(done, total)`` callback while streaming.

        Returns:
            Path of the verified final file.

        Raises:
            ModelError: offline mode, HTTP failure, size mismatch or sha256
                mismatch. The final path is never written on failure.
        """
        entry = self._manifest.get(model_id)
        if self._offline:
            raise ModelError(
                f"RDV_OFFLINE is set; refusing to download model {model_id!r}. "
                "Unset RDV_OFFLINE to pull it."
            )
        self._model_dir.mkdir(parents=True, exist_ok=True)
        destination = self._model_dir / _file_name(entry)
        if destination.is_file():
            if sha256_file(destination) == entry.sha256:
                return destination
            destination.unlink()

        part = destination.with_name(destination.name + ".part")
        start = part.stat().st_size if part.is_file() else 0
        if start > entry.size:
            part.unlink()
            start = 0
        if start and start == entry.size:
            return self._finish(part, destination, entry)

        headers = {"Range": f"bytes={start}-"} if start else {}
        downloaded = start
        client = self._client or httpx.Client(
            follow_redirects=True,
            timeout=self._timeout,
        )
        try:
            try:
                with client.stream(
                    "GET", entry.url, headers=request_headers(entry.url, headers)
                ) as response:
                    if response.status_code == 416 and start:
                        # The server says the range is already complete.
                        return self._finish(part, destination, entry)
                    if start and response.status_code == 200:
                        # The server ignored Range: restart from zero.
                        start = 0
                        downloaded = 0
                    response.raise_for_status()
                    mode = "ab" if start and response.status_code == 206 else "wb"
                    with part.open(mode) as handle:
                        for chunk in response.iter_bytes(_CHUNK_SIZE):
                            handle.write(chunk)
                            downloaded += len(chunk)
                            if progress is not None:
                                progress(downloaded, entry.size)
            except httpx.HTTPError as exc:
                raise ModelError(
                    f"Download failed for model {model_id!r}: {exc}. "
                    "Run `rdv models pull` again to resume."
                ) from exc
        finally:
            if self._client is None:
                client.close()

        return self._finish(part, destination, entry)

    def _finish(self, part: Path, destination: Path, entry: ModelEntry) -> Path:
        if not part.is_file():
            raise ModelError(f"Download did not produce {part}")
        if part.stat().st_size != entry.size:
            raise ModelError(
                f"Size mismatch for model {entry.id!r}: expected {entry.size} bytes, "
                f"got {part.stat().st_size}. Run `rdv models pull` again to resume."
            )
        digest = sha256_file(part)
        if digest != entry.sha256:
            part.unlink(missing_ok=True)
            raise ModelError(
                f"sha256 mismatch for model {entry.id!r}: expected {entry.sha256}, got {digest}"
            )
        part.replace(destination)
        return destination

    def verify(self, path: Path, model_id: str | None = None) -> ModelEntry:
        """Check ``path`` against its manifest entry.

        Args:
            path: local ONNX file.
            model_id: entry to compare against; when omitted, the entry whose
                sha256 matches is selected.

        Raises:
            ModelError: the file is missing, does not match, or no manifest
                entry matches it.
        """
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"Model file not found: {path}")
        digest = sha256_file(path)
        if model_id is not None:
            entry = self._manifest.get(model_id)
            if digest != entry.sha256:
                raise ModelError(
                    f"sha256 mismatch for {path}: expected {entry.sha256}, got {digest}"
                )
            return entry
        for entry in self._manifest.models:
            if digest == entry.sha256:
                return entry
        raise ModelError(
            f"sha256 {digest} does not match any model in the manifest; "
            f"{path} is not a published artifact"
        )
