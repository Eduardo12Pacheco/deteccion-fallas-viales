"""Seekable HTTP-range access to the pinned Figshare RDD2022 archive (D14).

The dataset is published as one 13 GB ZIP that contains one ZIP per country.
Downloading the outer file is wasteful: the inner members are stored
uncompressed inside the outer ZIP, so a byte range of an inner archive is also
a byte range of the outer file. This adapter exposes the outer archive as a
seekable file object and can read a nested member through small ranges, which
is how the 10.6 GB Norway archive is sampled without downloading it.

The Figshare download endpoint redirects to a presigned URL that expires after
ten seconds, so the signed URL is re-resolved on a short TTL and whenever a
range request returns 403. Transient failures (403/429/5xx) are retried with
backoff, and ``RDV_OFFLINE=1`` fails before the first request (S20).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import time
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from rdv.errors import DatasetError

DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_CACHE_BLOCKS = 64
RETRYABLE_STATUS = frozenset({403, 429, 500, 502, 503, 504})
ProgressCallback = Callable[[int, int], None]

LOCAL_HEADER_SIZE = 30


@dataclass(frozen=True, slots=True)
class CountryPin:
    """Pinned location and checksums of one inner country archive."""

    name: str
    member: str
    header_offset: int
    size: int
    crc32: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetLock:
    """Validated ``data/rdd2022.lock.json`` content (D14)."""

    schema: str
    url: str
    outer_size: int
    outer_md5: str
    countries: Mapping[str, CountryPin]

    @classmethod
    def load(cls, path: Path) -> DatasetLock:
        """Read the pinned archive metadata.

        Raises:
            DatasetError: the file is missing, malformed, or has an unsupported
                schema. The message names the path.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DatasetError(f"Cannot read dataset lock {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Malformed dataset lock {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != "rdv.dataset-lock/v1":
            raise DatasetError(f"Unsupported dataset lock {path}: expected rdv.dataset-lock/v1")
        try:
            url = str(payload["url"])
            outer = payload["outer"]
            outer_size = int(outer["size"])
            outer_md5 = str(outer["md5"])
            raw_countries = payload["countries"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"Malformed dataset lock {path}: {exc}") from exc
        if not isinstance(raw_countries, dict):
            raise DatasetError(f"Malformed dataset lock {path}: 'countries' must be an object")

        countries: dict[str, CountryPin] = {}
        for name, raw in raw_countries.items():
            if not isinstance(raw, dict):
                raise DatasetError(
                    f"Malformed dataset lock {path}: country {name!r} is not an object"
                )
            try:
                crc_text = str(raw["crc32"])
                sha256 = raw.get("sha256")
                countries[str(name)] = CountryPin(
                    name=str(name),
                    member=str(raw["member"]),
                    header_offset=int(raw["header_offset"]),
                    size=int(raw["size"]),
                    crc32=int(crc_text, 16) if not crc_text.isdigit() else int(crc_text),
                    sha256=str(sha256) if sha256 else None,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetError(
                    f"Malformed dataset lock {path}: invalid country {name!r}: {exc}"
                ) from exc
        return cls(
            schema="rdv.dataset-lock/v1",
            url=url,
            outer_size=outer_size,
            outer_md5=outer_md5,
            countries=countries,
        )

    def country(self, name: str) -> CountryPin:
        """Return one country pin or raise a clear :class:`DatasetError`."""
        try:
            return self.countries[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.countries))
            raise DatasetError(f"Country {name!r} is not pinned; known countries: {known}") from exc

    def with_sha256(self, country: str, sha256: str) -> DatasetLock:
        """Return a copy with the inner-archive sha256 recorded."""
        pin = self.country(country)
        updated = dict(self.countries)
        updated[country] = dataclasses.replace(pin, sha256=sha256)
        return dataclasses.replace(self, countries=updated)

    def write(self, path: Path) -> None:
        """Write the lock deterministically (indent 2, trailing newline)."""
        payload: dict[str, Any] = {
            "schema": self.schema,
            "dataset": "RDD2022",
            "url": self.url,
            "outer": {"size": self.outer_size, "md5": self.outer_md5},
            "countries": {
                name: {
                    "member": pin.member,
                    "header_offset": pin.header_offset,
                    "size": pin.size,
                    "crc32": f"{pin.crc32:08x}",
                    "sha256": pin.sha256,
                }
                for name, pin in self.countries.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of one verified download."""

    destination: Path
    size: int
    crc32: int
    sha256: str
    md5: str
    resumed_bytes: int


class _RemoteFile:
    """Seekable read-only view over a byte window of a :class:`RemoteZipArchive`."""

    def __init__(self, archive: RemoteZipArchive, base: int = 0, length: int | None = None) -> None:
        self._archive = archive
        self._base = base
        self._length = archive.size - base if length is None else length
        if self._length < 0 or base < 0 or base + self._length > archive.size:
            raise ValueError("remote file window is outside the archive")
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError(f"unsupported whence {whence!r}")
        if position < 0:
            raise ValueError("cannot seek before the start of the remote file")
        self._position = position
        return self._position

    def read(self, size: int = -1) -> bytes:
        remaining = self._length - self._position
        if size is None or size < 0:
            size = remaining
        size = min(size, remaining)
        if size <= 0:
            return b""
        data = self._archive.read(self._base + self._position, size)
        self._position += len(data)
        return data


class RemoteZipArchive:
    """HTTP-range reader for the pinned Figshare outer ZIP (D14).

    Args:
        url: stable download endpoint that redirects to a presigned URL.
        size: exact outer archive size in bytes (pinned in the dataset lock).
        block_size: cache block size; 4 MiB by default.
        cache_blocks: maximum number of cached blocks (LRU).
        offline: fail before any request, mirroring ``RDV_OFFLINE=1``.
        client: optional ``httpx.Client`` (tests pass a ``MockTransport`` one).
        timeout: per-request timeout in seconds.
        max_attempts: retries per range request (5 in production).
        resolve_ttl: seconds a presigned URL is reused before re-resolution.
        sleep: injected sleeper so tests keep retry/backoff instant.
        chunk_size: default chunk for :meth:`download_to`.
    """

    def __init__(
        self,
        url: str,
        *,
        size: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        cache_blocks: int = DEFAULT_CACHE_BLOCKS,
        offline: bool = False,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_attempts: int = 5,
        resolve_ttl: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if offline:
            raise DatasetError(
                "Offline mode (RDV_OFFLINE=1) forbids the network request needed for the dataset"
            )
        if size <= 0:
            raise ValueError("size must be positive")
        if block_size <= 0 or cache_blocks <= 0 or chunk_size <= 0:
            raise ValueError("block_size, cache_blocks and chunk_size must be positive")
        self.url = url
        self.size = size
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.resolve_ttl = resolve_ttl
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self._owns_client = client is None
        self._resolved_url: str | None = None
        self._resolved_at = 0.0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        self.requests = 0

    def close(self) -> None:
        """Close an internally created HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RemoteZipArchive:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def read(self, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset`` (clamped at the archive end)."""
        if offset < 0 or size < 0:
            raise ValueError("offset and size must be non-negative")
        if offset >= self.size or size == 0:
            return b""
        size = min(size, self.size - offset)
        if size >= self.block_size:
            return self._fetch(offset, offset + size - 1)

        chunks = bytearray()
        position = offset
        end = offset + size
        while position < end:
            block_index = position // self.block_size
            block = self._cache.get(block_index)
            if block is None:
                block_start = block_index * self.block_size
                block_end = min(block_start + self.block_size, self.size) - 1
                block = self._fetch(block_start, block_end)
                self._cache[block_index] = block
                self._cache.move_to_end(block_index)
                while len(self._cache) > self.cache_blocks:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(block_index)
            within = position - block_index * self.block_size
            take = min(end - position, len(block) - within)
            chunks.extend(block[within : within + take])
            position += take
        return bytes(chunks)

    def local_header_data_offset(self, header_offset: int) -> int:
        """Return the first data byte of a stored ZIP member at ``header_offset``."""
        header = self.read(header_offset, LOCAL_HEADER_SIZE)
        if len(header) < LOCAL_HEADER_SIZE or header[:4] != b"PK\x03\x04":
            raise DatasetError(
                f"No ZIP local header at offset {header_offset}; the dataset lock is stale"
            )
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        return header_offset + LOCAL_HEADER_SIZE + int(name_length) + int(extra_length)

    def open_nested(self, member: str | CountryPin) -> _RemoteFile:
        """Open a stored outer member (an inner ZIP) as a seekable file object."""
        name = member.member if isinstance(member, CountryPin) else member
        with zipfile.ZipFile(_RemoteFile(self)) as outer:
            try:
                info = outer.getinfo(name)
            except KeyError as exc:
                raise DatasetError(f"Archive member {name!r} not found in the outer ZIP") from exc
            if info.compress_type != zipfile.ZIP_STORED:
                raise DatasetError(f"Outer member {name!r} is not stored and cannot be ranged")
            data_offset = self.local_header_data_offset(info.header_offset)
            length = info.file_size
        return _RemoteFile(self, base=data_offset, length=length)

    def download_to(
        self,
        destination: Path,
        *,
        data_offset: int,
        size: int,
        expected_crc32: int | None = None,
        expected_md5: str | None = None,
        chunk_size: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> DownloadResult:
        """Download one range into ``destination`` with resume and verification.

        The bytes are written to ``<destination>.part`` and only renamed after
        the size, CRC32 and MD5 checks pass. A failed check removes the part.
        """
        if size < 0 or data_offset < 0 or data_offset + size > self.size:
            raise ValueError("requested range is outside the archive")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")

        resumed = part.stat().st_size if part.is_file() else 0
        if resumed > size:
            part.unlink()
            resumed = 0
        step = chunk_size or self.chunk_size
        with part.open("ab") as handle:
            position = resumed
            if progress is not None:
                progress(position, size)
            while position < size:
                take = min(step, size - position)
                handle.write(self.read(data_offset + position, take))
                position += take
                if progress is not None:
                    progress(position, size)

        digest = _hash_file(part)
        crc = _crc32_file(part)
        if part.stat().st_size != size:
            part.unlink(missing_ok=True)
            raise DatasetError(
                f"Archive range size mismatch for {destination.name}: "
                f"expected {size} bytes, got {part.stat().st_size if part.exists() else 0}"
            )
        if expected_crc32 is not None and crc != expected_crc32:
            part.unlink(missing_ok=True)
            raise DatasetError(
                f"CRC32 mismatch for {destination.name}: "
                f"expected {expected_crc32:08x}, got {crc:08x}"
            )
        if expected_md5 is not None and digest["md5"] != expected_md5:
            part.unlink(missing_ok=True)
            raise DatasetError(
                f"MD5 mismatch for {destination.name}: expected {expected_md5}, got {digest['md5']}"
            )
        os.replace(part, destination)
        return DownloadResult(
            destination=destination,
            size=size,
            crc32=crc,
            sha256=digest["sha256"],
            md5=digest["md5"],
            resumed_bytes=resumed,
        )

    def _resolve(self) -> str:
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            try:
                response = self._client.head(self.url)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                self._sleep(_backoff(attempt))
                continue
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise DatasetError(f"Dataset redirect from {self.url} has no Location header")
                self._resolved_url = str(httpx.URL(self.url).join(location))
                self._resolved_at = time.monotonic()
                return self._resolved_url
            if response.status_code == 200:
                self._resolved_url = self.url
                self._resolved_at = time.monotonic()
                return self._resolved_url
            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code} resolving the dataset URL"
                self._sleep(_backoff(attempt))
                continue
            raise DatasetError(
                f"Cannot resolve dataset URL {self.url}: HTTP {response.status_code}"
            )
        raise DatasetError(f"Cannot resolve dataset URL {self.url}: {last_error}")

    def _fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise ValueError("invalid byte range")
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            if (
                self._resolved_url is None
                or time.monotonic() - self._resolved_at > self.resolve_ttl
            ):
                self._resolve()
            try:
                response = self._client.get(
                    self._resolved_url or self.url,
                    headers={"Range": f"bytes={start}-{end}"},
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)
                self._sleep(_backoff(attempt))
                continue
            if response.status_code == 403:
                self._resolved_url = None
                last_error = "HTTP 403 (expired presigned URL)"
                self._sleep(_backoff(attempt))
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}"
                self._sleep(_backoff(attempt))
                continue
            if response.status_code == 206:
                payload = response.content
                if len(payload) != end - start + 1:
                    raise DatasetError(
                        f"Short range response from the dataset: expected {end - start + 1} "
                        f"bytes, got {len(payload)}"
                    )
                self.bytes_fetched += len(payload)
                self.requests += 1
                return payload
            if response.status_code == 200 and start == 0 and len(response.content) == self.size:
                self.bytes_fetched += len(response.content)
                self.requests += 1
                return response.content
            raise DatasetError(
                f"Unexpected HTTP {response.status_code} for bytes {start}-{end} of the dataset"
            )
        raise DatasetError(f"Cannot read bytes {start}-{end} of the dataset: {last_error}")


def _backoff(attempt: int) -> float:
    return min(8.0, 0.25 * float(2**attempt))


def _hash_file(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def _crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF
