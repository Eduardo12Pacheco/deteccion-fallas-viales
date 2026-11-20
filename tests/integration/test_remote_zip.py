"""Ranged nested-zip access, resume, CRC and URL-expiry behaviour (D14/S8).

The real archive is a 13 GB Figshare ZIP whose presigned URLs expire after a
few seconds. Tests use a synthetic nested ZIP served through
``httpx.MockTransport`` so no socket is ever opened.
"""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

import httpx
import pytest

from rdv.adapters.datasets.remote_zip import RemoteZipArchive
from rdv.errors import DatasetError

SOURCE_URL = "https://figshare.test/download"
SIGNED_URL = "https://signed.example/archive"
XML = b"""<?xml version="1.0"?>
<annotation>
  <folder>Japan</folder>
  <filename>Japan_000001.jpg</filename>
  <size><width>600</width><height>600</height><depth>3</depth></size>
  <object>
    <name>D40</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>120</ymax></bndbox>
  </object>
</annotation>
"""
IMAGE = b"jpeg-bytes-" * 20_000
PADDING = b"padding-" * 300_000


def _build_nested_zip() -> tuple[bytes, bytes]:
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w", zipfile.ZIP_STORED) as inner:
        inner.writestr("Japan/train/images/Japan_000001.jpg", IMAGE)
        inner.writestr("Japan/train/annotations/xmls/Japan_000001.xml", XML)
    inner_bytes = inner_buffer.getvalue()

    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w", zipfile.ZIP_STORED) as outer:
        outer.writestr("RDD2022/Japan.zip", inner_bytes)
        outer.writestr("padding/pad.bin", PADDING)
    return outer_buffer.getvalue(), inner_bytes


def _member_data_offset(payload: bytes, member: str) -> int:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(member)
    header = payload[info.header_offset : info.header_offset + 30]
    assert header[:4] == b"PK\x03\x04"
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_length + extra_length


class FakeFigshare:
    """Minimal transport that honours the range protocol and can expire URLs."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.resolve_count = 0
        self.signed_requests: list[tuple[int, int]] = []
        self.reject_next_signed = False
        self.bytes_served = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD" and request.url.host == "figshare.test":
            self.resolve_count += 1
            return httpx.Response(
                302, headers={"location": f"{SIGNED_URL}?signature={self.resolve_count}"}
            )
        if request.url.host == "signed.example":
            if self.reject_next_signed:
                self.reject_next_signed = False
                return httpx.Response(403)
            range_header = request.headers.get("range")
            assert range_header is not None, "the adapter must always request byte ranges"
            start_text, end_text = range_header.split("=", 1)[1].split("-", 1)
            start, end = int(start_text), int(end_text)
            body = self.payload[start : end + 1]
            self.signed_requests.append((start, end))
            self.bytes_served += len(body)
            return httpx.Response(
                206,
                content=body,
                headers={"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"},
            )
        return httpx.Response(404)


@pytest.fixture
def nested() -> tuple[bytes, bytes]:
    return _build_nested_zip()


def _archive(payload: bytes, server: FakeFigshare, **kwargs: object) -> RemoteZipArchive:
    client = httpx.Client(transport=httpx.MockTransport(server.handler))
    return RemoteZipArchive(
        SOURCE_URL, size=len(payload), client=client, block_size=64 * 1024, **kwargs
    )


def test_default_block_cache_is_four_mebibytes() -> None:
    archive = RemoteZipArchive(SOURCE_URL, size=10)

    assert archive.block_size == 4 * 1024 * 1024


def test_offline_mode_fails_before_any_request(nested: tuple[bytes, bytes]) -> None:
    payload, _ = nested
    server = FakeFigshare(payload)

    with pytest.raises(DatasetError, match="RDV_OFFLINE"):
        _archive(payload, server, offline=True)

    assert server.resolve_count == 0
    assert server.signed_requests == []


def test_resume_after_truncated_part(nested: tuple[bytes, bytes], tmp_path: Path) -> None:
    payload, inner = nested
    server = FakeFigshare(payload)
    archive = _archive(payload, server)
    data_offset = _member_data_offset(payload, "RDD2022/Japan.zip")
    destination = tmp_path / "downloads" / "Japan.zip"
    destination.parent.mkdir(parents=True)
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(inner[:100])

    result = archive.download_to(
        destination,
        data_offset=data_offset,
        size=len(inner),
        expected_crc32=zlib.crc32(inner),
        chunk_size=128 * 1024,
    )

    assert result.resumed_bytes == 100
    assert destination.read_bytes() == inner
    assert not part.exists()
    assert server.signed_requests[0][0] == data_offset + 100


def test_crc_mismatch_removes_part_and_raises(nested: tuple[bytes, bytes], tmp_path: Path) -> None:
    payload, inner = nested
    server = FakeFigshare(payload)
    archive = _archive(payload, server)
    data_offset = _member_data_offset(payload, "RDD2022/Japan.zip")
    destination = tmp_path / "Japan.zip"

    with pytest.raises(DatasetError, match="CRC32"):
        archive.download_to(
            destination,
            data_offset=data_offset,
            size=len(inner),
            expected_crc32=zlib.crc32(inner) ^ 0xFFFFFFFF,
        )

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


def test_expired_url_is_resolved_again(nested: tuple[bytes, bytes]) -> None:
    payload, _ = nested
    server = FakeFigshare(payload)
    archive = _archive(payload, server)

    first = archive.read(0, 64)
    server.reject_next_signed = True
    later = archive.read(len(payload) - 64, 64)

    assert first == payload[:64]
    assert later == payload[-64:]
    assert server.resolve_count == 2


def test_nested_member_read_fetches_less_than_the_archive(
    nested: tuple[bytes, bytes],
) -> None:
    payload, inner = nested
    server = FakeFigshare(payload)
    archive = _archive(payload, server)

    nested_archive = archive.open_nested("RDD2022/Japan.zip")
    with zipfile.ZipFile(nested_archive) as zipped:
        xml = zipped.read("Japan/train/annotations/xmls/Japan_000001.xml")
        image = zipped.read("Japan/train/images/Japan_000001.jpg")

    assert xml == XML
    assert image == IMAGE
    assert inner in payload
    assert archive.bytes_fetched < len(payload)
    assert server.bytes_served < len(payload)


def test_full_download_verifies_size_and_md5(nested: tuple[bytes, bytes], tmp_path: Path) -> None:
    payload, _ = nested
    server = FakeFigshare(payload)
    archive = _archive(payload, server)
    destination = tmp_path / "rdd2022.zip"

    result = archive.download_to(
        destination,
        data_offset=0,
        size=len(payload),
        expected_md5=_md5(payload),
    )

    assert destination.read_bytes() == payload
    assert result.sha256 == _sha256(payload)

    broken = tmp_path / "broken.zip"
    with pytest.raises(DatasetError, match="MD5"):
        archive.download_to(
            broken,
            data_offset=0,
            size=len(payload),
            expected_md5="0" * 32,
        )
    assert not broken.exists()


def _md5(payload: bytes) -> str:
    import hashlib

    return hashlib.md5(payload).hexdigest()


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
