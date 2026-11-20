"""Unit tests for the local-server security helpers (S14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdv.entrypoints.web.security import (
    is_local_host,
    origin_matches_host,
    parse_host_header,
)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("localhost", ("localhost", None)),
        ("localhost:16080", ("localhost", 16080)),
        ("127.0.0.1", ("127.0.0.1", None)),
        ("127.0.0.1:16080", ("127.0.0.1", 16080)),
        ("[::1]", ("::1", None)),
        ("[::1]:16080", ("::1", 16080)),
        ("::1", ("::1", None)),
        (" EXAMPLE.com:80 ", ("example.com", 80)),
        ("", None),
        ("[::1", None),
        ("[::1]garbage", None),
        ("[::1]:abc", None),
        (":16080", None),
    ],
)
def test_parse_host_header(header: str, expected: tuple[str, int | None] | None) -> None:
    assert parse_host_header(header) == expected


@pytest.mark.parametrize(
    ("header", "allowed"),
    [
        ("localhost", True),
        ("localhost:16080", True),
        ("127.0.0.1:1", True),
        ("[::1]:16080", True),
        ("::1", True),
        ("evil.example", False),
        ("127.0.0.1.evil.example", False),
        ("0.0.0.0", False),
        ("", False),
    ],
)
def test_is_local_host(header: str, allowed: bool) -> None:
    assert is_local_host(header) is allowed


@pytest.mark.parametrize(
    ("origin", "host", "expected"),
    [
        ("http://127.0.0.1:16080", "127.0.0.1:16080", True),
        ("http://localhost:16080", "localhost:16080", True),
        ("https://localhost", "localhost", True),
        ("http://localhost", "localhost:80", True),
        ("http://localhost:80", "localhost", True),
        ("http://evil.example:16080", "127.0.0.1:16080", False),
        ("http://127.0.0.1:16081", "127.0.0.1:16080", False),
        ("file:///etc/passwd", "127.0.0.1:16080", False),
        ("null", "127.0.0.1:16080", False),
        ("http://127.0.0.1:bad", "127.0.0.1:16080", False),
        ("http://[::1]:16080", "[::1]:16080", True),
    ],
)
def test_origin_matches_host(origin: str, host: str, expected: bool) -> None:
    assert origin_matches_host(origin, host) is expected


def test_security_module_does_not_import_adapters() -> None:
    source = Path(__file__).resolve().parents[3] / "src" / "rdv" / "entrypoints" / "web"
    text = (source / "security.py").read_text(encoding="utf-8")
    assert "rdv.adapters" not in text
