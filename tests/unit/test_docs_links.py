"""Every relative link and ``#anchor`` in README/docs must resolve (S19)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]

_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)",
)
_FENCE = re.compile(r"^\s*(```+|~~~+)")
_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(?P<text>.+?)\s*#*\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_EXTERNAL = ("http://", "https://", "mailto:", "tel:", "data:")


def _markdown_files() -> list[Path]:
    files = sorted(ROOT.glob("README*.md")) + sorted((ROOT / "docs").rglob("*.md"))
    return [path for path in files if path.is_file()]


def _without_code(text: str) -> str:
    """Drop fenced code blocks and HTML comments so examples are not links."""

    text = _COMMENT.sub("", text)
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def slugify(text: str) -> str:
    """Approximate GitHub's heading anchor slug for this repository's docs."""

    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "").replace("*", "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def anchors(path: Path) -> set[str]:
    text = _without_code(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    counts: dict[str, int] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _ATX_HEADING.match(line)
        heading: str | None = match.group("text") if match else None
        if heading is None and index + 1 < len(lines):
            underline = lines[index + 1].strip()
            if underline and set(underline) <= {"=", "-"} and line.strip():
                heading = line.strip()
        if heading is None:
            continue
        base = slugify(heading)
        if not base:
            continue
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        result.add(base if seen == 0 else f"{base}-{seen}")
    return result


def broken_references() -> list[str]:
    cache: dict[Path, set[str]] = {}
    broken: list[str] = []
    for path in _markdown_files():
        text = _without_code(path.read_text(encoding="utf-8"))
        for match in _LINK.finditer(text):
            raw = match.group("target")
            target = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
            if target.startswith(_EXTERNAL):
                continue
            file_part, _, anchor = target.partition("#")
            file_part = unquote(file_part)
            anchor = unquote(anchor)
            destination = path if not file_part else (path.parent / file_part).resolve()
            location = f"{path.relative_to(ROOT)} -> {target}"
            if not destination.exists():
                broken.append(f"{location} (missing file)")
                continue
            if not anchor:
                continue
            if destination.suffix.lower() != ".md":
                broken.append(f"{location} (anchor on a non-markdown target)")
                continue
            if destination not in cache:
                cache[destination] = anchors(destination)
            if anchor not in cache[destination]:
                broken.append(f"{location} (missing anchor)")
    return broken


def test_readme_and_docs_links_resolve() -> None:
    broken = broken_references()
    assert broken == [], "broken documentation references:\n" + "\n".join(broken)


def test_slugify_matches_github_style() -> None:
    assert slugify("Licensing and provenance") == "licensing-and-provenance"
    assert slugify("The `rdv` CLI (v1)") == "the-rdv-cli-v1"


def test_markdown_files_are_discovered() -> None:
    assert _markdown_files(), "README/docs must exist for the link check"
