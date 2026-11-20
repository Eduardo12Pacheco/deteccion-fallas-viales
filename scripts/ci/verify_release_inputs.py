"""Verify the release tag, project version and pinned model manifest (S18).

The release workflow runs this script before building anything, and again in
the bundle matrix with ``--model`` to prove the downloaded weights are exactly
the ones pinned in ``src/rdv/assets/models.json``. It is deliberately
stdlib-only so a verification failure never depends on the environment.

Usage:
    uv run python scripts/ci/verify_release_inputs.py v0.1.0
    uv run python scripts/ci/verify_release_inputs.py v0.1.0 --model models/*.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_MANIFEST = ROOT / "src" / "rdv" / "assets" / "models.json"
MANIFEST_SCHEMA = "rdv.models/v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_ENTRY_FIELDS = (
    "id",
    "version",
    "url",
    "sha256",
    "size",
    "classes",
    "input_size",
    "recommended_score_threshold",
    "license",
)


def project_version(pyproject: Path) -> str:
    """Read ``[project].version`` from a ``pyproject.toml``."""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read {pyproject}: {exc}") from exc
    version = payload.get("project", {}).get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{pyproject} does not declare [project].version")
    return version


def normalize_tag(tag: str) -> str:
    """Strip the optional ``v`` prefix from a release tag."""
    normalized = tag[1:] if tag.startswith("v") else tag
    if not normalized:
        raise ValueError("release tag is empty")
    return normalized


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a model manifest and return its raw entries."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} must declare schema {MANIFEST_SCHEMA!r}")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path} must pin at least one model in 'models'")
    if not all(isinstance(entry, dict) for entry in models):
        raise ValueError(f"{path} 'models' entries must be objects")
    return models


def manifest_errors(models: list[dict[str, Any]]) -> list[str]:
    """Return every problem that would make a release unverifiable."""
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(models):
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]
        if missing:
            errors.append(f"model entry {index} is missing {missing!r}")
            continue
        model_id = str(entry["id"])
        if model_id in seen:
            errors.append(f"duplicate model id {model_id!r}")
        seen.add(model_id)
        sha256 = entry["sha256"]
        if not isinstance(sha256, str) or not SHA256_PATTERN.match(sha256.lower()):
            errors.append(f"model {model_id!r} must pin a 64-character hex sha256")
        size = entry["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            errors.append(f"model {model_id!r} size must be a positive integer")
        url = entry["url"]
        if not isinstance(url, str) or not url.startswith("https://github.com/"):
            errors.append(f"model {model_id!r} url must be a GitHub release URL")
        classes = entry["classes"]
        if (
            not isinstance(classes, list)
            or not classes
            or not all(isinstance(code, str) and code for code in classes)
            or len(set(classes)) != len(classes)
        ):
            errors.append(f"model {model_id!r} classes must be unique non-empty strings")
        threshold = entry["recommended_score_threshold"]
        if (
            not isinstance(threshold, int | float)
            or isinstance(threshold, bool)
            or not 0.0 <= float(threshold) <= 1.0
        ):
            errors.append(f"model {model_id!r} recommended_score_threshold must be within [0, 1]")
        if not isinstance(entry["license"], str) or not entry["license"]:
            errors.append(f"model {model_id!r} license must be a non-empty string")
    return errors


def asset_errors(path: Path, models: list[dict[str, Any]]) -> list[str]:
    """Return why a downloaded ONNX file does not match any manifest entry."""
    if not path.is_file():
        return [f"model asset not found: {path}"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    for entry in models:
        if entry.get("sha256", "").lower() == digest and entry.get("size") == size:
            return []
    return [
        f"{path.name} does not match the manifest (sha256 {digest}, size {size}); "
        "expected " + ", ".join(f"{entry.get('id')}: {entry.get('sha256')}" for entry in models)
    ]


def verify(
    tag: str,
    *,
    pyproject: Path = DEFAULT_PYPROJECT,
    manifest: Path = DEFAULT_MANIFEST,
    model_paths: list[Path] | None = None,
) -> tuple[str, list[str]]:
    """Validate a release tag against the project version and pinned manifest."""
    errors: list[str] = []
    try:
        version = project_version(pyproject)
    except ValueError as exc:
        return "", [str(exc)]
    tag_version = normalize_tag(tag)
    if tag_version != version:
        errors.append(f"tag {tag} does not match pyproject version {version}")
    try:
        models = load_manifest(manifest)
    except ValueError as exc:
        return version, [*errors, str(exc)]
    errors.extend(manifest_errors(models))
    for model_path in model_paths or []:
        errors.extend(asset_errors(model_path, models))
    return version, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the release tag, project version and pinned model manifest."
    )
    parser.add_argument("tag", help="Git tag being released, for example v0.1.0.")
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        help="Downloaded ONNX asset to match against the manifest (repeatable).",
    )
    args = parser.parse_args(argv)

    version, errors = verify(
        args.tag,
        pyproject=args.pyproject,
        manifest=args.manifest,
        model_paths=list(args.model),
    )
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: release inputs for {args.tag} (version {version}, {len(args.model)} asset(s) checked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
