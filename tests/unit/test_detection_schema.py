"""The detections schema is valid and shipped identically inside the wheel."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SCHEMA = REPO_ROOT / "docs" / "schemas" / "detections.v1.schema.json"
ASSET_SCHEMA = REPO_ROOT / "src" / "rdv" / "assets" / "schemas" / "detections.v1.schema.json"


def test_schema_is_valid_draft_2020_12() -> None:
    Draft202012Validator.check_schema(json.loads(DOCS_SCHEMA.read_text(encoding="utf-8")))


def test_packaged_schema_is_byte_identical_to_the_documented_one() -> None:
    assert ASSET_SCHEMA.read_bytes() == DOCS_SCHEMA.read_bytes()


def test_schema_rejects_a_document_without_the_contract_shape() -> None:
    validator = Draft202012Validator(json.loads(DOCS_SCHEMA.read_text(encoding="utf-8")))

    errors = list(validator.iter_errors({"schema": "rdv.detections/v1"}))

    assert errors
