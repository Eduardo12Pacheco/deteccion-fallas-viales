"""The fixture generator is deterministic and locks every produced file."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tests.support.synthetic import write_synthetic_dataset

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_fixtures.py"


def _run(script: Path, dataset: Path, out: Path, samples: Path, lock: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(dataset),
            "--out",
            str(out),
            "--samples",
            str(samples),
            "--fixtures-lock",
            str(lock),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _hashes(*roots: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(root.parent))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return hashes


def test_fixture_generation_is_byte_identical_and_locked(tmp_path: Path) -> None:
    dataset = tmp_path / "rdd2022"
    write_synthetic_dataset(
        dataset,
        countries=("Japan", "India", "Czech"),
        images_per_country=40,
        width=200,
        height=150,
        seed=7,
    )
    out = tmp_path / "fixtures" / "rdd2022-mini"
    samples = tmp_path / "assets" / "samples"
    lock = tmp_path / "fixtures.lock.json"

    _run(SCRIPT, dataset, out, samples, lock)
    first = _hashes(out, samples)
    first_lock = lock.read_bytes()

    _run(SCRIPT, dataset, out, samples, lock)
    second = _hashes(out, samples)

    assert first == second
    assert lock.read_bytes() == first_lock
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["schema"] == "rdv.fixtures-lock/v1"
    assert len(payload["selection"]) == 16
    assert len(payload["files"]) >= 16 + 4
    assert len({entry["source"] for entry in payload["selection"]}) == 16
    for relative, digest in payload["files"].items():
        path = Path(relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    ids = [image for split in splits["splits"].values() for image in split]
    assert len(ids) == 16
