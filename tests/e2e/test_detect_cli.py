"""End-to-end ``rdv detect`` on the real mini fixture and tiny model (S5)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
SCHEMA = REPO_ROOT / "docs" / "schemas" / "detections.v1.schema.json"
MINI = REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini"
IMAGE = sorted((MINI / "Japan" / "train" / "images").glob("*.jpg"))[0]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rdv", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_detect_writes_all_s5_artifacts_and_is_byte_identical_twice(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = _run("detect", str(IMAGE), "--model", str(TINY_MODEL), "--out", str(first))
    assert result.returncode == 0, result.stderr

    assert sorted(path.name for path in first.iterdir()) == [
        f"{IMAGE.stem}.annotated.png",
        f"{IMAGE.stem}.detections.json",
        "summary.json",
        "summary.md",
    ]
    document = json.loads((first / f"{IMAGE.stem}.detections.json").read_text(encoding="utf-8"))
    from jsonschema import Draft202012Validator

    Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(document)
    assert document["schema"] == "rdv.detections/v1"
    assert document["source"]["kind"] == "image"
    assert document["summary"]["frames_analyzed"] == 1
    assert document["summary"]["counts"] == {"D00": 1, "D10": 0, "D20": 0, "D40": 1}
    assert [entry["class"] for entry in document["frames"][0]["detections"]] == ["D00", "D40"]

    annotated = np.asarray(Image.open(first / f"{IMAGE.stem}.annotated.png").convert("RGB"))
    assert annotated.shape[0] > 0 and annotated.shape[1] > 0
    assert annotated.dtype == np.uint8

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["images"] == 1
    assert summary["totals"]["detections"] == 2
    assert "Detection summary" in (first / "summary.md").read_text(encoding="utf-8")

    second_result = _run("detect", str(IMAGE), "--model", str(TINY_MODEL), "--out", str(second))
    assert second_result.returncode == 0, second_result.stderr
    for name in (f"{IMAGE.stem}.detections.json", "summary.json", "summary.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_detect_accepts_multiple_images_and_an_absolute_output_dir(tmp_path: Path) -> None:
    other = sorted((MINI / "Czech" / "train" / "images").glob("*.jpg"))[0]
    out = tmp_path / "batch"

    result = _run(
        "detect",
        str(IMAGE),
        str(other),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(out),
        "--conf",
        "0.9",
    )

    assert result.returncode == 0, result.stderr
    assert (out / f"{IMAGE.stem}.detections.json").is_file()
    assert (out / f"{other.stem}.detections.json").is_file()
    document = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert document["totals"]["images"] == 2
    assert document["params"]["score_threshold"] == pytest.approx(0.9)


def test_corrupt_image_exits_five_and_writes_nothing_for_it(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    out = tmp_path / "out"

    result = _run("detect", str(corrupt), "--model", str(TINY_MODEL), "--out", str(out))

    assert result.returncode == 5
    assert "error:" in result.stderr
    assert "Cannot decode image" in result.stderr
    if out.exists():
        assert not list(out.glob("corrupt.*"))


def test_missing_model_exits_four_with_the_pull_hint(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["RDV_MODEL_DIR"] = str(tmp_path / "empty-models")

    result = _run("detect", str(IMAGE), "--out", str(tmp_path / "out"), env=env)

    assert result.returncode == 4
    assert "rdv models pull" in result.stderr


@pytest.mark.skipif(
    "CUDAExecutionProvider" in ort.get_available_providers(),
    reason="a CUDA provider is available on this machine",
)
def test_explicit_cuda_without_provider_exits_four(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["RDV_DEVICE"] = "cuda"

    result = _run(
        "detect",
        str(IMAGE),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(tmp_path / "out"),
        env=env,
    )

    assert result.returncode == 4
    assert "CUDA" in result.stderr


def test_video_input_exits_five_with_a_clear_message(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    result = _run("detect", str(clip), "--model", str(TINY_MODEL), "--out", str(tmp_path / "out"))

    assert result.returncode == 5
    assert "Video detection is not available yet" in result.stderr


def test_no_inputs_exits_five() -> None:
    result = _run("detect")

    assert result.returncode == 5
    assert "No input images" in result.stderr
