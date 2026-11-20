"""End-to-end ``rdv detect`` on a synthetic video and the tiny model (S6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
from typer.testing import CliRunner

from rdv.entrypoints.cli.app import app
from tests.support.synthetic_video import write_moving_rectangle_video, write_truncated_video

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "models" / "tiny-detector.onnx"
runner = CliRunner()


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rdv", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _decode_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def test_video_detection_writes_mp4_json_and_summary(tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    out = tmp_path / "out"

    result = _run(
        "detect",
        str(clip),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(out),
        "--analysis-fps",
        "5",
    )

    assert result.returncode == 0, result.stderr
    assert (out / "clip.annotated.mp4").is_file()
    assert (out / "clip.detections.json").is_file()
    assert (out / "clip.summary.json").is_file()
    assert (out / "clip.summary.md").is_file()
    assert _decode_count(out / "clip.annotated.mp4") == 40
    document = json.loads((out / "clip.detections.json").read_text(encoding="utf-8"))
    assert document["summary"]["frames_analyzed"] == 20
    assert document["summary"]["unique_instances"] == {
        "D00": 1,
        "D10": 0,
        "D20": 0,
        "D40": 1,
    }


def test_max_frames_stops_early_in_the_cli(tmp_path: Path) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=40)
    out = tmp_path / "out"

    result = _run(
        "detect",
        str(clip),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(out),
        "--max-frames",
        "10",
    )

    assert result.returncode == 0, result.stderr
    assert _decode_count(out / "clip.annotated.mp4") == 10
    summary = json.loads((out / "clip.summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["frames_processed"] == 10
    assert summary["totals"]["truncated"] is True
    assert "Max frames: 10" in (out / "clip.summary.md").read_text(encoding="utf-8")


def test_truncated_video_exits_five_without_final_artifacts(tmp_path: Path) -> None:
    clip = write_truncated_video(tmp_path / "clip.mp4")
    out = tmp_path / "out"

    result = _run("detect", str(clip), "--model", str(TINY_MODEL), "--out", str(out))

    assert result.returncode == 5
    assert "error:" in result.stderr
    assert not (out / "clip.annotated.mp4").exists()
    assert not (out / "clip.detections.json").exists()


def test_video_flags_without_a_video_exit_five(tmp_path: Path) -> None:
    image = sorted(
        (REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini" / "Japan" / "train" / "images").glob(
            "*.jpg"
        )
    )[0]

    result = _run(
        "detect",
        str(image),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(tmp_path / "out"),
        "--max-frames",
        "10",
    )

    assert result.returncode == 5
    assert "video inputs" in result.stderr


def test_mixed_image_and_video_inputs_are_supported(tmp_path: Path) -> None:
    image = sorted(
        (REPO_ROOT / "tests" / "fixtures" / "rdd2022-mini" / "Japan" / "train" / "images").glob(
            "*.jpg"
        )
    )[0]
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=8)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "detect",
            str(image),
            str(clip),
            "--model",
            str(TINY_MODEL),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert (out / f"{image.stem}.detections.json").is_file()
    assert (out / "clip.detections.json").is_file()


@pytest.mark.parametrize("flag", ["--analysis-fps", "--max-frames"])
def test_non_positive_video_options_are_usage_errors(tmp_path: Path, flag: str) -> None:
    clip = write_moving_rectangle_video(tmp_path / "clip.mp4", frames=4)
    value = "0"

    result = _run(
        "detect",
        str(clip),
        "--model",
        str(TINY_MODEL),
        "--out",
        str(tmp_path / "out"),
        flag,
        value,
    )

    assert result.returncode == 2
