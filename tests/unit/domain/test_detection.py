"""Detection and frame results are validated value objects."""

from __future__ import annotations

import math

import pytest

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection, FrameResult
from rdv.domain.geometry import BBox


def test_detection_keeps_class_score_and_box() -> None:
    box = BBox(1.0, 2.0, 3.0, 4.0)
    detection = Detection(DamageClass.D40, 0.75, box)

    assert detection.damage_class is DamageClass.D40
    assert detection.score == 0.75
    assert detection.bbox == box


@pytest.mark.parametrize("score", [-0.01, 1.01, math.nan])
def test_detection_rejects_scores_outside_the_probability_range(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        Detection(DamageClass.D00, score, BBox(0, 0, 1, 1))


def test_frame_result_copies_detections_into_a_tuple() -> None:
    detection = Detection(DamageClass.D10, 0.5, BBox(0, 0, 1, 1))

    frame = FrameResult(frame_index=3, detections=[detection])  # type: ignore[arg-type]

    assert frame.frame_index == 3
    assert frame.detections == (detection,)
    assert frame.timestamp is None


def test_frame_result_carries_an_optional_timestamp() -> None:
    frame = FrameResult(frame_index=0, detections=(), timestamp=0.25)

    assert frame.timestamp == 0.25


@pytest.mark.parametrize("frame_index", [-1])
def test_frame_result_rejects_negative_frame_index(frame_index: int) -> None:
    with pytest.raises(ValueError, match="frame_index"):
        FrameResult(frame_index=frame_index, detections=())


@pytest.mark.parametrize("timestamp", [-0.1, math.nan, math.inf])
def test_frame_result_rejects_invalid_timestamps(timestamp: float) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        FrameResult(frame_index=0, detections=(), timestamp=timestamp)
