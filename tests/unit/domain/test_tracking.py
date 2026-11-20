"""Greedy IoU tracking and unique-instance counts (decision D20)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rdv.domain.classes import DamageClass
from rdv.domain.detection import Detection
from rdv.domain.geometry import BBox
from rdv.domain.tracking import IouTracker, TrackedDetection


def _detection(
    damage_class: DamageClass = DamageClass.D00,
    box: tuple[float, float, float, float] = (0, 0, 10, 10),
    score: float = 0.9,
) -> Detection:
    return Detection(damage_class, score, BBox(*box))


def test_tracked_detection_validates_id_and_hits() -> None:
    tracked = TrackedDetection(track_id=1, detection=_detection(), hits=2)

    assert tracked.track_id == 1
    assert tracked.hits == 2
    with pytest.raises(ValueError, match="track_id"):
        TrackedDetection(track_id=0, detection=_detection(), hits=1)
    with pytest.raises(ValueError, match="hits"):
        TrackedDetection(track_id=1, detection=_detection(), hits=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"iou_threshold": -0.1}, "iou_threshold"),
        ({"iou_threshold": 1.1}, "iou_threshold"),
        ({"max_age": -1}, "max_age"),
        ({"min_hits": 0}, "min_hits"),
    ],
)
def test_tracker_parameters_are_validated(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IouTracker(**kwargs)  # type: ignore[arg-type]


def test_first_frame_creates_tracks_in_input_order() -> None:
    tracker = IouTracker()
    detections = [
        _detection(DamageClass.D00, (0, 0, 10, 10)),
        _detection(DamageClass.D40, (50, 50, 60, 60)),
    ]

    tracked = tracker.update(detections)

    assert [item.track_id for item in tracked] == [1, 2]
    assert [item.detection for item in tracked] == detections
    assert all(item.hits == 1 for item in tracked)


def test_overlapping_detection_in_the_next_frame_keeps_the_track() -> None:
    tracker = IouTracker()
    first = tracker.update([_detection(box=(0, 0, 10, 10))])
    second = tracker.update([_detection(box=(1, 1, 11, 11))])

    assert second[0].track_id == first[0].track_id
    assert second[0].hits == 2
    assert tracker.unique_counts()[DamageClass.D00] == 1


def test_same_box_in_another_class_is_a_new_track() -> None:
    tracker = IouTracker(min_hits=1)
    first = tracker.update([_detection(DamageClass.D00)])
    second = tracker.update([_detection(DamageClass.D40)])

    assert second[0].track_id != first[0].track_id
    assert tracker.unique_counts()[DamageClass.D00] == 1
    assert tracker.unique_counts()[DamageClass.D40] == 1


def test_non_overlapping_detection_starts_a_new_track() -> None:
    tracker = IouTracker()
    first = tracker.update([_detection(box=(0, 0, 10, 10))])
    second = tracker.update([_detection(box=(100, 100, 110, 110))])

    assert second[0].track_id != first[0].track_id


def test_two_overlapping_detections_in_one_frame_get_distinct_tracks() -> None:
    tracker = IouTracker()
    tracker.update([_detection(box=(0, 0, 10, 10))])

    tracked = tracker.update(
        [
            _detection(box=(0, 0, 10, 10), score=0.9),
            _detection(box=(0, 0, 10, 10), score=0.8),
        ]
    )

    assert len({item.track_id for item in tracked}) == 2
    assert tracker.unique_counts()[DamageClass.D00] == 1


def test_unique_instances_require_min_hits() -> None:
    tracker = IouTracker(min_hits=2, max_age=1)
    tracker.update([_detection()])

    assert tracker.unique_counts()[DamageClass.D00] == 0


def test_retired_tracks_keep_their_unique_count() -> None:
    tracker = IouTracker(min_hits=2, max_age=2)
    tracker.update([_detection()])
    tracker.update([_detection()])

    for _ in range(3):  # one more than max_age analyzed frames without a match
        assert tracker.update([]) == []

    assert tracker.unique_counts()[DamageClass.D00] == 1


def test_short_lived_track_is_not_counted_after_retirement() -> None:
    tracker = IouTracker(min_hits=2, max_age=1)
    tracker.update([_detection()])

    for _ in range(2):
        tracker.update([])

    assert tracker.unique_counts()[DamageClass.D00] == 0


def test_track_stays_alive_until_after_max_age() -> None:
    tracker = IouTracker(min_hits=1, max_age=2)
    first = tracker.update([_detection()])
    tracker.update([])
    tracker.update([])
    revived = tracker.update([_detection()])

    assert revived[0].track_id == first[0].track_id
    assert revived[0].hits == 2


def test_unique_counts_always_include_all_classes() -> None:
    tracker = IouTracker()

    counts = tracker.unique_counts()

    assert set(counts) == set(DamageClass)
    assert all(value == 0 for value in counts.values())


def test_greedy_matching_picks_the_globally_best_pair() -> None:
    tracker = IouTracker()
    first = tracker.update(
        [
            _detection(DamageClass.D00, (0, 0, 10, 10)),
            _detection(DamageClass.D00, (10, 0, 20, 10)),
        ]
    )

    second = tracker.update(
        [
            _detection(DamageClass.D00, (9, 0, 19, 10)),
            _detection(DamageClass.D00, (0, 0, 10, 10)),
        ]
    )

    assert second[0].track_id == first[1].track_id
    assert second[1].track_id == first[0].track_id
    assert [item.hits for item in second] == [2, 2]


def test_tracking_is_deterministic() -> None:
    def run() -> list[TrackedDetection]:
        tracker = IouTracker()
        output: list[TrackedDetection] = []
        for offset in range(4):
            output.extend(
                tracker.update(
                    [
                        _detection(DamageClass.D00, (offset, 0, offset + 10, 10)),
                        _detection(DamageClass.D40, (100, 100, 110, 110)),
                    ]
                )
            )
        return output

    assert run() == run()


detection_strategy = st.builds(
    _detection,
    st.sampled_from(list(DamageClass)),
    st.tuples(
        st.floats(min_value=0, max_value=500, allow_nan=False),
        st.floats(min_value=0, max_value=500, allow_nan=False),
        st.floats(min_value=0, max_value=500, allow_nan=False),
        st.floats(min_value=0, max_value=500, allow_nan=False),
    ).map(
        lambda values: (
            min(values[0], values[2]),
            min(values[1], values[3]),
            max(values[0], values[2]),
            max(values[1], values[3]),
        )
    ),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)


@given(frames=st.lists(st.lists(detection_strategy, max_size=6), max_size=8))
def test_track_ids_and_unique_counts_are_consistent(
    frames: list[list[Detection]],
) -> None:
    tracker = IouTracker()
    id_to_class: dict[int, DamageClass] = {}
    total_detections = 0
    for frame in frames:
        tracked = tracker.update(frame)
        total_detections += len(frame)
        assert len({item.track_id for item in tracked}) == len(tracked)
        for item in tracked:
            previous = id_to_class.setdefault(item.track_id, item.detection.damage_class)
            assert previous is item.detection.damage_class
            assert item.hits >= 1

    unique_counts = tracker.unique_counts()
    assert set(unique_counts) == set(DamageClass)
    assert sum(unique_counts.values()) <= total_detections
