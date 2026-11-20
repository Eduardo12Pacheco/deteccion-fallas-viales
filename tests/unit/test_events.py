"""Progress events and the null observer (Observer pattern, D9)."""

from __future__ import annotations

import pytest

from rdv.application.events import NullProgressSink, ProgressEvent


def test_progress_event_reports_a_fraction() -> None:
    event = ProgressEvent(stage="detect", current=25, total=100, message="frame 25")

    assert event.fraction == pytest.approx(0.25)
    assert event.is_complete is False


def test_progress_event_without_a_total_has_no_fraction() -> None:
    event = ProgressEvent(stage="detect", current=1)

    assert event.fraction is None
    assert event.is_complete is False


def test_completed_progress_event() -> None:
    assert ProgressEvent(stage="detect", current=10, total=10).is_complete is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stage": "", "current": 0}, "stage"),
        ({"stage": "detect", "current": -1}, "current"),
        ({"stage": "detect", "current": 0, "total": 0}, "total"),
        ({"stage": "detect", "current": 11, "total": 10}, "current"),
    ],
)
def test_invalid_progress_events_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProgressEvent(**kwargs)  # type: ignore[arg-type]


def test_null_progress_sink_accepts_everything() -> None:
    sink = NullProgressSink()

    assert sink.emit(ProgressEvent(stage="detect", current=1, total=2)) is None
