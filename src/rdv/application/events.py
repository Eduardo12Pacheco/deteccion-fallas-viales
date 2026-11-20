"""Progress events emitted by long-running use cases.

Use cases never print and never know whether progress is shown in a terminal,
stored in a web job, or dropped entirely: they emit immutable events to a
``ProgressSink`` (Observer pattern, decision D9).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One observable step of a use case.

    Attributes:
        stage: stable stage name such as ``"detect"`` or ``"evaluate"``.
        current: number of completed units (frames, images, epochs).
        total: expected units, or ``None`` when the total is unknown.
        message: optional human-readable detail for the console/UI.
    """

    stage: str
    current: int
    total: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("ProgressEvent stage must not be empty")
        if self.current < 0:
            raise ValueError("ProgressEvent current must be non-negative")
        if self.total is not None:
            if self.total <= 0:
                raise ValueError("ProgressEvent total must be positive when provided")
            if self.current > self.total:
                raise ValueError("ProgressEvent current must not exceed total")

    @property
    def fraction(self) -> float | None:
        """Completed fraction in ``[0, 1]``, or ``None`` without a total."""
        if self.total is None:
            return None
        return self.current / self.total

    @property
    def is_complete(self) -> bool:
        """True once every expected unit has been processed."""
        return self.total is not None and self.current == self.total


class NullProgressSink:
    """Null Object implementation for callers that do not display progress."""

    def emit(self, event: ProgressEvent) -> None:
        """Discard the event."""
        del event
