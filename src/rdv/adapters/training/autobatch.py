"""Auto micro-batch probing with an OOM fallback ladder (decision D16).

The trainer calls :func:`choose_micro_batch` with a probe that runs the real
forward/backward for a few steps. The largest candidate that completes within
the VRAM cap wins; a CUDA OOM simply moves to the next candidate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from rdv.errors import RdvError

DEFAULT_LADDER: tuple[int, ...] = (16, 12, 8, 6, 4)
DEFAULT_VRAM_CAP_GB = 7.2
GIB = 1024**3


class MicroBatchError(RdvError):
    """No candidate micro-batch completed within the VRAM cap."""

    exit_code = 6


@dataclass(frozen=True, slots=True)
class AutoBatchResult:
    """Selected micro-batch and the memory the probe measured."""

    micro_batch: int
    peak_bytes: int


def default_retryable() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "this batch did not fit" rather than a real bug."""
    return (torch.cuda.OutOfMemoryError, MemoryError)


def choose_micro_batch(
    probe: Callable[[int], int],
    *,
    ladder: Sequence[int] = DEFAULT_LADDER,
    cap_bytes: int = int(DEFAULT_VRAM_CAP_GB * GIB),
    retryable: tuple[type[BaseException], ...] = (),
    empty_cache: Callable[[], None] | None = None,
) -> AutoBatchResult:
    """Return the first ladder size that fits the memory cap.

    Args:
        probe: runs the candidate and returns peak allocated bytes.
        ladder: candidate batch sizes, largest first.
        cap_bytes: maximum acceptable peak allocation.
        retryable: exceptions treated as "does not fit"; defaults to CUDA OOM
            and ``MemoryError``.
        empty_cache: optional cache flush called after each failed attempt.

    Raises:
        ValueError: the ladder is empty or contains non-positive sizes.
        MicroBatchError: every candidate failed or exceeded the cap.
    """
    candidates = tuple(int(size) for size in ladder)
    if not candidates:
        raise ValueError("micro-batch ladder must not be empty")
    if any(size <= 0 for size in candidates):
        raise ValueError("micro-batch ladder sizes must be positive")
    handled = retryable or default_retryable()
    clear = empty_cache if empty_cache is not None else _empty_cache
    failures: list[str] = []
    for size in candidates:
        try:
            peak = int(probe(size))
        except handled as exc:
            failures.append(f"{size}: {type(exc).__name__}")
            clear()
            continue
        if peak <= cap_bytes:
            return AutoBatchResult(micro_batch=size, peak_bytes=peak)
        failures.append(f"{size}: {peak} > {cap_bytes} bytes")
        clear()
    raise MicroBatchError(
        "No micro-batch in the ladder fit within the VRAM cap: " + ", ".join(failures)
    )


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
