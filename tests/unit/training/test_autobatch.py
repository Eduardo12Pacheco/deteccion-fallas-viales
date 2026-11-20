"""Auto micro-batch ladder: OOM fallback and the VRAM cap (D16)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from rdv.adapters.training.autobatch import (
    DEFAULT_LADDER,
    AutoBatchResult,
    MicroBatchError,
    choose_micro_batch,
)

pytestmark = pytest.mark.train


class _OutOfMemory(RuntimeError):
    pass


def test_ladder_order_is_the_documented_one() -> None:
    assert DEFAULT_LADDER == (16, 12, 8, 6, 4)


def test_picks_the_largest_size_that_fits_the_cap() -> None:
    def probe(size: int) -> int:
        return size * 1_000

    result = choose_micro_batch(probe, cap_bytes=12_500)

    assert result == AutoBatchResult(micro_batch=12, peak_bytes=12_000)


def test_falls_back_when_the_probe_runs_out_of_memory() -> None:
    attempted: list[int] = []

    def probe(size: int) -> int:
        attempted.append(size)
        if size > 8:
            raise _OutOfMemory("CUDA out of memory")
        return 1_000

    result = choose_micro_batch(probe, cap_bytes=10_000, retryable=(_OutOfMemory,))

    assert result.micro_batch == 8
    assert attempted == [16, 12, 8]


def test_raises_when_every_size_fails() -> None:
    def probe(size: int) -> int:
        raise _OutOfMemory("nope")

    with pytest.raises(MicroBatchError, match="micro-batch"):
        choose_micro_batch(probe, retryable=(_OutOfMemory,))


def test_empty_ladder_is_rejected() -> None:
    with pytest.raises(ValueError, match="ladder"):
        choose_micro_batch(lambda size: 0, ladder=())


def test_calls_empty_cache_between_attempts() -> None:
    calls: list[int] = []

    def probe(size: int) -> int:
        if size > 6:
            raise _OutOfMemory
        return 1

    choose_micro_batch(
        probe,
        retryable=(_OutOfMemory,),
        empty_cache=lambda: calls.append(1),
    )

    assert calls == [1, 1, 1]
