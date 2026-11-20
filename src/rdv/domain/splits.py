"""Deterministic seeded block splits (decision D3).

RDD2022 test annotations are not public and consecutive image ids often come
from the same capture run, so splitting individual ids would leak near
duplicates across partitions. We sort the ids, cut them into blocks of
consecutive ids, shuffle the blocks with a fixed seed, and assign blocks
greedily to the 80/10/10 targets. The assignment depends only on the set of
ids, the seed, the block size, and the ratios.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_SPLIT_NAMES = ("train", "val", "test")
_DEFAULT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Image ids assigned to the train, validation and test partitions."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        """Return a JSON-ready mapping of partition name to sorted ids."""
        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }


def _validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != len(_SPLIT_NAMES):
        raise ValueError(
            f"ratios must have exactly three values (train, val, test), got {len(ratios)}"
        )
    if not all(math.isfinite(ratio) and ratio >= 0.0 for ratio in ratios):
        raise ValueError(f"ratios must be finite and non-negative, got {tuple(ratios)!r}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)!r}")


def block_split(
    ids: Iterable[str],
    *,
    seed: int = 42,
    block_size: int = 100,
    ratios: Sequence[float] = _DEFAULT_RATIOS,
) -> SplitAssignment:
    """Split ``ids`` into train/val/test by shuffled blocks of consecutive ids.

    Args:
        ids: dataset image ids; duplicates are rejected.
        seed: seed of the block shuffle (the project default is 42).
        block_size: number of consecutive sorted ids per block.
        ratios: train/val/test proportions; must be three non-negative values
            that sum to one.

    Returns:
        Sorted, disjoint tuples for each partition covering every input id.

    Raises:
        ValueError: ids are duplicated or blank, the block size is not
            positive, or the ratios are not a valid three-way distribution.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size!r}")
    _validate_ratios(ratios)

    unique_ids = list(ids)
    seen: set[str] = set()
    for image_id in unique_ids:
        if not image_id.strip():
            raise ValueError("image ids must be non-empty strings")
        if image_id in seen:
            raise ValueError(f"duplicate image id: {image_id!r}")
        seen.add(image_id)
    if not unique_ids:
        return SplitAssignment((), (), ())

    ordered = sorted(unique_ids)
    blocks = [ordered[start : start + block_size] for start in range(0, len(ordered), block_size)]
    random.Random(seed).shuffle(blocks)

    targets = [len(ordered) * ratio for ratio in ratios]
    counts = [0, 0, 0]
    assigned: list[list[str]] = [[], [], []]
    for block in blocks:
        deficits = [targets[index] - counts[index] for index in range(len(targets))]
        chosen = max(range(len(targets)), key=lambda index: deficits[index])
        assigned[chosen].extend(block)
        counts[chosen] += len(block)

    train, val, test = (tuple(sorted(part)) for part in assigned)
    return SplitAssignment(train=train, val=val, test=test)
