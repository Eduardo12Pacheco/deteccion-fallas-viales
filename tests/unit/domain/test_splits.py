"""Seeded block splits keep near-duplicate frames in the same split (D3)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rdv.domain.splits import SplitAssignment, block_split


def _ids(count: int) -> list[str]:
    return [f"Japan/{index:06d}.jpg" for index in range(count)]


def test_split_counts_are_exact_for_a_thousand_ids() -> None:
    split = block_split(_ids(1000))

    assert len(split.train) == 800
    assert len(split.val) == 100
    assert len(split.test) == 100


def test_split_is_disjoint_and_covers_every_id() -> None:
    ids = _ids(231)
    split = block_split(ids, block_size=25, seed=7)

    assigned = set(split.train) | set(split.val) | set(split.test)
    assert assigned == set(ids)
    assert len(split.train) + len(split.val) + len(split.test) == len(ids)


def test_split_output_is_sorted_within_each_partition() -> None:
    split = block_split(_ids(500), seed=3)

    assert list(split.train) == sorted(split.train)
    assert list(split.val) == sorted(split.val)
    assert list(split.test) == sorted(split.test)


def test_split_is_independent_of_input_order() -> None:
    ids = _ids(420)

    assert block_split(ids, seed=42) == block_split(list(reversed(ids)), seed=42)
    assert block_split(ids, seed=42) == block_split(ids[100:] + ids[:100], seed=42)


def test_split_is_deterministic_for_the_same_seed() -> None:
    assert block_split(_ids(300), seed=42) == block_split(_ids(300), seed=42)


def test_different_seeds_produce_different_assignments() -> None:
    assert block_split(_ids(1000), seed=42) != block_split(_ids(1000), seed=7)


def test_ratios_stay_within_one_block_of_the_target() -> None:
    ids = _ids(1050)
    block_size = 100
    ratios = (0.8, 0.1, 0.1)
    split = block_split(ids, seed=11, block_size=block_size, ratios=ratios)

    for part, ratio in zip((split.train, split.val, split.test), ratios, strict=True):
        assert abs(len(part) - len(ids) * ratio) <= block_size


def test_empty_input_yields_empty_parts() -> None:
    assert block_split([]) == SplitAssignment((), (), ())


def test_as_dict_lists_all_three_parts() -> None:
    split = block_split(_ids(10))

    assert split.as_dict() == {
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
    }


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_size_must_be_positive(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size"):
        block_split(_ids(10), block_size=block_size)


@pytest.mark.parametrize(
    "ratios",
    [
        (0.8, 0.1),
        (0.8, 0.1, 0.1, 0.0),
        (-0.1, 0.6, 0.5),
        (0.8, 0.3, -0.1),
        (0.8, 0.1, 0.05),
        (0.5, 0.5, 0.5),
    ],
)
def test_ratios_must_form_a_valid_three_way_distribution(
    ratios: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="ratios"):
        block_split(_ids(10), ratios=ratios)  # type: ignore[arg-type]


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        block_split(["a.jpg", "a.jpg"])


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_ids_are_rejected(blank: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        block_split(["a.jpg", blank])


def test_zero_ratios_leave_a_partition_empty() -> None:
    split = block_split(_ids(30), ratios=(1.0, 0.0, 0.0))

    assert len(split.train) == 30
    assert split.val == ()
    assert split.test == ()


ids_strategy = st.lists(
    st.text(min_size=1, max_size=12).filter(lambda value: value.strip() == value),
    unique=True,
    max_size=60,
)


@given(
    ids=ids_strategy,
    block_size=st.integers(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_generated_splits_are_disjoint_covering_and_deterministic(
    ids: list[str], block_size: int, seed: int
) -> None:
    ratios = (0.8, 0.1, 0.1)
    split = block_split(ids, seed=seed, block_size=block_size, ratios=ratios)

    parts = [split.train, split.val, split.test]
    assert set().union(*map(set, parts)) == set(ids)
    assert sum(len(part) for part in parts) == len(ids)
    for part, ratio in zip(parts, ratios, strict=True):
        assert abs(len(part) - len(ids) * ratio) <= block_size
    assert split == block_split(list(reversed(ids)), seed=seed, block_size=block_size)
