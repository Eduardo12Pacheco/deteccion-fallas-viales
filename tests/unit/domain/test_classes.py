"""The four road-damage classes are a closed, ordered set (decision D4)."""

from __future__ import annotations

import pytest

from rdv.domain.classes import DamageClass


def test_class_ids_match_the_annotation_order() -> None:
    assert [int(damage_class) for damage_class in DamageClass] == [0, 1, 2, 3]
    assert [damage_class.code for damage_class in DamageClass] == ["D00", "D10", "D20", "D40"]


def test_class_display_names_are_human_readable() -> None:
    assert DamageClass.D00.display_name == "Longitudinal crack"
    assert DamageClass.D10.display_name == "Transverse crack"
    assert DamageClass.D20.display_name == "Alligator crack"
    assert DamageClass.D40.display_name == "Pothole"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("D00", DamageClass.D00),
        ("d10", DamageClass.D10),
        (" D20 ", DamageClass.D20),
        ("D40", DamageClass.D40),
    ],
)
def test_from_code_is_case_and_whitespace_insensitive(code: str, expected: DamageClass) -> None:
    assert DamageClass.from_code(code) is expected


@pytest.mark.parametrize("code", ["", "D30", "pothole", "D400"])
def test_from_code_rejects_unknown_labels(code: str) -> None:
    with pytest.raises(ValueError, match="Unknown damage class"):
        DamageClass.from_code(code)


def test_damage_class_constructs_from_its_id() -> None:
    assert DamageClass(2) is DamageClass.D20
    with pytest.raises(ValueError):
        DamageClass(4)
