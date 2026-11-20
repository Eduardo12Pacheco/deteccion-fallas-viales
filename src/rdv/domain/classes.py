"""The four road-damage classes used by the detector (decision D4).

``DamageClass`` is both the annotation label (``D00``..``D40``) and the model
class id (``0``..``3``); keeping one enum prevents two sources of truth between
the dataset parser, the decoder, and the reports.
"""

from __future__ import annotations

from enum import IntEnum


class DamageClass(IntEnum):
    """Road-damage categories, ordered exactly like the annotation ids."""

    D00 = 0
    D10 = 1
    D20 = 2
    D40 = 3

    @property
    def code(self) -> str:
        """Annotation label used in the RDD2022 XML files."""
        return self.name

    @property
    def display_name(self) -> str:
        """Human-readable name used in reports and the UI."""
        return _DISPLAY_NAMES[self]

    @classmethod
    def from_code(cls, code: str) -> DamageClass:
        """Resolve an annotation label, ignoring case and surrounding space.

        Raises:
            ValueError: the label is not one of the four supported classes.
        """
        normalized = code.strip().upper()
        try:
            return cls[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown damage class: {code!r}") from exc


_DISPLAY_NAMES: dict[DamageClass, str] = {
    DamageClass.D00: "Longitudinal crack",
    DamageClass.D10: "Transverse crack",
    DamageClass.D20: "Alligator crack",
    DamageClass.D40: "Pothole",
}
