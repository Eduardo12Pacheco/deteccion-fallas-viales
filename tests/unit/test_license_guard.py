"""The installed runtime stack must not declare AGPL/GPL licenses.

The scan resolves the project's runtime dependency closure from
``pyproject.toml`` (base dependencies plus every optional runtime extra) and
inspects the metadata of the installed distributions. Development and build
tooling groups are excluded on purpose: PyInstaller is GPL-2.0 with the
bootloader exception and is never shipped as a runtime dependency (S3).
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]

# A permissive/weak-copyleft exception that is acceptable in binary form and
# documented in THIRD_PARTY_LICENSES.md. The guard only rejects GPL/AGPL.
_COPYLEFT = re.compile(
    r"(?<![A-Za-z])(A?GPL(?:-?\d[\w.\-+]*)?|GNU (?:Affero |Lesser )?General Public License)",
    re.IGNORECASE,
)


class LicenseGuardError(AssertionError):
    """Raised when a runtime dependency declares AGPL/GPL."""


@dataclass(frozen=True)
class DistributionLicense:
    """The license facts the guard needs from one installed distribution."""

    name: str
    version: str
    expression: str | None
    classifiers: tuple[str, ...]
    raw: str


def _metadata_facts(distribution: Any) -> DistributionLicense:
    meta = distribution.metadata
    classifiers = tuple(meta.get_all("Classifier") or ())
    return DistributionLicense(
        name=meta.get("Name") or "unknown",
        version=meta.get("Version") or "0",
        expression=meta.get("License-Expression"),
        classifiers=classifiers,
        raw=(meta.get("License") or "").strip(),
    )


def license_facets(entry: DistributionLicense) -> list[str]:
    """Return the short license statements that describe a distribution.

    Long ``License`` fields often embed the full text of several bundled
    third-party licenses (SciPy is the clearest example), so the field is only
    used when it is a short statement and no classifier/expression exists.
    """

    facets: list[str] = []
    if entry.expression:
        facets.append(entry.expression)
    facets.extend(
        classifier.rsplit("::", 1)[-1].strip()
        for classifier in entry.classifiers
        if classifier.startswith("License")
    )
    if not facets and entry.raw and len(entry.raw) <= 160:
        facets.append(entry.raw)
    return facets


def copyleft_violations(entries: list[DistributionLicense]) -> list[DistributionLicense]:
    """Return the entries whose license facets declare GPL or AGPL."""

    violations: list[DistributionLicense] = []
    for entry in entries:
        for facet in license_facets(entry):
            for match in _COPYLEFT.finditer(facet):
                token = match.group(0).lower()
                if token.startswith("lgpl") or token.startswith("gnu lesser"):
                    continue
                violations.append(entry)
                break
            else:
                continue
            break
    return violations


def _project_roots() -> list[tuple[Requirement, bool]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    roots = [(Requirement(item), False) for item in project["dependencies"]]
    for requirements in project.get("optional-dependencies", {}).values():
        for item in requirements:
            requirement = Requirement(item)
            # Optional extras are only present when installed; the closure
            # below skips anything that is not, so their markers do not apply.
            requirement.marker = None
            roots.append((requirement, True))
    return roots


def runtime_distributions() -> list[DistributionLicense]:
    """Resolve the installed runtime dependency closure and its licenses."""

    environment = default_environment()
    seen: dict[str, DistributionLicense] = {}
    missing_required: list[str] = []
    stack = _project_roots()
    while stack:
        requirement, optional = stack.pop()
        try:
            distribution = metadata.distribution(requirement.name)
        except metadata.PackageNotFoundError:
            if optional:
                continue
            missing_required.append(requirement.name)
            continue
        facts = _metadata_facts(distribution)
        if facts.name in seen:
            continue
        seen[facts.name] = facts
        for raw in distribution.requires or ():
            dependency = Requirement(raw)
            if dependency.marker is not None and not dependency.marker.evaluate(
                {**environment, "extra": ""}
            ):
                continue
            stack.append((dependency, False))
    if missing_required:
        raise LicenseGuardError(
            "base runtime dependencies from pyproject.toml are not installed: "
            + ", ".join(sorted(missing_required))
        )
    return list(seen.values())


def assert_runtime_licenses_clean() -> None:
    """Fail when any installed runtime dependency declares AGPL/GPL."""

    violations = copyleft_violations(runtime_distributions())
    if violations:
        rendered = ", ".join(f"{item.name} {item.version}" for item in violations)
        raise LicenseGuardError(f"runtime dependencies with GPL/AGPL licenses: {rendered}")


def test_runtime_dependency_licenses_are_clean() -> None:
    entries = runtime_distributions()
    assert len(entries) >= 20
    violations = copyleft_violations(entries)
    assert violations == [], "runtime dependencies declare GPL/AGPL: " + ", ".join(
        f"{item.name} {item.version}" for item in violations
    )


def test_agpl_fake_distribution_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = DistributionLicense(
        name="fake-agpl-package",
        version="1.0.0",
        expression="AGPL-3.0-only",
        classifiers=(),
        raw="AGPL-3.0",
    )
    monkeypatch.setattr(
        "tests.unit.test_license_guard.runtime_distributions",
        lambda: [fake],
    )
    with pytest.raises(LicenseGuardError, match="fake-agpl-package"):
        assert_runtime_licenses_clean()


def test_lgpl_is_not_treated_as_gpl() -> None:
    entry = DistributionLicense(
        name="opencv-python-headless",
        version="5.1.0.95",
        expression=None,
        classifiers=(),
        raw="LGPL-2.1-or-later",
    )
    assert copyleft_violations([entry]) == []
