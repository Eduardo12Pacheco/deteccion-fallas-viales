"""Exit codes are part of the public error contract (decision D19)."""

from __future__ import annotations

import pytest

from rdv.errors import (
    ConfigError,
    DatasetError,
    MediaError,
    MissingExtraError,
    ModelError,
    RdvError,
)


@pytest.mark.parametrize(
    ("error_type", "exit_code"),
    [
        (DatasetError, 3),
        (ModelError, 4),
        (MediaError, 5),
        (ConfigError, 6),
        (MissingExtraError, 6),
    ],
)
def test_expected_errors_carry_their_exit_code(error_type: type[RdvError], exit_code: int) -> None:
    error = error_type("boom")

    assert isinstance(error, RdvError)
    assert error.exit_code == exit_code
    assert error.message == "boom"
    assert str(error) == "boom"
