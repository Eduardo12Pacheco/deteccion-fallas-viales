"""Explicit error hierarchy for every expected failure mode.

Each error carries the process exit code documented in the project plan so the
CLI and the library raise the same observable contract. Usage errors exit with
code 2 (handled by Typer), unexpected errors with code 1.
"""


class RdvError(Exception):
    """Base class for expected, user-facing failures."""

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DatasetError(RdvError):
    """A dataset archive, annotation, or split is missing or invalid."""

    exit_code = 3


class ModelError(RdvError):
    """A model file is missing, corrupt, or incompatible."""

    exit_code = 4


class MediaError(RdvError):
    """An image or video cannot be decoded or written."""

    exit_code = 5


class ConfigError(RdvError):
    """Configuration values are missing or invalid."""

    exit_code = 6


class MissingExtraError(RdvError):
    """An optional dependency extra is required for the requested operation."""

    exit_code = 6
