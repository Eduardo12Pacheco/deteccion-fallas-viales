"""road-damage-vision: reproducible road-damage detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("road-damage-vision")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.1.0"

__all__ = ["__version__"]
