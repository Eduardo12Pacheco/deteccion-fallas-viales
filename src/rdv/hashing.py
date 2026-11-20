"""File digest helpers shared by adapters.

Kept outside the hexagonal layers because both the inference adapter (model
identity) and the model store (download verification) need the same exact
hashing behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of ``path``.

    Raises:
        OSError: the file cannot be read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
