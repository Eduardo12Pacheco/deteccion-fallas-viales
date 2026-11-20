"""PyInstaller entry script for the portable desktop bundle (S15/D12).

The spec collects the package, the web assets and exactly one ONNX model, so
this script only hands control to the shared launcher in
``rdv.entrypoints.desktop``.
"""

from __future__ import annotations

import sys

from rdv.entrypoints.desktop import main

if __name__ == "__main__":
    sys.exit(main())
