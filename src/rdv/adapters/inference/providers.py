"""Execution-provider strategy for ONNX Runtime (decision D8).

``RDV_DEVICE=auto`` prefers CUDA when the runtime exposes it and otherwise
falls back to CPU. An explicit ``cuda`` request without the provider is a
``ModelError`` (exit 4) instead of a silent CPU fallback, so users never
believe they are benchmarking the GPU when they are not.
"""

from __future__ import annotations

from collections.abc import Sequence

import onnxruntime as ort

from rdv.errors import ModelError

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"

_SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda"})


def available_providers() -> tuple[str, ...]:
    """Execution providers compiled into the installed ONNX Runtime."""
    return tuple(ort.get_available_providers())


def resolve_providers(
    device: str,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Return the ONNX Runtime provider priority list for ``device``.

    Args:
        device: ``auto``, ``cpu`` or ``cuda``.
        available: provider names to consider; defaults to the runtime's list.

    Returns:
        Providers in priority order. CUDA is always followed by CPU so the
        runtime can fall back for operators without a GPU kernel when the
        device was auto-selected.

    Raises:
        ModelError: ``device`` is unknown or ``cuda`` was requested but the
            runtime has no CUDA provider.
    """
    normalized = device.strip().lower()
    if normalized not in _SUPPORTED_DEVICES:
        expected = ", ".join(sorted(_SUPPORTED_DEVICES))
        raise ModelError(f"Unsupported device {device!r}; expected one of {expected}")
    providers = tuple(available) if available is not None else available_providers()

    if normalized == "cpu":
        return [CPU_PROVIDER]
    if normalized == "cuda":
        if CUDA_PROVIDER not in providers:
            raise ModelError(
                "CUDA inference was requested (RDV_DEVICE=cuda) but ONNX Runtime has no "
                f"{CUDA_PROVIDER}; install the cuda backend or use RDV_DEVICE=cpu"
            )
        return [CUDA_PROVIDER, CPU_PROVIDER]
    if CUDA_PROVIDER in providers:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


def selected_device(providers: Sequence[str]) -> str:
    """Map a resolved provider list to the ``cpu``/``cuda`` device label."""
    return "cuda" if providers and providers[0] == CUDA_PROVIDER else "cpu"
