"""Execution-provider strategy (decision D8)."""

from __future__ import annotations

import pytest

from rdv.adapters.inference.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    resolve_providers,
    selected_device,
)
from rdv.errors import ModelError

BOTH = (CPU_PROVIDER, CUDA_PROVIDER)
CPU_ONLY = (CPU_PROVIDER,)


def test_auto_prefers_cuda_when_the_runtime_exposes_it() -> None:
    assert resolve_providers("auto", BOTH) == [CUDA_PROVIDER, CPU_PROVIDER]
    assert selected_device(resolve_providers("auto", BOTH)) == "cuda"


def test_auto_falls_back_to_cpu() -> None:
    assert resolve_providers("auto", CPU_ONLY) == [CPU_PROVIDER]
    assert selected_device(resolve_providers("auto", CPU_ONLY)) == "cpu"


def test_explicit_cpu_never_uses_cuda() -> None:
    assert resolve_providers("cpu", BOTH) == [CPU_PROVIDER]
    assert selected_device([CPU_PROVIDER]) == "cpu"


def test_explicit_cuda_without_the_provider_is_a_model_error() -> None:
    with pytest.raises(ModelError, match="CUDA inference was requested") as excinfo:
        resolve_providers("cuda", CPU_ONLY)

    assert excinfo.value.exit_code == 4


def test_explicit_cuda_with_the_provider_keeps_cpu_as_fallback() -> None:
    assert resolve_providers("cuda", BOTH) == [CUDA_PROVIDER, CPU_PROVIDER]


def test_device_names_are_case_insensitive() -> None:
    assert resolve_providers(" CPU ", BOTH) == [CPU_PROVIDER]


def test_unknown_device_is_a_model_error() -> None:
    with pytest.raises(ModelError, match="Unsupported device"):
        resolve_providers("tpu", BOTH)
