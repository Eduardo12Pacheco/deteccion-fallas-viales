"""Exponential moving average of the trainable weights (decision D16).

The shadow copy starts at the initial weights and is updated after every
optimizer step with ``decay_t = min(decay, (1 + t) / (10 + t))`` while the
warmup is active. The closed form makes the schedule testable without a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def decay_for_step(step: int, decay: float, *, warmup: bool) -> float:
    """Return the EMA decay applied at zero-based optimizer step ``step``."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step!r}")
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must be within (0, 1), got {decay!r}")
    if not warmup:
        return decay
    return min(decay, (1.0 + step) / (10.0 + step))


class ModelEMA:
    """Shadow copy of a model's floating-point parameters and buffers."""

    def __init__(
        self, model: torch.nn.Module, *, decay: float = 0.9999, warmup: bool = True
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be within (0, 1), got {decay!r}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.step = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: torch.nn.Module, step: int | None = None) -> None:
        """Blend the current model weights into the shadow copy."""
        current = step if step is not None else self.step
        decay = decay_for_step(current, self.decay, warmup=self.warmup)
        state = model.state_dict()
        with torch.no_grad():
            for name in self.shadow:
                value = state[name]
                if not value.is_floating_point():
                    continue
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step is None:
            self.step += 1

    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy the shadow weights into ``model`` (used to save the best EMA)."""
        state = model.state_dict()
        with torch.no_grad():
            for name, value in self.shadow.items():
                if name in state and state[name].shape == value.shape:
                    state[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        """Serializable state for ``last.pt``."""
        return {
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
            "step": self.step,
            "decay": self.decay,
            "warmup": self.warmup,
        }

    @classmethod
    def from_state_dict(
        cls,
        model: torch.nn.Module,
        state: Mapping[str, Any],
    ) -> ModelEMA:
        """Rebuild an EMA from :meth:`state_dict` output."""
        ema = cls(
            model,
            decay=float(state.get("decay", 0.9999)),
            warmup=bool(state.get("warmup", True)),
        )
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA state is missing the shadow mapping")
        ema.shadow = {
            name: value.detach().clone()
            for name, value in shadow.items()
            if isinstance(value, torch.Tensor)
        }
        ema.step = int(state.get("step", 0))
        return ema
