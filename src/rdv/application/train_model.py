"""Train use case: a thin command object over the ``Trainer`` port (T9, D16).

The application layer never imports torch or PyYAML. The concrete
``RtDetrTrainer`` adapter is built by the composition root and passed in, which
keeps the base install torch-free and makes the CLI order easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """What to do with an already configured trainer.

    Attributes:
        run_dir: directory receiving checkpoints and logs.
        resume: ``auto`` (use ``last.pt`` when present), ``none``, or an
            explicit checkpoint path.
        benchmark_only: measure throughput/memory instead of fine-tuning.
        dry_run: validate config, dataset and model instantiation only.
        max_steps: optional optimizer-step cap for smoke tests; training stops
            at the end of the epoch that reaches it.
    """

    run_dir: Path
    resume: str = "auto"
    benchmark_only: bool = False
    dry_run: bool = False
    max_steps: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None")
        if not self.resume.strip():
            raise ValueError("resume must be 'auto', 'none' or a path")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Outcome of a dry run, a benchmark or a fine-tune."""

    run_dir: Path
    epochs_completed: int
    steps: int
    last_loss: float | None = None
    best_metric: float | None = None
    benchmark: dict[str, Any] | None = None
    dry_run: bool = False
    losses: tuple[float, ...] = field(default_factory=tuple)


class Trainer(Protocol):
    """Port implemented by the fine-tuning adapter."""

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Run the requested training mode and report the result."""
        ...


def train_model(trainer: Trainer, request: TrainingRequest) -> TrainingResult:
    """Execute one training request through the injected adapter."""
    return trainer.train(request)
