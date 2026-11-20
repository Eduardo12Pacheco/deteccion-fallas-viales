"""Model EMA: warmup ramp and the exact closed-form update (D16)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from rdv.adapters.training.ema import ModelEMA, decay_for_step

pytestmark = pytest.mark.train


def test_decay_uses_the_documented_warmup_ramp() -> None:
    assert decay_for_step(0, 0.9999, warmup=True) == pytest.approx(0.1)
    assert decay_for_step(9, 0.9999, warmup=True) == pytest.approx(10 / 19)
    assert decay_for_step(500, 0.9999, warmup=True) == pytest.approx(501 / 510)
    assert decay_for_step(90_000, 0.9999, warmup=True) == pytest.approx(0.9999)
    assert decay_for_step(0, 0.9999, warmup=False) == pytest.approx(0.9999)


def test_update_matches_the_closed_form() -> None:
    model = torch.nn.Linear(2, 1, bias=True)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    ema = ModelEMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    ema.update(model)

    for name, ema_value in ema.shadow.items():
        model_value = model.state_dict()[name]
        assert torch.allclose(ema_value, 0.9 * initial[name] + 0.1 * model_value)


def test_update_honours_the_warmup_step() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    ema = ModelEMA(model, decay=0.999, warmup=True)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model, step=0)

    # decay_t = min(0.999, (1 + 0) / (10 + 0)) = 0.1
    expected = 0.1 * initial + 0.9 * torch.full_like(initial, 2.0)
    assert torch.allclose(ema.shadow["weight"], expected)


def test_state_dict_round_trips() -> None:
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)

    restored = ModelEMA.from_state_dict(model, ema.state_dict())

    for left, right in zip(ema.shadow.values(), restored.shadow.values(), strict=True):
        assert torch.equal(left, right)


def test_copy_to_restores_the_shadow_weights() -> None:
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.9, warmup=False)

    ema.copy_to(model)

    for shadow, value in zip(ema.shadow.values(), model.state_dict().values(), strict=True):
        assert torch.equal(shadow, value)
