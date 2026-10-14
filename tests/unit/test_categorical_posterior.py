"""Exact enumeration of the clean-posterior categorical reverse mixture."""

from __future__ import annotations

import pytest
import torch

from gpcr_ringgen.models.diffusion import DDPMConfig, DDPMNoiseSchedule, categorical_reverse_step
from gpcr_ringgen.utils.seed import make_torch_generator


def _schedule(steps: int, beta_start: float, beta_end: float) -> DDPMNoiseSchedule:
    return DDPMNoiseSchedule(DDPMConfig(
        num_timesteps=steps,
        beta_schedule="linear",
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type="epsilon",
        loss_reduction="mean",
    ))


def _enumerated_mixture(logits: torch.Tensor, current: int, timestep: int,
                        schedule: DDPMNoiseSchedule) -> torch.Tensor:
    """Build full transition matrices independently of the implementation."""
    classes = logits.shape[-1]
    clean_probabilities = logits.to(torch.float64).softmax(-1)
    if timestep == 0:
        return clean_probabilities
    identity = torch.eye(classes, dtype=torch.float64)
    uniform = torch.ones(classes, classes, dtype=torch.float64) / classes
    previous = identity.clone()
    for step in range(timestep):
        beta = schedule.betas[step].item()
        previous = previous @ ((1.0 - beta) * identity + beta * uniform)
    beta = schedule.betas[timestep].item()
    transition = (1.0 - beta) * identity + beta * uniform
    mixture = torch.zeros(classes, dtype=torch.float64)
    for clean in range(classes):
        conditional = previous[clean] * transition[:, current]
        conditional = conditional / conditional.sum()
        mixture += clean_probabilities[clean] * conditional
    return mixture


def test_reported_two_class_posterior_matches_exact_fractions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []

    def capture(probabilities, count, **kwargs):
        captured.append(probabilities.detach().clone())
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture)
    categorical_reverse_step(
        torch.tensor([[0]]), torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.tensor([1]), _schedule(2, 0.2, 0.5), torch.tensor([[True]]), 2,
    )
    assert torch.allclose(captured[0][0], torch.tensor([17 / 28, 11 / 28], dtype=torch.float64), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("classes", [2, 3, 4, 12])
@pytest.mark.parametrize("betas", [(0.2, 0.5), (1e-5, 0.02), (0.8, 0.999)], ids=["moderate", "small", "large"])
def test_reverse_mixture_matches_matrix_enumeration(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, classes: int, betas: tuple[float, float]
) -> None:
    schedule = _schedule(4, *betas)
    timestep = torch.arange(4)
    # Include every possible current label, plus a padded entry in each row.
    labels = torch.arange(classes + 1).remainder(classes).repeat(4, 1)
    labels[:, -1] = 0
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, -1] = False
    logits = torch.randn(4, classes + 1, classes, dtype=dtype, generator=make_torch_generator(31))
    # Also exercise an almost-deterministic clean prediction.
    logits[2, 0] = -1000
    logits[2, 0, -1] = 1000
    generator = make_torch_generator(7)
    captured = []

    def capture(probabilities, count, *, generator):
        assert count == 1
        captured.append((probabilities.detach().clone(), generator))
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", capture)
    original_labels, original_mask = labels.clone(), mask.clone()
    sampled = categorical_reverse_step(labels, logits, timestep, schedule, mask, classes, generator=generator)
    actual = captured[0][0].reshape_as(logits)
    expected = torch.stack([
        torch.stack([_enumerated_mixture(logits[batch, position], labels[batch, position].item(), batch, schedule)
                     for position in range(classes + 1)])
        for batch in range(4)
    ])
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(actual.to(torch.float64), expected, atol=tolerance, rtol=tolerance)
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    assert torch.allclose(actual.sum(-1), torch.ones_like(actual.sum(-1)), atol=tolerance, rtol=tolerance)
    assert captured[0][1] is generator
    assert torch.equal(labels, original_labels)
    assert torch.equal(mask, original_mask)
    assert sampled.shape == labels.shape
    assert torch.count_nonzero(sampled[~mask]) == 0


def test_reverse_step_seed_is_reproducible() -> None:
    logits = torch.randn(2, 32, 4, generator=make_torch_generator(12))
    labels = torch.arange(32).remainder(4).repeat(2, 1)
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[1, -4:] = False
    labels[~mask] = 0
    arguments = (labels, logits, torch.tensor([0, 3]), _schedule(4, 0.2, 0.5), mask, 4)
    first = categorical_reverse_step(*arguments, generator=make_torch_generator(21))
    second = categorical_reverse_step(*arguments, generator=make_torch_generator(21))
    assert torch.equal(first, second)
    assert torch.count_nonzero(first[~mask]) == 0
