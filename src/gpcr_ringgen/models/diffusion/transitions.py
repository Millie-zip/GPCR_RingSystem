"""Forward coordinate diffusion utilities for Stage 3 DDPM training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from gpcr_ringgen.core.types import DiffusionState, Stage3DiffusionState
from gpcr_ringgen.models.diffusion.schedules import DDPMNoiseSchedule


class DiffusionTransition(ABC):
    """Compatibility extension point for schema-free future transitions."""

    @abstractmethod
    def apply(self, state: DiffusionState, timestep: int) -> DiffusionState:
        """Return a transitioned schema-free state."""


def _validate_atom_tensor(name: str, values: Tensor, state: Stage3DiffusionState) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.shape != state.coordinates.shape:
        raise ValueError(f"{name} must have shape [B, N, 3]")
    if values.dtype != state.coordinates.dtype:
        raise TypeError(f"{name} dtype must match state coordinates")
    if values.device != state.coordinates.device:
        raise ValueError(f"{name} device must match state coordinates")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def project_zero_center_of_mass(values: Tensor, atom_mask: Tensor) -> Tensor:
    """Remove per-sample translation using only valid ligand atoms.

    Args:
        values: Shape ``[B, N, 3]``; floating-point on the active model device.
            It may represent coordinates, sampled noise, or predicted noise.
        atom_mask: Shape ``[B, N]``; ``torch.bool`` on the same device.
            ``True`` positions define atoms included in the center; ``False``
            positions are padding and are returned as exact zeros.

    Returns:
        Shape ``[B, N, 3]`` with the input dtype/device. Every sample's valid
        atom mean is zero up to floating-point error.
    """

    if not isinstance(values, Tensor) or values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("values must be a torch.Tensor with shape [B, N, 3]")
    if not torch.is_floating_point(values):
        raise TypeError("values must use a floating-point dtype")
    if not isinstance(atom_mask, Tensor) or atom_mask.ndim != 2:
        raise ValueError("atom_mask must be a torch.Tensor with shape [B, N]")
    if tuple(atom_mask.shape) != tuple(values.shape[:2]):
        raise ValueError("atom_mask shape must match values [B, N]")
    if atom_mask.dtype != torch.bool:
        raise TypeError("atom_mask must have dtype torch.bool")
    if atom_mask.device != values.device:
        raise ValueError("atom_mask and values must be on the same device")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values contain NaN or Inf")

    weights = atom_mask.unsqueeze(-1).to(values.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("Every sample must contain at least one valid atom")
    center = (values * weights).sum(dim=1, keepdim=True) / counts
    return (values - center) * weights


def sample_timesteps(
    batch_size: int,
    schedule: DDPMNoiseSchedule,
    *,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniformly sample discrete DDPM timesteps.

    Returns a ``torch.long`` tensor with shape ``[B]`` on ``device``. Pass a
    local seeded generator for reproducibility without changing global RNG
    state.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(schedule, DDPMNoiseSchedule):
        raise TypeError("schedule must be a DDPMNoiseSchedule")
    return torch.randint(
        0,
        schedule.num_timesteps,
        (batch_size,),
        dtype=torch.long,
        device=torch.device(device),
        generator=generator,
    )


def sample_coordinate_noise(
    state: Stage3DiffusionState,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample masked isotropic Gaussian noise and remove its translation mode.

    The returned tensor has shape ``[B, N, 3]`` and matches the state's
    coordinate dtype/device. Padding rows are zero and the valid-atom mean of
    every sample is zero.
    """

    if not isinstance(state, Stage3DiffusionState):
        raise TypeError("state must be a Stage3DiffusionState")
    state.validate()
    noise = torch.randn(
        state.coordinates.shape,
        dtype=state.coordinates.dtype,
        device=state.coordinates.device,
        generator=generator,
    )
    return project_zero_center_of_mass(noise, state.mask.atom_mask)


@dataclass(frozen=True)
class ForwardDiffusionOutput:
    """Result and supervised target from a Stage 3 forward noising step.

    Tensor fields:
        noisy_state.coordinates: Shape ``[B, N, 3]``; floating-point on the
            model device. Valid atoms contain ``x_t`` and padded atoms are zero.
        clean_coordinates: Shape ``[B, N, 3]``; same dtype/device. This is the
            input ``x_0`` after valid-atom center-of-mass removal. Padding is zero.
        noise: Shape ``[B, N, 3]``; same dtype/device. This is the centered
            epsilon prediction target. Padding is zero.
        timestep: Available as ``noisy_state.timestep``, shape ``[B]`` and
            ``torch.long``. Atom validity follows ``noisy_state.mask.atom_mask``.
    """

    noisy_state: Stage3DiffusionState
    clean_coordinates: Tensor
    noise: Tensor

    def validate(self) -> None:
        """Validate state alignment, numerical values, and padding semantics."""

        if not isinstance(self.noisy_state, Stage3DiffusionState):
            raise TypeError("noisy_state must be a Stage3DiffusionState")
        self.noisy_state.validate()
        _validate_atom_tensor(
            "noisy coordinates", self.noisy_state.coordinates, self.noisy_state
        )
        _validate_atom_tensor(
            "clean_coordinates", self.clean_coordinates, self.noisy_state
        )
        _validate_atom_tensor("noise", self.noise, self.noisy_state)
        padding = ~self.noisy_state.mask.atom_mask.unsqueeze(-1)
        for name, values in (
            ("noisy coordinates", self.noisy_state.coordinates),
            ("clean_coordinates", self.clean_coordinates),
            ("noise", self.noise),
        ):
            if bool((values.masked_select(padding) != 0).any()):
                raise ValueError(f"{name} must be zero for padded atoms")


def q_sample(
    state: Stage3DiffusionState,
    timestep: Tensor,
    schedule: DDPMNoiseSchedule,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ForwardDiffusionOutput:
    r"""Sample ``q(x_t | x_0)`` for centered ligand coordinates.

    The conventional DDPM equation is
    ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon``.
    Before applying it, both ``x_0`` and epsilon are projected onto the
    zero-center-of-mass subspace defined by ``atom_mask``. This removes the
    arbitrary global translation mode and sets padded atoms to zero.

    Args:
        state: Clean Stage 3 state with coordinates shaped ``[B, N, 3]``.
        timestep: Shape ``[B]``; ``torch.long`` on the state device.
        schedule: Validated DDPM schedule.
        noise: Optional explicit epsilon shaped ``[B, N, 3]`` with the state
            dtype/device. It is centered by this function before use.
        generator: Optional local generator used only when ``noise`` is absent.
    """

    if not isinstance(state, Stage3DiffusionState):
        raise TypeError("state must be a Stage3DiffusionState")
    state.validate()
    if not isinstance(schedule, DDPMNoiseSchedule):
        raise TypeError("schedule must be a DDPMNoiseSchedule")
    if not bool(torch.isfinite(state.coordinates).all()):
        raise ValueError("state coordinates contain NaN or Inf")
    signal_coefficient, noise_coefficient = schedule.coefficients(
        timestep,
        state.coordinates,
    )
    if noise is not None and generator is not None:
        raise ValueError("noise and generator cannot be provided together")
    if noise is None:
        centered_noise = sample_coordinate_noise(state, generator=generator)
    else:
        _validate_atom_tensor("noise", noise, state)
        centered_noise = project_zero_center_of_mass(noise, state.mask.atom_mask)

    clean_coordinates = project_zero_center_of_mass(
        state.coordinates,
        state.mask.atom_mask,
    )
    coordinate_mask = state.mask.atom_mask.unsqueeze(-1).to(state.coordinates.dtype)
    noisy_coordinates = (
        signal_coefficient * clean_coordinates
        + noise_coefficient * centered_noise
    ) * coordinate_mask
    noisy_state = replace(
        state,
        coordinates=noisy_coordinates,
        timestep=timestep.clone(),
    )
    output = ForwardDiffusionOutput(
        noisy_state=noisy_state,
        clean_coordinates=clean_coordinates,
        noise=centered_noise,
    )
    output.validate()
    return output


def training_target(
    forward_output: ForwardDiffusionOutput,
    *,
    prediction_type: str = "epsilon",
) -> Tensor:
    """Return the configured coordinate prediction target.

    The current Stage 3 contract supports only epsilon prediction. The returned
    tensor has shape ``[B, N, 3]`` and inherits dtype/device/mask semantics from
    :attr:`ForwardDiffusionOutput.noise`.
    """

    if not isinstance(forward_output, ForwardDiffusionOutput):
        raise TypeError("forward_output must be a ForwardDiffusionOutput")
    forward_output.validate()
    if prediction_type != "epsilon":
        raise ValueError("prediction_type must be 'epsilon'")
    return forward_output.noise


__all__ = [
    "DiffusionTransition",
    "ForwardDiffusionOutput",
    "project_zero_center_of_mass",
    "q_sample",
    "sample_coordinate_noise",
    "sample_timesteps",
    "training_target",
]
