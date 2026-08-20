"""Fixed-topology categorical/coordinate diffusion for chemical realization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from gpcr_ringgen.core.types import Stage3DiffusionState, Stage3Input
from gpcr_ringgen.models.diffusion.schedules import DDPMNoiseSchedule
from gpcr_ringgen.models.diffusion.state import initialize_stage3_state
from gpcr_ringgen.models.diffusion.transitions import q_sample
from gpcr_ringgen.utils.seed import make_torch_generator


def _validate_category_tensor(
    name: str,
    values: Tensor,
    mask: Tensor,
    num_classes: int,
) -> None:
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise ValueError(f"{name} must be a torch.Tensor with shape [B, L]")
    if values.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if values.shape != mask.shape:
        raise ValueError(f"{name} shape must match its validity mask")
    if values.device != mask.device:
        raise ValueError(f"{name} and its validity mask must share a device")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    valid_values = values.masked_select(mask)
    if valid_values.numel() and bool(
        ((valid_values < 0) | (valid_values >= num_classes)).any()
    ):
        raise ValueError(f"{name} contains a category outside [0, {num_classes - 1}]")
    if bool((values.masked_select(~mask) != 0).any()):
        raise ValueError(f"{name} must be zero where its validity mask is false")


def _validate_timestep(
    timestep: Tensor,
    schedule: DDPMNoiseSchedule,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(timestep, Tensor) or timestep.ndim != 1:
        raise ValueError("timestep must be a torch.Tensor with shape [B]")
    if timestep.dtype != torch.long:
        raise TypeError("timestep must have dtype torch.long")
    if timestep.shape[0] != batch_size:
        raise ValueError("timestep must have shape [B]")
    if timestep.device != device:
        raise ValueError("timestep and categorical state must share a device")
    if bool(((timestep < 0) | (timestep >= schedule.num_timesteps)).any()):
        raise ValueError("timestep values are outside the configured schedule")


def _validate_fixed_chemical_topology(state: Stage3DiffusionState) -> None:
    """Require one stored edge per undirected chemical bond.

    The existing EGNN consumes every stored edge in both directions. Joint
    bond diffusion therefore assigns exactly one label to each unordered atom
    pair and rejects self-loops or duplicated reverse entries.
    """

    for batch_index in range(state.batch_size):
        valid = state.mask.edge_mask[batch_index]
        endpoints = state.edge_index[batch_index, :, valid].transpose(0, 1)
        if endpoints.numel() == 0:
            continue
        if bool((endpoints[:, 0] == endpoints[:, 1]).any()):
            raise ValueError("fixed chemical topology cannot contain self-loop edges")
        unordered = torch.sort(endpoints, dim=-1).values
        if torch.unique(unordered, dim=0).shape[0] != unordered.shape[0]:
            raise ValueError(
                "fixed chemical topology must store each undirected edge once"
            )


def sample_uniform_categories(
    mask: Tensor,
    num_classes: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample categorical labels on a frozen set of valid nodes or edges.

    Args:
        mask: Shape ``[B, L]``; ``torch.bool``. ``True`` entries are fixed
            topology nodes or edges that receive a label. ``False`` padding is
            returned as category zero and never becomes active.
        num_classes: Number of chemical label classes. This vocabulary does
            not include node/edge existence because topology is frozen.
    """

    if not isinstance(mask, Tensor) or mask.ndim != 2:
        raise ValueError("mask must be a torch.Tensor with shape [B, L]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must have dtype torch.bool")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    sampled = torch.randint(
        0,
        num_classes,
        mask.shape,
        dtype=torch.long,
        device=mask.device,
        generator=generator,
    )
    return sampled * mask.to(torch.long)


def q_sample_categorical(
    clean_labels: Tensor,
    timestep: Tensor,
    schedule: DDPMNoiseSchedule,
    mask: Tensor,
    num_classes: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Sample a uniform categorical corruption on fixed topology positions.

    The cumulative transition is
    ``q(z_t | z_0) = alpha_bar_t * one_hot(z_0) + (1-alpha_bar_t) / K``.
    Consequently labels approach a uniform categorical prior at large
    timesteps while the node/edge validity mask and ``edge_index`` never change.
    """

    _validate_category_tensor("clean_labels", clean_labels, mask, num_classes)
    _validate_timestep(
        timestep,
        schedule,
        batch_size=clean_labels.shape[0],
        device=clean_labels.device,
    )
    alpha_bar = schedule.alpha_bars.index_select(
        0,
        timestep.to(schedule.alpha_bars.device),
    ).to(device=clean_labels.device, dtype=torch.float64).unsqueeze(-1)
    keep = torch.rand(
        clean_labels.shape,
        dtype=torch.float64,
        device=clean_labels.device,
        generator=generator,
    ) < alpha_bar
    uniform = sample_uniform_categories(
        mask,
        num_classes,
        generator=generator,
    )
    noisy = torch.where(keep, clean_labels, uniform)
    return noisy * mask.to(torch.long)


@dataclass(frozen=True)
class JointForwardDiffusionOutput:
    """Forward target for fixed-topology chemical and coordinate diffusion.

    Tensor fields:
        noisy_state.atom_types: ``[B, N]`` long categorical labels. Padding is
            zero and ``atom_mask`` is unchanged.
        noisy_state.bond_types: ``[B, E]`` long labels on existing topology
            edges. Padding is zero; ``edge_index``/``edge_mask`` are unchanged.
        coordinate_noise: ``[B, N, 3]`` floating epsilon target.
        clean_atom_types: ``[B, N]`` long categorical reconstruction target.
        clean_bond_types: ``[B, E]`` long categorical reconstruction target.
    """

    noisy_state: Stage3DiffusionState
    coordinate_noise: Tensor
    clean_atom_types: Tensor
    clean_bond_types: Tensor

    def validate(self, schedule: DDPMNoiseSchedule) -> None:
        self.noisy_state.validate()
        config = schedule.config
        if config.mode != "joint_chemical":
            raise ValueError("joint output requires diffusion mode joint_chemical")
        if self.noisy_state.atom_types is None or self.noisy_state.bond_types is None:
            raise ValueError("joint noisy state requires atom_types and bond_types")
        if config.num_atom_types is None or config.num_bond_types is None:
            raise ValueError("joint diffusion class counts are required")
        _validate_category_tensor(
            "noisy atom_types",
            self.noisy_state.atom_types,
            self.noisy_state.mask.atom_mask,
            config.num_atom_types,
        )
        _validate_category_tensor(
            "noisy bond_types",
            self.noisy_state.bond_types,
            self.noisy_state.mask.edge_mask,
            config.num_bond_types,
        )
        _validate_category_tensor(
            "clean_atom_types",
            self.clean_atom_types,
            self.noisy_state.mask.atom_mask,
            config.num_atom_types,
        )
        _validate_category_tensor(
            "clean_bond_types",
            self.clean_bond_types,
            self.noisy_state.mask.edge_mask,
            config.num_bond_types,
        )
        if self.coordinate_noise.shape != self.noisy_state.coordinates.shape:
            raise ValueError("coordinate_noise must have shape [B, N, 3]")
        if self.coordinate_noise.dtype != self.noisy_state.coordinates.dtype:
            raise TypeError("coordinate_noise dtype must match coordinates")
        if self.coordinate_noise.device != self.noisy_state.coordinates.device:
            raise ValueError("coordinate_noise device must match coordinates")


def q_sample_joint(
    state: Stage3DiffusionState,
    timestep: Tensor,
    schedule: DDPMNoiseSchedule,
    *,
    coordinate_noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> JointForwardDiffusionOutput:
    """Jointly corrupt atom labels, fixed-edge bond labels, and coordinates."""

    state.validate()
    _validate_fixed_chemical_topology(state)
    config = schedule.config
    if config.mode != "joint_chemical":
        raise ValueError("q_sample_joint requires diffusion mode joint_chemical")
    if state.atom_types is None or state.bond_types is None:
        raise ValueError("clean state requires atom_types and bond_types")
    if config.num_atom_types is None or config.num_bond_types is None:
        raise ValueError("joint diffusion class counts are required")
    coordinate_output = q_sample(
        state,
        timestep,
        schedule,
        noise=coordinate_noise,
        generator=generator if coordinate_noise is None else None,
    )
    noisy_atom_types = q_sample_categorical(
        state.atom_types,
        timestep,
        schedule,
        state.mask.atom_mask,
        config.num_atom_types,
        generator=generator,
    )
    noisy_bond_types = q_sample_categorical(
        state.bond_types,
        timestep,
        schedule,
        state.mask.edge_mask,
        config.num_bond_types,
        generator=generator,
    )
    noisy_state = replace(
        coordinate_output.noisy_state,
        atom_types=noisy_atom_types,
        bond_types=noisy_bond_types,
    )
    output = JointForwardDiffusionOutput(
        noisy_state=noisy_state,
        coordinate_noise=coordinate_output.noise,
        clean_atom_types=state.atom_types.clone(),
        clean_bond_types=state.bond_types.clone(),
    )
    output.validate(schedule)
    return output


def initialize_joint_stage3_state(
    stage3_input: Stage3Input,
    schedule: DDPMNoiseSchedule,
    *,
    seed: int,
) -> Stage3DiffusionState:
    """Initialize uniform chemical labels and Gaussian coordinates on fixed topology."""

    if schedule.config.mode != "joint_chemical":
        raise ValueError("joint initialization requires diffusion mode joint_chemical")
    num_atom_types = schedule.config.num_atom_types
    num_bond_types = schedule.config.num_bond_types
    if num_atom_types is None or num_bond_types is None:
        raise ValueError("joint diffusion class counts are required")
    state = initialize_stage3_state(
        stage3_input,
        seed=seed,
        timestep=schedule.num_timesteps - 1,
    )
    _validate_fixed_chemical_topology(state)
    generator = make_torch_generator((seed + 1) % (2**64), device=state.coordinates.device)
    state = replace(
        state,
        atom_types=sample_uniform_categories(
            state.mask.atom_mask,
            num_atom_types,
            generator=generator,
        ),
        bond_types=sample_uniform_categories(
            state.mask.edge_mask,
            num_bond_types,
            generator=generator,
        ),
    )
    state.validate()
    return state


def categorical_reverse_step(
    noisy_labels: Tensor,
    predicted_clean_logits: Tensor,
    timestep: Tensor,
    schedule: DDPMNoiseSchedule,
    mask: Tensor,
    num_classes: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one exact uniform-categorical posterior using predicted ``z_0``.

    ``predicted_clean_logits`` parameterizes ``p_theta(z_0 | z_t)``. For
    ``t>0`` it is combined with the known uniform forward transition to obtain
    ``p_theta(z_{t-1} | z_t)``. At ``t=0`` a clean label is sampled directly
    from the prediction. Frozen validity masks are never updated.
    """

    _validate_category_tensor("noisy_labels", noisy_labels, mask, num_classes)
    _validate_timestep(
        timestep,
        schedule,
        batch_size=noisy_labels.shape[0],
        device=noisy_labels.device,
    )
    expected = (*noisy_labels.shape, num_classes)
    if not isinstance(predicted_clean_logits, Tensor) or tuple(
        predicted_clean_logits.shape
    ) != expected:
        raise ValueError("predicted_clean_logits must have shape [B, L, K]")
    if not torch.is_floating_point(predicted_clean_logits):
        raise TypeError("predicted_clean_logits must use a floating-point dtype")
    if predicted_clean_logits.device != noisy_labels.device:
        raise ValueError("categorical logits and labels must share a device")
    if not bool(torch.isfinite(predicted_clean_logits).all()):
        raise ValueError("predicted_clean_logits contain NaN or Inf")

    clean_probabilities = torch.softmax(predicted_clean_logits, dim=-1)
    schedule_device = schedule.alpha_bars.device
    alpha_t = schedule.alphas.index_select(
        0,
        timestep.to(schedule_device),
    ).to(device=noisy_labels.device, dtype=clean_probabilities.dtype)
    previous_timestep = torch.clamp(timestep - 1, min=0)
    alpha_bar_previous = schedule.alpha_bars.index_select(
        0,
        previous_timestep.to(schedule_device),
    ).to(device=noisy_labels.device, dtype=clean_probabilities.dtype)
    alpha_bar_previous = torch.where(
        timestep == 0,
        torch.ones_like(alpha_bar_previous),
        alpha_bar_previous,
    )
    prior_previous = (
        alpha_bar_previous[:, None, None] * clean_probabilities
        + (1.0 - alpha_bar_previous[:, None, None]) / num_classes
    )
    transition = torch.full_like(
        clean_probabilities,
        1.0,
    ) * ((1.0 - alpha_t)[:, None, None] / num_classes)
    transition.scatter_add_(
        -1,
        noisy_labels.unsqueeze(-1),
        alpha_t[:, None, None].expand(*noisy_labels.shape, 1),
    )
    posterior = prior_previous * transition
    posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(posterior.dtype).tiny
    )
    probabilities = torch.where(
        (timestep == 0)[:, None, None],
        clean_probabilities,
        posterior,
    )
    sampled = torch.multinomial(
        probabilities.reshape(-1, num_classes),
        1,
        generator=generator,
    ).reshape_as(noisy_labels)
    return sampled * mask.to(torch.long)


__all__ = [
    "JointForwardDiffusionOutput",
    "categorical_reverse_step",
    "initialize_joint_stage3_state",
    "q_sample_categorical",
    "q_sample_joint",
    "sample_uniform_categories",
]
