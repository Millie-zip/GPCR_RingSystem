"""Stage 3 state initialization without diffusion or denoising behavior."""

from __future__ import annotations

import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    DiffusionState,
    Stage3DiffusionState,
    Stage3Input,
)
from gpcr_ringgen.utils.seed import make_torch_generator


def make_dummy_state() -> DiffusionState:
    """Create the input state used solely by the Dummy Pipeline."""
    return DiffusionState(
        payload={"status": "dummy-initialized"},
        timestep=0,
        metadata={"dummy": True, "scientific": False},
    )


def _clone_batch_mask(mask: BatchMask) -> BatchMask:
    """Copy masks so the initialized state cannot mutate its input masks."""

    return BatchMask(
        atom_mask=mask.atom_mask.clone(),
        edge_mask=mask.edge_mask.clone(),
        descriptor_mask=(
            None if mask.descriptor_mask is None else mask.descriptor_mask.clone()
        ),
        pocket_mask=None if mask.pocket_mask is None else mask.pocket_mask.clone(),
        ring_atom_mask=(
            None if mask.ring_atom_mask is None else mask.ring_atom_mask.clone()
        ),
        ring_pair_mask=(
            None if mask.ring_pair_mask is None else mask.ring_pair_mask.clone()
        ),
        ring_edge_mask=(
            None if mask.ring_edge_mask is None else mask.ring_edge_mask.clone()
        ),
    )


def initialize_stage3_state(
    stage3_input: Stage3Input,
    *,
    seed: int,
    timestep: int = 0,
) -> Stage3DiffusionState:
    """Create a centered isotropic-Gaussian 3D state from Stage 3 input.

    Valid ligand atom coordinates are first sampled independently from
    ``N(0, I_3)`` and then centered per sample using only positions where
    ``atom_mask=True``. Padding coordinates are set to zero.

    Args:
        stage3_input: Validated padded Stage 3 input. Graph topology/features,
            descriptor rows, masks, and sample ordering are preserved.
        seed: Integer used by a local device-specific ``torch.Generator``. The
            global Python, NumPy, and PyTorch random states are not modified.
        timestep: Non-negative state label stored as shape ``[B]``
            ``torch.long``. Its default value does not define a diffusion noise
            schedule.

    Returns:
        A validated ``Stage3DiffusionState`` whose coordinates have shape
        ``[B, N, 3]``, the scaffold floating dtype/device, and zeros wherever
        ``atom_mask=False``.
    """

    if not isinstance(stage3_input, Stage3Input):
        raise TypeError("stage3_input must be a Stage3Input")
    stage3_input.validate()
    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise TypeError("timestep must be an integer")
    if timestep < 0:
        raise ValueError("timestep must be non-negative")

    atom_mask = stage3_input.mask.atom_mask
    valid_atom_counts = atom_mask.sum(dim=1)
    if bool((valid_atom_counts == 0).any()):
        raise ValueError("Every sample must contain at least one valid ligand atom")

    atom_features = stage3_input.scaffold.atom_features
    generator = make_torch_generator(seed, device=atom_features.device)
    coordinates = torch.randn(
        (*atom_features.shape[:2], 3),
        dtype=atom_features.dtype,
        device=atom_features.device,
        generator=generator,
    )
    coordinate_mask = atom_mask.unsqueeze(-1).to(dtype=coordinates.dtype)
    valid_counts = coordinate_mask.sum(dim=1, keepdim=True)
    center = (coordinates * coordinate_mask).sum(dim=1, keepdim=True) / valid_counts
    coordinates = (coordinates - center) * coordinate_mask

    state = Stage3DiffusionState(
        atom_features=atom_features.clone(),
        edge_index=stage3_input.scaffold.edge_index.clone(),
        edge_features=stage3_input.scaffold.edge_features.clone(),
        coordinates=coordinates,
        descriptor=stage3_input.descriptor.values.clone(),
        timestep=torch.full(
            (stage3_input.batch_size,),
            timestep,
            dtype=torch.long,
            device=atom_features.device,
        ),
        mask=_clone_batch_mask(stage3_input.mask),
        metadata=stage3_input.metadata,
        atom_types=(
            None
            if stage3_input.scaffold.atom_types is None
            else stage3_input.scaffold.atom_types.clone()
        ),
        bond_types=(
            None
            if stage3_input.scaffold.bond_types is None
            else stage3_input.scaffold.bond_types.clone()
        ),
        formal_charge_types=(
            None
            if stage3_input.scaffold.formal_charge_types is None
            else stage3_input.scaffold.formal_charge_types.clone()
        ),
        chiralities=(
            None
            if stage3_input.scaffold.chiralities is None
            else stage3_input.scaffold.chiralities.clone()
        ),
    )
    state.validate()
    return state


def clean_stage3_state(
    stage3_input: Stage3Input,
    *,
    timestep: int = 0,
) -> Stage3DiffusionState:
    """Build a training state from persisted ``x0`` coordinates.

    This is deliberately separate from :func:`initialize_stage3_state`, whose
    Gaussian initialization is an inference prior.  Training must never use
    that prior as a substitute for database coordinates.
    """

    if not isinstance(stage3_input, Stage3Input):
        raise TypeError("stage3_input must be a Stage3Input")
    stage3_input.validate()
    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise TypeError("timestep must be an integer")
    if timestep < 0:
        raise ValueError("timestep must be non-negative")
    coordinates = stage3_input.scaffold.coordinates
    if coordinates is None:
        raise ValueError("training input must contain persisted coordinates")
    state = Stage3DiffusionState(
        atom_features=stage3_input.scaffold.atom_features.clone(),
        edge_index=stage3_input.scaffold.edge_index.clone(),
        edge_features=stage3_input.scaffold.edge_features.clone(),
        coordinates=coordinates.clone(),
        descriptor=stage3_input.descriptor.values.clone(),
        timestep=torch.full(
            (stage3_input.batch_size,),
            timestep,
            dtype=torch.long,
            device=coordinates.device,
        ),
        mask=_clone_batch_mask(stage3_input.mask),
        metadata=stage3_input.metadata,
        atom_types=(
            None
            if stage3_input.scaffold.atom_types is None
            else stage3_input.scaffold.atom_types.clone()
        ),
        bond_types=(
            None
            if stage3_input.scaffold.bond_types is None
            else stage3_input.scaffold.bond_types.clone()
        ),
        formal_charge_types=(
            None
            if stage3_input.scaffold.formal_charge_types is None
            else stage3_input.scaffold.formal_charge_types.clone()
        ),
        chiralities=(
            None
            if stage3_input.scaffold.chiralities is None
            else stage3_input.scaffold.chiralities.clone()
        ),
    )
    state.validate()
    return state


__all__ = [
    "DiffusionState",
    "Stage3DiffusionState",
    "clean_stage3_state",
    "initialize_stage3_state",
    "make_dummy_state",
]
