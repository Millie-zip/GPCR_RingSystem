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
        diffusion_mask=(
            None if mask.diffusion_mask is None else mask.diffusion_mask.clone()
        ),
        descriptor_mask=(
            None if mask.descriptor_mask is None else mask.descriptor_mask.clone()
        ),
        pocket_mask=None if mask.pocket_mask is None else mask.pocket_mask.clone(),
    )


def initialize_stage3_state(
    stage3_input: Stage3Input,
    *,
    seed: int,
    timestep: int = 0,
) -> Stage3DiffusionState:
    """Create a state with fixed ring coordinates and noisy non-ring atoms.

    ``diffusion_mask=True`` atoms are initialized from ``N(0, I_3)``.
    Fixed atoms copy the pocket-local scaffold coordinates exactly. Padding is
    zero. Thus ring coordinates never enter the random initialization.

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
    scaffold_coordinates = stage3_input.scaffold.coordinates
    if scaffold_coordinates is None:
        raise ValueError("scaffold coordinates are required for fixed ring atoms")
    diffusion_mask = stage3_input.mask.diffusion_mask
    if diffusion_mask is None:
        raise ValueError("diffusion_mask is required for Stage 3 initialization")
    generator = make_torch_generator(seed, device=atom_features.device)
    random_coordinates = torch.randn(
        (*atom_features.shape[:2], 3),
        dtype=atom_features.dtype,
        device=atom_features.device,
        generator=generator,
    )
    noise_mask = diffusion_mask.unsqueeze(-1).to(dtype=random_coordinates.dtype)
    noise_counts = noise_mask.sum(dim=1, keepdim=True)
    center = (random_coordinates * noise_mask).sum(dim=1, keepdim=True) / noise_counts
    centered_random = (random_coordinates - center) * noise_mask
    fixed_mask = (atom_mask & ~diffusion_mask).unsqueeze(-1)
    random_coordinates = torch.where(
        fixed_mask.any(dim=1, keepdim=True),
        random_coordinates * noise_mask,
        centered_random,
    )
    coordinates = torch.where(
        fixed_mask,
        scaffold_coordinates,
        random_coordinates,
    )
    coordinates = coordinates * atom_mask.unsqueeze(-1).to(coordinates.dtype)

    state = Stage3DiffusionState(
        atom_features=atom_features.clone(),
        atom_types=stage3_input.scaffold.atom_types.clone(),
        edge_index=stage3_input.scaffold.edge_index.clone(),
        bond_types=stage3_input.scaffold.bond_types.clone(),
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
    )
    state.validate()
    return state


__all__ = [
    "DiffusionState",
    "Stage3DiffusionState",
    "initialize_stage3_state",
    "make_dummy_state",
]
