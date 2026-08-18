"""Route-A side-chain count generation and dynamic ring-core expansion.

The count model has unbounded non-negative integer support.  After a count is
sampled, graphs are padded only to the largest realized graph in that batch;
there is no vocabulary-level or project-wide ``N_max``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    MolecularGraph,
    Stage3Input,
)


@dataclass(frozen=True)
class OuterAtomCountConfig:
    """Dimensions for the conditional Poisson outer-atom count model."""

    atom_feature_dim: int
    descriptor_dim: int
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        for name in ("atom_feature_dim", "descriptor_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "OuterAtomCountConfig":
        """Load dimensions from combined ``model`` and ``route_a`` sections."""

        model = config.get("model")
        route_a = config.get("route_a")
        if not isinstance(model, Mapping) or not isinstance(route_a, Mapping):
            raise TypeError("count config requires model and route_a mappings")
        count = route_a.get("count_model")
        if not isinstance(count, Mapping):
            raise TypeError("route_a.count_model must be a mapping")
        if count.get("type") != "conditional_poisson":
            raise ValueError("Route A count model type must be 'conditional_poisson'")
        descriptor_dim = model.get("descriptor_dim")
        if descriptor_dim is None:
            raise ValueError("Route A count model requires model.descriptor_dim")
        return cls(
            atom_feature_dim=model.get("atom_feature_dim"),
            descriptor_dim=descriptor_dim,
            hidden_dim=count.get("hidden_dim", 64),
        )


class ConditionalOuterAtomCountModel(nn.Module):
    """Generate a finite outer-atom count without a fixed categorical maximum.

    The Poisson rate is conditioned on mean-pooled ring-core atom features and
    the existing descriptor vector.  Poisson support is ``0, 1, ...``; a
    caller may impose a runtime-only allocation guard, but the model itself has
    no maximum atom-count class.
    """

    def __init__(self, config: OuterAtomCountConfig) -> None:
        super().__init__()
        if not isinstance(config, OuterAtomCountConfig):
            raise TypeError("config must be an OuterAtomCountConfig")
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(
                config.atom_feature_dim + config.descriptor_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        atom_features: Tensor,
        ring_atom_mask: Tensor,
        descriptor: Tensor,
        descriptor_mask: Tensor,
    ) -> Tensor:
        """Return a positive Poisson rate shaped ``[B]``.

        ``atom_features`` is ``[B, N, F_atom]`` floating-point.
        ``ring_atom_mask`` is ``[B, N]`` bool and must select at least one atom
        per sample. ``descriptor`` and ``descriptor_mask`` are ``[B, D]``.
        All tensors must share a device; floating tensors share a dtype.
        """

        if atom_features.ndim != 3 or not torch.is_floating_point(atom_features):
            raise ValueError("atom_features must be floating with shape [B, N, F]")
        if atom_features.shape[-1] != self.config.atom_feature_dim:
            raise ValueError("atom feature dimension does not match count config")
        if ring_atom_mask.shape != atom_features.shape[:2]:
            raise ValueError("ring_atom_mask must have shape [B, N]")
        if ring_atom_mask.dtype != torch.bool:
            raise TypeError("ring_atom_mask must have dtype torch.bool")
        if descriptor.ndim != 2 or descriptor.shape[0] != atom_features.shape[0]:
            raise ValueError("descriptor must have shape [B, D]")
        if descriptor.shape[-1] != self.config.descriptor_dim:
            raise ValueError("descriptor dimension does not match count config")
        if descriptor_mask.shape != descriptor.shape:
            raise ValueError("descriptor_mask must match descriptor")
        if descriptor_mask.dtype != torch.bool:
            raise TypeError("descriptor_mask must have dtype torch.bool")
        for name, tensor in (
            ("ring_atom_mask", ring_atom_mask),
            ("descriptor", descriptor),
            ("descriptor_mask", descriptor_mask),
        ):
            if tensor.device != atom_features.device:
                raise ValueError(f"{name} must be on the atom_features device")
        if descriptor.dtype != atom_features.dtype:
            raise TypeError("descriptor and atom_features must share a dtype")
        counts = ring_atom_mask.sum(dim=1, keepdim=True)
        if bool((counts == 0).any()):
            raise ValueError("every sample must contain at least one ring atom")
        weights = ring_atom_mask.unsqueeze(-1).to(atom_features.dtype)
        pooled = (atom_features * weights).sum(dim=1) / counts.to(
            atom_features.dtype
        )
        masked_descriptor = descriptor * descriptor_mask.to(descriptor.dtype)
        raw_log_rate = self.network(torch.cat((pooled, masked_descriptor), dim=-1))
        # Numerical guard only; Poisson samples retain unbounded integer support.
        rate = F.softplus(raw_log_rate.squeeze(-1).clamp(min=-30.0, max=30.0))
        rate = rate + torch.finfo(rate.dtype).eps
        if not bool(torch.isfinite(rate).all()):
            raise ValueError("outer atom count rate contains NaN or Inf")
        return rate

    @staticmethod
    def loss(rate: Tensor, target_count: Tensor, *, reduction: str = "mean") -> Tensor:
        """Return Poisson negative log likelihood for counts shaped ``[B]``."""

        if rate.ndim != 1 or not torch.is_floating_point(rate):
            raise ValueError("rate must be floating with shape [B]")
        if target_count.shape != rate.shape or target_count.dtype != torch.long:
            raise ValueError("target_count must be torch.long with shape [B]")
        if target_count.device != rate.device:
            raise ValueError("target_count and rate must share a device")
        if bool((target_count < 0).any()):
            raise ValueError("target_count must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        return F.poisson_nll_loss(
            torch.log(rate),
            target_count.to(rate.dtype),
            log_input=True,
            full=True,
            reduction=reduction,
        )

    @staticmethod
    def sample(rate: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Sample non-negative ``torch.long`` counts with unbounded support."""

        if rate.ndim != 1 or not torch.is_floating_point(rate):
            raise ValueError("rate must be floating with shape [B]")
        if bool((~torch.isfinite(rate) | (rate <= 0)).any()):
            raise ValueError("rate must be finite and positive")
        return torch.poisson(rate, generator=generator).to(torch.long)


def _validate_outer_counts(
    outer_counts: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if not isinstance(outer_counts, Tensor) or outer_counts.ndim != 1:
        raise ValueError("outer_counts must be a tensor with shape [B]")
    if outer_counts.dtype != torch.long:
        raise TypeError("outer_counts must have dtype torch.long")
    if outer_counts.shape[0] != batch_size:
        raise ValueError("outer_counts must have shape [B]")
    if outer_counts.device != device:
        raise ValueError("outer_counts must be on the Stage3Input device")
    if bool((outer_counts < 0).any()):
        raise ValueError("outer_counts must be non-negative")


def expand_ring_core_batch(
    stage3_input: Stage3Input,
    outer_counts: Tensor,
    *,
    runtime_atom_limit: int | None = None,
) -> Stage3Input:
    """Build batch-local dynamic complete candidate graphs around ring cores.

    The input may contain a complete training molecule, but only nodes selected
    by ``ring_atom_mask`` and fixed edges selected by ``ring_edge_mask`` are
    copied.  ``outer_counts[b]`` zero-feature node slots are appended. Every
    unordered pair is then enumerated once. Core-core non-edges remain fixed
    absent; any pair touching an outer node is a generative bond candidate and
    may form new side-chain rings.

    ``runtime_atom_limit`` is an optional memory-safety guard, not a chemical
    vocabulary limit. Omitting it leaves the model contract uncapped.
    """

    if not isinstance(stage3_input, Stage3Input):
        raise TypeError("stage3_input must be a Stage3Input")
    stage3_input.validate()
    ring_atom_mask = stage3_input.mask.ring_atom_mask
    ring_edge_mask = stage3_input.mask.ring_edge_mask
    if ring_atom_mask is None or ring_edge_mask is None:
        raise ValueError("Route A requires ring_atom_mask and ring_edge_mask")
    graph = stage3_input.scaffold
    if graph.atom_types is None or graph.bond_types is None:
        raise ValueError("Route A requires atom_types and bond_types")
    if graph.formal_charge_types is None or graph.chiralities is None:
        raise ValueError("Route A requires formal_charge_types and chiralities")
    _validate_outer_counts(
        outer_counts,
        batch_size=stage3_input.batch_size,
        device=stage3_input.device,
    )
    if runtime_atom_limit is not None:
        if isinstance(runtime_atom_limit, bool) or not isinstance(runtime_atom_limit, int):
            raise TypeError("runtime_atom_limit must be an integer or None")
        if runtime_atom_limit <= 0:
            raise ValueError("runtime_atom_limit must be positive")

    core_counts = ring_atom_mask.sum(dim=1).to(torch.long)
    total_counts = core_counts + outer_counts
    if bool((total_counts == 0).any()):
        raise ValueError("expanded samples must contain at least one atom")
    if runtime_atom_limit is not None and bool((total_counts > runtime_atom_limit).any()):
        raise ValueError("sampled atom count exceeds the runtime-only safety limit")

    batch_size = stage3_input.batch_size
    max_atoms = int(total_counts.max().item())
    edge_counts = total_counts * (total_counts - 1) // 2
    max_edges = int(edge_counts.max().item())
    dtype = graph.atom_features.dtype
    device = graph.atom_features.device
    atom_feature_dim = graph.atom_features.shape[-1]
    edge_feature_dim = graph.edge_features.shape[-1]

    atom_features = torch.zeros(
        (batch_size, max_atoms, atom_feature_dim), dtype=dtype, device=device
    )
    edge_features = torch.zeros(
        (batch_size, max_edges, edge_feature_dim), dtype=dtype, device=device
    )
    edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long, device=device)
    atom_types = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=device)
    bond_types = torch.zeros((batch_size, max_edges), dtype=torch.long, device=device)
    formal_charge_types = torch.zeros_like(atom_types)
    chiralities = torch.zeros_like(atom_types)
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=device)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)
    new_ring_atom_mask = torch.zeros_like(atom_mask)
    ring_pair_mask = torch.zeros_like(edge_mask)
    new_ring_edge_mask = torch.zeros_like(edge_mask)

    for batch_index in range(batch_size):
        old_core_indices = torch.nonzero(
            ring_atom_mask[batch_index], as_tuple=False
        ).squeeze(-1)
        num_core = int(old_core_indices.numel())
        num_total = int(total_counts[batch_index].item())
        num_edges = int(edge_counts[batch_index].item())
        atom_mask[batch_index, :num_total] = True
        new_ring_atom_mask[batch_index, :num_core] = True
        atom_features[batch_index, :num_core] = graph.atom_features[
            batch_index, old_core_indices
        ]
        atom_types[batch_index, :num_core] = graph.atom_types[
            batch_index, old_core_indices
        ]
        formal_charge_types[batch_index, :num_core] = graph.formal_charge_types[
            batch_index, old_core_indices
        ]
        chiralities[batch_index, :num_core] = graph.chiralities[
            batch_index, old_core_indices
        ]

        old_to_new = {
            int(old_index.item()): new_index
            for new_index, old_index in enumerate(old_core_indices)
        }
        fixed_edges: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        valid_fixed = ring_edge_mask[batch_index]
        for old_edge_position in torch.nonzero(valid_fixed, as_tuple=False).squeeze(-1):
            old_source = int(graph.edge_index[batch_index, 0, old_edge_position].item())
            old_target = int(graph.edge_index[batch_index, 1, old_edge_position].item())
            if old_source not in old_to_new or old_target not in old_to_new:
                raise ValueError("ring_edge_mask references a non-ring atom")
            pair = tuple(sorted((old_to_new[old_source], old_to_new[old_target])))
            if pair in fixed_edges:
                raise ValueError("ring core contains a duplicated undirected edge")
            fixed_edges[pair] = (
                graph.edge_features[batch_index, old_edge_position],
                graph.bond_types[batch_index, old_edge_position],
            )

        if num_edges:
            pairs = torch.triu_indices(num_total, num_total, offset=1, device=device)
            edge_index[batch_index, :, :num_edges] = pairs
            edge_mask[batch_index, :num_edges] = True
            core_pairs = (pairs[0] < num_core) & (pairs[1] < num_core)
            ring_pair_mask[batch_index, :num_edges] = core_pairs
            for edge_position in range(num_edges):
                pair = (
                    int(pairs[0, edge_position].item()),
                    int(pairs[1, edge_position].item()),
                )
                fixed = fixed_edges.get(pair)
                if fixed is None:
                    continue
                new_ring_edge_mask[batch_index, edge_position] = True
                edge_features[batch_index, edge_position] = fixed[0]
                bond_types[batch_index, edge_position] = fixed[1]

    records = tuple(
        {
            **dict(record),
            "route_a_outer_atom_count": int(outer_counts[index].item()),
            "route_a_total_atom_count": int(total_counts[index].item()),
        }
        for index, record in enumerate(stage3_input.metadata.records)
    )
    expanded = Stage3Input(
        scaffold=MolecularGraph(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            atom_types=atom_types,
            bond_types=bond_types,
            formal_charge_types=formal_charge_types,
            chiralities=chiralities,
        ),
        descriptor=stage3_input.descriptor,
        pocket=stage3_input.pocket,
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            descriptor_mask=stage3_input.mask.descriptor_mask,
            pocket_mask=stage3_input.mask.pocket_mask,
            ring_atom_mask=new_ring_atom_mask,
            ring_pair_mask=ring_pair_mask,
            ring_edge_mask=new_ring_edge_mask,
        ),
        metadata=BatchMetadata(
            sample_ids=stage3_input.metadata.sample_ids,
            records=records,
        ),
        pocket_field=stage3_input.pocket_field,
    )
    expanded.validate()
    return expanded


def outer_atom_counts(atom_mask: Tensor, ring_atom_mask: Tensor) -> Tensor:
    """Return per-sample outer-atom counts shaped ``[B]`` as ``torch.long``."""

    if atom_mask.shape != ring_atom_mask.shape or atom_mask.ndim != 2:
        raise ValueError("atom_mask and ring_atom_mask must share shape [B, N]")
    if atom_mask.dtype != torch.bool or ring_atom_mask.dtype != torch.bool:
        raise TypeError("atom masks must have dtype torch.bool")
    if atom_mask.device != ring_atom_mask.device:
        raise ValueError("atom masks must share a device")
    if bool((ring_atom_mask & ~atom_mask).any()):
        raise ValueError("ring_atom_mask must be a subset of atom_mask")
    return (atom_mask & ~ring_atom_mask).sum(dim=1).to(torch.long)


def build_route_a_training_batch(
    stage3_input: Stage3Input,
    *,
    bond_type_offset: int = 1,
) -> Stage3Input:
    """Convert real variable-size molecules to complete Route-A candidates.

    Existing atoms are reordered core-first and retained. Every unordered atom
    pair is enumerated once. Existing sparse bonds receive their categorical
    label plus ``bond_type_offset``; absent pairs receive category zero
    (``NO_BOND``). This is the training counterpart of
    :func:`expand_ring_core_batch`.
    """

    if not isinstance(stage3_input, Stage3Input):
        raise TypeError("stage3_input must be a Stage3Input")
    stage3_input.validate()
    if isinstance(bond_type_offset, bool) or not isinstance(bond_type_offset, int):
        raise TypeError("bond_type_offset must be an integer")
    if bond_type_offset < 1:
        raise ValueError("bond_type_offset must reserve category zero for NO_BOND")
    graph = stage3_input.scaffold
    ring_atom_mask = stage3_input.mask.ring_atom_mask
    if ring_atom_mask is None:
        raise ValueError("Route-A training conversion requires ring_atom_mask")
    if graph.atom_types is None or graph.bond_types is None:
        raise ValueError("Route-A training conversion requires atom and bond labels")
    if graph.formal_charge_types is None or graph.chiralities is None:
        raise ValueError("Route-A training conversion requires charge and chirality labels")

    batch_size = stage3_input.batch_size
    atom_counts = stage3_input.mask.atom_mask.sum(dim=1).to(torch.long)
    edge_counts = atom_counts * (atom_counts - 1) // 2
    max_atoms = int(atom_counts.max().item())
    max_edges = int(edge_counts.max().item())
    device = stage3_input.device
    dtype = graph.atom_features.dtype
    atom_feature_dim = graph.atom_features.shape[-1]
    edge_feature_dim = graph.edge_features.shape[-1]

    atom_features = torch.zeros(
        (batch_size, max_atoms, atom_feature_dim), dtype=dtype, device=device
    )
    edge_features = torch.zeros(
        (batch_size, max_edges, edge_feature_dim), dtype=dtype, device=device
    )
    edge_index = torch.zeros((batch_size, 2, max_edges), dtype=torch.long, device=device)
    atom_types = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=device)
    bond_types = torch.zeros((batch_size, max_edges), dtype=torch.long, device=device)
    formal_charge_types = torch.zeros_like(atom_types)
    chiralities = torch.zeros_like(atom_types)
    coordinates = (
        None
        if graph.coordinates is None
        else torch.zeros((batch_size, max_atoms, 3), dtype=dtype, device=device)
    )
    atom_mask = torch.zeros((batch_size, max_atoms), dtype=torch.bool, device=device)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool, device=device)
    new_ring_atom_mask = torch.zeros_like(atom_mask)
    ring_pair_mask = torch.zeros_like(edge_mask)
    ring_edge_mask = torch.zeros_like(edge_mask)

    for batch_index in range(batch_size):
        valid_atoms = torch.nonzero(
            stage3_input.mask.atom_mask[batch_index], as_tuple=False
        ).squeeze(-1)
        core_atoms = valid_atoms[ring_atom_mask[batch_index, valid_atoms]]
        outer_atoms = valid_atoms[~ring_atom_mask[batch_index, valid_atoms]]
        ordering = torch.cat((core_atoms, outer_atoms))
        num_core = int(core_atoms.numel())
        num_atoms = int(ordering.numel())
        num_edges = int(edge_counts[batch_index].item())
        if num_core == 0:
            raise ValueError("every Route-A training sample needs a ring core")
        atom_mask[batch_index, :num_atoms] = True
        new_ring_atom_mask[batch_index, :num_core] = True
        atom_features[batch_index, :num_atoms] = graph.atom_features[
            batch_index, ordering
        ]
        atom_types[batch_index, :num_atoms] = graph.atom_types[batch_index, ordering]
        formal_charge_types[batch_index, :num_atoms] = graph.formal_charge_types[
            batch_index, ordering
        ]
        chiralities[batch_index, :num_atoms] = graph.chiralities[batch_index, ordering]
        if coordinates is not None:
            coordinates[batch_index, :num_atoms] = graph.coordinates[
                batch_index, ordering
            ]
        old_to_new = {
            int(old_index.item()): new_index
            for new_index, old_index in enumerate(ordering)
        }
        existing_edges: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
        for old_edge_position in torch.nonzero(
            stage3_input.mask.edge_mask[batch_index], as_tuple=False
        ).squeeze(-1):
            source = int(graph.edge_index[batch_index, 0, old_edge_position].item())
            target = int(graph.edge_index[batch_index, 1, old_edge_position].item())
            if source not in old_to_new or target not in old_to_new:
                raise ValueError("a valid training edge references a padded atom")
            pair = tuple(sorted((old_to_new[source], old_to_new[target])))
            if pair[0] == pair[1]:
                raise ValueError("Route-A training graphs cannot contain self loops")
            if pair in existing_edges:
                raise ValueError("Route-A training graphs store each undirected edge once")
            existing_edges[pair] = (
                graph.edge_features[batch_index, old_edge_position],
                graph.bond_types[batch_index, old_edge_position] + bond_type_offset,
            )

        if num_edges:
            pairs = torch.triu_indices(num_atoms, num_atoms, offset=1, device=device)
            edge_index[batch_index, :, :num_edges] = pairs
            edge_mask[batch_index, :num_edges] = True
            core_pairs = (pairs[0] < num_core) & (pairs[1] < num_core)
            ring_pair_mask[batch_index, :num_edges] = core_pairs
            for edge_position in range(num_edges):
                pair = (
                    int(pairs[0, edge_position].item()),
                    int(pairs[1, edge_position].item()),
                )
                existing = existing_edges.get(pair)
                if existing is None:
                    continue
                edge_features[batch_index, edge_position] = existing[0]
                bond_types[batch_index, edge_position] = existing[1]
                if pair[1] < num_core:
                    ring_edge_mask[batch_index, edge_position] = True

    converted = Stage3Input(
        scaffold=MolecularGraph(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_features=edge_features,
            coordinates=coordinates,
            atom_types=atom_types,
            bond_types=bond_types,
            formal_charge_types=formal_charge_types,
            chiralities=chiralities,
        ),
        descriptor=stage3_input.descriptor,
        pocket=stage3_input.pocket,
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            descriptor_mask=stage3_input.mask.descriptor_mask,
            pocket_mask=stage3_input.mask.pocket_mask,
            ring_atom_mask=new_ring_atom_mask,
            ring_pair_mask=ring_pair_mask,
            ring_edge_mask=ring_edge_mask,
        ),
        metadata=stage3_input.metadata,
        pocket_field=stage3_input.pocket_field,
    )
    converted.validate()
    return converted


__all__ = [
    "ConditionalOuterAtomCountModel",
    "OuterAtomCountConfig",
    "build_route_a_training_batch",
    "expand_ring_core_batch",
    "outer_atom_counts",
]
