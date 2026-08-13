"""Minimal E(n)-equivariant message-passing layer."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class EquivariantLayer(ABC):
    """Future layer contract."""

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        """Return transformed features."""


class EGNNMessageLayer(nn.Module, EquivariantLayer):
    """Update scalar node states and coordinates using relative geometry.

    Node and edge features are treated as rotation/translation-invariant
    scalars. Each valid molecular edge is used in both directions internally.
    The coordinate contribution for target ``i`` from neighbor ``j`` is
    ``unit(x_i - x_j) * phi_x(message_ij)``, which is E(n)-equivariant.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_feature_dim: int,
        coordinate_scale: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        edge_input_dim = 2 * hidden_dim + edge_feature_dim + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.coordinate_scale = coordinate_scale
        self.epsilon = epsilon

    def forward(
        self,
        node_features: Tensor,
        coordinates: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        atom_mask: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply one vectorized padded-batch equivariant message pass."""

        batch_size, num_atoms, hidden_dim = node_features.shape
        num_edges = edge_index.shape[-1]
        flat_nodes = node_features.reshape(batch_size * num_atoms, hidden_dim)
        flat_coordinates = coordinates.reshape(batch_size * num_atoms, 3)

        batch_indices = (
            torch.arange(batch_size, device=edge_index.device)
            .unsqueeze(1)
            .expand(batch_size, num_edges)
        )
        valid_edges = edge_mask.reshape(-1)
        edge_batches = batch_indices.reshape(-1)[valid_edges]
        source_local = edge_index[:, 0, :].reshape(-1)[valid_edges]
        target_local = edge_index[:, 1, :].reshape(-1)[valid_edges]
        source = source_local + edge_batches * num_atoms
        target = target_local + edge_batches * num_atoms
        valid_edge_features = edge_features.reshape(
            batch_size * num_edges,
            edge_features.shape[-1],
        )[valid_edges]

        # Molecular bonds are consumed in both directions without changing the
        # stored scaffold topology or edge tensor.
        directed_source = torch.cat((source, target), dim=0)
        directed_target = torch.cat((target, source), dim=0)
        directed_edge_features = torch.cat(
            (valid_edge_features, valid_edge_features),
            dim=0,
        )

        relative = (
            flat_coordinates[directed_target] - flat_coordinates[directed_source]
        )
        squared_distance = torch.sum(relative * relative, dim=-1, keepdim=True)
        messages = self.edge_mlp(
            torch.cat(
                (
                    flat_nodes[directed_target],
                    flat_nodes[directed_source],
                    directed_edge_features,
                    squared_distance,
                ),
                dim=-1,
            )
        )

        flat_size = batch_size * num_atoms
        aggregated_messages = torch.zeros(
            flat_size,
            hidden_dim,
            dtype=node_features.dtype,
            device=node_features.device,
        )
        aggregated_messages.index_add_(0, directed_target, messages)
        degree = torch.zeros(
            flat_size,
            1,
            dtype=node_features.dtype,
            device=node_features.device,
        )
        degree.index_add_(
            0,
            directed_target,
            torch.ones_like(messages[:, :1]),
        )
        aggregated_messages = aggregated_messages / degree.clamp_min(1.0)

        direction = relative / torch.sqrt(squared_distance + self.epsilon)
        coordinate_messages = (
            direction * self.coordinate_mlp(messages) * self.coordinate_scale
        )
        coordinate_update = torch.zeros(
            flat_size,
            3,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        coordinate_update.index_add_(0, directed_target, coordinate_messages)
        coordinate_update = coordinate_update / degree.clamp_min(1.0)

        flat_atom_mask = atom_mask.reshape(flat_size, 1)
        updated_nodes = self.node_norm(
            flat_nodes
            + self.node_mlp(torch.cat((flat_nodes, aggregated_messages), dim=-1))
        )
        updated_nodes = updated_nodes * flat_atom_mask.to(updated_nodes.dtype)
        updated_coordinates = flat_coordinates + coordinate_update * flat_atom_mask.to(
            coordinate_update.dtype
        )
        return (
            updated_nodes.reshape(batch_size, num_atoms, hidden_dim),
            updated_coordinates.reshape(batch_size, num_atoms, 3),
        )


__all__ = ["EGNNMessageLayer", "EquivariantLayer"]
