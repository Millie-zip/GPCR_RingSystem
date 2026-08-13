"""Feature embedding boundaries and scalar timestep embeddings."""

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class FeatureEmbedding(ABC):
    """Future feature embedding contract."""

    @abstractmethod
    def encode(self, values: Any) -> Any:
        """Encode values after the representation is selected."""


class SinusoidalTimestepEmbedding(nn.Module):
    """Embed scalar timesteps without introducing spatial coordinates.

    Input is a finite scalar tensor with shape ``[B]``. Output has shape
    ``[B, D_time]`` and uses the embedding module's floating dtype/device.
    Timesteps are invariant scalar conditions, so this operation does not
    affect coordinate equivariance.
    """

    def __init__(self, embedding_dim: int, *, max_period: float = 10_000.0) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if not math.isfinite(max_period) or max_period <= 1.0:
            raise ValueError("max_period must be finite and greater than 1")

        half_dim = embedding_dim // 2
        exponent_denominator = max(half_dim - 1, 1)
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / exponent_denominator
        )
        self.embedding_dim = embedding_dim
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timestep: Tensor) -> Tensor:
        if not isinstance(timestep, Tensor):
            raise TypeError("timestep must be a torch.Tensor")
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        if not bool(torch.isfinite(timestep).all()):
            raise ValueError("timestep contains NaN or Inf")

        arguments = timestep.to(dtype=self.frequencies.dtype).unsqueeze(-1)
        arguments = arguments * self.frequencies.unsqueeze(0)
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat(
                (
                    embedding,
                    torch.zeros_like(embedding[:, :1]),
                ),
                dim=-1,
            )
        return embedding


__all__ = ["FeatureEmbedding", "SinusoidalTimestepEmbedding"]
