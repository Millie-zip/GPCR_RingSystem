"""Embedding extension points and scalar timestep embedding."""

from gpcr_ringgen.models.embeddings.embeddings import (
    FeatureEmbedding,
    SinusoidalTimestepEmbedding,
)

__all__ = ["FeatureEmbedding", "SinusoidalTimestepEmbedding"]
