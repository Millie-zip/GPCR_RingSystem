"""GPCR pocket representation and optional guidance extension points."""

from gpcr_ringgen.core.interfaces import PocketFieldProvider
from gpcr_ringgen.core.types import PocketBatch, PocketField, PocketRepresentation
from gpcr_ringgen.pocket.fields import (
    FieldChannel,
    FieldQueryResult,
    GeometryFieldConfig,
    QueryablePocketField,
    build_geometry_field,
    load_geometry_field_config,
)
from gpcr_ringgen.pocket.preprocessing import (
    adapt_pocket_atoms,
    inverse_transform_coordinates,
    preprocess_pocket,
    transform_coordinates,
)
from gpcr_ringgen.pocket.representation import (
    PocketAtomMetadata,
    PocketCoordinateTransform,
    PocketGeometry,
    ProcessedPocketBatch,
)

__all__ = [
    "PocketAtomMetadata",
    "PocketBatch",
    "PocketCoordinateTransform",
    "PocketField",
    "PocketFieldProvider",
    "FieldChannel",
    "FieldQueryResult",
    "GeometryFieldConfig",
    "PocketGeometry",
    "PocketRepresentation",
    "ProcessedPocketBatch",
    "QueryablePocketField",
    "adapt_pocket_atoms",
    "build_geometry_field",
    "inverse_transform_coordinates",
    "load_geometry_field_config",
    "preprocess_pocket",
    "transform_coordinates",
]
