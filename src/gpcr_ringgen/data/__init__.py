"""Real-data loading and Stage 3 batching APIs."""

from gpcr_ringgen.data.collate import collate_molecules
from gpcr_ringgen.data.datasets import JsonlMolecularDataset, MolecularDataset
from gpcr_ringgen.data.manifest import FeatureManifest
from gpcr_ringgen.data.materialize import materialize_training_jsonl
from gpcr_ringgen.data.records import TrainingRecord
from gpcr_ringgen.data.vocabulary import (
    ATOM_TYPES,
    BOND_TYPES,
    ATOM_TYPE_TO_ID,
    BOND_TYPE_TO_ID,
)
from gpcr_ringgen.data.stage3_inference import (
    Stage3InferenceRequest,
    load_stage3_inference_request,
)
from gpcr_ringgen.data.transforms import stage2_output_to_stage3_input

__all__ = [
    "MolecularDataset",
    "JsonlMolecularDataset",
    "TrainingRecord",
    "FeatureManifest",
    "materialize_training_jsonl",
    "collate_molecules",
    "ATOM_TYPES",
    "BOND_TYPES",
    "ATOM_TYPE_TO_ID",
    "BOND_TYPE_TO_ID",
    "Stage3InferenceRequest",
    "load_stage3_inference_request",
    "stage2_output_to_stage3_input",
]
