"""Plain-data contracts binding checkpoints to their scientific semantics."""

from __future__ import annotations

import copy
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from gpcr_ringgen.core.exceptions import ConfigurationError
from gpcr_ringgen.data.manifest import FeatureManifest
from gpcr_ringgen.models.diffusion import ChemicalVocabulary, DDPMConfig
from gpcr_ringgen.models.equivariant import EquivariantDenoiserConfig, FieldConditioningConfig
from gpcr_ringgen.pocket import GeometryFieldConfig
from gpcr_ringgen.vocabulary import ATOM_ID_TO_TYPE, BOND_ID_TO_TYPE


def plain_data(value: Any) -> Any:
    """Normalize tuples to lists and prohibit non-JSON metadata."""
    return json.loads(json.dumps(value, allow_nan=False))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve ordered feature definitions, independent of file location."""
    inline = config.get("feature_manifest")
    if isinstance(inline, Mapping):
        return FeatureManifest.from_mapping(inline).to_mapping()
    data = config.get("data", {})
    path = data.get("feature_manifest") if isinstance(data, Mapping) else None
    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError("checkpoint validation requires data.feature_manifest or an inline feature_manifest")
    return FeatureManifest.from_file(path).to_mapping()


def chemical_semantics(config: Mapping[str, Any], *, coordinate: bool = False) -> dict[str, Any]:
    if coordinate:
        model = EquivariantDenoiserConfig.from_mapping(config)
        atoms = model.num_atom_types or model.atom_type_vocab_size
        bonds = model.num_bond_types or model.bond_type_vocab_size
        if atoms > len(ATOM_ID_TO_TYPE) or bonds > len(BOND_ID_TO_TYPE):
            raise ConfigurationError("coordinate model vocabulary exceeds canonical labels")
        return {"atom_types": list(ATOM_ID_TO_TYPE[:atoms]), "bond_types": list(BOND_ID_TO_TYPE[:bonds])}
    vocabulary = ChemicalVocabulary.from_mapping(config)
    return {"atom_types": list(vocabulary.atom_types), "bond_types": list(vocabulary.bond_types)}


def generator_semantics(config: Mapping[str, Any]) -> dict[str, Any]:
    model = EquivariantDenoiserConfig.from_mapping(config)
    diffusion = DDPMConfig.from_mapping(config)
    field = FieldConditioningConfig.from_mapping(
        config.get("field_conditioning", {"mode": "none", "channels": []})
    )
    model_values = asdict(model)
    # The baseline sizes are unused when prediction heads specify the sizes.
    model_values["atom_type_vocab_size"] = model.num_atom_types or model.atom_type_vocab_size
    model_values["bond_type_vocab_size"] = model.num_bond_types or model.bond_type_vocab_size
    features = feature_semantics(config)
    manifest = FeatureManifest.from_mapping(features)
    if (model.atom_feature_dim != manifest.atom_feature_dim
            or model.edge_feature_dim != manifest.edge_feature_dim
            or (model.descriptor_dim or 0) != manifest.descriptor_dim):
        raise ConfigurationError("checkpoint model dimensions do not match feature definitions")
    chemistry = chemical_semantics(config, coordinate=diffusion.mode == "coordinate_only")
    if diffusion.mode == "joint_chemical":
        if (model.num_atom_types != diffusion.num_atom_types
                or model.num_bond_types != diffusion.num_bond_types
                or len(chemistry["atom_types"]) != diffusion.num_atom_types
                or len(chemistry["bond_types"]) != diffusion.num_bond_types):
            raise ConfigurationError("model, diffusion and chemical vocabularies must match")
    return plain_data({
        "version": 1,
        "kind": "stage3_generator",
        "model": model_values,
        "diffusion": asdict(diffusion),
        "chemical_vocabulary": chemistry,
        "features": features,
        "field_conditioning": asdict(field),
        "field": asdict(GeometryFieldConfig.from_mapping(config)) if field.mode != "none" else None,
    })


def training_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    return {
        "records_sha256": file_sha256(data["path"]),
        "max_records": data.get("max_records"),
        "train_fraction": float(config["training"].get("train_fraction", 0.9)),
        "seed": config["seed"],
    }


def require_match(actual: Any, expected: Any, name: str) -> None:
    """Report the field that differs without dumping records or weights."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ConfigurationError(f"checkpoint semantics mismatch at {name}: missing or unexpected fields")
        for key, value in expected.items():
            require_match(actual[key], value, f"{name}.{key}")
    elif actual != expected:
        raise ConfigurationError(f"checkpoint semantics mismatch at {name}")


def validate_generator_checkpoint(
    checkpoint: object,
    config: Mapping[str, Any],
    *,
    resume_provenance: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ConfigurationError("checkpoint must be a mapping")
    expected = generator_semantics(config)
    saved = checkpoint.get("semantics")
    version = checkpoint.get("checkpoint_version", 1)
    if version not in (1, 2):
        raise ConfigurationError("unsupported generator checkpoint version")
    if saved is None:
        if version == 2 or not isinstance(checkpoint.get("config"), Mapping):
            raise ConfigurationError("checkpoint lacks sufficient semantic metadata")
        old_config = copy.deepcopy(dict(checkpoint["config"]))
        old_model = old_config.get("model")
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
        # Recover the historical coordinate embedding size, rather than applying
        # the current 12-class default to a 9-class checkpoint.
        if isinstance(old_model, dict) and old_model.get("num_atom_types") is None:
            for field, key in (("atom_type_vocab_size", "backbone.atom_type_embedding.weight"),
                               ("bond_type_vocab_size", "backbone.bond_type_embedding.weight")):
                if field not in old_model:
                    weight = state.get(key) if isinstance(state, Mapping) else None
                    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
                        raise ConfigurationError("legacy checkpoint cannot establish categorical IDs")
                    old_model[field] = int(weight.shape[0])
        if not isinstance(old_config.get("feature_manifest"), Mapping):
            data = config.get("data", {})
            path = data.get("feature_manifest") if isinstance(data, Mapping) else None
            if not isinstance(path, str) or checkpoint.get("feature_manifest_hash") != file_sha256(path):
                raise ConfigurationError("legacy checkpoint cannot establish feature definitions")
            # A matching historical byte hash establishes the current definition;
            # the original manifest path need not exist on this machine.
            old_config["feature_manifest"] = expected["features"]
        try:
            saved = generator_semantics(old_config)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ConfigurationError(f"legacy checkpoint lacks sufficient semantic metadata: {exc}") from exc
    require_match(saved, expected, "generator")
    if resume_provenance is not None:
        require_match(checkpoint.get("training_provenance"), resume_provenance, "training_provenance")


def label_cache_provenance(path: str | Path) -> dict[str, Any]:
    """Fingerprint logical labels, including WAL contents, using a read-only DB."""
    digest = hashlib.sha256()
    count = 0
    versions: set[str] = set()
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for smiles, score, version in connection.execute(
            "SELECT canonical_smiles, ra_score, teacher_version FROM rascore_labels ORDER BY canonical_smiles"
        ):
            if not isinstance(version, str) or not version.strip():
                raise ConfigurationError("label cache has no valid teacher version")
            versions.add(version)
            digest.update(json.dumps([smiles, score, version], ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
            count += 1
    if count == 0 or len(versions) != 1:
        raise ConfigurationError("label cache requires exactly one teacher version and at least one label")
    return {"teacher_version": versions.pop(), "labels_sha256": digest.hexdigest(), "label_count": count}


def reward_semantics(config: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    return plain_data({
        "version": 1,
        "kind": "ra_reward",
        "model": dict(model_config),
        "features": feature_semantics(config),
        "chemical_vocabulary": chemical_semantics(config),
    })


def validate_reward_metadata(
    semantics: object, provenance: object, model_config: Mapping[str, Any]
) -> None:
    if not isinstance(semantics, Mapping) or not isinstance(provenance, Mapping):
        raise ConfigurationError("RA reward checkpoint lacks semantic metadata and teacher/data provenance")
    if semantics.get("version") != 1 or semantics.get("kind") != "ra_reward":
        raise ConfigurationError("unsupported RA reward semantic metadata version")
    require_match(semantics.get("model"), dict(model_config), "reward.model")
    manifest = FeatureManifest.from_mapping(semantics.get("features"))
    vocabulary = ChemicalVocabulary.from_mapping(semantics.get("chemical_vocabulary"))
    if (manifest.atom_feature_dim != model_config["atom_feature_dim"]
            or manifest.edge_feature_dim != model_config["edge_feature_dim"]
            or vocabulary.num_atom_types != model_config["num_atom_types"]
            or vocabulary.num_bond_types != model_config["num_bond_types"]):
        raise ConfigurationError("RA reward metadata dimensions do not match model")
    if not isinstance(provenance.get("teacher_version"), str) or not provenance["teacher_version"].strip():
        raise ConfigurationError("RA reward provenance requires teacher_version")
    for key in ("records_sha256", "labels_sha256"):
        digest = provenance.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ConfigurationError(f"RA reward provenance requires a valid {key}")
    plain_data(semantics)
    plain_data(provenance)


def validate_reward_guidance_metadata(
    semantics: Mapping[str, Any], provenance: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    require_match(semantics["features"], feature_semantics(config), "reward.features")
    require_match(semantics["chemical_vocabulary"], chemical_semantics(config), "reward.chemical_vocabulary")
    section = config["ra_reward"]
    expected = section.get("expected_provenance")
    keys = ("teacher_version", "labels_sha256", "records_sha256")
    if not isinstance(expected, Mapping) or set(expected) != set(keys):
        raise ConfigurationError("ra_reward.expected_provenance requires teacher_version, labels_sha256 and records_sha256")
    require_match({key: provenance[key] for key in keys}, dict(expected), "reward.provenance")
