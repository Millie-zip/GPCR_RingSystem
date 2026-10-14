"""Tests for the independent RA reward surrogate and inference guidance."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import torch
from gpcr_ringgen.checkpoint_semantics import file_sha256, label_cache_provenance, reward_semantics
from gpcr_ringgen.vocabulary import ATOM_ID_TO_TYPE, BOND_ID_TO_TYPE

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    Stage3DiffusionState,
)
from gpcr_ringgen.models.equivariant import (
    DenoiserOutput,
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
)
from gpcr_ringgen.sampling import (
    RAScoreRewardGuidance,
    RAScoreRewardGuidanceConfig,
)
from gpcr_ringgen.training.ra_reward import (
    RARewardModel,
    RARewardModelConfig,
    load_ra_reward_checkpoint,
    save_ra_reward_checkpoint,
)


def _state(*, dtype: torch.dtype = torch.float32) -> Stage3DiffusionState:
    state = Stage3DiffusionState(
        atom_features=torch.tensor(
            [
                [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=dtype,
        ),
        edge_index=torch.tensor(
            [
                [[0, 1, 2], [1, 2, 0]],
                [[0, 1, -1], [1, 0, -1]],
            ],
            dtype=torch.long,
        ),
        edge_features=torch.tensor(
            [[[1.0], [1.0], [1.0]], [[1.0], [1.0], [0.0]]],
            dtype=dtype,
        ),
        coordinates=torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=dtype,
        ),
        descriptor=torch.ones(2, 1, dtype=dtype),
        timestep=torch.zeros(2, dtype=torch.long),
        mask=BatchMask(
            atom_mask=torch.tensor([[True, True, True], [True, True, False]]),
            edge_mask=torch.tensor([[True, True, True], [True, True, False]]),
            descriptor_mask=torch.ones(2, 1, dtype=torch.bool),
        ),
        metadata=BatchMetadata(sample_ids=("ra-0", "ra-1"), records=({}, {})),
        atom_types=torch.tensor([[0, 1, 2], [3, 4, 0]]),
        bond_types=torch.tensor([[0, 1, 3], [1, 0, 0]]),
    )
    state.validate()
    return state


def _reward_model() -> RARewardModel:
    torch.manual_seed(7)
    return RARewardModel(
        RARewardModelConfig(
            atom_feature_dim=3,
            edge_feature_dim=1,
            hidden_dim=8,
            num_layers=2,
            num_atom_types=12,
            num_bond_types=4,
        )
    )


def _denoiser() -> EquivariantDenoiserBackbone:
    return EquivariantDenoiserBackbone(
        EquivariantDenoiserConfig(
            atom_feature_dim=3,
            edge_feature_dim=1,
            hidden_dim=8,
            num_layers=1,
            timestep_embedding_dim=8,
            coordinate_scale=0.1,
            epsilon=1.0e-8,
            max_time_period=100.0,
            descriptor_dim=1,
            num_atom_types=12,
            num_bond_types=4,
        )
    )


def test_reward_model_accepts_hard_and_soft_chemical_labels() -> None:
    state = _state()
    model = _reward_model().eval()
    hard = model(state)
    soft = model(
        state,
        atom_type_probabilities=torch.nn.functional.one_hot(
            state.atom_types, num_classes=12
        ).float(),
        bond_type_probabilities=torch.nn.functional.one_hot(
            state.bond_types, num_classes=4
        ).float(),
    )
    assert hard.shape == (2,)
    assert torch.isfinite(hard).all()
    assert torch.allclose(hard, soft)
    assert bool(((hard >= 0.0) & (hard <= 1.0)).all())


def test_reward_guidance_changes_only_chemical_logits() -> None:
    state = _state()
    denoiser = _denoiser().eval()
    prediction = denoiser(state)
    guidance = RAScoreRewardGuidance(
        _reward_model(),
        RAScoreRewardGuidanceConfig(
            guidance_scale=1.0,
            start_ratio=0.0,
            eval_every=1,
            gradient_clip_norm=1.0,
        ),
    )
    result = guidance.apply(
        state,
        prediction,
        timestep_index=0,
        num_timesteps=3,
    )
    assert result.reward is not None
    assert torch.isfinite(result.reward).all()
    assert torch.equal(result.prediction.coordinate_noise, prediction.coordinate_noise)
    assert torch.isfinite(result.prediction.atom_type_logits).all()
    assert torch.isfinite(result.prediction.bond_type_logits).all()
    assert torch.count_nonzero(
        result.prediction.atom_type_logits[~state.mask.atom_mask]
    ) == 0
    assert torch.count_nonzero(
        result.prediction.bond_type_logits[~state.mask.edge_mask]
    ) == 0
    assert torch.equal(state.edge_index, _state().edge_index)


def test_reward_guidance_scale_zero_is_a_noop() -> None:
    state = _state()
    prediction = _denoiser().eval()(state)
    guidance = RAScoreRewardGuidance(
        _reward_model(),
        RAScoreRewardGuidanceConfig(guidance_scale=0.0),
    )
    result = guidance.apply(
        state,
        prediction,
        timestep_index=0,
        num_timesteps=3,
    )
    assert result.reward is None
    assert result.prediction is prediction


def test_reward_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _reward_model()
    records = tmp_path / "synthetic_records.txt"
    records.write_text("synthetic round-trip test fixture", encoding="utf-8")
    labels = tmp_path / "labels.db"
    with sqlite3.connect(labels) as connection:
        connection.execute("CREATE TABLE rascore_labels(canonical_smiles TEXT, ra_score REAL, teacher_version TEXT)")
        connection.execute("INSERT INTO rascore_labels VALUES ('CC', 0.5, 'test_teacher_v1')")
    semantics = reward_semantics({
        "feature_manifest": {"schema_version": "test_reward_features_v1", "atom_features": ["a", "b", "c"], "edge_features": ["edge"], "descriptor_features": []},
        "chemical_vocabulary": {"atom_types": list(ATOM_ID_TO_TYPE), "bond_types": list(BOND_ID_TO_TYPE)},
    }, model.config.to_mapping())
    provenance = {"records_sha256": file_sha256(records), **label_cache_provenance(labels)}
    path = save_ra_reward_checkpoint(model, tmp_path / "ra.ckpt", epoch=2, semantics=semantics, provenance=provenance)
    loaded = load_ra_reward_checkpoint(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert loaded.config == model.config
    assert loaded.checkpoint_semantics == semantics
    assert loaded.training_provenance == provenance
    assert torch.allclose(loaded(_state()), model(_state()))
