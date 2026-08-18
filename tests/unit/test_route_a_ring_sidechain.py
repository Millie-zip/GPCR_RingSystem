from dataclasses import replace

import torch

from gpcr_ringgen.core.types import (
    BatchMask,
    BatchMetadata,
    DescriptorBatch,
    MolecularGraph,
    PocketBatch,
    Stage3Input,
)
from gpcr_ringgen.models.diffusion import (
    ConditionalOuterAtomCountModel,
    DDPMConfig,
    DDPMNoiseSchedule,
    OuterAtomCountConfig,
    RingSidechainClassConfig,
    build_route_a_training_batch,
    clean_ring_sidechain_state_from_input,
    expand_ring_core_batch,
    q_sample_ring_sidechain,
)
from gpcr_ringgen.models.diffusion.state import initialize_stage3_state
from gpcr_ringgen.models.equivariant import (
    EquivariantDenoiserBackbone,
    EquivariantDenoiserConfig,
)
from gpcr_ringgen.sampling import RouteAStage3Sampler, materialized_bond_mask
from gpcr_ringgen.training import RingSidechainDiffusionLoss
from gpcr_ringgen.utils.seed import make_torch_generator


def _ring_input() -> Stage3Input:
    # Four-node square core. The two diagonals are absent and remain fixed absent.
    edge_index = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 0]]])
    atom_mask = torch.ones((1, 4), dtype=torch.bool)
    edge_mask = torch.ones((1, 4), dtype=torch.bool)
    ring_atom_mask = atom_mask.clone()
    ring_pair_mask = edge_mask.clone()
    ring_edge_mask = edge_mask.clone()
    value = Stage3Input(
        scaffold=MolecularGraph(
            atom_features=torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]
            ),
            edge_index=edge_index,
            edge_features=torch.ones((1, 4, 1)),
            atom_types=torch.tensor([[0, 1, 2, 0]]),
            # Route-A vocabulary reserves zero for NO_BOND.
            bond_types=torch.tensor([[1, 2, 1, 2]]),
            formal_charge_types=torch.tensor([[2, 2, 1, 2]]),
            chiralities=torch.tensor([[0, 1, 0, 2]]),
        ),
        descriptor=DescriptorBatch(torch.tensor([[0.2, -0.3]])),
        pocket=PocketBatch(
            coordinates=torch.tensor([[[0.0, 0.0, 0.0]]]),
            features=torch.tensor([[[1.0]]]),
        ),
        mask=BatchMask(
            atom_mask=atom_mask,
            edge_mask=edge_mask,
            descriptor_mask=torch.ones((1, 2), dtype=torch.bool),
            pocket_mask=torch.ones((1, 1), dtype=torch.bool),
            ring_atom_mask=ring_atom_mask,
            ring_pair_mask=ring_pair_mask,
            ring_edge_mask=ring_edge_mask,
        ),
        metadata=BatchMetadata(sample_ids=("ring",), records=({},)),
    )
    value.validate()
    return value


def _schedule() -> DDPMNoiseSchedule:
    return DDPMNoiseSchedule(
        DDPMConfig(
            mode="joint_chemical",
            num_timesteps=3,
            beta_schedule="linear",
            beta_start=0.01,
            beta_end=0.05,
            prediction_type="epsilon",
            loss_reduction="mean",
            num_atom_types=4,
            num_bond_types=5,
        )
    )


def _classes() -> RingSidechainClassConfig:
    return RingSidechainClassConfig(
        num_atom_types=4,
        num_bond_states=5,
        num_formal_charge_types=5,
        num_chirality_types=3,
    )


def _denoiser() -> EquivariantDenoiserBackbone:
    return EquivariantDenoiserBackbone(
        EquivariantDenoiserConfig(
            atom_feature_dim=2,
            edge_feature_dim=1,
            hidden_dim=16,
            num_layers=2,
            timestep_embedding_dim=8,
            coordinate_scale=0.1,
            epsilon=1.0e-8,
            max_time_period=10000.0,
            descriptor_dim=2,
            num_atom_types=4,
            num_bond_types=5,
            num_formal_charge_types=5,
            num_chirality_types=3,
        )
    )


def test_count_model_is_seeded_and_has_no_count_class_cap() -> None:
    model = ConditionalOuterAtomCountModel(
        OuterAtomCountConfig(atom_feature_dim=2, descriptor_dim=2, hidden_dim=8)
    )
    request = _ring_input()
    rate = model(
        request.scaffold.atom_features,
        request.mask.ring_atom_mask,
        request.descriptor.values,
        request.mask.descriptor_mask,
    )
    first = model.sample(rate, generator=make_torch_generator(17, device=rate.device))
    second = model.sample(rate, generator=make_torch_generator(17, device=rate.device))
    assert torch.equal(first, second)
    assert first.shape == (1,)
    assert bool((first >= 0).all())
    loss = model.loss(rate, torch.tensor([3]))
    assert loss.ndim == 0 and bool(torch.isfinite(loss))


def test_dynamic_expansion_uses_batch_local_size_and_all_candidate_pairs() -> None:
    expanded = expand_ring_core_batch(_ring_input(), torch.tensor([2]))
    assert expanded.scaffold.atom_features.shape[:2] == (1, 6)
    assert expanded.scaffold.edge_index.shape == (1, 2, 15)
    assert int(expanded.mask.ring_atom_mask.sum()) == 4
    assert int(expanded.mask.ring_pair_mask.sum()) == 6
    assert int(expanded.mask.ring_edge_mask.sum()) == 4
    fixed_non_edges = expanded.mask.ring_pair_mask & ~expanded.mask.ring_edge_mask
    assert torch.equal(
        expanded.scaffold.bond_types[fixed_non_edges],
        torch.zeros(2, dtype=torch.long),
    )
    assert expanded.metadata.records[0]["route_a_outer_atom_count"] == 2


def test_training_conversion_marks_real_bonds_and_no_bond_candidates() -> None:
    base = _ring_input()
    full = Stage3Input(
        scaffold=MolecularGraph(
            atom_features=torch.cat(
                (base.scaffold.atom_features, torch.tensor([[[0.2, 0.8]]])), dim=1
            ),
            edge_index=torch.tensor([[[0, 1, 2, 3, 1], [1, 2, 3, 0, 4]]]),
            edge_features=torch.ones((1, 5, 1)),
            coordinates=torch.zeros((1, 5, 3)),
            atom_types=torch.tensor([[0, 1, 2, 0, 3]]),
            # Sparse training input uses the legacy real-bond IDs 0..3.
            bond_types=torch.tensor([[0, 1, 0, 1, 2]]),
            formal_charge_types=torch.tensor([[2, 2, 1, 2, 3]]),
            chiralities=torch.tensor([[0, 1, 0, 2, 1]]),
        ),
        descriptor=base.descriptor,
        pocket=base.pocket,
        mask=BatchMask(
            atom_mask=torch.ones((1, 5), dtype=torch.bool),
            edge_mask=torch.ones((1, 5), dtype=torch.bool),
            descriptor_mask=base.mask.descriptor_mask,
            pocket_mask=base.mask.pocket_mask,
            ring_atom_mask=torch.tensor([[True, True, True, True, False]]),
            ring_pair_mask=torch.tensor([[True, True, True, True, False]]),
            ring_edge_mask=torch.tensor([[True, True, True, True, False]]),
        ),
        metadata=base.metadata,
    )
    full.validate()
    converted = build_route_a_training_batch(full)
    assert converted.scaffold.edge_index.shape[-1] == 10
    assert int((converted.scaffold.bond_types != 0).sum()) == 5
    assert int(converted.mask.ring_pair_mask.sum()) == 6
    assert int(converted.mask.ring_edge_mask.sum()) == 4
    assert torch.equal(converted.scaffold.coordinates, full.scaffold.coordinates)
    clean_state = clean_ring_sidechain_state_from_input(converted)
    assert torch.equal(clean_state.coordinates, full.scaffold.coordinates)


def test_partitioned_forward_diffusion_and_loss() -> None:
    expanded = expand_ring_core_batch(_ring_input(), torch.tensor([2]))
    clean = initialize_stage3_state(expanded, seed=5, timestep=0)
    outer = clean.mask.atom_mask & ~clean.mask.ring_atom_mask
    clean = replace(
        clean,
        atom_types=torch.tensor([[0, 1, 2, 0, 3, 1]]),
        formal_charge_types=torch.where(
            outer, torch.tensor([[0, 0, 0, 0, 3, 4]]), clean.formal_charge_types
        ),
        chiralities=torch.where(
            outer, torch.tensor([[0, 0, 0, 0, 1, 2]]), clean.chiralities
        ),
    )
    target = q_sample_ring_sidechain(
        clean,
        torch.tensor([2]),
        _schedule(),
        _classes(),
        generator=make_torch_generator(19, device=clean.coordinates.device),
    )
    ring = clean.mask.ring_atom_mask
    assert torch.equal(
        target.noisy_state.formal_charge_types[ring], clean.formal_charge_types[ring]
    )
    assert torch.equal(target.noisy_state.chiralities[ring], clean.chiralities[ring])
    fixed_non_edges = clean.mask.ring_pair_mask & ~clean.mask.ring_edge_mask
    assert bool((target.noisy_state.bond_types[fixed_non_edges] == 0).all())
    assert bool((target.noisy_state.bond_types[clean.mask.ring_edge_mask] > 0).all())

    model = _denoiser()
    prediction = model(target.noisy_state, target.noisy_state.timestep)
    assert prediction.formal_charge_logits.shape == (1, 6, 5)
    assert prediction.chirality_logits.shape == (1, 6, 3)
    loss = RingSidechainDiffusionLoss()(prediction, target)
    loss.total.backward()
    assert bool(torch.isfinite(loss.total))
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_route_a_sampler_runs_and_preserves_ring_existence() -> None:
    request = _ring_input()
    count_model = ConditionalOuterAtomCountModel(
        OuterAtomCountConfig(atom_feature_dim=2, descriptor_dim=2, hidden_dim=8)
    )
    for parameter in count_model.parameters():
        torch.nn.init.zeros_(parameter)
    output = RouteAStage3Sampler(_schedule(), _classes()).sample(
        request,
        _denoiser(),
        count_model,
        seed=23,
        runtime_atom_limit=20,
    )
    state = output.final_state
    assert bool((state.bond_types[state.mask.ring_edge_mask] > 0).all())
    fixed_non_edges = state.mask.ring_pair_mask & ~state.mask.ring_edge_mask
    assert bool((state.bond_types[fixed_non_edges] == 0).all())
    assert torch.equal(
        state.formal_charge_types[state.mask.ring_atom_mask],
        request.scaffold.formal_charge_types[request.mask.ring_atom_mask],
    )
    assert materialized_bond_mask(state).shape == state.mask.edge_mask.shape
    assert bool(torch.isfinite(state.coordinates).all())
