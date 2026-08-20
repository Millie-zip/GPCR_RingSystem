"""Stage 3 DDPM reverse diffusion and local-coordinate sampling."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from gpcr_ringgen.core.types import (
    BatchMetadata,
    MolecularGraph,
    Stage3DiffusionState,
    Stage3Input,
)
from gpcr_ringgen.models.diffusion import (
    DDPMNoiseSchedule,
    initialize_stage3_state,
    project_zero_center_of_mass,
)
from gpcr_ringgen.models.diffusion.chemical import (
    categorical_reverse_step,
    initialize_joint_stage3_state,
)
from gpcr_ringgen.models.diffusion.ring_sidechain import (
    RingSidechainClassConfig,
    initialize_ring_sidechain_state,
    reverse_route_categories,
)
from gpcr_ringgen.models.diffusion.route_a import (
    ConditionalOuterAtomCountModel,
    expand_ring_core_batch,
)
from gpcr_ringgen.models.equivariant import DenoiserOutput
from gpcr_ringgen.pocket import (
    PocketCoordinateTransform,
    QueryablePocketField,
    inverse_transform_coordinates,
)
from gpcr_ringgen.utils.seed import make_torch_generator


def _validate_prediction(
    predicted_noise: Tensor,
    state: Stage3DiffusionState,
) -> None:
    if not isinstance(predicted_noise, Tensor):
        raise TypeError("predicted_noise must be a torch.Tensor")
    if predicted_noise.shape != state.coordinates.shape:
        raise ValueError("predicted_noise must have shape [B, N, 3]")
    if predicted_noise.dtype != state.coordinates.dtype:
        raise TypeError("predicted_noise dtype must match state coordinates")
    if predicted_noise.device != state.coordinates.device:
        raise ValueError("predicted_noise device must match state coordinates")
    if not bool(torch.isfinite(predicted_noise).all()):
        raise ValueError("predicted_noise contains NaN or Inf")


def _extract_schedule_value(
    values: Tensor,
    timestep: Tensor,
    reference: Tensor,
) -> Tensor:
    selected = values.to(
        device=reference.device,
        dtype=reference.dtype,
    ).index_select(0, timestep)
    return selected.reshape((reference.shape[0],) + (1,) * (reference.ndim - 1))


@dataclass(frozen=True)
class ReverseDiffusionStepOutput:
    """One sampled DDPM posterior transition.

    Tensor fields:
        previous_state.coordinates: Shape ``[B, N, 3]``; floating-point on the
            model device. Valid atoms contain ``x_{t-1}`` (or the final
            ``x_0`` estimate when ``t=0``); padded atoms are exactly zero.
        posterior_mean: Shape ``[B, N, 3]``; same dtype/device. The epsilon
            parameterized DDPM mean before stochastic posterior noise. Padding
            is zero and the valid-atom center of mass is zero.
        posterior_variance: Shape ``[B, 1, 1]``; same dtype/device. Values are
            non-negative. The value is zero for samples at ``t=0``.
    """

    previous_state: Stage3DiffusionState
    posterior_mean: Tensor
    posterior_variance: Tensor

    def validate(self) -> None:
        """Validate shapes, devices, finiteness, mask, and variance range."""

        if not isinstance(self.previous_state, Stage3DiffusionState):
            raise TypeError("previous_state must be a Stage3DiffusionState")
        self.previous_state.validate()
        coordinates = self.previous_state.coordinates
        if self.posterior_mean.shape != coordinates.shape:
            raise ValueError("posterior_mean must have shape [B, N, 3]")
        if self.posterior_variance.shape != (coordinates.shape[0], 1, 1):
            raise ValueError("posterior_variance must have shape [B, 1, 1]")
        for name, tensor in (
            ("posterior_mean", self.posterior_mean),
            ("posterior_variance", self.posterior_variance),
        ):
            if tensor.dtype != coordinates.dtype:
                raise TypeError(f"{name} dtype must match state coordinates")
            if tensor.device != coordinates.device:
                raise ValueError(f"{name} device must match state coordinates")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if bool((self.posterior_variance < 0).any()):
            raise ValueError("posterior_variance must be non-negative")
        padding = ~self.previous_state.mask.atom_mask.unsqueeze(-1)
        for name, tensor in (
            ("previous coordinates", coordinates),
            ("posterior_mean", self.posterior_mean),
        ):
            if bool((tensor.masked_select(padding) != 0).any()):
                raise ValueError(f"{name} must be zero for padded atoms")


def ddpm_reverse_step(
    state: Stage3DiffusionState,
    predicted_noise: Tensor,
    schedule: DDPMNoiseSchedule,
    *,
    generator: torch.Generator | None = None,
) -> ReverseDiffusionStepOutput:
    r"""Sample one standard epsilon-parameterized DDPM reverse transition.

    The posterior mean is
    ``mu = alpha_t^-1/2 * (x_t - beta_t / sqrt(1-alpha_bar_t) * epsilon)``.
    For ``t>0``, centered Gaussian noise is scaled by posterior variance
    ``beta_t * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)``. For ``t=0`` the
    variance is zero and the mean is returned directly.

    Coordinates, predicted noise, and stochastic noise are projected into the
    valid-atom zero-center-of-mass subspace. Thus padding is never updated and
    global ligand translation is never sampled.
    """

    if not isinstance(state, Stage3DiffusionState):
        raise TypeError("state must be a Stage3DiffusionState")
    state.validate()
    if not isinstance(schedule, DDPMNoiseSchedule):
        raise TypeError("schedule must be a DDPMNoiseSchedule")
    _validate_prediction(predicted_noise, state)
    timestep = state.timestep
    if bool((timestep >= schedule.num_timesteps).any()):
        raise ValueError(
            f"state timestep values must be below {schedule.num_timesteps}"
        )

    coordinates = project_zero_center_of_mass(
        state.coordinates,
        state.mask.atom_mask,
    )
    epsilon = project_zero_center_of_mass(
        predicted_noise,
        state.mask.atom_mask,
    )
    beta_t = _extract_schedule_value(schedule.betas, timestep, coordinates)
    alpha_t = _extract_schedule_value(schedule.alphas, timestep, coordinates)
    alpha_bar_t = _extract_schedule_value(
        schedule.alpha_bars,
        timestep,
        coordinates,
    )
    previous_timestep = torch.clamp(timestep - 1, min=0)
    alpha_bar_previous = _extract_schedule_value(
        schedule.alpha_bars,
        previous_timestep,
        coordinates,
    )
    at_initial_step = (timestep == 0).reshape(-1, 1, 1)
    alpha_bar_previous = torch.where(
        at_initial_step,
        torch.ones_like(alpha_bar_previous),
        alpha_bar_previous,
    )

    posterior_mean = torch.rsqrt(alpha_t) * (
        coordinates
        - beta_t / torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )
    posterior_mean = project_zero_center_of_mass(
        posterior_mean,
        state.mask.atom_mask,
    )
    posterior_variance = beta_t * (1.0 - alpha_bar_previous) / (
        1.0 - alpha_bar_t
    )
    posterior_variance = torch.where(
        at_initial_step,
        torch.zeros_like(posterior_variance),
        posterior_variance,
    ).clamp_min(0.0)

    stochastic_noise = torch.randn(
        coordinates.shape,
        dtype=coordinates.dtype,
        device=coordinates.device,
        generator=generator,
    )
    stochastic_noise = project_zero_center_of_mass(
        stochastic_noise,
        state.mask.atom_mask,
    )
    sampled_coordinates = posterior_mean + torch.sqrt(
        posterior_variance
    ) * stochastic_noise
    sampled_coordinates = project_zero_center_of_mass(
        sampled_coordinates,
        state.mask.atom_mask,
    )
    # The terminal t=0 transition is deterministic. Selecting the already
    # centered mean also avoids an unnecessary second projection and its
    # possible last-bit floating-point perturbation.
    previous_coordinates = torch.where(
        at_initial_step,
        posterior_mean,
        sampled_coordinates,
    )
    previous_state = replace(
        state,
        coordinates=previous_coordinates,
        timestep=previous_timestep,
    )
    output = ReverseDiffusionStepOutput(
        previous_state=previous_state,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
    )
    output.validate()
    return output


@dataclass(frozen=True)
class Stage3SamplingOutput:
    """Local-coordinate Stage 3 reverse-diffusion result.

    Tensor fields:
        coordinates: Shape ``[B, N, 3]`` through ``final_state`` and ``graph``;
            floating-point on the model device. Valid atoms are generated local
            coordinates with zero center of mass. Padded atoms are zero.
        graph.atom_features: Shape ``[B, N, F_atom]``; unchanged scaffold
            features. ``graph.edge_index`` is ``[B, 2, E]`` and
            ``graph.edge_features`` is ``[B, E, F_edge]``; topology is unchanged.
        trajectory: Optional tuple of coordinate tensors, each ``[B, N, 3]``
            with the same dtype/device/mask semantics. It starts with the
            initialized Gaussian state and contains every reverse-step result.
            It is ``None`` by default to avoid retaining ``T+1`` tensors.
    """

    final_state: Stage3DiffusionState
    graph: MolecularGraph
    metadata: BatchMetadata
    trajectory: tuple[Tensor, ...] | None = None
    coordinate_frame: str = "local"

    @property
    def coordinates(self) -> Tensor:
        """Return generated local coordinates shaped ``[B, N, 3]``."""

        return self.final_state.coordinates

    def validate(self, *, num_timesteps: int | None = None) -> None:
        """Validate the output contract and optional trajectory."""

        self.final_state.validate()
        self.graph.validate()
        self.metadata.validate(self.final_state.batch_size)
        if self.coordinate_frame != "local":
            raise ValueError("Stage3SamplingOutput coordinate_frame must be 'local'")
        if self.graph.coordinates is None:
            raise ValueError("sampling graph coordinates are required")
        state = self.final_state
        for name, graph_tensor, state_tensor in (
            ("atom_features", self.graph.atom_features, state.atom_features),
            ("edge_index", self.graph.edge_index, state.edge_index),
            ("edge_features", self.graph.edge_features, state.edge_features),
            ("coordinates", self.graph.coordinates, state.coordinates),
        ):
            if not torch.equal(graph_tensor, state_tensor):
                raise ValueError(f"graph {name} must match final_state")
        for name, graph_tensor, state_tensor in (
            ("atom_types", self.graph.atom_types, state.atom_types),
            ("bond_types", self.graph.bond_types, state.bond_types),
            (
                "formal_charge_types",
                self.graph.formal_charge_types,
                state.formal_charge_types,
            ),
            ("chiralities", self.graph.chiralities, state.chiralities),
        ):
            if (graph_tensor is None) != (state_tensor is None):
                raise ValueError(f"graph {name} presence must match final_state")
            if graph_tensor is not None and state_tensor is not None:
                if not torch.equal(graph_tensor, state_tensor):
                    raise ValueError(f"graph {name} must match final_state")
        if self.metadata is not state.metadata:
            raise ValueError("sampling metadata must be the final_state metadata")
        padding = ~state.mask.atom_mask.unsqueeze(-1)
        if bool((state.coordinates.masked_select(padding) != 0).any()):
            raise ValueError("generated coordinates must be zero for padded atoms")
        if not bool(torch.isfinite(state.coordinates).all()):
            raise ValueError("generated coordinates contain NaN or Inf")
        if self.trajectory is not None:
            if num_timesteps is not None and len(self.trajectory) != num_timesteps + 1:
                raise ValueError("trajectory must contain initialization plus every step")
            for coordinates in self.trajectory:
                if coordinates.shape != state.coordinates.shape:
                    raise ValueError("trajectory coordinates must have shape [B, N, 3]")
                if coordinates.dtype != state.coordinates.dtype:
                    raise TypeError("trajectory dtype must match final coordinates")
                if coordinates.device != state.coordinates.device:
                    raise ValueError("trajectory device must match final coordinates")
                if not bool(torch.isfinite(coordinates).all()):
                    raise ValueError("trajectory contains NaN or Inf")
                if bool((coordinates.masked_select(padding) != 0).any()):
                    raise ValueError("trajectory must be zero for padded atoms")
            if not torch.equal(self.trajectory[-1], state.coordinates):
                raise ValueError("trajectory final coordinates must match final_state")


class DDPMStage3Sampler:
    """Run trained-compatible Stage 3 denoisers through a full DDPM chain."""

    def __init__(self, schedule: DDPMNoiseSchedule) -> None:
        if not isinstance(schedule, DDPMNoiseSchedule):
            raise TypeError("schedule must be a DDPMNoiseSchedule")
        if schedule.config.prediction_type != "epsilon":
            raise ValueError("DDPMStage3Sampler requires epsilon prediction")
        if schedule.config.mode != "coordinate_only":
            raise ValueError(
                "DDPMStage3Sampler requires coordinate_only mode; use "
                "JointDDPMStage3Sampler for joint_chemical mode"
            )
        self.schedule = schedule

    def _predict_noise(
        self,
        denoiser: nn.Module,
        state: Stage3DiffusionState,
        pocket_field: QueryablePocketField | None,
    ) -> Tensor:
        if pocket_field is None:
            output = denoiser(state, state.timestep)
        else:
            output = denoiser(
                state,
                state.timestep,
                pocket_field=pocket_field,
            )
        if not isinstance(output, DenoiserOutput):
            raise TypeError("denoiser must return DenoiserOutput")
        output.validate(state)
        return output.coordinate_noise

    def sample(
        self,
        stage3_input: Stage3Input,
        denoiser: nn.Module,
        *,
        seed: int,
        pocket_field: QueryablePocketField | None = None,
        return_trajectory: bool = False,
    ) -> Stage3SamplingOutput:
        """Generate a padded batch of local ligand coordinates.

        Args:
            stage3_input: Validated Stage 2 graph, descriptor, pocket, masks,
                and metadata. Graph topology and descriptor row ordering are
                preserved in the returned state.
            denoiser: Trained-compatible base or Field-conditioned module that
                returns :class:`DenoiserOutput`. Randomly initialized weights
                are valid only for structural smoke tests.
            seed: Controls both centered ``X_T`` initialization and all reverse
                posterior noise through local generators. Global RNG state is
                not changed.
            pocket_field: Optional stable query field. Required by an additive
                Field-conditioned denoiser and omitted for no-field ablation.
                If omitted, ``stage3_input.pocket_field`` is used when it
                implements :class:`QueryablePocketField`.
            return_trajectory: Retain initialization and all reverse states if
                ``True``. Defaults to ``False`` to avoid ``O(T*B*N)`` memory.

        Returns:
            :class:`Stage3SamplingOutput` with generated local coordinates,
            unchanged graph topology/features, sample metadata, and optional
            trajectory.
        """

        if not isinstance(stage3_input, Stage3Input):
            raise TypeError("stage3_input must be a Stage3Input")
        stage3_input.validate()
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module")
        if not isinstance(return_trajectory, bool):
            raise TypeError("return_trajectory must be a bool")
        effective_field = pocket_field
        if effective_field is None and isinstance(
            stage3_input.pocket_field,
            QueryablePocketField,
        ):
            effective_field = stage3_input.pocket_field
        if effective_field is not None:
            effective_field.validate(
                batch_size=stage3_input.batch_size,
                device=stage3_input.device,
            )

        state = initialize_stage3_state(
            stage3_input,
            seed=seed,
            timestep=self.schedule.num_timesteps - 1,
        )
        reverse_seed = (seed + 1) % (2**64)
        generator = make_torch_generator(
            reverse_seed,
            device=state.coordinates.device,
        )
        trajectory: list[Tensor] | None = (
            [state.coordinates.clone()] if return_trajectory else None
        )

        was_training = denoiser.training
        denoiser.eval()
        try:
            with torch.no_grad():
                for timestep_index in range(
                    self.schedule.num_timesteps - 1,
                    -1,
                    -1,
                ):
                    timestep = torch.full(
                        (state.batch_size,),
                        timestep_index,
                        dtype=torch.long,
                        device=state.coordinates.device,
                    )
                    state = replace(state, timestep=timestep)
                    predicted_noise = self._predict_noise(
                        denoiser,
                        state,
                        effective_field,
                    )
                    state = ddpm_reverse_step(
                        state,
                        predicted_noise,
                        self.schedule,
                        generator=generator,
                    ).previous_state
                    if trajectory is not None:
                        trajectory.append(state.coordinates.clone())
        finally:
            denoiser.train(was_training)

        graph = MolecularGraph(
            atom_features=state.atom_features,
            edge_index=state.edge_index,
            edge_features=state.edge_features,
            coordinates=state.coordinates,
            atom_types=state.atom_types,
            bond_types=state.bond_types,
        )
        result = Stage3SamplingOutput(
            final_state=state,
            graph=graph,
            metadata=state.metadata,
            trajectory=None if trajectory is None else tuple(trajectory),
        )
        result.validate(num_timesteps=self.schedule.num_timesteps)
        return result


class JointDDPMStage3Sampler:
    """Jointly sample atom labels, fixed-edge bond labels, and coordinates.

    The topology tensors ``edge_index``, ``atom_mask``, and ``edge_mask`` are
    copied from :class:`Stage3Input` and never modified. Atom and bond labels
    start from uniform categorical priors while coordinates start from the
    existing centered Gaussian prior.
    """

    def __init__(self, schedule: DDPMNoiseSchedule) -> None:
        if not isinstance(schedule, DDPMNoiseSchedule):
            raise TypeError("schedule must be a DDPMNoiseSchedule")
        if schedule.config.mode != "joint_chemical":
            raise ValueError("JointDDPMStage3Sampler requires joint_chemical mode")
        if schedule.config.prediction_type != "epsilon":
            raise ValueError("JointDDPMStage3Sampler requires epsilon prediction")
        if schedule.config.num_atom_types is None:
            raise ValueError("joint sampler requires num_atom_types")
        if schedule.config.num_bond_types is None:
            raise ValueError("joint sampler requires num_bond_types")
        self.schedule = schedule

    @staticmethod
    def _predict(
        denoiser: nn.Module,
        state: Stage3DiffusionState,
        pocket_field: QueryablePocketField | None,
    ) -> DenoiserOutput:
        if pocket_field is None:
            output = denoiser(state, state.timestep)
        else:
            output = denoiser(state, state.timestep, pocket_field=pocket_field)
        if not isinstance(output, DenoiserOutput):
            raise TypeError("denoiser must return DenoiserOutput")
        output.validate(state)
        if output.atom_type_logits is None or output.bond_type_logits is None:
            raise ValueError("joint denoiser must predict atom and bond type logits")
        return output

    def sample(
        self,
        stage3_input: Stage3Input,
        denoiser: nn.Module,
        *,
        seed: int,
        pocket_field: QueryablePocketField | None = None,
        return_trajectory: bool = False,
    ) -> Stage3SamplingOutput:
        """Run fixed-topology reverse diffusion from chemical/coordinate noise."""

        if not isinstance(stage3_input, Stage3Input):
            raise TypeError("stage3_input must be a Stage3Input")
        stage3_input.validate()
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module")
        if not isinstance(return_trajectory, bool):
            raise TypeError("return_trajectory must be a bool")
        effective_field = pocket_field
        if effective_field is None and isinstance(
            stage3_input.pocket_field,
            QueryablePocketField,
        ):
            effective_field = stage3_input.pocket_field
        if effective_field is not None:
            effective_field.validate(
                batch_size=stage3_input.batch_size,
                device=stage3_input.device,
            )

        state = initialize_joint_stage3_state(
            stage3_input,
            self.schedule,
            seed=seed,
        )
        generator = make_torch_generator(
            (seed + 2) % (2**64),
            device=state.coordinates.device,
        )
        trajectory: list[Tensor] | None = (
            [state.coordinates.clone()] if return_trajectory else None
        )
        num_atom_types = self.schedule.config.num_atom_types
        num_bond_types = self.schedule.config.num_bond_types
        if num_atom_types is None or num_bond_types is None:
            raise RuntimeError("joint class counts are unavailable")

        was_training = denoiser.training
        denoiser.eval()
        try:
            with torch.no_grad():
                for timestep_index in range(
                    self.schedule.num_timesteps - 1,
                    -1,
                    -1,
                ):
                    timestep = torch.full(
                        (state.batch_size,),
                        timestep_index,
                        dtype=torch.long,
                        device=state.coordinates.device,
                    )
                    state = replace(state, timestep=timestep)
                    prediction = self._predict(denoiser, state, effective_field)
                    if state.atom_types is None or state.bond_types is None:
                        raise RuntimeError("joint state labels are unavailable")
                    previous_atom_types = categorical_reverse_step(
                        state.atom_types,
                        prediction.atom_type_logits,
                        timestep,
                        self.schedule,
                        state.mask.atom_mask,
                        num_atom_types,
                        generator=generator,
                    )
                    previous_bond_types = categorical_reverse_step(
                        state.bond_types,
                        prediction.bond_type_logits,
                        timestep,
                        self.schedule,
                        state.mask.edge_mask,
                        num_bond_types,
                        generator=generator,
                    )
                    coordinate_step = ddpm_reverse_step(
                        state,
                        prediction.coordinate_noise,
                        self.schedule,
                        generator=generator,
                    )
                    state = replace(
                        coordinate_step.previous_state,
                        atom_types=previous_atom_types,
                        bond_types=previous_bond_types,
                    )
                    state.validate()
                    if trajectory is not None:
                        trajectory.append(state.coordinates.clone())
        finally:
            denoiser.train(was_training)

        graph = MolecularGraph(
            atom_features=state.atom_features,
            edge_index=state.edge_index,
            edge_features=state.edge_features,
            coordinates=state.coordinates,
            atom_types=state.atom_types,
            bond_types=state.bond_types,
        )
        result = Stage3SamplingOutput(
            final_state=state,
            graph=graph,
            metadata=state.metadata,
            trajectory=None if trajectory is None else tuple(trajectory),
        )
        result.validate(num_timesteps=self.schedule.num_timesteps)
        return result


class RouteAStage3Sampler:
    """Generate an outer count, expand the ring core, then run joint DDPM."""

    def __init__(
        self,
        schedule: DDPMNoiseSchedule,
        classes: RingSidechainClassConfig,
    ) -> None:
        if not isinstance(schedule, DDPMNoiseSchedule):
            raise TypeError("schedule must be a DDPMNoiseSchedule")
        if not isinstance(classes, RingSidechainClassConfig):
            raise TypeError("classes must be a RingSidechainClassConfig")
        if schedule.config.prediction_type != "epsilon":
            raise ValueError("Route A requires epsilon coordinate prediction")
        self.schedule = schedule
        self.classes = classes

    @staticmethod
    def _predict(
        denoiser: nn.Module,
        state: Stage3DiffusionState,
        pocket_field: QueryablePocketField | None,
    ) -> DenoiserOutput:
        output = (
            denoiser(state, state.timestep)
            if pocket_field is None
            else denoiser(state, state.timestep, pocket_field=pocket_field)
        )
        if not isinstance(output, DenoiserOutput):
            raise TypeError("denoiser must return DenoiserOutput")
        output.validate(state)
        required = (
            output.atom_type_logits,
            output.bond_type_logits,
            output.formal_charge_logits,
            output.chirality_logits,
        )
        if any(values is None for values in required):
            raise ValueError("Route-A denoiser must predict all chemical logits")
        return output

    def sample(
        self,
        ring_input: Stage3Input,
        denoiser: nn.Module,
        count_model: ConditionalOuterAtomCountModel,
        *,
        seed: int,
        runtime_atom_limit: int | None = None,
        pocket_field: QueryablePocketField | None = None,
        return_trajectory: bool = False,
    ) -> Stage3SamplingOutput:
        """Sample variable-size side chains around a fixed ring-core subgraph."""

        if not isinstance(ring_input, Stage3Input):
            raise TypeError("ring_input must be a Stage3Input")
        ring_input.validate()
        if ring_input.mask.ring_atom_mask is None:
            raise ValueError("Route A requires ring_atom_mask")
        if ring_input.mask.descriptor_mask is None:
            raise ValueError("Route A requires descriptor_mask")
        if not isinstance(denoiser, nn.Module):
            raise TypeError("denoiser must be a torch.nn.Module")
        if not isinstance(count_model, ConditionalOuterAtomCountModel):
            raise TypeError("count_model must be ConditionalOuterAtomCountModel")
        if not isinstance(return_trajectory, bool):
            raise TypeError("return_trajectory must be a bool")

        count_generator = make_torch_generator(seed, device=ring_input.device)
        count_was_training = count_model.training
        denoiser_was_training = denoiser.training
        count_model.eval()
        denoiser.eval()
        try:
            with torch.no_grad():
                rate = count_model(
                    ring_input.scaffold.atom_features,
                    ring_input.mask.ring_atom_mask,
                    ring_input.descriptor.values,
                    ring_input.mask.descriptor_mask,
                )
                outer_counts = count_model.sample(rate, generator=count_generator)
                expanded_input = expand_ring_core_batch(
                    ring_input,
                    outer_counts,
                    runtime_atom_limit=runtime_atom_limit,
                )
                effective_field = pocket_field
                if effective_field is None and isinstance(
                    expanded_input.pocket_field, QueryablePocketField
                ):
                    effective_field = expanded_input.pocket_field
                if effective_field is not None:
                    effective_field.validate(
                        batch_size=expanded_input.batch_size,
                        device=expanded_input.device,
                    )
                state = initialize_ring_sidechain_state(
                    expanded_input,
                    self.schedule,
                    self.classes,
                    seed=(seed + 1) % (2**64),
                )
                generator = make_torch_generator(
                    (seed + 2) % (2**64), device=state.coordinates.device
                )
                trajectory: list[Tensor] | None = (
                    [state.coordinates.clone()] if return_trajectory else None
                )
                for timestep_index in range(
                    self.schedule.num_timesteps - 1, -1, -1
                ):
                    timestep = torch.full(
                        (state.batch_size,),
                        timestep_index,
                        dtype=torch.long,
                        device=state.coordinates.device,
                    )
                    state = replace(state, timestep=timestep)
                    prediction = self._predict(denoiser, state, effective_field)
                    atom, bond, charge, chirality = reverse_route_categories(
                        state,
                        atom_logits=prediction.atom_type_logits,
                        bond_logits=prediction.bond_type_logits,
                        formal_charge_logits=prediction.formal_charge_logits,
                        chirality_logits=prediction.chirality_logits,
                        timestep=timestep,
                        schedule=self.schedule,
                        classes=self.classes,
                        generator=generator,
                    )
                    coordinate_step = ddpm_reverse_step(
                        state,
                        prediction.coordinate_noise,
                        self.schedule,
                        generator=generator,
                    )
                    state = replace(
                        coordinate_step.previous_state,
                        atom_types=atom,
                        bond_types=bond,
                        formal_charge_types=charge,
                        chiralities=chirality,
                    )
                    state.validate()
                    if trajectory is not None:
                        trajectory.append(state.coordinates.clone())
        finally:
            count_model.train(count_was_training)
            denoiser.train(denoiser_was_training)

        graph = MolecularGraph(
            atom_features=state.atom_features,
            edge_index=state.edge_index,
            edge_features=state.edge_features,
            coordinates=state.coordinates,
            atom_types=state.atom_types,
            bond_types=state.bond_types,
            formal_charge_types=state.formal_charge_types,
            chiralities=state.chiralities,
        )
        result = Stage3SamplingOutput(
            final_state=state,
            graph=graph,
            metadata=state.metadata,
            trajectory=None if trajectory is None else tuple(trajectory),
        )
        result.validate(num_timesteps=self.schedule.num_timesteps)
        return result


def materialized_bond_mask(state: Stage3DiffusionState) -> Tensor:
    """Return ``[B, E]`` mask for sampled non-``NO_BOND`` candidate pairs."""

    state.validate()
    if state.bond_types is None:
        raise ValueError("state.bond_types are required")
    return state.mask.edge_mask & (state.bond_types != 0)


def restore_global_coordinates(
    output: Stage3SamplingOutput,
    transform: PocketCoordinateTransform,
) -> Tensor:
    """Restore generated valid atoms from pocket-local to global coordinates.

    The returned tensor has shape ``[B, N, 3]`` and the sampling dtype/device.
    Valid atoms use ``x_global = x_local + pocket_center``. Padded atoms remain
    exact zeros instead of receiving a meaningless translated padding value.
    No files are written.
    """

    if not isinstance(output, Stage3SamplingOutput):
        raise TypeError("output must be a Stage3SamplingOutput")
    output.validate()
    global_coordinates = inverse_transform_coordinates(
        output.coordinates,
        transform,
    )
    return global_coordinates * output.final_state.mask.atom_mask.unsqueeze(-1).to(
        global_coordinates.dtype
    )


__all__ = [
    "DDPMStage3Sampler",
    "JointDDPMStage3Sampler",
    "RouteAStage3Sampler",
    "ReverseDiffusionStepOutput",
    "Stage3SamplingOutput",
    "ddpm_reverse_step",
    "restore_global_coordinates",
    "materialized_bond_mask",
]
