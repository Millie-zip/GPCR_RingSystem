# Public interfaces

## Shared data structures

- `RingTopology`: a topology identifier plus schema-free metadata.
- `ScaffoldCondition`: a scaffold identifier linked to the canonical `RingTopology`.
- `DiffusionState`: a schema-free payload, timestep, and metadata container.
- `PocketRepresentation`: a pocket identifier plus undecided feature and metadata mappings.
- `GeneratedMolecule`: an inspectable generated record; Dummy outputs are explicitly non-scientific.

Atom/bond vocabularies, feature meanings, pocket resolution, and scientific
representations remain intentionally undecided. Stage 3 fixes only the padded
batch transport shapes below.

## Stage 3 data contract

| Structure | Core tensor shapes | Mask semantics |
| --- | --- | --- |
| `MolecularGraph` | atoms `[B, N, F_atom]`, edges `[B, E, F_edge]`, edge index `[B, 2, E]`, optional coordinates `[B, N, 3]` | `atom_mask` and `edge_mask` use `True` for real entries |
| `DescriptorBatch` | values `[B, D]` | `descriptor_mask=True` means the value is present |
| `PocketBatch` | coordinates `[B, P, 3]`, features `[B, P, F_pocket]` | `pocket_mask=True` means a real pocket point/atom |
| `Stage3DiffusionState` | graph tensors plus required coordinates, descriptor `[B, D]`, timestep `[B]` | Carries `BatchMask`; padded values must be ignored |

Floating feature/coordinate tensors are expected on the active model device.
`edge_index` and `timestep` use `torch.long`; all masks use `torch.bool`.
`BatchMetadata` carries one unique sample ID and one metadata mapping per sample.

`Stage2Output` and plain mapping mocks can be adapted with
`stage2_output_to_stage3_input`. The adapter validates data only; it does not run
Stage 2, calculate descriptors, preprocess pockets, construct fields, diffuse,
or sample.

## Stage 3 initial state

`initialize_stage3_state(stage3_input, seed, timestep=0)` converts the existing
padded Stage 3 boundary object into `Stage3DiffusionState`; no alternate graph
or descriptor structure is introduced. With atom mask `m` and independent
Gaussian samples `Z`:

    Z[b, n, :] ~ N(0, I_3)
    center[b, :] = sum_n m[b, n] * Z[b, n, :] / sum_n m[b, n]
    X_init[b, n, :] = m[b, n] * (Z[b, n, :] - center[b, :])

The resulting coordinates have shape `[B, N, 3]`, the same floating dtype and
device as scaffold atom features. `atom_mask=True` rows are valid ligand atoms;
padding rows are exactly zero. Every sample must contain at least one valid
atom. Graph features, edge topology, descriptors, masks, sample IDs, and batch
ordering are copied without changing their values.

The required `seed` initializes a local device-specific `torch.Generator`, so
identical input and seed produce identical coordinates without mutating global
random state. The optional non-negative `timestep` is stored as `[B]`
`torch.long`; it is a state label only and does not define a diffusion noise
schedule.

## Pocket-independent equivariant denoiser

The public backbone call is:

    output = denoiser(state, timestep)
    coordinate_noise = output.coordinate_noise

`state` is the existing `Stage3DiffusionState`. `timestep` may be a
non-negative integer shared by the batch, a tensor `[B]`, or omitted to use
`state.timestep`. The output contract is:

| Tensor | Shape | dtype/device | Mask semantics |
| --- | --- | --- | --- |
| atom features | `[B, N, F_atom]` | floating; model device | scalar inputs, ignored where `atom_mask=False` |
| edge index/features | `[B, 2, E]`, `[B, E, F_edge]` | `long`, floating | only `edge_mask=True` bonds participate |
| noisy coordinates | `[B, N, 3]` | floating; model device | vector input, padding ignored |
| timestep | `[B]` | scalar numeric; model device | one invariant condition per sample |
| coordinate noise | `[B, N, 3]` | matches coordinates | zero on padding and centered over valid atoms |

The base architecture is EGNN-style. For valid atoms `i, j`, scalar messages
depend only on scalar node/bond features and squared distance:

    m_ij = phi_e(h_i, h_j, e_ij, ||x_i - x_j||^2)
    delta_x_i = sum_j unit(x_i - x_j) * phi_x(m_ij)

Both distance and learned coefficients are rotation/translation invariant;
only the relative direction carries a vector representation. Therefore
rotating coordinates rotates the predicted coordinate noise, while translating
all coordinates leaves the prediction unchanged. A final valid-atom
center-of-mass removal eliminates net translational drift.

Atom and edge features must be invariant scalars; vector-valued input feature
irreps are not supported in this base implementation. Sinusoidal timestep
embedding is also scalar. Descriptor conditioning is an optional masked linear
projection selected by `descriptor_dim`; it is disabled by default. No pocket
or Field data is read by this backbone.

Model dimensions and numerical constants are configured in
`configs/model/equivariant_denoiser.yaml`. Diffusion scheduling, forward
noising, losses, reverse dynamics, and sampling remain outside this module.

`PocketField` deliberately fixes no tensor or spatial layout. Implementations
expose batch size, device, and validation compatibility through the stable
protocol. Queryable implementations additionally satisfy
`QueryablePocketField`; Stage 3 code should use this protocol instead of a
specific kernel class.

## Pocket geometry preprocessing

`ProcessedPocketBatch` composes the existing `PocketBatch` and `BatchMetadata`
rather than introducing another Stage 3 pocket tensor schema. It stores:

- local `PocketBatch` coordinates/features;
- `pocket_mask [B, P]`;
- atom type, element, residue name/ID, and optional chain ID annotations;
- per-sample IDs and metadata;
- translation-only `PocketCoordinateTransform.center [B, 3]`;
- local bounding-box minimum/maximum `[B, 3]` and radius `[B]`.

For valid pocket atoms, preprocessing computes:

    center = sum(mask * global_coordinates) / sum(mask)
    local_coordinates = global_coordinates - center
    global_coordinates = local_coordinates + center

The bounding radius is the maximum Euclidean distance from the saved center.
No PCA alignment or fixed orientation is part of this contract. Because no protein parser dependency is present,
`adapt_pocket_atoms` consumes batched output from an upstream parser while
`read_pocket_file` remains an explicit unimplemented parser boundary.

## Basic geometry field

`build_geometry_field(processed_pocket, config)` consumes the existing
`ProcessedPocketBatch`; it does not parse or recenter a pocket. The returned
object implements `QueryablePocketField.query(query_points, query_mask)`.

Inputs and outputs use the following tensor contract:

| Tensor | Shape | dtype/device | Mask semantics |
| --- | --- | --- | --- |
| single `query_points` | `[Q, 3]` with field batch size 1 | floating; exactly matches pocket coordinates | positions with `query_mask=False` are padding |
| batched `query_points` | `[B, Q, 3]` | floating; exactly matches pocket coordinates | padded points use `query_mask=False` |
| single/batched `query_mask` | `[Q]` or `[B, Q]` | `torch.bool`; query device | `True` means a valid query |
| `FieldQueryResult.values` | `[Q, C]` or `[B, Q, C]` | matches pocket coordinates | invalid-query rows are zero |

The enabled channels have a stable order recorded in
`FieldQueryResult.channels`. `FieldQueryResult.channel_metadata` provides an
ordered `FieldChannel` definition containing each channel's name, description,
and unit. `channel_index(name)` and `select_channel(name)` allow downstream code
to avoid fixed numeric indices. The current implementation supports only:

    occupancy(x) = sum_i 1[d_i <= cutoff] * exp(-d_i^2 / (2 * max(sigma^2, epsilon)))
    nearest_distance(x) = min_i d_i
    d_i = ||x - r_i||_2

Only atoms with `pocket_mask=True` participate. `cutoff` applies to occupancy;
the nearest-distance channel remains the actual nearest valid-atom distance.
All pairwise distances are evaluated with vectorized PyTorch operations. The
parameters and channel switches are defined in
`configs/pocket/geometry_field.yaml`.

With both default channels enabled, channel order is `occupancy`, then
`nearest_distance`. Disabled channels are omitted while preserving this
relative order. Empty query tensors with `Q=0` are valid and produce
`[0, C]` or `[B, 0, C]` values with matching empty masks. The implementation
currently materializes pairwise distances with `O(B * Q * P)` memory; this
calculation is isolated behind an internal batched method so future chunking
does not require a public API change.

Voxelization, Gaussian grids, electrostatic, hydrophobic, hydrogen-bond,
neural, and other pharmacochemical fields are not implemented.

## Abstract contracts

- `TopologyProvider`
- `ScaffoldProvider`
- `DiffusionModel`
- `Constraint`
- `Guidance`
- `Sampler`
- `Evaluator`

Sampling is a separate service rather than a `DiffusionModel.sample()` method. This permits models, constraints, guidance plugins, and sampling policies to evolve independently.

## Optional Pocket Guidance

Pocket Guidance must remain optional. The base software path must run with:

    sampler.sample(model=model, condition=condition, guidance=None)

Future `PocketGuidance` implementations must implement the shared `Guidance` contract and must not become a required dependency for unconditional or scaffold-only generation.
