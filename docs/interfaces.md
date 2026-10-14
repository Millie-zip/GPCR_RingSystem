# Public interfaces

## Shared data structures

- `RingTopology`: a topology identifier plus schema-free metadata.
- `ScaffoldCondition`: a scaffold identifier linked to the canonical `RingTopology`.
- `DiffusionState`: a schema-free payload, timestep, and metadata container.
- `PocketRepresentation`: a pocket identifier plus undecided feature and metadata mappings.
- `GeneratedMolecule`: an inspectable generated record; Dummy outputs are explicitly non-scientific.

Continuous feature meanings, pocket resolution, and other scientific
representations remain intentionally undecided. Atom and bond categorical
vocabularies belong to each configured diffusion mode, alongside the shared
padded Stage 3 transport shapes. The persisted coordinate-only schema uses the
frozen IDs listed below.

## Stage 3 data contract

| Structure | Core tensor shapes | Mask semantics |
| --- | --- | --- |
| `MolecularGraph` | atoms `[B, N, F_atom]`, edges `[B, E, F_edge]`, edge index `[B, 2, E]`, optional coordinates `[B, N, 3]`, optional categorical atom/bond labels `[B, N]` / `[B, E]` | `atom_mask` and `edge_mask` use `True` for fixed topology entries |
| `DescriptorBatch` | values `[B, D]` | `descriptor_mask=True` means the value is present |
| `PocketBatch` | coordinates `[B, P, 3]`, features `[B, P, F_pocket]` | `pocket_mask=True` means a real pocket point/atom |
| `Stage3DiffusionState` | graph tensors, coordinates, optional noisy atom/bond labels, descriptor `[B, D]`, timestep `[B]` | Carries `BatchMask`; topology masks remain frozen and padded values are zero |

Floating feature/coordinate tensors are expected on the active model device.
`edge_index` and `timestep` use `torch.long`; all masks use `torch.bool`.
`BatchMetadata` carries one unique sample ID and one metadata mapping per sample.

In `joint_chemical` diffusion mode, topology means the unlabeled fixed graph:
`N`, `edge_index`, `atom_mask`, `edge_mask`, edge existence, and ring
connectivity do not change. `atom_types [B, N]`, `bond_types [B, E]` on those
existing edges, and `coordinates [B, N, 3]` are the generated variables.
`chemical_vocabulary` records the stable category ordering; the existing-edge
bond vocabulary deliberately has no `NO_BOND` category.

Route A is a separate variable-side-chain contract layered on these base
types. `ring_atom_mask [B,N]` fixes core membership;
`ring_pair_mask [B,E]` identifies all core-core candidate pairs; and
`ring_edge_mask [B,E]` identifies the fixed-present subset. After a
non-negative outer count is sampled, the graph is padded only to the current
batch maximum and contains each unordered candidate pair once. Bond category
zero is `NO_BOND`. Core edges may change bond type but never become
`NO_BOND`; core non-edges remain `NO_BOND`; every pair touching an outer atom
is generated. `formal_charge_types [B,N]` and `chiralities [B,N]` are generated
only outside the core and copied unchanged inside it. Atom type and coordinates
are generated for all real atoms.

The persisted coordinate-only training schema uses frozen categorical IDs: `C=0`, `N=1`, `O=2`, `S=3`, `P=4`, `F=5`,
`Cl=6`, `Br=7`, `I=8`, `H=9`, `B=10`, `Si=11` for atoms and `SINGLE=0`, `DOUBLE=1`, `TRIPLE=2`,
`AROMATIC=3` for bonds. Padding uses ID 0 and is excluded by masks. There is
no unknown category in that persisted schema: its adapters must reject unsupported chemistry. Joint and Route-A label bounds follow their configured vocabularies. `diffusion_mask` defaults to `atom_mask`; an explicit mask can preserve fixed coordinate anchors in coordinate-only sampling.

The canonical IDs live in `gpcr_ringgen.vocabulary` and are re-exported by
`core.vocab` and `data.vocabulary`. Offline asset builders use the same labels
without importing torch. Coordinate models default to 12 atom embeddings.
To load an existing 9-class coordinate checkpoint, explicitly set
`model.atom_type_vocab_size: 9` and keep its original model dimensions. No
weights are expanded or reinitialized during loading; such a checkpoint still
supports only IDs 0 through 8. A default 12-class model rejects its incompatible
embedding shape. Supporting H/B/Si requires a model trained with the expanded
vocabulary.

The checkpoint must also contain sufficient semantic metadata as described
below; a matching 9-entry tensor shape alone does not establish compatibility.

The standard JSON adapter accepts optional `atom_types` / `bond_types` label
sequences. The inference pipeline supplies the configured joint vocabulary,
or the canonical prefix matching a coordinate model's embedding size. Names
are converted with that order and numeric IDs are checked against its bounds.
With no explicit sequences the adapter uses the canonical 12/4 vocabulary.

## Checkpoint semantics

Generator checkpoints now use `checkpoint_version: 2` and store `semantics`
as plain JSON-compatible data: normalized model/diffusion parameters, ordered
chemical labels, the complete feature manifest, field conditioning and field
parameters. Training also stores `training_provenance` with the JSONL SHA-256,
record limit, split fraction and seed. SHA-256 is streamed once at setup, not
on every epoch or batch. Source files are expected to remain fixed during a run.

Record inference and the standard JSON pipeline compare the semantic contract
before loading tensors. The JSON pipeline requires `data.feature_manifest`
or a top-level inline `feature_manifest` when loading a checkpoint; random
smoke runs remain usable without a manifest. Resume also compares the data
fingerprint and split settings. Inference may use new records; the training
data fingerprint remains provenance rather than an inference input restriction.
Output/input locations, device, dtype, inference seed and trajectory output
can change. Epoch count, output directory and optimizer settings can change
for resume, but its data version, seed, split and scientific contract must match.
Loading an optimizer state retains its saved optimizer settings.

A version-1 generator can be loaded for inference only if its saved config
establishes the model, schedule and vocabularies, and either contains feature
definitions or has a historical manifest byte hash matching the current file.
Its original manifest path need not exist. For legacy coordinate models with
an omitted embedding count, the saved embedding shape establishes the frozen
canonical prefix. An old file without recorded data provenance cannot resume
training under the verified contract. Bare weights or incomplete metadata are
rejected. No switch bypasses mismatched scientific semantics.

Reward training additionally requires a feature manifest and ordered chemical
vocabulary. Its version-2 checkpoint records those definitions, model settings,
the training JSONL fingerprint, teacher version and a logical label-cache
fingerprint. The latter hashes ordered `(SMILES, score, teacher_version)` rows,
so it includes SQLite WAL data and is independent of SQLite file layout or
location. Empty caches and caches with mixed or missing teacher versions are
rejected. The teacher version is the label producer's declared identity; it
does not independently attest the external teacher implementation or weights.

When reward guidance is enabled, set `ra_reward.expected_provenance` to the
`teacher_version`, `labels_sha256` and `records_sha256` values recorded in the
reward training `summary.json`. Loading checks these pinned values and the
generator's full feature and chemical vocabulary definitions. Guidance scale,
start ratio, temperature and evaluation frequency remain runtime choices.
Do not derive these expected pins from the checkpoint being validated. Old
reward files containing only model dimensions lack enough information to
establish compatibility and are rejected. The standalone checkpoint writer
also requires `semantics` and `provenance` arguments. All torch loads retain
`weights_only=True`; matching metadata does not validate model performance.

`Stage2Output` and plain mapping mocks can be adapted with
`stage2_output_to_stage3_input`. The adapter validates data only; it does not run
Stage 2, calculate descriptors, preprocess pockets, construct fields, diffuse,
or sample.

## Stage 3 initial state

`initialize_stage3_state(stage3_input, seed, timestep=0)` converts the existing
padded Stage 3 boundary object into `Stage3DiffusionState`; no alternate graph
or descriptor structure is introduced. Let `d` be `diffusion_mask`, where ring
atoms and padding are false, and let `X_ring` be the pocket-local input:

    Z[b, n, :] ~ N(0, I_3)
    center[b, :] = sum_n d[b, n] * Z[b, n, :] / sum_n d[b, n]
    X_init = where(d, Z - center, X_ring)

The resulting coordinates have shape `[B, N, 3]`, the same floating dtype and
device as scaffold atom features. Fixed ring rows are copied exactly, padding
rows are zero, and every sample must contain at least one atom to diffuse.
Graph features, topology, descriptors, masks, IDs, and ordering are preserved.

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
| coordinate noise | `[B, N, 3]` | matches coordinates | nonzero only where `diffusion_mask=True`; fixed ring/padding are zero |

The base architecture is EGNN-style. For valid atoms `i, j`, scalar messages
depend only on scalar node/bond features and squared distance:

    m_ij = phi_e(h_i, h_j, e_ij, ||x_i - x_j||^2)
    delta_x_i = sum_j unit(x_i - x_j) * phi_x(m_ij)

Both distance and learned coefficients are rotation/translation invariant;
only the relative direction carries a vector representation. Therefore
rotating coordinates rotates the predicted coordinate noise, while translating
all coordinates leaves the prediction unchanged. Coordinate updates and the
final centering are restricted to `diffusion_mask=True`, so the ring is fixed.

Atom and edge features must be invariant scalars; vector-valued input feature
irreps are not supported in this base implementation. Atom and bond type IDs
are embedded separately instead of being mixed into continuous features.
Sinusoidal timestep embedding is also scalar. Descriptor conditioning is an
optional masked linear projection selected by `descriptor_dim`; it is disabled
by default. No pocket or Field data is read by this backbone.

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
