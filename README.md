# gpcr-ringgen

`gpcr-ringgen` is a research codebase scaffold for future GPCR ring-system molecular generation work.

> Current status: Stage 3 data contracts, reproducible three-dimensional
> initialization, reversible pocket geometry preprocessing, and a basic
> queryable geometry field and pocket-independent equivariant denoiser
> backbone are available. The Stage 3 forward DDPM process and masked
> epsilon-prediction loss are also available for training integration, together
> with optional Field-conditioned equivariant denoising and Stage 3 DDPM
> reverse sampling. A configuration-driven independent Stage 3 inference
> Pipeline is available for fixed scaffold and descriptor inputs.

The real-data Stage 3 path is now available: an immutable source SQLite
database can be converted to a QC-audited 3D asset DB, materialized into
versioned JSONL records, trained with the existing DDPM denoiser, and sampled
again from a saved checkpoint. Training is pocket-disabled and does not apply
ring/geometry weak constraints; those masks are retained for audit. Sampling
guidance, docking, chemical evaluation, and molecule export remain separate
future work.

## Real-data training path

Create the independent assets and JSONL records with the GPCR interpreter:

    python scripts/preprocess.py \
      --source-db <source.db> \
      --output-db <stage3_assets.db> \
      --training-jsonl <stage3_training.jsonl> \
      --manifest configs/data/stage3_feature_manifest.yaml \
      --progress <stage3_progress.jsonl>

Then prepare a YAML from the model and diffusion examples, set
`data.path`, `data.feature_manifest`, `training.output_dir`, and
`field_conditioning.mode: none`, and run:

    python scripts/train.py --config <training.yaml>

The resulting `best.ckpt` can be used with `scripts/infer.py` for
pocket-disabled inference from a record in the same JSONL library. The
checkpoint stores and verifies the feature-manifest hash.

## Environment requirements

- Recommended Python version: Python 3.10
- Minimum supported Python version: Python 3.10
- Environment manager: Conda
- Conda environment name: `GPCR`

Python 3.10 is the default development and test environment. Python 3.11+ is not artificially blocked, but users of other versions must confirm dependency compatibility in their own environment.

All packages required during project development must be recorded in `requirements.txt`. Do not install undeclared project dependencies without updating that file.

## First-time setup

Linux/macOS:

    git clone <repository>
    cd gpcr-ringgen
    conda create -n GPCR python=3.10
    conda activate GPCR
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install --no-deps -e .

## Subsequent usage

This checkout already uses the configured `GPCR` environment. Do not recreate it during project development. Activate the existing environment:

    conda activate GPCR
    python scripts/validate_environment.py
    python -m pytest

## Common commands

    python scripts/validate_environment.py
    python -m pytest
    python -m pytest tests/unit/test_stage3_types.py
    python -m pytest tests/unit/test_stage3_initialization.py
    python -m pytest tests/unit/test_equivariant_denoiser.py
    python -m pytest tests/unit/test_diffusion_process.py
    python -m pytest tests/unit/test_field_conditioned_denoiser.py
    python -m pytest tests/unit/test_reverse_diffusion_sampling.py
    python -m pytest tests/integration/test_stage3_inference_pipeline.py
    python -m pytest tests/unit/test_pocket_preprocessing.py
    python -m pytest tests/unit/test_geometry_field.py
    python scripts/sample.py --config configs/experiment/dummy.yaml
    python scripts/run_stage3_inference.py --config configs/pipeline/stage3_smoke.yaml

The Dummy Pipeline writes an inspectable placeholder result to `outputs/dummy_sample.json`. It does not generate a chemically meaningful molecule.

## Stage 3 three-dimensional initialization

`initialize_stage3_state(stage3_input, seed=..., timestep=0)` converts the
existing padded Stage 3 input into `Stage3DiffusionState`. For every sample it
draws valid ligand atom coordinates from isotropic `N(0, I_3)`, subtracts the
valid-atom center, and sets padded coordinates to zero. The result shape is
`[B, N, 3]` with the scaffold atom-feature dtype/device.

Pass the existing top-level experiment `seed` value to this function. A local
PyTorch generator makes repeated calls with the same input and seed
reproducible without changing global RNG state. `timestep` initializes the
state label; the forward diffusion step replaces it with the sampled training
timestep.

## Stage 3 forward diffusion and loss

`DDPMNoiseSchedule`, `sample_timesteps(...)`, and `q_sample(...)` implement the
training-side DDPM process
`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`. The current
prediction type is `epsilon`; `training_target(...)` returns the exact centered
noise used to construct `x_t`. Timesteps are zero-based, and `t=0` is the
weakest configured noising step rather than a separate clean-state sentinel.

Both the input coordinates and sampled Gaussian noise are centered per sample
using valid ligand atoms only. Padded coordinate rows are zero. The
`NoisePredictionLoss` computes MSE only where `atom_mask=True`, with `mean`
reduction dividing by three times the number of valid atoms.

The example `configs/diffusion/ddpm.yaml` exposes:

- `num_timesteps`: number of discrete forward steps;
- `beta_schedule`: currently `linear`;
- `beta_start` / `beta_end`: endpoints of the linear beta schedule;
- `prediction_type`: currently `epsilon`;
- `loss_reduction`: `mean` or `sum`.

Pass local generators created by `make_torch_generator(seed, device)` to
`sample_timesteps` and `q_sample` for reproducible sampling without mutating the
global PyTorch random state. Reverse diffusion and sampling are intentionally
implemented separately from the training-side API.

## Stage 3 reverse diffusion sampling

`DDPMStage3Sampler(schedule).sample(stage3_input, denoiser, seed=...)` starts
from the existing centered Gaussian Stage 3 initialization and applies every
reverse timestep from `T-1` through `0`. It uses the same epsilon-prediction and
linear-beta schedule configured by `configs/diffusion/ddpm.yaml`; DDIM or other
sampler formulas are not mixed into this path.

For a model prediction `epsilon_theta`, one reverse step uses:

    mu_t = 1/sqrt(alpha_t) *
           (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
    beta_tilde_t = beta_t * (1-alpha_bar_(t-1)) / (1-alpha_bar_t)
    x_(t-1) = mu_t + sqrt(beta_tilde_t) * z       when t > 0
    x_0     = mu_0                                 when t = 0

Coordinates, predicted noise, and posterior noise are kept in the valid-atom
zero-center-of-mass subspace. Padded atoms remain zero at every step. The
sampler temporarily calls `model.eval()`, runs under `torch.no_grad()`, and
restores the model's previous training/evaluation state afterward.

The output contains generated local coordinates `[B, N, 3]`, the unchanged
graph/topology tensors, descriptor-bearing final state, and per-sample
metadata. Pass `return_trajectory=True` to retain initialization plus all
reverse states; it defaults to `False` to avoid `O(T*B*N)` memory. Pass a
`QueryablePocketField` for Field-conditioned inference or use conditioning
`mode: none` with no field for ablation.

`restore_global_coordinates(output, pocket_transform)` applies the saved
pocket center to valid ligand atoms and returns `[B, N, 3]` global coordinates
without writing files. Sampling output remains local by default. Randomly
initialized models can validate the structure of this process, but their
coordinates must not be interpreted as chemically or pharmacologically valid.

## Independent Stage 3 inference Pipeline

The independent entrypoint runs the current Stage 3 stack without a Stage 1 or
Stage 2 model:

    conda activate GPCR
    python scripts/run_stage3_inference.py --config configs/pipeline/stage3_smoke.yaml

Its orchestration is deliberately thin:

    standard JSON input
    -> preprocess_pocket
    -> build_geometry_field
    -> stage2_output_to_stage3_input
    -> DDPMStage3Sampler.sample
    -> restore_global_coordinates
    -> structured JSON output

`examples/inputs/stage3_smoke.json` is the standard single-sample input
example. It contains global pocket coordinates and annotations, a fixed/mock
scaffold represented by atom features plus bond topology, a descriptor vector,
masks, a sample ID, and metadata. The loader only converts this JSON to the
existing project contracts; it does not calculate descriptors or call a Stage
1/Stage 2 model.

`configs/pipeline/stage3_smoke.yaml` keeps all runtime settings together:

- `pipeline.input_path`, `checkpoint_path`, `output_path`, and
  `allow_random_weights`;
- `device.type` and `device.dtype`;
- `seed` and `sampling.return_trajectory`;
- model dimensions and Field-conditioning mode;
- Field parameters/channels and DDPM schedule parameters.

When `checkpoint_path` is null, `allow_random_weights: true` is required. This
explicit mode is only for structural smoke testing and the output is marked
`model_weights: random_untrained` and `scientific_status:
pipeline_smoke_test_only`. A checkpoint must contain the configured
`FieldConditionedEquivariantDenoiser` state dict, either directly or under
`model_state_dict` / `state_dict`.

The JSON output contains the sample ID, padded local and restored global
coordinates, atom mask, fixed atom/bond topology, descriptor and mask,
metadata, pocket transform center, Field channel order, seed, checkpoint
status, and a complete configuration snapshot/reference. Trajectory is omitted
by default. The project has no reliable SDF/MOL reconstruction writer, so this
Pipeline does not invent one.

## Base equivariant denoiser

`EquivariantDenoiserBackbone(config)(state, timestep)` returns a
`DenoiserOutput` containing coordinate noise shaped `[B, N, 3]`. The backbone
uses atom scalars, bond scalars, squared inter-atom distances, relative
directions, and a sinusoidal scalar timestep embedding. Absolute xyz values are
never passed to an MLP. Predictions for padded atoms are zero and valid-atom
predictions are centered per sample.

Hyperparameters are illustrated in
`configs/model/equivariant_denoiser.yaml`. Atom and edge feature dimensions
must match the runtime `Stage3DiffusionState`; their scientific vocabularies
remain external to the backbone. Optional descriptor conditioning is enabled
only when `descriptor_dim` is set; the default configuration disables it. The
backbone has no Pocket Field, diffusion scheduler, noising, loss, or sampling
dependency.

## Field-conditioned equivariant denoiser

`FieldConditionedEquivariantDenoiser` wraps the unchanged scientific operations
of the base backbone and exposes two modes through
`configs/model/field_conditioning.yaml`:

- `mode: none` bypasses Field Query and exactly follows the base denoiser path;
- `mode: additive` queries `F(X_t)` and injects encoded scalar field features
  into the initial node hidden state.

The additive data flow is:

    X_t + atom_mask -> QueryablePocketField.query -> [B, N, C_field]
    field features -> learned scalar encoder -> [B, N, hidden_dim]
    atom + timestep + descriptor + field hidden states -> EGNN -> epsilon prediction

`field_conditioning.channels` defines the required channel names and order. It
must match `PocketField.channels`; the learned encoder adapts `C_field` to the
backbone hidden dimension. Field values at padded ligand atoms are zero and do
not enter message passing.

The existing descriptor path remains a separate batch-aligned global scalar
condition. `descriptor_mask` is applied before its linear projection. This
module does not recompute or normalize descriptors; any descriptor
normalization must be completed upstream and recorded with the descriptor
pipeline.

The current geometry channels depend on pocket-query relative distances, so
they are invariant when pocket and ligand undergo the same rigid transform.
They are injected only as scalar node conditions; coordinate updates continue
to use relative ligand displacement vectors. This preserves joint rotation
equivariance and translation consistency without passing absolute xyz values
to an MLP.

## Basic pocket geometry field

The example configuration is `configs/pocket/geometry_field.yaml`:

- `sigma`: Gaussian width in the same units as the pocket coordinates;
- `cutoff`: maximum atom-query distance that contributes to occupancy;
- `epsilon`: positive numerical lower bound for the Gaussian denominator;
- `channels.occupancy`: enables the Gaussian occupancy/density channel;
- `channels.nearest_distance`: enables nearest-pocket-atom distance.

The field is built from the local-coordinate `ProcessedPocketBatch` produced by
pocket preprocessing. Query points must use that same local coordinate frame,
dtype, and device. This implementation is continuous and query-based; it does
not construct a voxel grid.

The public `QueryablePocketField.query(query_points, query_mask=None)` API
accepts `[Q, 3]` for a single pocket or padded `[B, Q, 3]` coordinates. It
returns a `FieldQueryResult` with features shaped `[Q, C]` or `[B, Q, C]` and a
matching boolean query mask. `result.channels`, `result.channel_metadata`, and
`result.select_channel(name)` expose channel ordering without relying on
hard-coded indices. Empty `Q=0` queries return correctly shaped empty features.
