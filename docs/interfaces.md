# Public interfaces

## Shared data structures

- `RingTopology`: a topology identifier plus schema-free metadata.
- `ScaffoldCondition`: a scaffold identifier linked to the canonical `RingTopology`.
- `DiffusionState`: a schema-free payload, timestep, and metadata container.
- `PocketRepresentation`: a pocket identifier plus undecided feature and metadata mappings.
- `GeneratedMolecule`: an inspectable generated record; Dummy outputs are explicitly non-scientific.

Tensor shapes, graph encoding, vocabularies, pocket resolution, and timestep representation remain intentionally undecided.

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
