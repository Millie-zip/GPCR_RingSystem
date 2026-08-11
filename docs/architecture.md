# Architecture

This repository is an executable software skeleton, not a scientific implementation.

## Layers

1. **Infrastructure Layer** — `core`, configuration, `data`, `utils`, and `tests` provide stable types, contracts, and basic runtime services.
2. **Domain Layer** — `topology`, `scaffold`, `pocket`, and `constraints` describe extension points without choosing scientific representations.
3. **Algorithm Layer** — `models`, `sampling`, and `training` depend on shared contracts. `scripts` remain thin entrypoints above these layers.

The intended dependency direction is:

    core -> domain modules -> algorithm modules -> scripts

Domain modules must not import concrete diffusion implementations. Pocket behavior reaches sampling through the shared `Guidance` interface; it must not control model internals. Shared types live only in `core/types.py` to avoid incompatible parallel schemas and circular imports.

## Module ownership

- Developer 1: Topology / Scaffold / Diversity
- Developer 2: Models / Constraints / Sampling / Training
- Developer 3: Pocket / Guidance / Target Evaluation

The developers collaborate through `core.types` and `core.interfaces`. Changes to these shared boundaries should be reviewed by all module owners before scientific implementations are added.
