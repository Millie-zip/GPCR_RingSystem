# Data directories

- `raw/`: immutable source data acquired for future experiments.
- `interim/`: temporary products created during future preprocessing.
- `processed/`: future model-ready inputs.

The current project skeleton contains no real training data, and the final
training-data schema remains a research decision. Large datasets must not be
committed to Git. Keep only lightweight documentation and directory markers in
the repository.

The lightweight standard input for independent Stage 3 inference is documented
by `examples/inputs/stage3_smoke.json`. It is a single-sample JSON record with:

- global pocket coordinates, features, atom/residue annotations, and mask;
- a fixed scaffold with atom features, edge index/features, and masks;
- an upstream-provided descriptor vector and mask;
- a sample ID and JSON-compatible metadata.

This is an inference interchange example, not a real training dataset or a
replacement for a future protein structure parser.
