# Preprocessing command entry points

Implementations live in `src/gpcr_ringgen/preprocessing`; this directory
contains thin commands and external source-ingestion jobs.

Pipeline order:

1. `source_ingestion/` creates fused-ring SQLite databases per source.
2. `merge_ring_sources.py` merges and standardizes source databases.
3. `build_ring_library.py` extracts the master fused-ring library.
4. `filter_ring_index.py` creates the eligible-molecule index.
5. `analyze_global_scaffolds.py` and `merge_global_scaffolds.py` build the
   global scaffold index.
6. `run_pocket_prefilter.py` performs target-pocket filtering.
7. `build_diffusion_dataset.py` materializes the model dataset.

Set `GPCR_RINGGEN_DATA_ROOT` to the external production-data directory.

