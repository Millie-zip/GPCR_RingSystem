# Preprocessing architecture and migration map

The former numbered root directories have been integrated into two layers:

- `src/gpcr_ringgen/preprocessing`: importable implementations;
- `scripts/preprocessing`: executable commands and source ingestion.

| Former path | New path |
| --- | --- |
| `01_four_download/ChEMBL_Fused_Ring.py` | `scripts/preprocessing/source_ingestion/chembl.py` |
| `01_four_download/coconut_fused_ring.py` | `scripts/preprocessing/source_ingestion/coconut.py` |
| `01_four_download/pubchem_fused_ring.py` | `scripts/preprocessing/source_ingestion/pubchem.py` |
| `01_four_download/zinc_fused_ring.py` | `scripts/preprocessing/source_ingestion/zinc.py` |
| `01_four_download/zinc_fused_ring_server.py` | `scripts/preprocessing/source_ingestion/zinc_server.py` |
| `02_fused-ring master library/four_fused_ringdata_server.py` | `src/gpcr_ringgen/preprocessing/ring_library/merge_sources.py` |
| `02_fused-ring master library/build_fused_ring_master_library_server.py` | `src/gpcr_ringgen/preprocessing/ring_library/build_master.py` |
| `02_fused-ring master library/filter_fused_ring_parts_13_35_3plus_v2.py` | `src/gpcr_ringgen/preprocessing/ring_library/filter_parts.py` |
| `02_fused-ring master library/filter_fused_ring_parts_index_v3.py` | `src/gpcr_ringgen/preprocessing/ring_library/filter_index.py` |
| `03_global scaffold/global_scaffold_analysis_strict_v4.py` | `src/gpcr_ringgen/preprocessing/global_scaffold/analyze.py` |
| `03_global scaffold/global_scaffold_merge_v4_4_no_weights.py` | `src/gpcr_ringgen/preprocessing/global_scaffold/merge.py` |
| `04_pocket prefilter/gpcr_prefilter_final_parallel_v1.py` | `src/gpcr_ringgen/preprocessing/pocket_prefilter/pipeline.py` |
| `05_3d_diffusion_dataset/build_5ht2c_3d_diffusion_library_v1_COMPLETE.py` | `src/gpcr_ringgen/preprocessing/diffusion_dataset/build.py` |

The offline pocket prefilter remains separate from the model-time
`gpcr_ringgen.pocket` package. SQLite dataset materialization remains separate
from model-time loading and collation in `gpcr_ringgen.data`.

