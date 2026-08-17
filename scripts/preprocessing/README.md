# Preprocessing command entry points

## Export the ring-fixed Stage 3 database

### Step 1: build the 3D pose database and pocket JSON

`build_stage3_assets.py` defaults to the reported server paths and writes both
files beside `5ht2c_3d_diffusion_library_v1.db`:

- `8dpf_docked_poses.db`;
- `8dpf_13A_pocket.json`.

It extracts 8DPF receptor heavy atoms within 13 Å of the mean lorcaserin
heavy-atom position. For each source molecule it generates ETKDGv3/UFF starting
conformers, invokes an external docking command, checks that docking preserved
the original heavy-atom order, and applies the existing native-calibrated
cavity/anchor/clash/span PASS test. RDKit conformers are initialization only;
they are never saved directly as training `x0`.

The docking engine differs between servers, so copy
`configs/preprocessing/docking_command.example.json`, replace the executable
and arguments with the installed engine's command, and retain the four path
placeholders. The engine must accept SDF input and write SDF output without
reordering heavy atoms.

Start with 100 molecules:

```bash
python scripts/preprocessing/build_stage3_assets.py \
  --docking-config configs/preprocessing/docking_command.json \
  --limit 100
```

The builder writes `stage3_build_progress.json` next to the output database and
refreshes it every 15 seconds, including while RDKit conformer generation or the
docking subprocess is busy. Results are committed after every molecule by
default. Watch the build from another terminal with:

```bash
python scripts/preprocessing/watch_stage3_progress.py \
  /path/to/output/stage3_build_progress.json --interval 15
```

The live record reports the current molecule, current phase, time spent in that
phase, completed/PASS/FAIL counts, and the last error. `--heartbeat-seconds`
and `--commit-every` can tune heartbeat and transaction frequency.

For a fast CPU interface test, reduce only the test-time conformer effort with
`--max-conformers 2 --embed-max-iterations 500 --uff-max-iters 100`. Production
defaults remain 10 conformers, 2000 embedding iterations, and 250 UFF steps.
Conformer generation runs in an isolated subprocess, so the heartbeat remains
live even inside RDKit; `--conformer-timeout-seconds` bounds a pathological
molecule and records it as a row-level failure instead of blocking the build.

If interrupted, continue with the same files using `--resume`. Do not start the
full library until the docking command and the first 100 PASS/FAIL records have
been reviewed. The old random rigid-placement prefilter is not treated as
docking and is not used as an `x0` label.

### Step 2: export the model-ready database

`export_stage3_dataset.py` reads the existing
`5ht2c_3d_diffusion_library_v1.db` without modifying it and writes a separate
normalized Stage 3 database. It joins:

- `selected_molecules.atom_facts_json` and `bond_facts_json`;
- `selected_ring_systems.ring_atom_indices_json`;
- a separate `docked_poses` table containing atom-index-aligned `x0 [N,3]`;
- one reusable 13 Å 8DPF pocket JSON.

The coordinate database must contain:

```sql
CREATE TABLE docked_poses (
    source_part TEXT,
    molecule_uid TEXT,
    pose_id TEXT,
    coordinates_json TEXT,
    status TEXT,
    cavity_frac REAL,
    anchor_frac REAL,
    span_excess_A REAL,
    collision_frac REAL
);
```

Only `status='PASS'` poses are exported. `coordinates_json` must use the exact
atom order stored by `atom_facts_json`. The exporter does not use an undocked
RDKit conformer as a pocket-conditioned target.

First run a small server-side export:

```bash
python scripts/preprocessing/export_stage3_dataset.py \
  --source-db /share/home/weiliao02/lyx/zinc_project/5ht2c_3d_diffusion_library_v1/5ht2c_3d_diffusion_library_v1.db \
  --poses-db /path/to/8dpf_docked_poses.db \
  --pocket-json /path/to/8dpf_13A_pocket.json \
  --output-db /path/to/5ht2c_stage3_ready_test.db \
  --limit 100
```

After inspecting those 100 records, remove `--limit` and choose a new output
path. Existing output files are never overwritten. Continuous atom/edge
features are explicitly zero placeholders until their scientific schema is
frozen; categorical atom and bond IDs remain active, and descriptor values are
disabled with an all-false mask.

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

