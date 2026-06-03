# scripts/

Shell helpers for setup, indexing, and cloud deployment. All scripts assume they're invoked from the repo root and that `server/venv/` exists (run `setup_models.sh` first if it doesn't).

## Layout

- `setup_models.sh` — creates `server/venv/`, installs Python deps, downloads SAM3 weights.
- `download_visual_genome.sh` — fetches the VG image corpus + JSON metadata into `data/visual_genome/`.
- `build_index.sh` — builds the Visual Genome per-image FAISS index (the main search index).
- `build_all_indexes.sh` — convenience wrapper: checks the VG data is present, then calls `build_index.sh`.
- `cloud_bootstrap.sh` — provisions a fresh cloud VM end-to-end (clone, deps, weights, indexes, server). Used by `deploy/setup-cloud.sh`.

The new SAM region precompute is a Python module, not a shell script — invoke it directly:
`./server/venv/bin/python -m src.precompute.build_region_index`.
