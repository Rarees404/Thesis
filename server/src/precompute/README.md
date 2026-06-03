# server/src/precompute/

One-off offline scripts that produce indexes and other artifacts the runtime server depends on. Each script is idempotent and resumable where possible — you can re-run after a crash and it picks up where it left off.

Outputs land under the top-level `faiss/` tree (or wherever `config.py` points), so the runtime server doesn't need this folder at all to boot — only to *rebuild* its inputs.

## Layout

- `build_region_index.py` — for every Visual Genome image in `image_paths.txt`, crops up to `--max-regions-per-image` VG-annotated boxes, encodes each crop with SigLIP, and writes:
  - `region_index.faiss` — `IndexFlatIP` of L2-normalized embeddings
  - `region_meta.jsonl` — one line per row: `{image_path, region_idx, phrase, bbox, source}`
  - `region_index.progress.txt` — last image index processed (resume cursor)

  Used at runtime by `services/region_index.py` for hard-filter blacklist / boostlist lookup during relevance feedback.

## Running

```
cd server
./venv/bin/python -m src.precompute.build_region_index \
    --image-paths ../faiss/visual_genome/google/siglip-large-patch16-256/image_paths.txt \
    --vg-dir      ../data/visual_genome \
    --output-dir  ../faiss/visual_genome/google/siglip-large-patch16-256 \
    --device auto
```

Add `--limit 100` for a quick smoke test.

### Resuming after a crash or partial run

The script reads `region_index.progress.txt` (just one integer — the last processed corpus index) and skips ahead. To resume, run the **exact same command** with no extra flags. Don't delete `region_meta.jsonl`; new rows are appended.

To start over from scratch, delete:

```
rm faiss/.../region_index.faiss faiss/.../region_meta.jsonl faiss/.../region_index.progress.txt
```

### Current state of the index in this checkout

If `progress.txt < len(image_paths.txt) - 1`, the index covers only a prefix of the corpus. The runtime server still loads it and performs hard filtering, but blacklist/boostlist will only ever fire on images in the covered prefix — for images past the cursor, the filter is a silent no-op.

Verify with:
```
cd server
venv/bin/python -c "import faiss; print(faiss.read_index('../faiss/visual_genome/google/siglip-large-patch16-256/region_index.faiss').ntotal)"
cat ../faiss/visual_genome/google/siglip-large-patch16-256/region_index.progress.txt
wc -l ../faiss/visual_genome/google/siglip-large-patch16-256/image_paths.txt
```

The runtime test `test_region_index_artifact_consistency_if_present` checks `ntotal == len(meta)` so a corrupt index fails CI without needing to be loaded into FAISS by the server.
