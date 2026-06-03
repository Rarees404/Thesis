# server/src/services/

Orchestration layer. Combines the model wrappers in `models/` and the helpers in `utils/` into the two top-level pipelines the API exposes.

## Layout

- `retrieval_service.py` — owns the lifecycle of search and feedback:
  - `search(query, top_k)` — text → SigLIP → FAISS image index → top-K.
  - `process_and_apply_feedback(...)` — collects user signals (SAM masks, full-image labels, text hints), runs Rocchio over segment embeddings, optionally hard-filters with the region index, returns the new batch.
  - `region_index.py` (added with the SAM region precompute) — loads the per-object FAISS index and exposes kNN lookup for hard filtering.
