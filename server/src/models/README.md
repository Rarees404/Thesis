# server/src/models/

Thin wrappers around external models. Each module exposes a load function and the primitives the rest of the system uses; no orchestration logic here.

## Layout

- `siglip.py` — SigLIP image / text encoder. Returns 768-dim embeddings.
- `sam.py` — SAM3 wrapper. Point-prompted segmentation; RLE encoding helpers (`mask_to_rle`, `rle_to_mask`).
- `ollama_vision.py` — Llama 3.2-Vision client used for on-demand region captioning.
- `relevance_feedback.py` — Rocchio-style query update from segment embeddings + text hints. The merge of SAM regions, bounding boxes, and full-image labels happens here.
- `vlm_wrapper.py` — uniform interface so the retrieval service can call either text or vision-language models without branching.
- `configs.py` — per-model config dataclasses loaded from `configs/demo/*.yaml`.
