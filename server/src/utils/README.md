# server/src/utils/

Helpers used across models and services.

## Layout

- `image_utils.py` — PIL ↔ numpy ↔ base64 conversions, resize/crop helpers, mask-to-region cropping.
- `utils.py` — small shared primitives (timing, hashing, path validation).
- `vg_regions.py` — Visual Genome region phrase index. Mask-IoU lookup of region phrases per image.
- `write_faiss_index.py` — used by the offline indexers to persist a FAISS index plus its row → metadata sidecar.
