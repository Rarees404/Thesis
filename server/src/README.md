# server/src/

Backend Python source.

## Layout

- `retrieval_server_visual.py` — FastAPI app. All HTTP endpoints live here. Loads models at startup, validates requests, dispatches into `services/`.
- `config.py` — paths and toggles (FAISS index location, SAM checkpoint, IMG_SIZE, etc.).
- `models/` — thin wrappers around external models (SigLIP, SAM3, Ollama, relevance feedback).
- `services/` — orchestration: combines models into the search and feedback pipelines.
- `utils/` — shared helpers: image preprocessing, VG region IoU, FAISS index writing.
- `precompute/` — one-off offline scripts that produce indexes and other artifacts.
- `test_commands.sh` — quick curl examples for hitting the running server.
