# server/

FastAPI backend for VisualRef. Loads the SigLIP image/text encoder, the SAM3 segmenter, the Visual Genome region index, and (optionally) Ollama for vision captioning. Serves the `/search`, `/segment`, `/apply_feedback`, `/caption`, `/metrics` endpoints consumed by the Next.js client.

## Layout

- `src/` — Python source. Entry point is `src/retrieval_server_visual.py`.
- `sam3/` — vendored Meta SAM3 source (used as a Python package).
- `checkpoints/` — model checkpoints (created on first run; usually empty in git).
- `tests/` — pytest suite (`test_smoke.py`, `test_changes.py`).
- `requirements.txt` / `requirements.macos.txt` — Python deps. macOS file pins MPS-compatible builds.
- `venv/` — local virtualenv (gitignored). Created by `scripts/setup_models.sh`.

## Running

```
./venv/bin/python -m src.retrieval_server_visual
```

…or via the project root `start.sh`, which boots the server and the Next.js client together.
