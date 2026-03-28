# VisualReF — Visual Relevance Feedback for Interactive Image Retrieval

Prototype from the RecSys ’25 demo paper: users search with natural language, click on image regions to mark what they want more or less of (SAM 3 segmentation), and optionally use **Ollama + Llama 3.2 Vision** to auto-caption those regions. Feedback updates the query with **Rocchio**-style refinement over **SigLIP** embeddings and a **FAISS** index.

**Stack today:** FastAPI backend (`server/`) · Next.js 16 frontend (`client-next/`) · SigLIP retrieval · SAM 3 · Ollama `llama3.2-vision`.

---

## Citation

```bibtex
@inproceedings{10.1145/3705328.3759341,
  author    = {Khaertdinov, Bulat and Popa, Mirela and Tintarev, Nava},
  title     = {{VisualReF}: Interactive Image Search Prototype with Visual Relevance Feedback},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3705328.3759341},
  booktitle = {Proceedings of the Nineteenth {ACM} Conference on Recommender Systems},
  series    = {RecSys '25}
}
```

Example figures: `./assets/`.

---

## Repository layout

| Path | Role |
|------|------|
| `start.sh` | **Main entry:** starts FastAPI (port 8001) + Next.js (port 3000), health checks |
| `server/` | Python backend: SigLIP, FAISS, SAM 3, Ollama client, `/search`, `/segment`, `/apply_feedback`, `/caption`, `/health` |
| `server/.env` | **Required:** dataset config path, FAISS index path, Ollama URL/model, `SAM_BACKEND=sam3` |
| `server/sam3/` | SAM 3 package (clone + `pip install -e .`; often gitignored — run `scripts/setup_models.sh`) |
| `server/venv/` | Python virtualenv (you create this; not committed) |
| `client-next/` | Next.js UI; proxies `/api/*` to the backend via `NEXT_PUBLIC_SERVER_URL` |
| `configs/demo/*.yaml` | Retrieval settings: model id, `IMG_SIZE`, corpus hints |
| `data/` | **Your image files** (not in git): e.g. `data/coco/val2014/*.jpg` |
| `faiss/` | **Built artifacts:** per-dataset folders with `image_index.faiss` + `image_paths.txt` |
| `scripts/setup_models.sh` | Installs SAM 3 from `server/sam3`, checks Ollama, pulls `llama3.2-vision` |
| `scripts/build_index.sh` | Builds FAISS index from a folder of images (SigLIP or CLIP) |
| `scripts/build_all_indexes.sh` | Runs `build_index.sh` for every dataset present under `data/` + combined |
| `deploy/` | Cloud GPU notes (`DEPLOY.md`), `setup-cloud.sh`, optional `Dockerfile` |

Legacy Gradio client, Docker Compose, and old launch scripts were removed; this README describes the supported path only.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10–3.12 recommended (3.11 used in development) |
| **Node.js** | 18+ (for Next.js) |
| **Disk** | Plan for datasets + HF cache + Ollama model (SAM 3 and SigLIP download from Hugging Face; `llama3.2-vision` is multi‑GB via Ollama) |
| **GPU** | Optional but strongly recommended: CUDA (Linux/Windows) or Apple **MPS** (macOS). CPU works but is slow for indexing and SAM 3 |
| **Ollama** | Optional for vision captions: [ollama.com](https://ollama.com) — install and `ollama pull llama3.2-vision` |
| **Hugging Face** | SAM 3 may be gated: log in with `huggingface-cli login` if downloads fail |

---

## End-to-end: clone → run

### 1. Clone

```bash
git clone <your-fork-or-upstream-url> visualref
cd visualref
```

### 2. Image data layout

The indexer walks a directory tree and collects `jpg` / `png` (case variants). Put corpora under **`data/`** at the repo root (create the folder if needed).

**COCO (example — val2014 only):**

```text
data/
└── coco/
    └── val2014/
        ├── COCO_val2014_000000000042.jpg
        └── ...
```

**Visual Genome:**

```text
data/
└── visual_genome/
    └── …/*.jpg
```

**Retail (example):**

```text
data/
└── retail/
    └── …/*.jpg
```

Paths written into `image_paths.txt` during indexing must still exist when you run the server (the API opens files by path).

### 3. Python environment (backend)

```bash
cd server
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. SAM 3 (segmentation)

From the **repository root**:

```bash
bash scripts/setup_models.sh
```

This clones/installs `server/sam3` if needed (`pip install -e .`) and pulls **Llama 3.2 Vision** via Ollama. If `server/sam3` is missing and the script clones it, ensure you have **git** and (for gated models) **Hugging Face** access.

Manual equivalent:

```bash
cd server/sam3
pip install -e .
cd ../..
```

### 5. Build the FAISS index

Still from **repository root**, with `server/venv` created:

```bash
bash scripts/build_index.sh coco          # default: SigLIP
# bash scripts/build_index.sh coco clip  # optional: CLIP + matching yaml in configs/demo
```

Outputs (SigLIP example):

```text
faiss/coco/google/siglip-large-patch16-256/
├── image_index.faiss
└── image_paths.txt
```

Each line in `image_paths.txt` is an **absolute** path to an image file.

To build everything you have under `data/` plus a combined index:

```bash
bash scripts/build_all_indexes.sh
```

### 6. Configure `server/.env`

Paths below are **relative to the `server/` directory** (as in the default template).

```env
CONFIG_PATH=../configs/demo/coco_siglip.yaml
INDEX_PATH=../faiss/coco/google/siglip-large-patch16-256/image_index.faiss
LOGS_PATH=../logs

OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2-vision
OLLAMA_ENABLED=true

SAM_BACKEND=sam3
```

- **`CONFIG_PATH`** — YAML that sets `VLM_MODEL_FAMILY`, `VLM_MODEL_NAME`, `IMG_SIZE`, etc. Must match the **same** SigLIP/CLIP model used to build the index.
- **`INDEX_PATH`** — must point at `image_index.faiss` next to its `image_paths.txt` in the same folder.

Switching datasets = change both `CONFIG_PATH` and `INDEX_PATH` to the matching pair (see `configs/demo/`).

### 7. Demo YAML (`configs/demo/*.yaml`)

Important keys (retrieval server):

- `VLM_MODEL_FAMILY` / `VLM_MODEL_NAME` — must match index builder.
- `IMG_SIZE` — square resize for search thumbnails; click coordinates from the UI are in this space.
- `IMAGE_CORPUS_PATH` / `INDEX_PATH` inside YAML — legacy hints; the running server uses **`INDEX_PATH` from `.env`** for loading FAISS.

### 8. Frontend environment

```bash
cd client-next
echo 'NEXT_PUBLIC_SERVER_URL=http://127.0.0.1:8001' > .env.local
npm install
cd ..
```

For a remote API, set `NEXT_PUBLIC_SERVER_URL` to that host (scheme + port).

### 9. Ollama (optional but recommended)

```bash
ollama serve    # terminal 1, or run as a service
ollama pull llama3.2-vision
```

If Ollama is down, the backend still runs; feedback uses image embeddings only until vision is available.

### 10. Start everything

From the **repository root**:

```bash
chmod +x start.sh scripts/*.sh   # once
./start.sh
```

- Backend: `http://localhost:8001` (e.g. `GET /health`, `GET /sam_status`, `GET /ollama_status`)
- Frontend: `http://localhost:3000`

First backend start can take several minutes while SigLIP and SAM 3 load and HF caches fill.

---

## Operations cheat sheet

| Task | Command / location |
|------|---------------------|
| Start app | `./start.sh` |
| Logs | `.logs/server.log`, `.logs/client.log` |
| Rebuild index after new images | `bash scripts/build_index.sh <dataset>` |
| Change corpus | New index + update `CONFIG_PATH` / `INDEX_PATH` in `server/.env` |
| Ports busy | `start.sh` kills listeners on 8001 and 3000; or set `SERVER_PORT` / `CLIENT_PORT` |

---

## HTTP API (overview)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/search` | Text query → top‑k image paths + base64 previews |
| POST | `/segment` | SAM 3 mask from points + optional preview-space coords |
| POST | `/apply_feedback` | Rocchio update from SAM regions + text + optional Ollama captions |
| POST | `/caption` | Ollama caption for one base64 image region |
| GET | `/health`, `/sam_status`, `/ollama_status`, `/metrics` | Status |

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `FAISS index file not found` | `INDEX_PATH` in `server/.env`; file exists relative to `server/` |
| `Image path does not exist` | Paths in `image_paths.txt` still valid (same machine or synced data) |
| SAM 3 import error | `cd server/sam3 && pip install -e .` |
| SAM 3 / HF 401 | `huggingface-cli login`; accept model terms on Hugging Face |
| Ollama “Vision OFF” | `ollama serve` + `ollama pull llama3.2-vision` |
| Next.js `ETIMEDOUT` / API errors | `NEXT_PUBLIC_SERVER_URL` must match where uvicorn runs |
| Next.js lockfile warning | A `package-lock.json` outside this repo can confuse Turbopack; remove stray lockfiles or set `turbopack.root` in `client-next/next.config.ts` per Next.js docs |

---

## Cloud GPU

See **[deploy/DEPLOY.md](deploy/DEPLOY.md)** for rsync, `deploy/setup-cloud.sh`, and running uvicorn on `0.0.0.0:8001` with the Next.js app on your laptop.

Optional container build from **repo root** (adjust `COPY` paths if you customize):

```bash
docker build -f deploy/Dockerfile -t visualref-api .
```

---

## Config directories (research / prompts)

- `configs/demo/` — retrieval backbones (SigLIP / CLIP) and dataset defaults.
- `configs/captioning/` — used by older LLaVA-based server paths; the current Next.js + visual server stack uses **Ollama** for captions, configured in `server/.env`.

---

## License

See [LICENSE](LICENSE).
