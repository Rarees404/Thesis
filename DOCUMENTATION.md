# VisualReF v2 — Project Documentation

Comprehensive technical reference for the VisualReF v2 interactive image-retrieval
system. For installation and quick-start, see [README.md](README.md). This document
explains **how the system is built and why**, component by component.

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The retrieval & feedback pipeline](#3-the-retrieval--feedback-pipeline)
4. [Running modes: with / without Llama 3.2 Vision](#4-running-modes-with--without-llama-32-vision)
5. [Data and indexes](#5-data-and-indexes)
6. [Configuration reference](#6-configuration-reference)
7. [HTTP API reference](#7-http-api-reference)
8. [The user-study system](#8-the-user-study-system)
9. [Frontend](#9-frontend)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Repository map](#12-repository-map)

---

## 1. Overview

VisualReF is an **interactive image search engine** driven by relevance feedback.
The user types a query, gets a grid of results, then refines the search not by
retyping but by **clicking on the objects they want more (or less) of**. Each click
becomes a precise object mask; the system learns from the masked region and re-ranks.

The central design idea is a **shared embedding space**: SigLIP maps both images and
text into the same 1024-dimensional vector space, so "a dog" and a picture of a dog
land close together. Search is nearest-neighbour lookup in that space (via FAISS), and
feedback is a vector nudge (via Rocchio) toward what the user liked and away from what
they rejected.

VisualReF v2 extends the original RecSys '25 demo with: pixel-precise SAM 3
segmentation (replacing whole-image / crop-box feedback), optional asynchronous VLM
captioning, a Visual Genome region-phrase signal, adaptive multimodal fusion,
full-image binary labels, an offline region index, and session-scoped hard filtering.

**Corpus:** Visual Genome, ~108k images.

---

## 2. Architecture

Two-tier client/server. The browser calls the FastAPI backend **directly** (not
through the Next.js dev proxy) so long operations — segmentation, feedback, VLM
captioning — are not killed by a proxy timeout.

```
Browser (Next.js + React + Zustand) :3000
   │  fetch JSON, direct call via NEXT_PUBLIC_SERVER_URL (long timeout)
   ▼
FastAPI backend :8001  (server/src/retrieval_server_visual.py)
   ├── SigLIP encoder         text & image → 1024-d ℓ2-normalized vectors
   ├── FAISS image index      flat inner-product NN search over ~108k vectors
   ├── SAM 3 segmenter        point-click → binary object mask
   ├── VG region phrases      human-written descriptions, matched by mask IoU
   ├── Ollama / Llama 3.2-V    optional region captioning (async)
   └── Region index           optional, 259k region crops, for hard filtering
   │  reads image files by path
   ▼
Repo on disk: data/ (images + VG JSON), faiss/ (indexes + image_paths.txt)
```

**Backend** — FastAPI on Uvicorn, port 8001. Entry point
`server/src/retrieval_server_visual.py`. Core retrieval lives in
`server/src/services/retrieval_service.py`.

**Frontend** — Next.js + React + Zustand, port 3000, in `client-next/`.

**Models** — SigLIP `google/siglip-large-patch16-256`; SAM 3 (local package in
`server/sam3/`, installed editable); Llama 3.2 Vision via Ollama (optional).

**Devices** — CUDA, Apple MPS, or CPU. SigLIP and SAM inference run with autograd
disabled, both to avoid state leaking between rounds and to sidestep autograd ops
unsupported on MPS.

---

## 3. The retrieval & feedback pipeline

### 3.1 Offline indexing

Every corpus image is resized to 256×256, encoded by SigLIP, ℓ2-normalized, and added
to a flat inner-product FAISS index. Because vectors are unit-norm, inner product
equals cosine similarity, so the index ranks directly with no score transform. Image
filesystem paths are written, in the **same row order** as the vectors, to
`image_paths.txt`. Built by `scripts/build_index.sh`.

### 3.2 Search (runtime)

`POST /search` → SigLIP encodes the query text → FAISS returns the top-k nearest image
vectors → paths resolved, images base64-encoded (cached), returned to the browser. A
fresh search resets per-session feedback state (accumulated query, blacklist,
boostlist).

### 3.3 Segmentation (SAM 3)

`POST /segment` turns a click into a mask. Per-click pipeline:

1. **Coordinate scaling** — browser CSS-pixel coords → full-resolution image pixels.
2. **Bounding-box prompt** — a tight box around the points, padded 20% for foreground
   (Relevant) clicks and 13% for background-only (Irrelevant) clicks, localising SAM.
3. **Mask prediction** — multi-output (3 candidates) on the first click of an image;
   single-output refinement on follow-up clicks.
4. **Mask selection** — highest-scoring candidate; any mask covering >28% of the image
   is rejected as over-segmented and the next-best is used (guards the single-click
   "selected the whole scene" failure mode).
5. **Encoding** — mask downsampled to preview resolution and sent as RLE
   (pycocotools-format run-length encoding; **not** the COCO dataset).

**Logit cache** — SAM's previous logits are reused as a soft prior on the next click
for the same image (better incremental refinement). FIFO, capped at 50 entries
(`_LOGIT_CACHE_MAX`); a memory bound, not a tuned value. Invalidated when the user adds
a negative click, since the old foreground prior no longer reflects intent.

### 3.4 Signal extraction (feedback)

`POST /apply_feedback` builds positive and negative feedback vectors from up to three
aligned signals per annotated region:

- **Image signal** — for each mask, non-masked pixels are set to neutral grey (RGB 128),
  the bounding box is cropped, resized to 256×256, and SigLIP-encoded. Per side, image
  embeddings are averaged.
- **Text signal** — three sources, aggregated and deduplicated, capped at 20 entries
  per side, then mean-pooled:
  - **VG phrases** — human-written Visual Genome region descriptions matched to the mask
    by IoU at segment time (instant, no VLM).
  - **VLM captions** — Llama 3.2 Vision descriptions of the crop (optional; async cache).
  - **User hints** — free-form positive/negative text typed in the feedback panel.
- **Full-image fallback** — a whole-image Relevant/Irrelevant label (no click) encodes
  the entire image as one virtual segment. Strict priority cascade per image:
  **SAM mask ≻ bounding box ≻ full-image label** (a finer annotation overrides a coarser
  one).

**VG priority rule** — when VG phrases exist and the user typed no hint, VLM captioning
is skipped: the phrases already give grounded text at zero latency.

### 3.5 Ollama captioning & the async pipeline

A single Llama 3.2 Vision call takes 20–40 s on consumer hardware, so it never runs on
the blocking feedback path. Instead, captioning fires **the moment a mask is produced**
(inside `/segment`), in a background task, before the user has even decided the region
is relevant. The caption is stored in an in-memory cache keyed by
`(image_path, label, query, hint)`, capped at 200 entries (`_CAPTION_CACHE_MAX`, FIFO).
By the time the user clicks Apply Feedback, the caption is usually already cached and is
retrieved in ~1 ms. If it isn't ready, the system falls back to the VG phrases returned
synchronously. The caption prompt sends both the isolated crop and the full scene with
a highlighted box; temperature 0.1, max 60 tokens (deterministic, faithful captions are
preferred since the text is consumed as a feedback signal, not read by a human).

### 3.6 Adaptive multimodal fusion

Image and text signals are fused by weighted addition, `d = w_img·d_img + w_txt·d_txt`,
with weights that adapt to signal richness:

| Condition | (w_img, w_txt) |
|---|---|
| VLM captions present | (0.4, 0.6) — text carries more, it's more specific |
| otherwise | (0.5, 0.5) |

If only one side is present, only that side is used.

### 3.7 Rocchio query update

The fused positive and negative vectors update the accumulated session query:

```
q(t+1) = ℓ2-normalize( α·q(t) + β·d⁺ − γ·d⁻ )
```

with **α = 0.8, β = 0.5, γ = 0.15**. High α preserves the original intent; β > γ
because users give more positive than negative feedback and over-penalising risks
over-correction. Output is normalized so it stays comparable under FAISS inner product.

- **User-text pass** — typed hints additionally drive a *second*, dedicated Rocchio
  pass so they are never diluted by auto-generated phrases. When text is the only signal
  (no clicks, no auto-text) it uses stronger weights (β=0.6, γ=0.3) so one hint visibly
  moves the query in a single step; otherwise softer (β=0.4, γ=0.2).
- **Session accumulation** — q(t+1) is stored per session UUID; later rounds refine it
  rather than the original query, so refinement is cumulative.
- **Drift mitigation** — an optional "anchor to original query" toggle averages the
  current query with the initial query before the update, `½(q(t) + q(0))`, to limit
  drift across rounds.
- **Null signal** — with no feedback the update reduces to re-searching the accumulated
  query.

### 3.8 Hard filtering (optional, session-scoped)

A deterministic complement to the soft Rocchio update: rejected content is *guaranteed*
not to reappear within the session.

- **Phase 1 (lookup).** Each segment embedding queries the offline region index (kNN).
  Parent images of regions similar to a **negative** segment go on a session
  **blacklist**; parents of regions similar to a **positive** segment go on a
  **boostlist**. Thresholds: negatives k=50 / cosine ≥ 0.70 (image side), 0.20 (text
  side); positives k=20 / 0.75 (image), 0.25 (text). Image-side bars are higher because
  SigLIP image-image cosine for related concepts sits ~0.5–0.9, while text-image cosine
  is lower.
- **Phase 2 (filtering).** FAISS over-fetches `max(6k, k + |blacklist|)` candidates,
  blacklisted paths are removed, boostlisted paths get a small additive score bonus
  (+0.05 — enough to break near-ties, too small to override a genuinely better match),
  and the top-k is returned.
- Both lists persist across all feedback rounds and are cleared only on a new search.

---

## 4. Running modes: with / without Llama 3.2 Vision

VLM auto-captioning is powerful but expensive (20–40 s/region, significant RAM/GPU).
The system is designed to run **fully without it**. The single switch is
`OLLAMA_ENABLED` in `server/.env`.

| Mode | `OLLAMA_ENABLED` | Ollama running? | Text signal source |
|---|---|---|---|
| **Lightweight** (default) | `false` | not needed | Visual Genome region phrases (instant) |
| **Full** | `true` | yes, model pulled | AI-generated region captions (+ VG phrases) |
| Safe fallback | `true` | no | Auto-falls back to VG phrases |

The fallback is automatic: at startup the server calls `check_ollama` (is Ollama
reachable *and* is the model pulled?). Only then is captioning enabled; otherwise it
silently uses VG phrases. A missing Ollama never crashes or hangs the server.

**Lightweight:** set `OLLAMA_ENABLED=false`, skip all Ollama setup. Start normally.

**Full:**
1. Install Ollama (`brew install ollama`, or ollama.com/download).
2. `ollama serve` and `ollama pull llama3.2-vision` (~7.8 GB, one time).
3. In `server/.env`: `OLLAMA_ENABLED=true`, `OLLAMA_URL=http://127.0.0.1:11434`,
   `OLLAMA_MODEL=llama3.2-vision`.
4. Start; verify `curl -s localhost:8001/health` shows `"ollama_available": true`.

Flip the flag and restart to switch modes anytime.

---

## 5. Data and indexes

```text
data/visual_genome/
├── VG_100K/                  # images
├── VG_100K_2/                # more images
├── region_descriptions.json  # human-written region phrases (the VG text signal)
├── objects.json              # scene-graph objects + WordNet synsets (study ground truth)
├── image_data.json           # image metadata
├── attributes.json           # (full VG dump)
└── relationships.json        # (full VG dump)

faiss/visual_genome/google/siglip-large-patch16-256/
├── image_index.faiss         # ~108k image vectors (searchable)
├── image_paths.txt           # one path per row, same order as the index
├── region_index.faiss        # optional — hard filtering (259k region crops)
└── region_meta.jsonl         # optional — hard-filter metadata (path, box, phrase)
```

**Build the image index:** `bash scripts/build_index.sh` — encodes every image (1–2+ h
on MPS/CPU, much faster on CUDA). One-time; re-run only if the image set changes. If
`region_descriptions.json` is present it is loaded at startup for the VG phrase signal.

**Build the optional region index (hard filtering):**
```bash
cd server
./venv/bin/python -m src.precompute.build_region_index \
    --image-paths ../faiss/visual_genome/google/siglip-large-patch16-256/image_paths.txt \
    --vg-dir ../data/visual_genome \
    --output-dir ../faiss/visual_genome/google/siglip-large-patch16-256
# --limit 100 for a quick smoke test
```
Resumable: re-run the same command to continue after an interruption. The server
auto-detects `region_index.faiss` + `region_meta.jsonl` at startup; absent → hard
filtering is silently disabled.

> If images move or are re-downloaded, **rebuild the index** so `image_paths.txt` stays
> aligned with the vectors.

---

## 6. Configuration reference

**`server/.env`** (paths relative to `server/`):

| Key | Meaning |
|---|---|
| `CONFIG_PATH` | Corpus/model YAML (`../configs/demo/vg_siglip.yaml`) |
| `INDEX_PATH` | The `.faiss` image index to search |
| `LOGS_PATH` | Where retrieval and study logs are written |
| `OLLAMA_ENABLED` | `true` = full mode (AI captions), `false` = lightweight |
| `OLLAMA_URL` / `OLLAMA_MODEL` | Ollama endpoint and model name |
| `SAM_BACKEND` | Segmentation backend (`sam3`) |
| `CORS_ORIGINS` | Allowed frontend origins (comma-separated) |

**`configs/demo/vg_siglip.yaml`** — corpus + model: `IMAGE_CORPUS_PATH`, `INDEX_PATH`,
`VLM_MODEL_NAME` (`google/siglip-large-patch16-256`), `IMG_SIZE` (256), `PATCH_SIZE`
(16), `TOP_K`.

**`client-next/.env.local`** — `NEXT_PUBLIC_SERVER_URL`: the backend URL the browser
calls directly (default `http://127.0.0.1:8001`).

**Key tuning constants** (in `server/src/services/retrieval_service.py`): Rocchio
`alpha/beta/gamma`, fusion `img_w/txt_w`, hard-filter thresholds (`HARD_FILTER_*`),
`SEARCH_OVERFETCH_FACTOR`. All are commented inline.

---

## 7. HTTP API reference

Base URL `http://localhost:8001`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/search` | Text query → top-k image paths + base64 previews + scores |
| POST | `/segment` | SAM 3 mask for a click (RLE); adds VG phrases; fires async caption |
| POST | `/apply_feedback` | Rocchio update from clicks, hints, phrases, captions → new ranking + hard-filter telemetry |
| POST | `/caption` | On-demand Ollama caption for one base64 region |
| GET | `/health` | `status`, `gpu_available`, `ollama_available`, `vg_index_loaded`, `region_index_loaded`, `region_index_size` |
| GET | `/sam_status` | Whether SAM loaded and which backend |
| GET | `/ollama_status` | Ollama reachability, model, url |
| GET | `/caption_cache_status` | Debug: pre-computed / in-flight caption counts |
| GET | `/caption_lookup` | Debug: fetch a cached caption by key |
| GET | `/metrics` | Index size, GPU backend, timings |
| GET | `/study/tasks` | Build/serve the study tasks (reads `eval/data/queries.json`) |
| POST | `/study/log` | Append a study event to the participant's JSONL log |
| POST | `/study/form` | Save a participant's post-task questionnaire |

**Safety** — every client-supplied image path is validated against the canonical
corpus path set; out-of-corpus paths are rejected (HTTP 403), preventing path traversal.

**Concurrency** — SAM keeps shared mutable state, so all SAM inference is serialised
through one lock. Ollama runs on a worker thread to keep the event loop responsive.

---

## 8. The user-study system

A guided target-finding flow used to evaluate the system, at
`http://localhost:3000/study`. **Requires `server/src/eval/data/queries.json`** — if it
is missing, `/study/tasks` returns `503 {"detail":"Study query set not built"}`.

**Task definitions** — `queries.json` holds `{"category": [...], "compositional": []}`.
Each task: `qid` (a WordNet synset, e.g. `cat_zebra.n.01`), `text` (the displayed query,
e.g. `zebra`), and `relevant` (a set of VG image IDs containing that synset, used as
ground truth and to pick the target). The three study tasks are zebra (1492 relevant,
target 134), tower (1750, target 9), and eyelid/"lid" (2003, target 6). The relevant
sets are derived from `objects.json` by matching each image's object synsets to the
task synset.

**Task serving** (`_build_study_tasks`) — loads the pool, and for each task picks the
**lowest relevant image ID that resolves to a corpus path** as the target, renders an
"archivist" cover story (`_STUDY_STORIES`) around the query text, base64-encodes the
target preview, and returns it. Deterministic, so every participant sees the same tasks.

**Logging** — each participant gets `logs/study/{id}.jsonl`: `session_start`,
`task_start`, `search_submitted`, `segment_click`, `feedback_applied`, `task_finished`,
each with result rankings, scores, and latency. The questionnaire is saved to
`logs/study/{id}.form.json`.

**Analysis** — `server/src/eval/analyze_study.py` joins the JSONL logs and form files
with the `queries.json` ground truth, computes objective metrics (success, rounds,
duration, precision@{5,10} at stop) and subjective Likert aggregates, and writes
`study_summary.json` + `study_tables.tex`:
```bash
cd server
./venv/bin/python -m src.eval.analyze_study \
    --logs-dir ../logs/study --queries src/eval/data/queries.json \
    --out-dir src/eval/study_results
```
Supporting modules: `metrics.py` (precision@k, nDCG@k, MRR), `questions_meta.py`
(Likert item → research-question mapping).

> `queries.json` is generated from VG data and is not tracked by git. To regenerate it,
> see the reconstruction approach in `analyze_study.py`'s ground-truth loader and the
> task synsets above.

---

## 9. Frontend

Next.js + React + Zustand (`client-next/`). Global state (results, SAM annotations,
round, history, hint text) lives in `src/lib/store.ts`; the server contract types are in
`src/lib/types.ts`; API calls in `src/lib/api.ts`.

Key components:
- **`image-card.tsx`** — renders each result with an HTML-canvas mask overlay (40%
  opacity); captures clicks relative to the rendered image rect and converts to natural
  pixel space before sending. A ResizeObserver keeps the overlay aligned as the browser
  scales the image.
- **`feedback-panel.tsx`** — positive/negative hint fields, the "anchor to original
  query" toggle, asynchronously-arriving VLM caption suggestions with a one-click "Use",
  and the Apply Feedback button; shows hard-filter telemetry from the previous round.
- **`search-bar.tsx`**, **`image-gallery.tsx`**, **`header.tsx`**,
  **`server-dashboard.tsx`** — query input, result grid, status.
- **`study/`** — the study flow: `task-selector`, `task-briefing`, `task-banner`,
  `questionnaire`, `study-complete`.

The study page hardcodes results-per-search via `STUDY_TOP_K` (overrides the store's
default of 5).

---

## 10. Testing

```bash
cd server
venv/bin/python -m pytest -q              # all tests
venv/bin/python -m pytest tests/test_changes.py -q          # core pipeline (30 tests)
venv/bin/python -m pytest tests/test_study_endpoints.py -q  # study endpoints (4 tests)
```
Always use `server/venv/bin/python`, not the system Python — it has the dependencies.
Tests cover RLE round-trips, Rocchio (positive/negative/none independently), SAM
coordinate clamping, the region index, endpoint validation (empty query, top_k range,
path traversal), logit-cache eviction, and the study task/log/form endpoints.

Frontend type/build checks (from `client-next/`): `npx tsc --noEmit`, `npx next build`.

---

## 11. Deployment

For a cloud-GPU split (remote uvicorn, local Next.js) see `deploy/DEPLOY.md`.
`scripts/cloud_bootstrap.sh` provisions a fresh GPU VM end-to-end (clone, deps, model
weights, index build, server), invoked by `deploy/setup-cloud.sh`. Point the local
frontend's `NEXT_PUBLIC_SERVER_URL` at the remote backend; ensure `CORS_ORIGINS` on the
server allows the frontend origin.

---

## 12. Repository map

| Path | Role |
|---|---|
| `start.sh` | One command to launch backend + frontend with a health checklist |
| `server/src/retrieval_server_visual.py` | FastAPI app — all endpoints, study system |
| `server/src/services/retrieval_service.py` | Search + Rocchio feedback + fusion + hard filtering |
| `server/src/services/region_index.py` | Region kNN index wrapper (hard filtering) |
| `server/src/models/sam.py` | SAM 3 segmenter, RLE encode/decode, logit cache |
| `server/src/models/siglip.py` | SigLIP wrapper (text/image embeddings) |
| `server/src/models/ollama_vision.py` | Ollama captioning client |
| `server/src/models/relevance_feedback.py` | Rocchio algorithm + mask/box extraction |
| `server/src/utils/vg_regions.py` | VG region-phrase lookup by IoU |
| `server/src/precompute/build_region_index.py` | Builds the optional hard-filter index |
| `server/src/eval/` | Study analysis: `analyze_study.py`, `metrics.py`, `questions_meta.py`, `data/queries.json` |
| `client-next/` | Next.js + React + Zustand frontend |
| `configs/demo/vg_siglip.yaml` | Corpus/model config |
| `scripts/` | Setup, download, indexing, cloud bootstrap |
| `data/` | Image corpus + VG metadata (downloaded, not in git) |
| `faiss/` | Built indexes (not in git) |
| `logs/study/` | Per-participant study data |
| `deploy/` | Cloud-GPU deployment notes |
