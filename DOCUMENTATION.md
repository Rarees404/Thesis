# VisualReF v2 — Technical Documentation

A complete technical reference for the VisualReF v2 interactive image-retrieval
system. The [README](README.md) covers installation and running; this document
explains **how the system works internally** — the retrieval logic, how a click
becomes a learning signal, how text feedback becomes a learning signal, how the two
are combined, and how each round produces a better result set.

## Contents

1. [Overview](#1-overview)
2. [System architecture](#2-system-architecture)
3. [Foundation: the shared embedding space](#3-foundation-the-shared-embedding-space)
4. [Stage 1 — Text search](#4-stage-1--text-search)
5. [Stage 2 — Region segmentation (the visual signal)](#5-stage-2--region-segmentation-the-visual-signal)
6. [Stage 3 — Text feedback (the semantic signal)](#6-stage-3--text-feedback-the-semantic-signal)
7. [Stage 4 — Combining the signals](#7-stage-4--combining-the-signals)
8. [Stage 5 — Producing the next batch](#8-stage-5--producing-the-next-batch)
9. [End-to-end worked example](#9-end-to-end-worked-example)
10. [Running modes: with / without Llama 3.2 Vision](#10-running-modes-with--without-llama-32-vision)
11. [Data and indexes](#11-data-and-indexes)
12. [Configuration reference](#12-configuration-reference)
13. [HTTP API reference](#13-http-api-reference)
14. [The user-study system](#14-the-user-study-system)
15. [Frontend](#15-frontend)
16. [Testing](#16-testing)
17. [Deployment](#17-deployment)
18. [Repository map](#18-repository-map)

---

## 1. Overview

VisualReF is an **interactive image search engine** driven by relevance feedback.
The user types a query, receives a grid of results, and then refines the search not by
retyping but by **clicking on the objects they want more (or less) of**. Each click is
turned into a precise object mask; the system learns from the masked region — and from
any optional typed hints — and re-ranks the corpus. Over a few rounds the search
converges on the user's intent, even when that intent was hard to express up front.

The whole system rests on one idea: **images and text live in the same vector space.**
Once a picture and a phrase are both points in that space, "similarity" is just a dot
product, "search" is nearest-neighbour lookup, and "feedback" is moving the query point
toward what the user liked and away from what they rejected. Everything below is an
elaboration of that single mechanism.

A feedback round is a five-stage pipeline:

```
Stage 1  Text search          query text  → ranked images
Stage 2  Segmentation         a click     → object mask → visual signal
Stage 3  Text feedback        phrases/caption/hint → semantic signal
Stage 4  Combine              visual + semantic, positive + negative → new query
Stage 5  Next batch           re-search + deterministic hard filter → new grid
```

Sections 4–8 walk through these stages in order.

---

## 2. System architecture

Two-tier client/server. The browser talks to the FastAPI backend **directly** (not
through the Next.js dev proxy), so the long operations — segmentation, feedback, and
VLM captioning — are never cut off by a proxy timeout.

```
Browser  (Next.js + React + Zustand)  :3000
   │  fetch JSON — direct call via NEXT_PUBLIC_SERVER_URL, long timeout
   ▼
FastAPI backend  :8001   (server/src/retrieval_server_visual.py)
   ├── SigLIP encoder        text & image → 1024-d, ℓ2-normalized vectors
   ├── FAISS image index     flat inner-product nearest-neighbour search (~108k vectors)
   ├── SAM 3 segmenter        point-click → binary object mask
   ├── VG region phrases      human-written descriptions, matched to a mask by IoU
   ├── Ollama / Llama 3.2-V    optional region captioning (runs asynchronously)
   └── Region index           optional, ~259k region crops, powers hard filtering
   │  reads image files by path
   ▼
Disk:  data/ (images + VG JSON)   faiss/ (indexes + image_paths.txt)
```

| Layer | Technology | Location |
|---|---|---|
| Backend | FastAPI on Uvicorn, port 8001 | `server/src/retrieval_server_visual.py` |
| Retrieval core | SigLIP + FAISS + Rocchio + fusion + hard filter | `server/src/services/retrieval_service.py` |
| Segmentation | SAM 3 (editable local package) | `server/src/models/sam.py`, `server/sam3/` |
| Captioning | Llama 3.2 Vision via Ollama (optional) | `server/src/models/ollama_vision.py` |
| Frontend | Next.js + React + Zustand, port 3000 | `client-next/` |

All neural inference runs with autograd disabled — both to keep state from leaking
between rounds and to sidestep autograd operations unsupported on Apple MPS.

---

## 3. Foundation: the shared embedding space

The backbone is **SigLIP** (`google/siglip-large-patch16-256`), a contrastively-trained
image–text model. It provides two encoders:

- `f_txt(s)` maps a text string `s` to a vector in ℝ¹⁰²⁴
- `f_img(x)` maps an image `x` to a vector in the **same** ℝ¹⁰²⁴

Both outputs are **ℓ2-normalized** to unit length (`F.normalize(..., p=2, dim=-1)` in the
SigLIP wrapper). Two consequences follow, and the entire system depends on them:

1. **Cosine similarity = dot product.** For unit vectors `a` and `b`,
   `cos(a, b) = a · b`. So comparing any two items — image↔image, text↔text, or
   image↔text — is a single dot product, and a larger value means "more similar."
2. **One space for everything.** Because SigLIP was trained to pull matching
   image–caption pairs together, a photo of a zebra and the words "a zebra" land near
   each other. This is what lets a *typed* hint and a *clicked* region both push the
   query in a meaningful direction.

A query, an image, a clicked crop, a VG phrase, and a VLM caption are therefore all the
same kind of object: a unit vector in ℝ¹⁰²⁴. Retrieval and feedback are pure vector
arithmetic over these points.

---

## 4. Stage 1 — Text search

When the user submits a query (`POST /search`):

1. The query text is encoded: `q = f_txt(query)` — a unit vector.
2. FAISS searches the image index for the `k` corpus vectors `e_i` with the largest
   inner product `q · e_i` (i.e. highest cosine similarity).
3. The corresponding image paths are looked up in `image_paths.txt` (same row order as
   the index), the images are base64-encoded (with an LRU cache), and returned with
   their scores.

The index is **flat** (exhaustive) and **inner-product** based; because every stored
vector is unit-norm, the inner product is exactly cosine similarity, so no score
transformation is needed. A fresh search also **resets the per-session feedback state**
(the accumulated query, the blacklist, and the boostlist), so each new query starts
clean.

The index itself is built offline (`scripts/build_index.sh`): every corpus image is
resized to 256×256, encoded by `f_img`, normalized, and appended to the index, with its
filesystem path written to `image_paths.txt` in the same order.

---

## 5. Stage 2 — Region segmentation (the visual signal)

The point of segmentation is **isolation**. If the user marks a whole image of "a zebra
on a plain near a jeep" as relevant, a whole-image embedding also reinforces *plain* and
*jeep*. By contrast, a mask around just the zebra yields an embedding of the zebra
alone. Segmentation is how VisualReF turns a vague "I like this picture" into a precise
"I like *this object*."

### 5.1 From click to prompt (`server/src/models/sam.py`)

A click arrives as an image coordinate plus a label (`1` = Relevant/foreground, `0` =
Irrelevant/background). Before calling SAM, the system derives a **bounding-box prompt**
that localises the model (`_compute_prompt_box`):

- If there is at least one **positive** click, the box is drawn around the positive
  points; padding is **20%** of the image dimension (min 24 px).
- If the clicks are **only negative**, the box is drawn around all clicked points with a
  **tighter 13%** padding.

The asymmetry is deliberate: positive boxes are padded generously so the object is not
clipped at its edge, while negative-only boxes are kept tight so the prompt stays on the
rejected region and does not leak adjacent foreground into the mask. Without a box, a
single click often makes SAM segment the whole scene.

### 5.2 Mask prediction and selection

SAM 3's interactive predictor is called with the point coordinates, labels, and the
prompt box:

- **First click on an image** → *multimask* mode: SAM returns **three** candidate masks
  plus confidence scores. `_select_best_mask` discards any candidate covering **> 28%**
  of the image area (the classic "selected the whole background" failure), then picks the
  highest-scoring survivor. Masks are logit floats, thresholded at 0 (sigmoid > 0.5).
- **Follow-up clicks on the same image** → *single-mask* mode: SAM returns one refined
  mask conditioned on the previous output, and the highest-scoring mask is taken
  directly.

The 28% area cap protects the *purpose* of segmentation: a mask spanning the whole scene
would make the resulting crop nearly identical to the full image, defeating the point of
region-level feedback.

### 5.3 Incremental refinement (the logit cache)

SAM can reuse the raw logits of a previous prediction as a soft prior on the next click,
which sharpens masks when the user refines a selection click-by-click. VisualReF keeps a
per-image FIFO cache of the most recent logits (`_logit_cache`, capped at 50 entries,
`_LOGIT_CACHE_MAX`) and feeds them back as `mask_input` on the next click for the same
image. The cap is a memory bound, not a tuned value. **A negative click invalidates the
cache entry**, because the old foreground prior no longer reflects the new intent.

### 5.4 From mask to a visual feedback vector

A binary mask is not yet a learning signal — it has to become a vector. The conversion
(`ImageBasedVLMRelevanceFeedback._extract_sam_segments` in
`server/src/models/relevance_feedback.py`):

1. **Grey-out.** Every pixel *outside* the mask is replaced with neutral grey (RGB 128):
   `masked = np.where(mask, image, full(image, 128))`. This removes background context
   while keeping the object's true colours.
2. **Crop.** The image is cropped to the mask's bounding box (`rows.min():rows.max()`,
   `cols.min():cols.max()`), discarding empty margins.
3. **Resize & encode.** The crop is resized to 256×256 and passed through `f_img` to
   produce a unit **visual embedding** of that one object.

If the user instead marks a **whole image** (a label with no click), the whole image is
encoded as a single "virtual segment" — the same pipeline, minus the mask. A strict
priority cascade decides, per image, which annotation wins: **SAM mask ≻ bounding box ≻
full-image label**. A finer annotation always overrides a coarser one for the same image.

The result of Stage 2 is a set of per-object visual embeddings, split into a positive
set (Relevant) and a negative set (Irrelevant).

---

## 6. Stage 3 — Text feedback (the semantic signal)

A clean crop still carries only *visual* information. Text adds *semantics* — and
because SigLIP aligns the two spaces, a textual description of the selected object is a
complementary, often more discriminative, signal than its pixels alone. "A golden
retriever near a stream" pins down a concept that the raw crop embedding leaves fuzzy.

### 6.1 Three sources of text

For each annotated region, up to three text strings may contribute, per polarity
(positive / negative):

1. **Visual Genome region phrases** — human-written descriptions from the VG dataset,
   matched to the mask by **bounding-box IoU** at segment time
   (`server/src/utils/vg_regions.py`). Instant, no model inference. This is the default
   text signal in lightweight mode.
2. **VLM captions** — a Llama 3.2 Vision description of the crop, generated
   asynchronously (see §10). The prompt sends both the isolated crop and the full scene
   with the region highlighted; temperature 0.1, max 60 tokens, so captions are short,
   deterministic, and faithful (the text is consumed by the search, not read by a human).
3. **User hints** — free-form positive/negative strings the user types in the feedback
   panel ("on a beach", "no people").

**Priority rule:** if VG phrases exist for a region and the user typed no hint, the VLM
call is skipped — the phrases already provide grounded text at zero latency. This is what
makes lightweight mode free of any VLM cost in the common case.

### 6.2 From text to a vector

Each text source is handled symmetrically to the visual side. The strings for a polarity
are collected, **deduplicated**, and **capped at 20** entries (so a handful of salient
phrases dominate rather than a long tail of near-duplicates, and so per-round encoding
stays bounded). Each surviving string `s` is encoded with `f_txt(s)` into a unit vector.

### 6.3 Why two signals instead of one

The visual and semantic signals fail in different ways, which is exactly why combining
them helps. A crop embedding captures appearance the user may not be able to name
(texture, exact colour, pose) but is blurry about category; a text embedding nails the
category but discards appearance. Fusing them (next stage) keeps the strengths of both.

The result of Stage 3 is a set of per-string text embeddings, again split into positive
and negative.

---

## 7. Stage 4 — Combining the signals

This is the heart of the system: turning the four embedding sets (visual±, text±) into a
single refined query. It happens in three steps — pool within a modality, fuse the
modalities, then apply the Rocchio update.

### 7.1 Pool within each modality

Within each polarity, the per-item vectors are averaged (mean-pooled) into one vector
per modality:

```
d⁺_img = mean of positive visual embeddings        d⁺_txt = mean of positive text embeddings
d⁻_img = mean of negative visual embeddings        d⁻_txt = mean of negative text embeddings
```

Averaging unit vectors yields the centroid of the relevant (or irrelevant) examples —
the "direction" they collectively point in.

### 7.2 Fuse visual + text (adaptive weights)

For each polarity, the image and text centroids are blended by weighted addition:

```
d±  =  w_img · d±_img  +  w_txt · d±_txt
```

with weights that adapt to how rich the text signal is:

| Condition | (w_img, w_txt) | Rationale |
|---|---|---|
| VLM captions present | (0.4, 0.6) | A specific generated caption is highly discriminative — lean on text. |
| otherwise | (0.5, 0.5) | VG phrases are useful but generic — weight the modalities evenly. |

If one modality is absent (e.g. a typed hint with no click, or a click in lightweight
mode with no phrase), only the present modality is used. This produces a single
**positive feedback vector `d⁺`** and a single **negative feedback vector `d⁻`**.

### 7.3 Positive + negative → the Rocchio update

The fused vectors update the query via the classic Rocchio rule
(`server/src/models/relevance_feedback.py`):

```
q(t+1)  =  ℓ2-normalize(  α · q(t)  +  β · d⁺  −  γ · d⁻  )
```

with **α = 0.8, β = 0.5, γ = 0.15**:

- **α (query inertia)** keeps most of the current query, so refinement is gradual and the
  original intent is not thrown away.
- **β (positive pull)** moves the query *toward* the relevant centroid.
- **γ (negative push)** moves it *away* from the irrelevant centroid. It is deliberately
  smaller than β — users give more positive than negative feedback, and over-penalising
  risks over-correcting past the target.

The final ℓ2-normalization returns the query to the unit sphere so it stays directly
comparable to the corpus vectors under FAISS inner-product search. If neither `d⁺` nor
`d⁻` exists, the update degenerates to `normalize(α·q)` — effectively re-searching the
current query.

### 7.4 Session accumulation and drift control

The updated query is stored **per session** (keyed by a session UUID) and is the `q(t)`
that the *next* round refines — so improvement is **cumulative**, not reset each round.

To prevent the query from drifting away from the user's original goal over many rounds,
an optional **"anchor to original query"** toggle averages the current query with the
initial one before applying Rocchio:

```
q(t)_anchored = ½ · ( q(t) + q(0) )      # q(0) = embedding of the first text query
```

The Rocchio update is then applied to `q(t)_anchored`. The user controls this with a
checkbox in the feedback panel.

### 7.5 The dedicated user-text pass

Typed hints are important enough that the system gives them a **second, separate Rocchio
pass** after the main one, so they are never diluted by the (often numerous)
auto-generated VG/VLM phrases when the mean is taken. The weights of this pass adapt to
context:

- **Text is the only signal** (no clicks, no auto-text): β = 0.6, γ = 0.3 — strong, so a
  single typed hint visibly moves the query in one round.
- **Other signals also present:** β = 0.4, γ = 0.2 — softer, so the hint refines rather
  than overrides the visual feedback.

---

## 8. Stage 5 — Producing the next batch

### 8.1 Re-search

The refined query `q(t+1)` is sent back through FAISS exactly as in Stage 1, producing a
new ranked grid. Because the query has moved toward the relevant centroid and away from
the irrelevant one, the new top-k is biased toward what the user marked relevant.

### 8.2 Hard filtering — a deterministic guarantee (optional)

Rocchio is a *soft* nudge: a rejected image can still creep back if the query drifts. For
cases where the user wants a hard "never show me this again," VisualReF adds a
deterministic filter, active when the optional **region index** is built
(`server/src/services/region_index.py`).

**Phase 1 — lookup.** Each per-object feedback embedding is used as a kNN query against
the region index (~259k VG region crops). For a **negative** segment, the parent images
of its nearest regions are added to a session **blacklist**; for a **positive** segment,
the parents of its nearest regions are added to a session **boostlist**. Thresholds:

| | k | image-side cosine | text-side cosine |
|---|---|---|---|
| Negative → blacklist | 50 | ≥ 0.70 | ≥ 0.20 |
| Positive → boostlist | 20 | ≥ 0.75 | ≥ 0.25 |

Image-side bars are higher because SigLIP image–image cosine for related concepts sits
around 0.5–0.9, whereas text–image cosine is systematically lower.

**Phase 2 — filter and re-rank.** FAISS over-fetches `max(6·k, k + |blacklist|)`
candidates (headroom so a full top-k survives), then:

- candidates whose path is on the **blacklist** are dropped outright;
- candidates on the **boostlist** get a small additive bonus of **+0.05** to their score
  — enough to win near-ties but too small to override a genuinely better match;
- the list is re-sorted and the top-k returned.

Both lists **persist across all rounds** of a session and are cleared only on a new
search. The response also carries telemetry (blacklist/boostlist sizes, how many images
were dropped or boosted this round), which the UI surfaces.

---

## 9. End-to-end worked example

Goal: find a specific photo of a **zebra on a dry plain**.

1. **Search** "zebra". `q(0) = f_txt("zebra")`. FAISS returns zebras — but also a horse
   in a stable and a zebra-print rug.
2. **Segment.** The user clicks the zebra's body in a good result. SAM returns a mask;
   the background (sky, fence) is greyed out, the crop is encoded → a positive visual
   embedding of *this zebra*. VG matches the region to the phrase "a zebra standing", → a
   positive text embedding. The user also clicks the rug (Irrelevant) → a negative visual
   embedding.
3. **Hint (optional).** The user types positive hint "dry grass".
4. **Combine.** Positives pool to `d⁺_img` (zebra crop) and `d⁺_txt` (mean of "a zebra
   standing" + "dry grass"); the rug gives `d⁻`. With no VLM, fusion uses (0.5, 0.5):
   `d⁺ = 0.5·d⁺_img + 0.5·d⁺_txt`. Rocchio:
   `q(1) = normalize(0.8·q(0) + 0.5·d⁺ − 0.15·d⁻)`; then the user-text pass nudges `q(1)`
   further toward "dry grass".
5. **Next batch.** FAISS re-searches with `q(1)`. The rug and visually similar prints are
   blacklisted by hard filtering and cannot return; the grid is now dominated by zebras
   on dry plains. One or two more rounds converge on the target.

---

## 10. Running modes: with / without Llama 3.2 Vision

VLM auto-captioning is powerful but expensive (20–40 s per region, significant RAM/GPU),
so the system runs **fully without it**. The single switch is `OLLAMA_ENABLED` in
`server/.env`.

| Mode | `OLLAMA_ENABLED` | Ollama running? | Text signal source |
|---|---|---|---|
| **Lightweight** (default) | `false` | not needed | Visual Genome region phrases (instant) |
| **Full** | `true` | yes, model pulled | AI-generated region captions (plus VG phrases) |
| Safe fallback | `true` | no | Auto-falls back to VG phrases |

**The async pipeline.** A VLM call never runs on the blocking feedback path. Captioning
fires the moment a mask is produced inside `/segment`, in a background task, *before* the
user has even decided the region is relevant. The caption is stored in an in-memory cache
keyed by `(image_path, label, query, hint)` (FIFO, capped at 200,
`_CAPTION_CACHE_MAX`). By the time the user clicks Apply Feedback, the caption is usually
already cached and is read in ~1 ms; if not ready, the system falls back to the VG
phrases returned synchronously.

**The fallback is automatic.** At startup the server calls `check_ollama` (is Ollama
reachable *and* is the model pulled?). Only then is captioning enabled; otherwise it
silently uses VG phrases. A missing Ollama never crashes or hangs the server.

- **Lightweight:** set `OLLAMA_ENABLED=false`; skip all Ollama setup; start normally.
- **Full:** install Ollama → `ollama serve` → `ollama pull llama3.2-vision` (~7.8 GB,
  once) → set `OLLAMA_ENABLED=true`, `OLLAMA_URL=http://127.0.0.1:11434`,
  `OLLAMA_MODEL=llama3.2-vision` → start → verify `curl -s localhost:8001/health` shows
  `"ollama_available": true`.

Flip the flag and restart to switch modes anytime.

---

## 11. Data and indexes

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
├── image_index.faiss         # ~108k image vectors (the search index)
├── image_paths.txt           # one path per row, same order as the index
├── region_index.faiss        # optional — hard filtering (~259k region crops)
└── region_meta.jsonl         # optional — hard-filter metadata (path, box, phrase)
```

**Build the image index:** `bash scripts/build_index.sh` — encodes every image (1–2+ h on
MPS/CPU, much faster on CUDA). One-time; re-run only if the image set changes. If
`region_descriptions.json` is present it is loaded at startup for the VG phrase signal.

**Build the optional region index (hard filtering):**
```bash
cd server
./venv/bin/python -m src.precompute.build_region_index \
    --image-paths ../faiss/visual_genome/google/siglip-large-patch16-256/image_paths.txt \
    --vg-dir ../data/visual_genome \
    --output-dir ../faiss/visual_genome/google/siglip-large-patch16-256
# add --limit 100 for a quick smoke test
```
It crops every VG bounding box, encodes it with `f_img`, and writes `region_index.faiss`
+ `region_meta.jsonl`. **Resumable** — re-run the same command to continue after an
interruption. The server auto-detects both files at startup; absent → hard filtering is
silently disabled.

> If images move or are re-downloaded, **rebuild the index** so `image_paths.txt` stays
> aligned, row-for-row, with the vectors.

---

## 12. Configuration reference

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

**`configs/demo/vg_siglip.yaml`** — `IMAGE_CORPUS_PATH`, `INDEX_PATH`, `VLM_MODEL_NAME`
(`google/siglip-large-patch16-256`), `IMG_SIZE` (256), `PATCH_SIZE` (16), `TOP_K`.

**`client-next/.env.local`** — `NEXT_PUBLIC_SERVER_URL`: the backend URL the browser
calls directly (default `http://127.0.0.1:8001`).

**Algorithm tuning constants** (`server/src/services/retrieval_service.py`, all commented
inline): Rocchio `alpha/beta/gamma`; fusion `img_w/txt_w`; hard-filter `HARD_FILTER_*`
thresholds and `HARD_FILTER_BOOST`; `SEARCH_OVERFETCH_FACTOR`. SAM constants
(`_LOGIT_CACHE_MAX`, area caps, padding) live in `server/src/models/sam.py`.

---

## 13. HTTP API reference

Base URL `http://localhost:8001`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/search` | Text query → top-k image paths + base64 previews + scores |
| POST | `/segment` | SAM 3 mask for a click (RLE); adds VG phrases; fires async caption |
| POST | `/apply_feedback` | Full Stage-4/5 update → new ranking + hard-filter telemetry |
| POST | `/caption` | On-demand Ollama caption for one base64 region |
| GET | `/health` | `status`, `gpu_available`, `ollama_available`, `vg_index_loaded`, `region_index_loaded`, `region_index_size` |
| GET | `/sam_status` | Whether SAM loaded and which backend |
| GET | `/ollama_status` | Ollama reachability, model, url |
| GET | `/caption_cache_status` · `/caption_lookup` | Debug: async caption-cache state |
| GET | `/metrics` | Index size, GPU backend, timings |
| GET | `/study/tasks` | Build/serve the study tasks (reads `eval/data/queries.json`) |
| POST | `/study/log` | Append a study event to the participant's JSONL log |
| POST | `/study/form` | Save a participant's post-task questionnaire |

**Safety** — every client-supplied image path is validated against the canonical corpus
path set; out-of-corpus paths are rejected with HTTP 403, preventing path traversal.

**Concurrency** — SAM keeps shared mutable state, so all SAM inference is serialised
through a single lock. Ollama runs on a worker thread so the event loop stays responsive
during the tens of seconds a caption can take.

---

## 14. The user-study system

A guided target-finding flow used to evaluate the system, at
`http://localhost:3000/study`. **It requires `server/src/eval/data/queries.json`** — if
that file is missing, `/study/tasks` returns `503 {"detail":"Study query set not built"}`.

**Task definitions.** `queries.json` holds `{"category": [...], "compositional": []}`.
Each task carries `qid` (a WordNet synset, e.g. `cat_zebra.n.01`), `text` (the displayed
query, e.g. `zebra`), and `relevant` (the set of VG image IDs containing that synset —
ground truth and the source of the target). The three study tasks are zebra (1492
relevant, target 134), tower (1750, target 9), and eyelid/"lid" (2003, target 6). The
relevant sets are derived from `objects.json` by matching each image's object synsets to
the task synset.

**Task serving** (`_build_study_tasks`). The pool is loaded, and for each task the
**lowest relevant image ID that resolves to a corpus path** is chosen as the target; an
"archivist" cover story (`_STUDY_STORIES`) is rendered around the query text; the target
preview is base64-encoded and returned. Deterministic, so every participant sees the same
tasks.

**Logging.** Each participant gets `logs/study/{id}.jsonl` with `session_start`,
`task_start`, `search_submitted`, `segment_click`, `feedback_applied`, and
`task_finished` events (result rankings, scores, latency). The questionnaire is saved to
`logs/study/{id}.form.json`.

**Analysis.** `server/src/eval/analyze_study.py` joins the logs and forms with the
`queries.json` ground truth, computing objective metrics (success, rounds, duration,
precision@{5,10} at stop) and subjective Likert aggregates, and writes
`study_summary.json` + `study_tables.tex`:
```bash
cd server
./venv/bin/python -m src.eval.analyze_study \
    --logs-dir ../logs/study --queries src/eval/data/queries.json \
    --out-dir src/eval/study_results
```
Supporting modules: `metrics.py` (precision@k, nDCG@k, MRR), `questions_meta.py` (Likert
item → research-question mapping).

> `queries.json` is generated from VG data and is not tracked by git. To regenerate it,
> scan `objects.json` for images whose object synsets match each task synset (see the
> ground-truth loader in `analyze_study.py`).

---

## 15. Frontend

Next.js + React + Zustand (`client-next/`). Global state (results, SAM annotations,
round, history, hint text) lives in `src/lib/store.ts`; the server-contract types in
`src/lib/types.ts`; API calls in `src/lib/api.ts`.

Key components:
- **`image-card.tsx`** — renders each result with an HTML-canvas mask overlay (40%
  opacity); captures clicks relative to the rendered image rectangle and converts them to
  natural pixel space before sending. A ResizeObserver keeps the overlay aligned as the
  browser rescales the image.
- **`feedback-panel.tsx`** — positive/negative hint fields, the "anchor to original
  query" toggle, asynchronously-arriving VLM caption suggestions with a one-click "Use",
  the Apply Feedback button, and the previous round's hard-filter telemetry.
- **`search-bar.tsx`**, **`image-gallery.tsx`**, **`header.tsx`**,
  **`server-dashboard.tsx`** — query input, result grid, status.
- **`study/`** — the study flow: `task-selector`, `task-briefing`, `task-banner`,
  `questionnaire`, `study-complete`.

The study page hardcodes results-per-search via `STUDY_TOP_K` (overriding the store's
default of 5).

---

## 16. Testing

```bash
cd server
venv/bin/python -m pytest -q                                 # all tests
venv/bin/python -m pytest tests/test_changes.py -q           # core pipeline (30 tests)
venv/bin/python -m pytest tests/test_study_endpoints.py -q   # study endpoints (4 tests)
```
Always use `server/venv/bin/python`, not the system Python — it has the dependencies.
Coverage includes RLE round-trips, Rocchio (positive/negative/none independently), SAM
coordinate clamping, the region index, endpoint validation (empty query, `top_k` range,
path traversal), logit-cache eviction, and the study task/log/form endpoints.

Frontend checks (from `client-next/`): `npx tsc --noEmit`, `npx next build`.

---

## 17. Deployment

For a cloud-GPU split (remote uvicorn, local Next.js) see `deploy/DEPLOY.md`.
`scripts/cloud_bootstrap.sh` provisions a fresh GPU VM end-to-end (clone, deps, model
weights, index build, server), invoked by `deploy/setup-cloud.sh`. Point the local
frontend's `NEXT_PUBLIC_SERVER_URL` at the remote backend and ensure the server's
`CORS_ORIGINS` allows the frontend origin.

---

## 18. Repository map

| Path | Role |
|---|---|
| `start.sh` | One command to launch backend + frontend with a health checklist |
| `server/src/retrieval_server_visual.py` | FastAPI app — all endpoints, study system |
| `server/src/services/retrieval_service.py` | Search + Rocchio feedback + fusion + hard filtering |
| `server/src/services/region_index.py` | Region kNN index wrapper (hard filtering) |
| `server/src/models/sam.py` | SAM 3 segmenter, prompt box, mask selection, logit cache, RLE |
| `server/src/models/siglip.py` | SigLIP wrapper (`f_txt`, `f_img`, normalization) |
| `server/src/models/ollama_vision.py` | Ollama captioning client |
| `server/src/models/relevance_feedback.py` | Rocchio update + mask/box → crop extraction |
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
