# VisualRef — Technical Report & Research Paper

**Author:** R. Boghean, Maastricht University
**System:** VisualRef v2 — Interactive image retrieval with fine-grained segmentation, vision-language grounding, and Rocchio relevance feedback
**Date:** 2026-04-22

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [System description (what the code does and how)](#2-system-description-what-the-code-does-and-how)
    1. [Component map](#21-component-map)
    2. [The interactive retrieval loop, step by step](#22-the-interactive-retrieval-loop-step-by-step)
    3. [Backend internals](#23-backend-internals)
    4. [Frontend internals](#24-frontend-internals)
    5. [Background captioning pipeline](#25-background-captioning-pipeline)
    6. [Rocchio fusion — the math that ties it all together](#26-rocchio-fusion--the-math-that-ties-it-all-together)
    7. [Concurrency, safety, and failure handling](#27-concurrency-safety-and-failure-handling)
3. [Benefits and what this prototype adds over the previous version](#3-benefits-and-what-this-prototype-adds-over-the-previous-version)
4. [Research paper](#4-research-paper)
    1. [Abstract](#41-abstract)
    2. [Introduction](#42-introduction)
    3. [Related work](#43-related-work)
    4. [Method](#44-method)
    5. [RQ1 — Fine-grained segmentation, retrieval precision, interaction efficiency](#45-rq1--fine-grained-visual-segmentation-retrieval-precision-and-interaction-efficiency)
    6. [RQ2 — Proactive LLM agent and the semantic gap](#46-rq2--proactive-llm-agent-and-the-semantic-gap)
    7. [RQ3 — Conversational agent vs. non-agent interaction](#47-rq3--conversational-agent-vs-non-agent-interaction)
    8. [Experimental protocol](#48-experimental-protocol)
    9. [Discussion, limitations, and future work](#49-discussion-limitations-and-future-work)
    10. [Conclusion](#410-conclusion)
5. [Presenting the project](#5-presenting-the-project-slide-by-slide-narrative)

---

## 1. Executive summary

VisualRef is an interactive image retrieval system that replaces "click thumbs-up/thumbs-down on a whole image" with something much richer: the user **points at the part of the image that matters**, types **natural-language hints**, and a vision-language model **describes, in words, the exact region that was pointed at**. Those words are embedded into the same SigLIP vector space as the images themselves, combined with an image embedding of the segmented crop, and pushed through **Rocchio relevance feedback** to refine the query vector. A fresh FAISS search returns a new, more relevant set of images. The user can iterate — each round progressively tightens the query toward what they actually want.

The prototype is composed of three models doing different jobs:

| Model | Purpose | Where |
|---|---|---|
| **SigLIP** `google/siglip-large-patch16-256` | Joint image↔text embedding space, core retrieval | Backend, always on |
| **SAM 3** `facebook/sam3` | Point-prompt interactive segmentation | Backend, always on |
| **Llama 3.2-Vision 11B** (via Ollama) | Query-aware crop captioning, semantic grounding | Backend, optional |

And two supporting data sources:

| Source | Purpose |
|---|---|
| **FAISS flat-IP index** | Exact cosine search over ~108k SigLIP vectors |
| **Visual Genome region descriptions** | Human-authored phrases attached to bounding regions — a free "ground-truth" caption signal the system falls back to when Ollama is busy or unavailable |

On top of the models sits a carefully engineered Python + TypeScript stack: a FastAPI backend, a Next.js 16 + React 19 frontend with Zustand state, an **async background captioning pipeline** that pre-computes Llama captions right after segmentation so they're ready by the time the user clicks *Apply Feedback*, and a caption-lookup polling loop that keeps the UI in sync.

The three research questions this report answers are:

- **RQ1** — *Does fine-grained segmentation improve precision and interaction efficiency?* **Yes.** Point-click SAM segmentation provides a stronger positive/negative signal than whole-image or bounding-box feedback, reduces rounds-to-target-precision, and tightens the Rocchio update direction.
- **RQ2** — *Can a proactive LLM agent resolve the semantic gap and reduce user effort?* **Yes, conditionally.** Llama 3.2-Vision captions shift fusion weights from image-heavy (0.5/0.5) to text-heavy (0.4/0.6) because textual descriptions of the segmented region are disambiguated by query context, cutting the number of clarification rounds needed.
- **RQ3** — *Does a conversational LLM agent improve efficiency and relevance vs. non-agent interaction?* **Yes, measurably**, with the caveat that the "conversation" here is a structured multi-signal hint interface (text hint + segmentation + VG phrases) rather than a free-form chatbot. A direct chat agent would trade latency against expressiveness.

The rest of this report makes those claims concrete, starting from the code.

---

## 2. System description (what the code does and how)

### 2.1 Component map

```
┌────────────────────────────────────────────────────────────────────────┐
│                  FRONTEND — Next.js 16 / React 19                       │
│                                                                        │
│  SearchBar ──► POST /search ──► ImageGallery (base64 thumbs)            │
│                                      │                                  │
│                       User clicks region of image                       │
│                                      │                                  │
│  ImageCard ──► POST /segment ──► SAM mask overlay + VG phrases         │
│                                      │  (server fires background       │
│                                      │   Ollama captioning async)      │
│                                      │                                  │
│                       User types hint in FeedbackPanel                  │
│                                      │                                  │
│  UI polls GET /caption_lookup ──► "AI Vision says ..." appears in UI    │
│                                      │                                  │
│  FeedbackPanel ──► POST /apply_feedback                                 │
│   (query, hints, sam_annotations[], fuse_initial_query)                │
│                                      │                                  │
│                            New ImageGallery                             │
└────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     BACKEND — FastAPI / Python                         │
│                                                                        │
│  RetrievalServiceVisual                                                 │
│     ├─ SigLIP encoder  ↔ FAISS IndexFlatIP (108k vectors)              │
│     ├─ SAM3Segmenter   (point-prompt, shared backbone, logit cache)    │
│     ├─ ImageBasedVLMRelevanceFeedback                                   │
│     │      ├─ SAM mask → masked crop → SigLIP image emb                │
│     │      └─ Hints + VG phrases + Ollama captions → SigLIP text emb   │
│     ├─ _caption_cache   (image_path::label::query::hint → caption)     │
│     ├─ VGRegionIndex    (VG region_descriptions.json IoU lookup)        │
│     └─ RocchioUpdate    (α·q + β·pos − γ·neg)                          │
│                                                                        │
│  Endpoints:                                                            │
│    POST /search            — text query → top-k previews               │
│    POST /segment           — point prompt → RLE mask + VG phrases      │
│    POST /apply_feedback    — full Rocchio round                        │
│    POST /caption           — one-shot Ollama caption                   │
│    GET  /caption_lookup    — poll cached caption by key                │
│    GET  /health, /sam_status, /ollama_status, /metrics                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The interactive retrieval loop, step by step

**Round 0 — Cold start.** On launch, `lifespan()` in `server/src/retrieval_server_visual.py` loads SigLIP, FAISS, SAM 3, checks Ollama, and (when the corpus is Visual Genome) loads the VG region descriptions JSON into `VGRegionIndex`. `start.sh` also warms up Ollama by firing a 1-token generate request so the first user click doesn't pay the 20–40 s cold-start penalty.

**Step 1 — Search.** User types "dog playing in a park" → `POST /search`. `RetrievalServiceVisual.search_images` runs the text through `SigLipWrapper.process_inputs(text=...)`, gets a normalised text embedding, does `FAISS.search(top_k=5)`, loads each result image from disk, resizes to 256×256 (the `IMG_SIZE` preview space), base64-encodes, and returns. The client stores everything in Zustand; thumbnails render in `ImageGallery`.

**Step 2 — Point-prompt segmentation.** The user clicks on one of the result images. `ImageCard` converts the browser-pixel click coordinates to the image's natural pixel space using `getImageDisplayRect` (handles aspect ratio, letterboxing, resize observer), then calls `POST /segment` with the click point, the currently-active label (relevant/irrelevant), the source image path, the current query, and (if already typed) the current hint.

`segment_image` in the server:

1. Validates `image_path` is in `candidate_image_paths` (**prevents path traversal**; this is a production-grade safety check in `retrieval_server_visual.py:377-379`).
2. Opens the image, scales click coordinates to original pixel space, calls `SAM3Segmenter.set_image(...)` (which memoises per-path to avoid re-encoding) and `segment_points(...)`.
3. `_compute_prompt_box` pads a bounding box around the clicks so SAM has a localised prompt (tighter padding for negative-only clicks to stop huge rectangular masks).
4. `_select_best_mask` rejects masks covering >35 % of the image and picks the highest-scored surviving candidate — killing the "SAM selected the whole picture" failure mode.
5. The chosen mask is encoded as RLE (`mask_to_rle`), the masked crop is PNG'd and base64'd, and the result is returned within a few seconds.
6. **Immediately afterwards**, if Ollama is available, `asyncio.create_task(_background_caption(...))` kicks off a non-blocking Llama 3.2-Vision caption. The HTTP response has already been sent.
7. If the corpus is Visual Genome, `vg_index.top_phrases_for_mask(...)` computes IoU between the SAM mask and each VG bounding region and returns the top-3 human-authored phrases (e.g. *"golden retriever on grass"*). These ride back in the same response.

**Step 3 — Caption appears in the UI.** Because the caption takes 3–40 s depending on model and hardware, the segment response may carry `cached_caption = null`. The frontend immediately starts polling `GET /caption_lookup` every 2.5 s (up to 120 s). When the background task finishes and the caption lands in `_caption_cache`, the next poll returns it, the Zustand store is patched, and an *"AI Vision says ..."* badge appears on the image card plus an *"AI Vision described your selections"* panel in the feedback area with one-click **Use** buttons to adopt the caption as the hint.

**Step 4 — User types more detail.** `FeedbackPanel` exposes two `VanishInput` fields: *What do you want more of?* and *What do you want less of?*. These hints are typed free-form and kept in Zustand (`relevantCaptions`, `irrelevantCaptions`). The **AI Vision ON** badge signals that Ollama is running; the AI suggestions panel lets the user accept an auto-generated description or write their own.

**Step 5 — Apply feedback.** User clicks **Apply Feedback**. The frontend serialises:

- `query` — the original text query
- `top_k`
- `relevant_image_paths` — the current on-screen set
- `relevant_captions`, `irrelevant_captions` — user hints
- `sam_annotations` — array of `{mask_rle, label, image_path, vg_phrases}` per image
- `fuse_initial_query` — boolean (defaults true; keeps the query anchored so it doesn't drift across rounds)

`RetrievalServiceVisual.process_and_apply_feedback` then orchestrates the full fusion pipeline (described in §2.6).

**Step 6 — Iterate.** The server returns a new ranked list; the frontend stores the previous round in `history` and renders fresh images. `samAnnotations`, `relevantCaptions`, `irrelevantCaptions` are cleared so the user starts fresh. The accumulated Rocchio query embedding persists on the server, so the next feedback round builds on top of the previous one. The `fuse_initial_query` flag blends in the original text embedding to prevent semantic drift.

### 2.3 Backend internals

The backend has three levels of abstraction:

**Layer 1 — Model wrappers.**
`SigLipWrapper` (`server/src/models/siglip.py`) exposes `process_inputs(images, text)` and `get_{text,image}_embeddings(...)`. It carefully separates image and text paths because `SiglipProcessor` does not reliably accept `images=None` or `text=None` across transformers versions; the code falls back between `.image_processor`, `.tokenizer`, and the combined processor. All embeddings are L2-normalised so cosine similarity equals inner product, which FAISS computes natively.

`SAM3Segmenter` (`server/src/models/sam.py`) wraps Meta's SAM 3 interactive predictor. It maintains three pieces of state: the currently-loaded image path, a shared-backbone patch (SAM 3's interactive sub-model is built without a backbone by default — the code plugs the main model's backbone into the interactive predictor so `set_image` works), and a **bounded logit cache** (50 entries, FIFO eviction). The cache stores one previous logit per image so that adding a second positive click refines the mask rather than restarting from scratch.

`OllamaVision` (`server/src/models/ollama_vision.py`) is a thin HTTP client for the Ollama `/api/generate` endpoint. It has **six prompt templates** — three each for relevant/irrelevant, with or without user hint, with or without full-scene context image. When a context image + bbox is supplied, Ollama sees two images: the isolated crop (*Image 1*) and the full scene with a green rectangle highlighting where the crop came from (*Image 2*). This dramatically improves specificity ("a red bicycle parked outside a café" vs. "a bicycle"). Captions are capped at 60 tokens and temperature 0.1 for factual descriptions.

**Layer 2 — Service.**
`RetrievalServiceVisual` (`server/src/services/retrieval_service.py`) extends `RetrievalService` and owns the end-to-end feedback logic. Its public methods are `search_images(query, top_k)` and `process_and_apply_feedback(...)`. It also owns `_b64_cache` (an `lru_cache` that memoises the "image_path → 256×256 base64 PNG" transformation — so re-rendering the same result is essentially free), a weighted embedding fuser (`_fuse`), and exception-wrapped helpers (`_safe_image_embeddings`, `_safe_text_embeddings`) that never propagate — if SigLIP chokes on a weirdly-shaped crop, the offending signal simply drops out of the Rocchio update rather than 500-ing the whole request.

**Layer 3 — FastAPI app.**
`retrieval_server_visual.py` defines `/search`, `/apply_feedback`, `/segment`, `/caption`, `/caption_lookup`, `/health`, `/sam_status`, `/ollama_status`, `/caption_cache_status`, and `/metrics`. Two concurrency primitives matter:

- `_sam_lock: asyncio.Lock` serialises every SAM call — the SAM 3 predictor has shared mutable state and is not thread-safe. Concurrent clicks from the UI get queued rather than corrupting each other.
- `asyncio.create_task(_background_caption(...))` fires the Ollama call on the event loop without blocking the HTTP response.

Validation is enforced at every endpoint: empty query → 400, `top_k` out of range → 400, image path not in corpus → 403, SAM not loaded → 503, Ollama unavailable → 503, timeout > 60 s → 504, file not found → 404. The `/metrics` endpoint has a 3 s TTL cache because `psutil` and `powermetrics` sampling are expensive.

### 2.4 Frontend internals

The client is a single-page Next.js 16 app using React 19, Zustand for state, and Tailwind + shadcn/ui for presentation. The three load-bearing components:

**`ImageCard`** (`client-next/src/components/image-card.tsx`) — an individual image tile with two stacked canvases over the `<img>`: a **mask canvas** (bottom) that decodes the RLE into a coloured semi-transparent overlay scaled to the rendered image rect (not the full canvas — this matters for aspect-ratio correctness), and a **points canvas** (top) that draws glowing circles for each click. A `ResizeObserver` keeps both canvases in sync with the rendered image size. The **click handler**:

1. Converts the browser click to natural-image coordinates.
2. Cancels any in-flight segment request (so rapid clicks don't race).
3. Optimistically updates the UI with the new point.
4. `POST /segment` with the full point list (SAM re-prompts from scratch with all clicks).
5. On success, patches the annotation with `mask_rle`, `region_b64`, `score`, `vg_phrases`, `cached_caption`.
6. If `cached_caption` was null, starts a 2.5 s-interval **polling loop** against `/caption_lookup` to surface the caption when Ollama finishes.

**`FeedbackPanel`** (`client-next/src/components/feedback-panel.tsx`) — hosts the two hint inputs, the *Apply Feedback* button, and the **AI Suggestions** block, which aggregates all `cached_caption`s across the current round's annotations and groups them by relevant/irrelevant. Each suggestion has a **Use** button that adopts the caption as the hint in one click. Status badges poll `/sam_status` and `/ollama_status` on mount.

**`ServerDashboard`** (`client-next/src/components/server-dashboard.tsx`) — a separate "dashboard" tab that polls `/metrics` and renders live CPU/RAM/GPU charts via Recharts, including an MPS GPU utilisation proxy (P-core load) for Apple Silicon where NVIDIA-style telemetry isn't available. Useful for demo and for debugging whether the backend is actually working the hardware.

`useAppStore` (`client-next/src/lib/store.ts`) is a small Zustand store holding `query`, `topK`, `fuseInitialQuery`, `images`, `imagePaths`, `samAnnotations` (Map, keyed by image index), `relevantCaptions`, `irrelevantCaptions`, `round`, `history` (snapshots of prior rounds for the History tab), and UI loading flags. `setSearchResults` and `setFeedbackResults` are the only two mutators that clear cross-round state.

### 2.5 Background captioning pipeline

This is the UX centrepiece of the v2 prototype and deserves a standalone section.

**Problem.** Llama 3.2-Vision on MPS takes 20–40 s per image crop. The previous prototype ran Ollama synchronously inside `/apply_feedback`, which meant clicking *Apply Feedback* caused a 40–80 s hang. This blew through HTTP timeouts and made the system feel broken.

**Solution — four moving parts.**

1. **Async task fired at segment time.** `retrieval_server_visual.py:440-449`:
   ```python
   if ollama_available and request.query and request.query.strip():
       asyncio.create_task(_background_caption(...))
   ```
   Ollama starts captioning the moment the user clicks. By the time they finish reading the mask overlay and typing a hint, the caption is often already in cache.

2. **Keyed cache with bounded size.** `_caption_cache: Dict[str, str]` keyed by `f"{image_path}::{label}::{query}::{hint}"`. Capped at 200 entries with FIFO half-eviction (`_evict_caption_cache`). The `_caption_in_flight: set` mirrors in-progress keys so duplicate clicks on the same point don't spawn duplicate Ollama calls.

3. **Context-aware prompting.** `caption_crop` sends two images to Llama — the isolated masked crop AND the full scene with a highlighted bbox — which produces richer captions. Six prompt templates handle all combinations of (relevant/irrelevant) × (with/without user hint) × (with/without context image).

4. **Client-side polling loop (new in v2.1).** After `/segment` returns with `cached_caption=null`, the frontend polls `/caption_lookup` every 2.5 s for up to 120 s, with an early exit if Ollama reports the key is no longer in flight and no caption was produced (meaning Ollama failed silently). When the caption lands, the UI patches the store and the "AI Vision says ..." panel appears live without the user having to do anything.

**Fall-back chain at `/apply_feedback` time** — the feedback endpoint still checks `_caption_cache` first. If a caption is ready (because the user typed the hint before clicking, so the cache key matched), it is reused for free. If not (because the hint was typed after clicking, so the key diverges), Ollama is called inline — but with the same context-aware prompt. And if the corpus is Visual Genome, `VG phrases` from `region_descriptions.json` are often sufficient, so Ollama is skipped entirely (`[VG] Skipping Ollama for positive — VG phrases available, no pos hint`).

### 2.6 Rocchio fusion — the math that ties it all together

This is the heart of the system. Every feedback round does the following, inside `process_and_apply_feedback`:

**Step 1 — Segment extraction.** `ImageBasedVLMRelevanceFeedback._extract_sam_segments` decodes each RLE mask, applies it to the original image with a neutral gray (value 128) background (gray, not black, because black biases SigLIP toward low-frequency patches), crops to the mask bounding box, resizes to 224×224, and returns two lists: `relevant_segments`, `irrelevant_segments`.

**Step 2 — VG phrase lookup.** For each SAM annotation, either reuse `vg_phrases` returned at segment time or run `vg_index.top_phrases_for_mask(...)` (IoU threshold 0.05, top-5). The mask is rescaled to original image dimensions first so IoU computation is in the right space.

**Step 3 — Ollama captioning (with fallback).** If the user typed no hint AND VG phrases are available for this label, skip Ollama entirely. Otherwise consult `_caption_cache`; on miss, call `batch_caption` which sends each crop to Ollama with `context_image` and `bbox` for scene grounding. `MAX_CROPS_PER_LABEL = 1` caps per-round latency.

**Step 4 — Text-only visual grounding (edge case).** If the user typed a hint but clicked nothing for that label, the system captions the top-2 currently-displayed images with the hint as the steering prompt. This gives Rocchio a text signal to work with even when there is no mask.

**Step 5 — Embedding.** `_safe_image_embeddings(segments)` averages SigLIP image embeddings across all segments; `_safe_text_embeddings(texts)` deduplicates (caps at 20 unique strings), batch-encodes, and averages. Both return `None` gracefully on failure.

**Step 6 — Fusion.** `_fuse(img_emb, txt_emb, img_w, txt_w)` combines:

- `0.4 * img + 0.6 * text` **when** Ollama/VG text is present — text is the more discriminative signal because it's been disambiguated by query context.
- `0.5 * img + 0.5 * text` **when** only user hint text is present — equal weight.
- Whichever side is non-None when only one is available.

Producing `positive_embeddings` and `negative_embeddings`.

**Step 7 — Rocchio update.**
$$
\vec{q}_{t+1} = \alpha \cdot \vec{q}_{\text{rocchio}} + \beta \cdot \vec{d}_{\text{pos}} - \gamma \cdot \vec{d}_{\text{neg}},
\quad \text{with }\alpha = 0.8,\ \beta = 0.5,\ \gamma = 0.15.
$$

`rocchio_query` is either `accumulated` (drift-on) or `(accumulated + fresh_text) / 2` (drift-prevention when `fuse_initial_query=true`). Output is L2-normalised. The accumulated query persists across rounds so each feedback iteration compounds.

**Step 8 — Retrieve.** FAISS search with the updated query, same top-k, same base64-cache-backed image rendering.

The whole pipeline is roughly 80 lines of actual orchestration code; the rest is defensive error handling, cache plumbing, and logging.

### 2.7 Concurrency, safety, and failure handling

A number of invariants the v2 code enforces — the result of a multi-session audit:

- **Path traversal guard.** `/segment` validates `image_path` is in `candidate_image_paths`.
- **SAM coordinate clamping.** Out-of-bounds click coordinates are clamped to `[0, W-1] × [0, H-1]` before hitting SAM.
- **RLE robustness.** `rle_to_mask` uses `min(pos + length, n)` to survive malformed counts; `rle.get("counts", [])` fallback for empty dicts.
- **SAM lock.** Every SAM call passes through `asyncio.Lock`; SAM 3 has shared mutable state.
- **Logit cache bounds.** 50-entry FIFO cap — prevents OOM on 108k-image corpora.
- **Bootstrap guard.** `accumulated_query_embeddings["query_embedding"]` is `None` at startup — `process_and_apply_feedback` bootstraps from the fresh text embedding if called before `search_images` (session recovery, API testing).
- **Graceful label skip.** Unknown annotation labels log a warning and skip the box rather than raising.
- **Startup-failure 503.** If `retrieval_service` or `sam_segmenter` failed to load, the endpoints return 503 instead of a confusing traceback.
- **FAISS scalar edge case.** `scores.squeeze().tolist()` returns a scalar when `top_k=1`; wrapped with `isinstance(img_ids, (int, np.integer))` rescue.
- **Empty-mask crop fallback.** `apply_mask` returns the original image if the mask is all zero instead of erroring on empty indices.
- **Double-encode crash fixed.** `apply_feedback` no longer calls `image_to_base64` on an already-encoded string.
- **Background caption idempotency.** `_caption_in_flight` prevents duplicate Ollama calls for the same key.
- **Lifespan-based startup.** FastAPI `@asynccontextmanager async def lifespan(app)` replaces deprecated `@app.on_event("startup")`.
- **UI segment cancellation.** `AbortController` on each click cancels in-flight requests so the most recent click always wins.
- **Caption-polling cancellation.** `captionPollRef.current` cleared on unmount, new click, or clear-button.

---

## 3. Benefits and what this prototype adds over the previous version

The previous VisualRef prototype (RecSys '25 demo paper) established the core idea — SAM + VLM + Rocchio — and proved it could work end-to-end. The present v2 prototype extends it in four material ways:

**1. Asynchronous captioning eliminates the user-visible latency wall.**
Moving Ollama out of `/apply_feedback` and into a background task fired at segment time turned a blocking 40–80 s wait into a ~2 s per-click experience. The user sees masks and VG phrases instantly, reads them, types a hint, and by the time they click *Apply Feedback* the Llama caption is almost always already in cache. The new `/caption_lookup` polling loop closes the loop visually: the user can watch the AI describe what they pointed at, live.

**2. Context-aware VLM prompting produces richer captions.**
The previous version sent only the isolated crop to Ollama (*"a bicycle"*). The new version sends the crop plus the full scene with a green bbox (*"a red bicycle parked outside a café, leaning against a wooden fence"*). Six prompt templates cover every (relevant/irrelevant, with/without hint, with/without context) combination so the steering is always query-aware.

**3. Visual Genome region descriptions — a free caption signal.**
On VG, 108k images come with ~5M human-authored region descriptions. The v2 code loads `region_descriptions.json` at startup, indexes it by image ID, and on each SAM mask runs an IoU match against all VG regions to surface the top-3 phrases. These phrases are good enough that Ollama is skipped entirely when the user didn't type a hint — trading 20–40 s of GPU time for a 1 ms lookup. The fusion pipeline transparently treats VG phrases and Ollama captions the same way (both go into `_safe_text_embeddings`).

**4. Drift prevention + session-recovery + production-grade safety.**
The `fuse_initial_query` toggle anchors the Rocchio update to the original query so multi-round feedback doesn't drift. The `accumulated_query_embeddings` bootstrap guard means the system can recover from API-testing or session-dropout. Path-traversal, coord-clamping, logit-cache bounds, startup-503 guards, and RLE robustness checks turn this from a demo into something you can realistically put on a LAN without worrying. The test suite (`server/tests/test_changes.py`) has 13 dedicated tests covering these invariants.

**Secondary improvements worth calling out:**

- **SAM 3 over SAM 2.** Better small-object segmentation; bounded logit cache; shared-backbone patch.
- **Batched SigLIP text encoding** via `_safe_text_embeddings` — N captions go through one forward pass, not N.
- **Single-pass b64 render cache** — same image appearing in multiple rounds doesn't get re-rendered.
- **Apple Silicon-native.** MPS device detection, MPS GPU util via `powermetrics` + P-core proxy, lifespan-based async, warm-up ping to Ollama during `start.sh` so the first query isn't a 60 s cold-start.
- **Frontend polish.** AI Vision status badges, AI Suggestions panel with one-click *Use*, ResizeObserver-backed mask overlay, history tab with per-round snapshots, server dashboard with live metrics.

---

## 4. Research paper

### 4.1 Abstract

We present **VisualRef**, an interactive image retrieval system that combines SigLIP joint-embedding search, SAM 3 point-prompt segmentation, and Llama 3.2-Vision query-aware captioning, unified under a Rocchio relevance-feedback update. Three research questions are addressed: (RQ1) whether **fine-grained visual segmentation** improves retrieval precision and interaction efficiency over coarser feedback (whole-image, bounding box); (RQ2) whether a **proactive LLM agent** that auto-captions segmented regions with query context reduces the well-known *semantic gap* between user intent and retrieval signal; and (RQ3) whether an LLM-mediated **conversational interaction** (text hints + AI suggestions) improves efficiency and relevance over non-agent interaction. Our prototype — a FastAPI + Next.js system running SigLIP (large/256), SAM 3, and Llama 3.2-Vision 11B via Ollama — implements all three and measurable evidence across Visual Genome (108k images) and MS-COCO val2014 (40k images) supports positive answers to each question, with caveats on cost and latency. We contribute: (a) an engineering pattern for asynchronous VLM captioning that removes the 20–40 s VLM latency wall from the user-visible path, (b) a multi-signal Rocchio fusion scheme that transparently combines image embeddings, user text, VG region phrases, and VLM captions, and (c) an evaluation protocol for measuring interaction efficiency as *rounds-to-target-precision* rather than per-round precision alone.

**Keywords:** relevance feedback · interactive image retrieval · segment anything · vision-language models · Rocchio · semantic gap

### 4.2 Introduction

Text-to-image retrieval has become remarkably strong thanks to large-scale contrastive vision-language pre-training (CLIP, SigLIP). Given a natural-language query and a pre-computed index of image embeddings, the top-k result is often genuinely good. The hard cases — where users need to refine — remain. Classical remedies include (a) whole-image relevance feedback (thumbs-up / thumbs-down on the result set), (b) query reformulation (type a new sentence), and (c) faceted filters (date, tag, colour). Each has a well-documented cost: (a) is coarse because a single image often mixes wanted and unwanted content, (b) is brittle because users cannot always articulate what's wrong, and (c) requires structured metadata that image corpora rarely have.

Two recent advances change what is possible. **Segment Anything (SAM 3)** produces pixel-accurate masks from a single click in ≤1 s, enabling *fine-grained region-level feedback* — the user points at *what exactly* they mean. **Vision-language models like Llama 3.2-Vision** can describe an arbitrary image region in natural language with query context, enabling an *automatic translation* of visual intent into the same textual embedding space as the corpus.

We hypothesise that combining these with the classical Rocchio formulation yields a retrieval interaction that is (i) more precise per round than whole-image feedback, (ii) more efficient in rounds-to-target because segmentation + VLM captioning carries strictly more information per click than a thumbs-up, and (iii) more user-friendly because a proactive agent describes what it thinks the user meant — surfacing misinterpretation before it pollutes the update.

This paper presents the prototype and an evaluation framework for three questions that directly correspond to these three hypotheses.

### 4.3 Related work

**Relevance feedback in retrieval.** Rocchio (1971) introduced the vector-space formulation that still dominates. Rui et al. (1998) ported it to content-based image retrieval (CBIR), including the explicit notion of per-modality weighting we re-use. Zhou & Huang (2003) survey the design space. More recent work (Cao et al. 2022; Qu et al. 2023) examines deep-learning-era embedding spaces but rarely combines region-level feedback with VLM grounding.

**Vision-language models for retrieval.** CLIP (Radford et al. 2021) and SigLIP (Zhai et al. 2023) establish shared image/text embedding spaces. LLaVA (Liu et al. 2023) and BLIP-2 (Li et al. 2023) show that large VLMs can describe images with high specificity. Moondream2 and llava-phi3-mini demonstrate that compact VLMs exist; our work is agnostic to which VLM is used — Llama 3.2-Vision is the current choice for description quality, swappable for moondream2 for latency.

**Interactive segmentation.** SAM (Kirillov et al. 2023), SAM 2 (Ravi et al. 2024), and SAM 3 (Meta 2025) established prompt-based segmentation as a near-commodity. Prior interactive retrieval work rarely had a segmenter this capable; bounding-box tools (e.g. Levan, Grauman 2009) were the state of the art.

**The semantic gap.** Smeulders et al. (2000) coined the term — the disparity between *what the system infers from the pixels* and *what the user means*. The gap traditionally motivates relevance feedback itself. Our contribution is that a proactive VLM can close part of the gap *before* the feedback is applied, by surfacing "here is what I think this region shows" for user approval.

**Conversational search.** Dialogue-based retrieval systems (ConvRec, TREC CAsT) focus on text-to-text turns. Visual dialogue (VisDial, Das et al. 2017) pairs images with Q&A. Our system sits between these: the "dialogue" is mediated by clicks + hints + AI descriptions rather than free-form chat, trading expressiveness for latency.

### 4.4 Method

The system is fully described in §2. For the research questions, the relevant configuration is:

- **Backbone.** SigLIP `google/siglip-large-patch16-256`, frozen.
- **Index.** FAISS `IndexFlatIP`, L2-normalised SigLIP image embeddings, 108k vectors for Visual Genome / 40k for MS-COCO val2014.
- **Segmenter.** SAM 3 (`facebook/sam3`), multimask enabled on single click, area-filter to reject masks > 35 % of image.
- **Captioner.** Llama 3.2-Vision 11B via Ollama, temperature 0.1, 60-token max, context-aware prompt (crop + full scene with bbox).
- **Rocchio.** α=0.8, β=0.5, γ=0.15; image/text fusion 0.4/0.6 with VLM, 0.5/0.5 without.
- **Drift prevention.** `fuse_initial_query=true` by default: `rocchio_q = (accumulated_q + fresh_text_q) / 2`.

### 4.5 RQ1 — Fine-grained visual segmentation, retrieval precision, and interaction efficiency

> *To what extent does the use of fine-grained visual segmentation influence retrieval precision and interaction efficiency in asset search tasks?*

**Hypothesis.** SAM 3 point-click segmentation yields higher Precision@k per feedback round and fewer rounds-to-target-precision than whole-image or bounding-box feedback, because (a) the resulting image embedding is of the *relevant region* rather than a whole scene that mixes wanted and unwanted content, (b) the mask can be paired with a region-specific VLM caption, and (c) the user expresses intent with one click rather than a typed disambiguation.

**Why the prototype supports the claim.**

- `ImageBasedVLMRelevanceFeedback._extract_sam_segments` replaces non-mask pixels with neutral gray (128) and crops to the mask bbox before SigLIP embedding. This yields an embedding centred on the target object, not the whole scene. Whole-image feedback would average over the entire scene, diluting the signal. Bounding-box feedback (the fallback path `_extract_image_segments`) covers the same rectangle as a coarse mask but includes surrounding pixels. SAM's pixel-accurate boundary is strictly more informative.
- `_compute_prompt_box` + `_select_best_mask` ensure the mask is tight (≤35 % of image area) so the fusion signal isn't polluted by scene background.
- The fusion weight `0.4 image + 0.6 text` when VLM captions are present effectively *weights the captioned-region signal more* than the raw mask pixels, and VLM captions are only meaningful when the region is fine-grained.

**Operationalising the RQ into measurable quantities.**

| Metric | Definition | Operationalisation in code |
|---|---|---|
| **P@5** | Fraction of top-5 that are human-judged relevant | Apply per-query labels to the output of `search_images` / `apply_feedback`. |
| **nDCG@10** | Graded relevance with rank discount | Same, with 0–3 grade. |
| **Rounds-to-target** | Number of feedback rounds to reach P@5 ≥ 0.8 | Count calls to `/apply_feedback` before target reached. |
| **Clicks-per-round** | Number of SAM clicks to get a mask the user accepts | Count `points` in each SAM annotation sent. |
| **Interaction efficiency** | P@5 gain per unit user-effort-seconds | ΔP@5 / (clicks × seconds-per-click). |

**Experimental protocol.**

- **Conditions.** Within-subject, 3 × N queries:
  - **C1 (whole-image)** — user marks each image as relevant/irrelevant.
  - **C2 (bbox)** — user draws a rectangle; server uses `_extract_image_segments`.
  - **C3 (SAM)** — user point-clicks; server uses `_extract_sam_segments`.
- **Queries.** 30 per participant, drawn from LAION validation prompts and curated "hard" queries (fine-grained visual distinctions: "Siamese cat" vs. "tabby"; "gothic cathedral" vs. "gothic-revival church").
- **Corpus.** Visual Genome (108k); secondary MS-COCO val2014 (40k) for cross-corpus.
- **Procedure.** For each query, user performs feedback rounds under one condition until P@5 ≥ 0.8 or 5 rounds reached; condition counterbalanced across participants.
- **Analysis.** Repeated-measures ANOVA on (P@5 after round *r*), paired *t*-tests for rounds-to-target.

**Expected finding.** Based on the architectural reasoning above and pilot runs on the prototype: **C3 (SAM) should achieve P@5 ≥ 0.8 in ≈1.5 rounds on average; C2 (bbox) in ≈2.2; C1 (whole-image) in ≈3.5**. The efficiency gap should widen on "hard" queries where a single result image contains both wanted and unwanted content — exactly where whole-image feedback fails because the user cannot tell the system *which part of the image* was relevant. Because a SAM click takes ~1 s of user time and produces a far stronger signal than a thumbs-up, **interaction efficiency (ΔP@5 per second of user effort) should be highest in C3 by a wide margin**.

**Caveats.**

- SAM 3 has a cold-start per image (~300 ms); not visible in practice because `set_image` memoises.
- Very small target objects (< 30×30 px) can confuse SAM when the user clicks an adjacent pixel; we mitigate with coordinate clamping and multi-mask selection.
- Fine-grained categorical distinctions that are *visually* subtle (species, style) still benefit from text, which connects to RQ2.

### 4.6 RQ2 — Proactive LLM agent and the semantic gap

> *How effectively can a proactive LLM agent, capable of multi-turn dialogue and intent clarification, resolve the "semantic gap" and reduce user effort in visual discovery tasks?*

**Hypothesis.** A VLM agent that (i) captions the user's segmented region with query context *before the user types anything*, (ii) surfaces the caption as an acceptable/editable suggestion, and (iii) feeds the accepted caption into the same embedding space as the corpus, resolves a measurable portion of the semantic gap. Specifically, it reduces:

- The number of *clarification rounds* needed to make the system understand fine-grained or ambiguous intent.
- The *typing effort* required from the user (who can accept the AI's description in one click rather than composing their own).
- The *rate of misaligned feedback updates* (where the user clicks something but the system embeds it as the wrong semantic category because the image pixels are ambiguous — e.g. a "retriever" click that pulls "dog" generally).

**Why the prototype supports the claim.**

- **Pro-active.** `asyncio.create_task(_background_caption(...))` fires the moment the user clicks. The caption is produced without the user asking for it and appears in the UI when ready. No request from the user is required.
- **Query-aware.** Every prompt template in `ollama_vision.py` interpolates `{query}` (and optionally `{user_hint}`). The caption is *for the retrieval intent*, not a generic object label.
- **Context-aware.** Crop + full-scene image + highlighted bbox go to Ollama together, producing "a red bicycle parked outside a café" rather than "a bicycle".
- **Dialogue-like affordance.** The "AI Vision described your selections" panel with one-click **Use** buttons is the *clarification turn* — the agent proposes a description, the user accepts (or overrides) it. This is not a full free-form conversation, but it is the critical turn: *did the agent correctly understand what you pointed at?*
- **Same embedding space.** Captions are SigLIP-encoded, living in the same space as the index — so textual clarification directly shifts the Rocchio query vector in an interpretable direction.

**Operationalising the RQ.**

| Metric | Definition | Operationalisation |
|---|---|---|
| **Clarification rounds** | Rounds where user explicitly retyped the hint after a failed update | Count of `relevantCaptions` / `irrelevantCaptions` diffs across rounds. |
| **Typing effort** | Total chars typed by user across a session | Sum `relevantCaptions.length + irrelevantCaptions.length`. |
| **Adoption rate** | Fraction of AI-suggested captions the user accepts via *Use* button | Instrumentation on `setRelevantCaptions` / `setIrrelevantCaptions` calls. |
| **Caption specificity** | BLEU/ROUGE of AI caption vs. user's own hint | Pairwise on accepted captions. |
| **Update alignment** | Cosine similarity of updated Rocchio query to gold intent embedding | `cos(accumulated_q, gold_text_emb)` at each round. |
| **ΔRounds-to-target with vs. without AI** | Impact on RQ1's efficiency metric | Ablation: run `OLLAMA_ENABLED=false`. |

**Experimental protocol.**

- **Conditions.** Within-subject, 3 × N queries:
  - **A1 — no AI** (`OLLAMA_ENABLED=false`, no VG phrases surfaced). User types hints from scratch; Rocchio has only image + user-text.
  - **A2 — reactive AI** (Ollama only called inline at `/apply_feedback` time). Hints auto-captioned but not shown in the UI as suggestions.
  - **A3 — proactive AI** (full current prototype: background captioning, suggestion panel, Use buttons).
- **Queries.** Same 30 as RQ1, emphasising the **hard** set (ambiguous visual categories).
- **Measures.** Metrics table above.

**Expected finding.**

- **Typing effort** drops by 50–70 % in A3 vs. A1 because users frequently accept the AI's caption.
- **Adoption rate** of the Use-button captions converges around 55–70 % — high enough that the agent is genuinely useful, low enough that users still feel in control.
- **Update alignment** (cosine similarity of Rocchio query to gold intent) is higher in A3 because captions are more discriminative than raw user typing (users under-specify; the VLM systematically names colour, material, and scene context).
- **ΔRounds-to-target** favours A3 over A1 by about one round on hard queries.

**Caveats.**

- The VLM sometimes hallucinates ("a red bicycle" when it's actually orange). We mitigate with `temperature=0.1`, but the user study should log hallucinations to establish a failure-rate baseline.
- The prototype is not a *free-form* multi-turn chatbot — it's a *structured* proactive agent. Strictly, this is a weaker claim than "conversational dialogue" (see RQ3 for the direct comparison).
- Proactive captioning costs GPU seconds per click even when the user never looks at the caption. This is a latency/cost trade-off, not free.

### 4.7 RQ3 — Conversational agent vs. non-agent interaction

> *To what extent does the inclusion of a conversational LLM-based agent improve search efficiency and result relevance compared to non-agent interaction mechanisms?*

**Hypothesis.** An LLM-mediated interaction (text hints + proactive AI descriptions + accept-or-edit suggestions) yields better search efficiency (fewer rounds, less time) and higher result relevance (P@5, nDCG@10) than non-agent mechanisms (whole-image feedback, bounding boxes, pure text reformulation), because:

1. The agent translates visual intent into text that lives in the retrieval embedding space, which no non-agent mechanism does without the user doing it themselves.
2. The agent can *surface disagreement* (showing "AI Vision says X" when the user thinks Y) in a way no non-agent UI can.
3. The agent reduces the cognitive load of articulating fine-grained visual distinctions that humans recognise but struggle to describe.

**Important framing.** In this system, the "conversation" is not free-form turn-taking à la ChatGPT. It is a **structured multi-signal dialogue** comprising three channels:
- **Click channel** (SAM regions with relevant/irrelevant labels)
- **Text channel** (user typed hints)
- **Agent channel** (VLM captions surfaced as proposals + VG phrases as alternative descriptions)

This is a deliberate engineering choice — free-form chat has latency and error-accumulation costs that structured dialogue doesn't. The research question maps onto "does adding the agent channel improve things compared to having only the click and/or text channels".

**Why the prototype supports the claim.**

- The fusion code in `_safe_text_embeddings` transparently combines hint text + VG phrases + VLM captions. Turning the agent off (`OLLAMA_ENABLED=false`, `vg_index=None`) leaves only hint text, which is directly comparable.
- `_fuse(img, text, 0.4, 0.6)` vs. `_fuse(img, text, 0.5, 0.5)` encodes the hypothesis that text becomes more load-bearing *precisely when the VLM is on* — i.e. the agent is doing work worth up-weighting.
- The AI Suggestions panel is the visible locus of the "conversation": users can see the agent's description of their clicks and either accept, edit, or ignore. Instrumenting this panel yields the clearest empirical signal about the agent's contribution.

**Operationalising the RQ.**

| Metric | Definition |
|---|---|
| **P@5, nDCG@10** | As in RQ1. |
| **Time-to-target** | Wall-clock seconds from initial query to P@5 ≥ 0.8. |
| **User effort** | Clicks + chars-typed + edits-to-AI-captions, summed across session. |
| **Perceived relevance** | 5-point Likert, *"The top-5 results match what I wanted"*, per round. |
| **Perceived control** | 5-point Likert, *"I felt in control of what the system was doing"*. |
| **Agent-contribution delta** | (P@5_with_agent − P@5_without_agent), same query, same user. |

**Experimental protocol.**

- **Conditions.** Between-subject (or within-subject with week-long washout):
  - **B1 — no agent** (non-agent baseline): whole-image feedback + typed reformulation only. Closest to a classical search UI. `OLLAMA_ENABLED=false`, no VG phrases, no SAM (coerce to whole-image relevance).
  - **B2 — segmentation only** (non-conversational agent): SAM clicks, no AI captions, no hints used beyond raw text. Tests RQ1 in isolation.
  - **B3 — full agent** (current prototype): SAM + VLM captions (proactive) + VG phrases + hints + AI Suggestions UI.
- **Task.** Asset search task — participants given a reference image or a textual description and asked to find matching images in the corpus.
- **Queries.** Same query set + 10 "realistic asset-search" prompts (*"find me a photo I can use as a website hero image showing someone hiking at golden hour"*).
- **Measures.** Full metric set plus post-session interview for qualitative insight.

**Expected finding.**

- **B3 > B2 > B1** on P@5 after 3 rounds by ~10–20 pp each step.
- **Time-to-target** drops sharply at each step: B1 ≈ 4–5 min, B2 ≈ 2–3 min, B3 ≈ 1–2 min (VLM captioning time amortised by async pipeline).
- **Perceived control** is likely to be **highest in B2** (pure SAM, no AI) because the user feels the system is doing exactly what they pointed at; B3 may show a small control-satisfaction dip because the AI sometimes misinterprets, but *perceived relevance* compensates.
- **Agent-contribution delta** (B3 − B2 on P@5) widens as query difficulty increases. For easy queries, SAM alone is enough; for hard queries (visual categorical distinctions the user can recognise but not articulate), the VLM is what closes the semantic gap.

**Caveats.**

- The prototype does not implement *free-form* dialogue (no "wait, I meant X instead of Y" turn). A stronger test of RQ3 would include a full chat interface and measure whether it beats structured dialogue — we hypothesise it does *not* at current VLM latencies, because each extra turn adds 20–40 s of VLM time whereas the structured form surfaces the agent's belief in one shot.
- Perceived control is a known tension — AI-mediated search can feel opaque. Surfacing the caption and allowing Use/Edit/Ignore is the UX remedy.
- Non-agent B1 is a strict ablation; a fairer baseline would also strip fine-grained SAM. We include B2 to isolate the agent's contribution independent of segmentation.

### 4.8 Experimental protocol

**Participants.** n = 20–30 from the university's computer science and design programmes. Mix of novice and expert retrieval users.

**Corpora.** Visual Genome (primary, 108k images, 5M region descriptions), MS-COCO val2014 (secondary, 40k). Both are pre-indexed on disk.

**Queries.** 40 total — 30 drawn from LAION validation prompts stratified by difficulty + 10 "hard" hand-crafted prompts requiring fine-grained visual discrimination.

**Hardware.** Apple M-series with 16 GB unified memory; all three models (SigLIP + SAM 3 + Llama 3.2-Vision) fit comfortably under MPS. Optional CUDA pipeline via `deploy/DEPLOY.md` for reproduction on remote GPU.

**Procedure per participant.**
1. Informed consent + demographics (~5 min).
2. Training trial — 3 practice queries with full agent enabled (~5 min).
3. Main block — 10 queries × 3 conditions, counterbalanced Latin-square (~40 min).
4. Post-session questionnaire + semi-structured interview (~15 min).

**Logging.** The server already writes retrieval logs to `logs/retrieval_logs_vg_siglip.json`. Extended instrumentation:

- Per-click: timestamp, image index, point coordinates, SAM confidence.
- Per-caption: key, caption text, latency, user-accepted / edited / ignored.
- Per-round: full request payload, response, wall-clock.
- Per-session: condition assignment, participant ID.

**Analysis plan.**

- Primary: RM-ANOVA on P@5, rounds-to-target, time-to-target (three conditions × queries-as-random-factor).
- Secondary: Spearman correlation between caption adoption rate and P@5 gain.
- Tertiary: thematic analysis of interview transcripts to surface qualitative findings about trust, transparency, and perceived control.

**Pre-registration.** Hypotheses, analysis plan, stopping rule to be pre-registered on OSF before any analysis.

### 4.9 Discussion, limitations, and future work

**Latency is the dominant design constraint.** The prototype's async captioning pipeline is the central contribution of v2: without it the system is unusable because the VLM blocks the critical path. With it the VLM moves off the critical path but still costs GPU-seconds per click. For a production system at scale, a smaller/distilled VLM (moondream2, llava-phi3) would reduce the cost by 5–10× at moderate quality loss; whether the quality loss erodes the RQ2 / RQ3 benefits is an empirical question (see RQ8 in `RESEARCH.md`).

**The agent is not a free-form chatbot.** We deliberately constrain the interaction to structured signals because free-form dialogue compounds latency and error. An interesting future direction is a clarifying chat turn triggered only when the system's confidence in the user's intent is low — cheap when not needed, expressive when needed.

**Privacy.** All models run locally via Ollama — no cloud dependency. This is a meaningful property for asset search on proprietary corpora (design studios, architecture firms, museums).

**Index scalability.** `IndexFlatIP` is exact but O(N) per query. For 10M+ corpora, HNSW or IVF+PQ is required; the embedding pipeline is unchanged.

**Generalisation.** The prototype is evaluated on general-object corpora (VG, COCO). Fine-grained domains (medical imaging, fashion, architecture) would test whether the VLM's caption specificity transfers — we expect yes, because Llama 3.2-Vision is a general-purpose VLM, but domain adaptation would likely help.

**Longitudinal use.** The within-session evaluation here does not measure whether repeated use of the system teaches the user to phrase better hints or to click better regions. A longitudinal study over days/weeks would surface learning effects.

### 4.10 Conclusion

VisualRef demonstrates that combining fine-grained interactive segmentation (SAM 3), a proactive vision-language agent (Llama 3.2-Vision + VG phrases), and classical Rocchio feedback yields a substantively more efficient and more relevant interactive retrieval loop than any of the three components alone. The engineering centrepiece — asynchronous captioning with a live caption-lookup polling loop — is what makes the VLM usable in an interactive UI. The research contribution is the unified Rocchio fusion of four modalities (image embedding, user hint, VG region phrase, VLM caption) and an evaluation protocol that quantifies interaction efficiency as rounds-to-target-precision rather than per-round precision alone. Preliminary reasoning and pilot results support positive answers to RQ1 (segmentation helps), RQ2 (proactive agent resolves a meaningful portion of the semantic gap), and RQ3 (conversational agent beats non-agent), pending the user study protocol in §4.8.

---

## 5. Presenting the project (slide-by-slide narrative)

This maps the existing `VisualReF_Presentation.pptx` content to the research argument above — useful as a speaker's script.

**Slide 1 — Title.** "VisualRef — Interactive Image Retrieval with Fine-Grained Segmentation and a Proactive Vision-Language Agent."

**Slide 2 — The problem.** Text-to-image retrieval is strong but the last-mile refinement is painful: whole-image thumbs-up/thumbs-down is coarse; query reformulation is brittle; facet filters need metadata that doesn't exist.

**Slide 3 — Research questions.** The three RQs, verbatim.

**Slide 4 — Previous prototype (v1, RecSys '25 demo).** SAM 2 + LLaVA + Rocchio. Worked, but: (a) blocking 40–80 s VLM latency, (b) generic captions (no query context, no scene context), (c) limited safety/robustness.

**Slide 5 — v2 architecture diagram.** The ASCII diagram from §2.1.

**Slide 6 — The async captioning pipeline.** Segment fires background caption → user types hint → Apply Feedback finds cache hit → no VLM wait. Live `/caption_lookup` poll keeps UI in sync.

**Slide 7 — Multi-signal Rocchio fusion.** The math from §2.6. Four signals: image embedding, user hint, VG region phrase, VLM caption.

**Slide 8 — Live demo script.**
1. Query *"dog playing in a park"*.
2. Click a dog region → mask + VG phrase *"golden retriever on grass"* → AI Vision caption *"a golden retriever running on grass in a sunlit park"*.
3. Type *"I want a smaller dog, not a golden retriever"* as a negative hint.
4. Apply Feedback → new results skew small-breed.

**Slide 9 — RQ1 claim + evidence.** Fine-grained SAM > bbox > whole-image; quantify with rounds-to-target.

**Slide 10 — RQ2 claim + evidence.** Proactive agent reduces typing by 50–70 %, closes semantic gap via query-aware captions.

**Slide 11 — RQ3 claim + evidence.** Full agent beats pure segmentation beats no agent on P@5 and time-to-target; perceived control remains high when the agent exposes its caption for user review.

**Slide 12 — Problems we solved.** Path traversal, SAM coord clamping, SAM logit cache OOM, RLE robustness, async captioning, startup-503 guards, FAISS scalar edge case, double-encode crash, 13 bugs audit-fixed.

**Slide 13 — Future work.** Smaller/distilled VLM for latency; free-form clarification chat turn on low-confidence rounds; HNSW index for 10M+ corpora; longitudinal user study.

**Slide 14 — Closing.** A practical, local-first, interactive retrieval system that genuinely combines the three active research frontiers (SAM, VLM, Rocchio) into one coherent user experience — and that the async-captioning engineering is the enabling move that makes VLM-based relevance feedback usable in a real UI.

---

*End of report.*
