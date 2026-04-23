# VisualRef — Research Document

## 1. Project Description

**VisualRef** is an interactive image retrieval system that combines three deep-learning components to let users iteratively refine search results through visual and textual feedback:

| Component | Model | Role |
|-----------|-------|------|
| **Embedding backbone** | SigLIP (`google/siglip-large-patch16-256`) | Encodes both images and text into a shared 1024-d embedding space. Provides the core retrieval signal. |
| **Interactive segmenter** | SAM 3 (`facebook/sam3`) | Lets users click on objects in result images. Produces pixel-accurate masks that isolate "what the user cares about" from the rest of the image. |
| **Vision-Language Model** | Llama 3.2 Vision 11B (via Ollama) | Auto-captions SAM-segmented crops with short, query-aware text descriptions. These captions are encoded by SigLIP into the same embedding space, enriching the feedback signal beyond what the image embedding alone provides. |

**The retrieval loop works as follows:**

1. **Initial search.** The user types a natural-language query (e.g., "dog playing in a park"). SigLIP encodes it into a text embedding. FAISS finds the top-k nearest images from a pre-built index of ~40k image embeddings.

2. **User feedback.** The user clicks on objects in the returned images. SAM 3 segments each click into a binary mask. The user can also type free-text hints describing what they want more or less of.

3. **Query refinement.** The system extracts SAM crops, optionally captions them via Ollama, computes image and text embeddings, and updates the query embedding using the **Rocchio algorithm**:

   \[
   \vec{q}' = \alpha \cdot \vec{q} + \beta \cdot \vec{d}_{\text{pos}} - \gamma \cdot \vec{d}_{\text{neg}}
   \]

   where \(\vec{d}_{\text{pos}}\) and \(\vec{d}_{\text{neg}}\) are weighted combinations of image embeddings (from SAM crops) and text embeddings (from user hints + Ollama captions).

4. **Refined search.** The updated query is L2-normalized and used for a new FAISS search, producing more relevant results.

5. **Iterate.** Steps 2–4 repeat as many times as the user wants.

---

## 2. Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Next.js)                    │
│                                                         │
│  SearchBar ──→ POST /search ──→ ImageGallery            │
│                                     │                   │
│                               User clicks on            │
│                               image objects             │
│                                     │                   │
│                               POST /segment ──→ SAM 3   │
│                               (mask overlay shown)      │
│                                     │                   │
│  FeedbackPanel ──→ POST /apply_feedback                 │
│  (text hints +                      │                   │
│   SAM annotations)                  ▼                   │
│                          Updated ImageGallery           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI)                      │
│                                                         │
│  SigLIP ───── text/image embeddings ────── FAISS index  │
│     │                                       (40k imgs)  │
│     │                                                   │
│  SAM 3 ────── point-prompt segmentation                 │
│     │                                                   │
│  Ollama ───── query-aware crop captioning               │
│  (llama3.2-vision)                                      │
│     │                                                   │
│  Rocchio ──── α·q + β·pos − γ·neg ──── updated query   │
└─────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| Rocchio α (query weight) | 0.8 | High — preserves original search intent |
| Rocchio β (positive weight) | 0.5 | Moderate — pulls toward relevant feedback |
| Rocchio γ (negative weight) | 0.15 | Low — gentle push away from irrelevant |
| Image/text fusion (with VLM) | 0.4 / 0.6 | Text signal weighted more when Ollama captions are available |
| Image/text fusion (no VLM) | 0.5 / 0.5 | Equal weighting when only image embeddings + user text |
| SigLIP image size | 256×256 | Preview thumbnails and embedding input size |
| FAISS index type | IndexFlatIP | Exact cosine similarity (brute-force) |
| Ollama temperature | 0.2 | Low creativity — factual descriptions |
| Ollama max tokens | 40 | Short captions (~15 words) |
| SAM box padding | 20% (fg), 13% (neg-only) | Bounding box prompt around click points |

---

## 3. Problems and Bottlenecks

### 3.1 Latency — Ollama Vision is the Critical Path

**The single biggest problem.** Llama 3.2 Vision (11B parameters) running on Apple Silicon MPS takes **20–40 seconds per image crop**. During `apply_feedback`, each SAM crop is sent to Ollama sequentially. Even with the current limit of 1 crop per label, a single feedback round takes 40–80 seconds of Ollama processing alone.

**Impact:** The frontend connection times out. Users experience long waits. The system feels unresponsive.

**Why it matters:** The VLM captioning is what makes this system novel — it bridges the "semantic gap" between what users see (visual) and what the retrieval space understands (text). Without it, the system falls back to image-only embeddings, which are less expressive for fine-grained feedback.

### 3.2 Sequential Processing

Multiple operations that could run in parallel are currently sequential:

- **Ollama captions:** Crops are captioned one at a time (no batching, no parallelism).
- **SigLIP text embeddings:** User text and Ollama captions are encoded in a Python for-loop, one string at a time. Could be batched into a single forward pass.
- **Image I/O:** Top-k result images are loaded sequentially with `Image.open()`.

### 3.3 No Pre-computation / Caching

- **Segment → Caption delay:** Ollama captioning happens only when the user clicks "Apply Feedback". It could start as soon as SAM produces a crop (at segment time), so captions are ready before the user finishes their selections.
- **Repeated embeddings:** If the user refines the same set of images across multiple rounds, SAM crops are re-embedded every time.
- **No embedding cache:** The same image crop produces the same SigLIP embedding every time.

### 3.4 FAISS Scalability

The current `IndexFlatIP` performs exact brute-force search. This works well for 40k images but will not scale:

| Corpus size | Approximate search time (flat) |
|-------------|-------------------------------|
| 40k | ~5ms |
| 1M | ~100ms |
| 10M | ~1s |
| 100M | ~10s |

For production-scale datasets, approximate nearest neighbor structures (HNSW, IVF) are needed.

### 3.5 Single-Process Python Backend

The FastAPI server runs in a single Python process. The GIL (Global Interpreter Lock) prevents true parallelism for CPU-bound work. The `asyncio.to_thread()` pattern helps for I/O-bound work but doesn't parallelize GPU inference.

### 3.6 Memory Pressure

Three large models run simultaneously in memory:

| Model | Approximate VRAM/RAM |
|-------|---------------------|
| SigLIP large | ~1.2 GB |
| SAM 3 | ~2.5 GB |
| Ollama llama3.2-vision | ~7.8 GB |

Total: ~11.5 GB. On a Mac with 16 GB unified memory, this leaves very little for the OS and other applications, causing memory pressure and swap thrashing.

### 3.7 Feedback Quality Uncertainty

The current system combines image and text embeddings with fixed weights (0.4/0.6). It is unclear whether these weights are optimal, and there is no mechanism to adapt them based on the quality of the feedback signal.

---

## 4. Proposed Infrastructure Improvements

### 4.1 Asynchronous Caption Pipeline (High Impact, Medium Effort)

**Idea:** Start Ollama captioning immediately when SAM produces a segment, not when the user clicks "Apply Feedback".

```
User clicks image → POST /segment → SAM mask returned to UI
                                  → Background: send crop to Ollama
                                  → Store caption in server-side cache

User clicks "Apply Feedback" → Captions already available → Fast response
```

**Implementation:** Add a `caption_cache: Dict[str, str]` to the server. After each `/segment` response, fire an `asyncio.create_task()` that calls Ollama and stores the result. The `/apply_feedback` endpoint checks the cache before calling Ollama.

**Expected improvement:** Eliminates 20–80s of waiting at feedback time.

### 4.2 Smaller/Faster VLM (High Impact, Low Effort)

Replace `llama3.2-vision` (11B, 7.8 GB) with a smaller model:

| Model | Size | Expected speed on MPS |
|-------|------|-----------------------|
| llama3.2-vision:11b | 7.8 GB | 20–40s per crop |
| moondream2 | 1.8 GB | 3–5s per crop |
| llava-phi3-mini | 2.7 GB | 5–10s per crop |
| minicpm-v | 3.0 GB | 5–8s per crop |

**Trade-off:** Smaller models produce less detailed captions but may be sufficient for the short (15-word) descriptions this system needs. This is an empirical question worth investigating.

### 4.3 Batch SigLIP Encoding (Medium Impact, Low Effort)

Currently, text strings are encoded one at a time in a for-loop. SigLIP supports batch encoding:

```python
# Current (slow):
for txt in all_pos_texts:
    emb = wrapper.get_text_embeddings(wrapper.process_inputs(text=txt))
    pos_text_embs.append(emb)

# Proposed (fast):
emb = wrapper.get_text_embeddings(wrapper.process_inputs(text=all_pos_texts))
positive_text_embeddings = emb.mean(dim=0)
```

**Expected improvement:** Reduces N forward passes to 1. Especially impactful when there are many Ollama captions + user text strings.

### 4.4 Progressive / Two-Phase Feedback (Medium Impact, Medium Effort)

Return results in two phases:

1. **Phase 1 (immediate, ~1s):** Use only image embeddings from SAM crops + user text embeddings. Return preliminary results.
2. **Phase 2 (async, 20–60s):** When Ollama captions are ready, refine the query and push updated results via WebSocket or SSE.

This gives the user immediate feedback while the VLM works in the background.

### 4.5 Embedding Cache (Low-Medium Impact, Low Effort)

Cache SigLIP embeddings keyed by image path + crop bounding box. Across multiple feedback rounds with the same images, this avoids redundant GPU inference.

### 4.6 GPU FAISS / HNSW Index (For Scale)

For corpora beyond 100k images:

- **HNSW** (`faiss.IndexHNSWFlat`): ~10x faster search with minimal recall loss.
- **GPU FAISS** (`faiss.GpuIndexFlatIP`): Leverages CUDA for massive parallelism.
- **IVF + PQ** (`faiss.IndexIVFPQ`): Compressed vectors for billion-scale search.

### 4.7 Multi-Worker Backend

Replace single-process uvicorn with:

```bash
uvicorn retrieval_server_visual:app --workers 2
```

Or use **Gunicorn** with uvicorn workers. Requires shared model state (e.g., loading models once and forking, or using a model server like Triton).

### 4.8 Model Quantization

- **SigLIP:** INT8 quantization via `torch.quantization` — ~2x speedup, minimal accuracy loss.
- **SAM 3:** FP16/BF16 inference on MPS — already partially supported.
- **Ollama:** Already supports quantized models (`llama3.2-vision:latest` is Q4_0 by default).

---

## 5. Research Questions

### RQ1: Does VLM-augmented relevance feedback improve retrieval precision over image-only or text-only feedback?

**Hypothesis:** Combining SAM crops (image signal) with Ollama captions (text signal) in the Rocchio framework produces more precise retrieval than using either modality alone.

**Method:** Ablation study comparing four conditions: (a) image embeddings only, (b) user text only, (c) image + user text, (d) image + user text + VLM captions. Measure Precision@k and nDCG across multiple feedback rounds on a labeled subset of COCO.

### RQ2: What is the optimal fusion strategy for multi-modal feedback signals?

**Hypothesis:** The fixed weights (0.4 image / 0.6 text) are suboptimal and a learned or adaptive weighting can improve results.

**Method:** Grid search or Bayesian optimization over (α, β, γ, img_w, txt_w) on a validation set. Compare fixed vs. adaptive weighting strategies.

### RQ3: How does VLM caption quality affect downstream retrieval performance?

**Hypothesis:** Caption specificity correlates with retrieval improvement — vague captions like "an animal" help less than specific ones like "golden retriever with red collar."

**Method:** Compare captioning models of varying capability (moondream2, llava-phi3, llama3.2-vision) and measure: (a) caption specificity (via BLEU/ROUGE against ground truth), (b) downstream retrieval metrics after feedback.

### RQ4: Can asynchronous/progressive feedback maintain user satisfaction while reducing perceived latency?

**Hypothesis:** Returning preliminary (image-only) results immediately, then updating with VLM-enhanced results asynchronously, produces comparable user satisfaction to waiting for the full result.

**Method:** User study (A/B test): synchronous full feedback vs. progressive two-phase feedback. Measure task completion time, number of feedback rounds, and subjective satisfaction (Likert scale).

### RQ5: How does segmentation granularity affect feedback quality?

**Hypothesis:** Fine-grained SAM point-click segmentation provides better feedback signal than coarse bounding boxes, because it isolates the relevant object more precisely.

**Method:** Compare three interaction modes on the same queries: (a) whole-image relevant/irrelevant, (b) bounding box selection, (c) SAM point-click segmentation. Measure retrieval precision improvement per feedback round.

### RQ6: What is the effect of multiple feedback rounds on query drift?

**Hypothesis:** Without anchoring to the original query (the `fuse_initial_query` mechanism), the query embedding drifts away from the user's true intent after 3+ rounds.

**Method:** Track the cosine similarity between the accumulated query embedding and the original query embedding across 1–10 feedback rounds. Correlate with retrieval relevance (human-judged).

### RQ7: How does this approach scale with corpus size?

**Hypothesis:** The Rocchio-based approach maintains effectiveness as the corpus grows from thousands to millions, but the VLM captioning benefit decreases because the embedding space becomes denser.

**Method:** Build indexes of varying sizes (10k, 40k, 100k, 500k images) from COCO, Open Images, or LAION subsets. Measure retrieval metrics at each scale with and without VLM feedback.

### RQ8: Can the VLM be replaced by a lightweight caption generator without loss of retrieval quality?

**Hypothesis:** A distilled or fine-tuned small VLM (< 3B parameters) can produce captions that are "good enough" for Rocchio feedback, achieving 90%+ of the retrieval improvement at 5–10x lower latency.

**Method:** Fine-tune a small model (e.g., moondream2) on the task of producing SigLIP-aligned captions for image crops. Compare embedding-space alignment (cosine similarity to ground-truth captions) and downstream retrieval metrics against the full llama3.2-vision model.

---

## 6. Related Work Directions

### 6.1 Relevance Feedback in Image Retrieval

- **Rocchio (1971):** Original vector space model for relevance feedback in text retrieval. Extended to CBIR (Content-Based Image Retrieval) in the late 1990s.
- **Rui et al. (1998):** "Relevance Feedback: A Power Tool for Interactive Content-Based Image Retrieval." Foundational paper on visual relevance feedback.
- **Zhou & Huang (2003):** "Relevance Feedback in Image Retrieval: A Comprehensive Review." Survey of approaches.

### 6.2 Vision-Language Models for Retrieval

- **CLIP (Radford et al., 2021):** Contrastive Language–Image Pre-training. Established the joint embedding space paradigm.
- **SigLIP (Zhai et al., 2023):** Sigmoid Loss for Language Image Pre-Training. Improved training efficiency over CLIP.
- **LLaVA (Liu et al., 2023):** Visual instruction tuning. Demonstrated that VLMs can describe image content with high fidelity.

### 6.3 Interactive Segmentation

- **SAM (Kirillov et al., 2023):** Segment Anything Model. Introduced prompt-based segmentation.
- **SAM 2 (Ravi et al., 2024):** Extended SAM to video with memory-based tracking.
- **SAM 3 (Meta, 2025):** Further improvements in segmentation quality and speed.

### 6.4 Multi-Modal Fusion for Search

- **Late fusion vs. early fusion:** Combining image and text signals at the embedding level (late) vs. at the model input level (early). VisualRef uses late fusion.
- **Adaptive fusion:** Learning to weight modalities based on query difficulty or signal quality.

---

## 7. Experimental Design (Suggested)

### 7.1 Datasets

| Dataset | Size | Domain | Use |
|---------|------|--------|-----|
| MS-COCO val2014 | 40,504 | General objects | Primary evaluation |
| Open Images V7 (subset) | 100k–500k | Diverse categories | Scale experiments |
| Flickr30k | 31,783 | Captioned photos | Caption quality evaluation |

### 7.2 Metrics

- **Precision@k** (k = 5, 10, 20): Fraction of retrieved images that are relevant.
- **nDCG@k:** Normalized Discounted Cumulative Gain — accounts for ranking position.
- **Mean Reciprocal Rank (MRR):** Position of first relevant result.
- **Rounds to target precision:** How many feedback rounds needed to reach P@5 > 0.8.
- **Latency:** Wall-clock time per feedback round.
- **User satisfaction:** Likert scale ratings (for user studies).

### 7.3 Baselines

1. **Text-only search:** SigLIP text embedding, no feedback.
2. **Rocchio with image embeddings only:** SAM crops → SigLIP image embedding → Rocchio.
3. **Rocchio with user text only:** User-typed descriptions → SigLIP text embedding → Rocchio.
4. **Full system (VisualRef):** SAM crops + Ollama captions + user text → fused embeddings → Rocchio.
5. **Re-ranking baseline:** Use a cross-encoder (e.g., CLIP reranker) to rerank initial results without iterative feedback.

---

## 8. Summary

VisualRef represents an intersection of three active research areas: interactive segmentation (SAM), vision-language understanding (Ollama/LLaVA), and relevance feedback (Rocchio). The core research contribution is **using a VLM to automatically bridge the semantic gap in visual relevance feedback** — the user points at what they want, the VLM describes it in natural language, and the retrieval system understands both signals.

The main engineering challenge is **latency**: the VLM is slow. The main research challenge is **proving that VLM-augmented feedback is worth the computational cost** compared to simpler approaches. The research questions above are designed to systematically address both dimensions.
