# VisualRef v2 — Diagram Description for AI Image Generation

Use this file as a prompt for any AI diagram tool (Eraser.io, Whimsical AI,
Miro AI, Lucidchart AI, etc.) or as a detailed reference for a graphic designer.
The description is structured from the top level down, covering layout, shapes,
colours, labels, and every connection.

---

## Overall layout

The diagram is a **left-to-right horizontal flowchart** with five clearly
separated horizontal swim-lanes stacked vertically. Each swim-lane has a
coloured background and a label on the left edge. The diagram is roughly
2400 × 1200 pixels. A clean white background behind everything.

The five swim-lanes from top to bottom are:

1. **USER** — light yellow background `#FEF9C3`
2. **FRONTEND** — light blue background `#DBEAFE`
3. **BACKEND** — light purple background `#EDE9FE`
4. **MODELS** — light red/pink background `#FEE2E2`
5. **STATE / CACHES** — light green background `#D1FAE5`

All swim-lanes span the full width. Swim-lane labels are written vertically
on the left side of each lane in dark grey bold text.

---

## Nodes (components)

Each node has a **shape**, **fill colour**, **border colour**, **bold label**,
and an optional smaller italic sub-label beneath.

### SWIM-LANE 1 — USER (top)

| ID | Shape | Fill | Border | Label | Sub-label |
|----|-------|------|--------|-------|-----------|
| U  | Rounded rectangle (stadium) | `#FCD34D` | `#92400E` | 👤 User | — |

The user node sits in the centre-left of the yellow lane.

---

### SWIM-LANE 2 — FRONTEND

| ID | Shape | Fill | Border | Label | Sub-label |
|----|-------|------|--------|-------|-----------|
| PAGE | Rectangle | `#BFDBFE` | `#1D4ED8` | page.tsx | top-level route |
| STORE | Cylinder | `#BFDBFE` | `#1D4ED8` | Zustand store | query · results · feedback · sam annotations · captions |
| CARD | Rectangle | `#BFDBFE` | `#1D4ED8` | image-card.tsx | Canvas overlay · click capture · mask renderer · caption polling |
| FEEDBACK | Rectangle | `#BFDBFE` | `#1D4ED8` | feedback-panel.tsx | free-text hint · Relevant / Irrelevant label |
| DASH | Rectangle | `#BFDBFE` | `#1D4ED8` | server-dashboard.tsx | health · model status · metrics |
| API | Rectangle with double left border | `#93C5FD` | `#1D4ED8` | lib/api.ts | fetch wrapper · AbortController · 180 s timeout |

Layout order left to right: PAGE → STORE → CARD → FEEDBACK → API (rightmost, close to backend boundary). DASH sits below CARD.

---

### SWIM-LANE 3 — BACKEND

| ID | Shape | Fill | Border | Label | Sub-label |
|----|-------|------|--------|-------|-----------|
| EP_SEARCH | Hexagon | `#DDD6FE` | `#5B21B6` | POST /search | — |
| EP_SEGMENT | Hexagon | `#DDD6FE` | `#5B21B6` | POST /segment | — |
| EP_APPLY | Hexagon | `#DDD6FE` | `#5B21B6` | POST /apply_feedback | — |
| EP_CAPLOOKUP | Hexagon | `#DDD6FE` | `#5B21B6` | GET /caption_lookup | — |
| EP_HEALTH | Hexagon | `#EDE9FE` | `#5B21B6` | GET /health · /sam_status · /ollama_status · /metrics | — |
| SVC | Large rounded rectangle | `#C4B5FD` | `#4C1D95` | RetrievalServiceVisual | Rocchio orchestrator |
| BGTASK | Rectangle with dashed border | `#E9D5FF` | `#7C3AED` | asyncio.create_task | background VLM captioner |

Layout: Hexagon endpoints run along the top edge of the backend lane from left to right. SVC is a large central box below the endpoints. BGTASK is to the right of SVC.

---

### SWIM-LANE 4 — MODELS

| ID | Shape | Fill | Border | Label | Sub-label |
|----|-------|------|--------|-------|-----------|
| SIGLIP | Rounded rectangle | `#FECACA` | `#B91C1C` | SigLIP-large 256 | joint image + text encoder |
| SAM | Rounded rectangle | `#FECACA` | `#B91C1C` | SAM 3 | point-prompt segmenter |
| OLLAMA | Rounded rectangle | `#FECACA` | `#B91C1C` | Llama 3.2-Vision 11B | via Ollama HTTP |
| VG | Rounded rectangle | `#FCA5A5` | `#B91C1C` | Visual Genome regions | IoU phrase lookup |

Layout left to right: SIGLIP · SAM · OLLAMA · VG, evenly spaced.

---

### SWIM-LANE 5 — STATE / CACHES (bottom)

| ID | Shape | Fill | Border | Label | Sub-label |
|----|-------|------|--------|-------|-----------|
| FAISS | Cylinder | `#A7F3D0` | `#047857` | FAISS IndexFlatIP | 108 k × 1152-d vectors |
| IMGS | Cylinder | `#A7F3D0` | `#047857` | Image files on disk | Visual Genome corpus |
| CAPCACHE | Cylinder | `#6EE7B7` | `#047857` | _caption_cache | bounded · FIFO eviction |
| INFLIGHT | Cylinder | `#D1FAE5` | `#047857` | _caption_in_flight | de-dup set |
| LOGITS | Cylinder | `#D1FAE5` | `#047857` | SAM logit cache | 50 entries · FIFO |
| B64 | Cylinder | `#D1FAE5` | `#047857` | image_b64 LRU cache | 2000 entries |

Layout left to right: FAISS · IMGS · CAPCACHE · INFLIGHT · LOGITS · B64.

---

## Connections (arrows)

Arrows are drawn with rounded elbows (not straight). Every arrow has a
**label** in a small white box sitting on top of the arrow line.
Solid lines = requests / actions. Dashed lines = responses / return values.

Use a numbered badge (circle with a number) beside each arrow label to show
which interaction phase it belongs to. Phases are ①②③④⑤⑥⑦.

### Phase ② — Initial query (dark blue arrows)

```
U ──①──► PAGE          label: "type query"
PAGE ─────► API         label: "onSearch()"
API ──②──► EP_SEARCH    label: "POST /search  {query, top_k}"   SOLID
EP_SEARCH ──► SVC        label: "search(query)"
SVC ──► SIGLIP           label: "encode_text(q)"
SIGLIP ──► SVC           label: "query embedding qᵥ"   (dashed return)
SVC ──► FAISS            label: "search(qᵥ, K)"
FAISS ──► SVC            label: "top-K indices"         (dashed return)
SVC ──► IMGS             label: "load image paths"
IMGS ──► SVC             label: "paths + b64"           (dashed return)
EP_SEARCH ──②──► API     label: "200 OK  images[]"      DASHED
API ──► STORE            label: "set results"
STORE ──► CARD           label: "render grid"
```

### Phase ③ — Click to segment (orange arrows)

```
U ──③──► CARD            label: "click pixel (x,y) on image"
CARD ──► CARD            label: "AbortController.abort(prev)" (self-loop, small curved arrow)
CARD ──► API             label: "segmentImage(path, point, query, hint, label)"
API ──③──► EP_SEGMENT    label: "POST /segment"   SOLID
EP_SEGMENT ──► SVC       label: "segment(path, points)"
SVC ──► SAM              label: "predict(points)"
SAM ──► LOGITS           label: "read/write logit cache"
LOGITS ──► SAM           label: "cached logits"     (dashed)
SAM ──► SVC              label: "mask RLE"          (dashed)
SVC ──► VG               label: "IoU bbox lookup"
VG ──► SVC               label: "top-3 phrases"     (dashed)
EP_SEGMENT ──③──► API    label: "200 OK  {mask_rle, region_b64, vg_phrases}"  DASHED
API ──► STORE            label: "setSamAnnotation()"
STORE ──► CARD           label: "overlay mask + VG chips"
```

### Phase ④ — Background caption (red dashed arrows)

```
EP_SEGMENT ──④──► BGTASK   label: "spawn task  (crop, query, hint, label)"  THICK DASHED
BGTASK ──► OLLAMA           label: "POST /api/generate  (select prompt template)"
OLLAMA ──► BGTASK           label: "caption text  (~2–4 s)"  (dashed)
BGTASK ──► CAPCACHE         label: "cache.set(key, caption)"
BGTASK ──► INFLIGHT         label: "in_flight.remove(key)"
```

Note: the arrow from EP_SEGMENT to BGTASK should be drawn as a **lightning bolt** or use a special "fire-and-forget" style (open arrowhead, thick dashed line).

### Phase ⑤ — Client polling (teal arrows, cyclic)

```
CARD ──⑤──► API              label: "lookupCachedCaption()  every 2.5 s"
API ──⑤──► EP_CAPLOOKUP      label: "GET /caption_lookup?..."
EP_CAPLOOKUP ──► CAPCACHE    label: "read(key)"
CAPCACHE ──► EP_CAPLOOKUP    label: "caption | null"   (dashed)
EP_CAPLOOKUP ──⑤──► API      label: "{caption, ready, in_flight}"  DASHED
API ──► STORE                 label: "update caption"
STORE ──► CARD                label: "caption chip appears"  (dashed)
```

Draw a small circular arrow around the polling section to show it is a loop.

### Phase ⑥ — Feedback entry (grey arrows)

```
U ──⑥──► FEEDBACK    label: "type hint · pick Relevant / Irrelevant"
FEEDBACK ──► STORE   label: "push feedback item"
```

### Phase ⑦ — Rocchio update (dark purple arrows)

```
U ──⑦──► FEEDBACK         label: "click Apply Feedback"
FEEDBACK ──► API           label: "applyFeedback({q, positives, negatives})"
API ──⑦──► EP_APPLY        label: "POST /apply_feedback"   SOLID
EP_APPLY ──► SVC           label: "apply(positives, negatives)"
SVC ──► SIGLIP             label: "encode_image(crop) × N"
SVC ──► SIGLIP             label: "encode_text(hint + phrase + caption) × N"
SIGLIP ──► SVC             label: "v_img, v_txt"   (dashed)
SVC ──► SVC                label: "fuse 0.4·v_img + 0.6·v_txt\nthen Rocchio:\nq_new = α·q + β·pos − γ·neg"  (self-loop annotation box)
SVC ──► FAISS              label: "search(q_new, top_k)"
FAISS ──► SVC              label: "new top-K"   (dashed)
EP_APPLY ──⑦──► API        label: "200 OK  new images[]"   DASHED
API ──► STORE              label: "set results"
STORE ──► CARD             label: "render new grid"
```

Add a large curved arrow from the bottom of the CARD node looping back up to
the User node, labelled **"loop until satisfied"** in italic.

### Dashboard (light grey arrows, thin)

```
DASH ──► API               label: "poll every 5 s"
API ──► EP_HEALTH          label: "GET /health · /sam_status · /ollama_status · /metrics · /caption_cache_status"
EP_HEALTH ──► DASH         label: "status responses"  (dashed, thin)
```

---

## Typography

- **Title** at the top of the diagram: `VisualRef v2 — Application Workflow`
  Font: bold, 28 pt, dark grey `#1F2937`
- **Swim-lane labels**: bold, 14 pt, rotated 90°, dark grey `#374151`
- **Node titles**: bold, 11 pt, `#111827`
- **Node sub-labels**: italic, 9 pt, `#6B7280`
- **Arrow labels**: 9 pt, white box with 1 pt grey border, `#374151`
- **Phase badge numbers**: 10 pt white text inside a small filled circle:
  ① blue `#2563EB`, ② blue, ③ orange `#EA580C`, ④ red `#DC2626`,
  ⑤ teal `#0D9488`, ⑥ grey `#6B7280`, ⑦ purple `#7C3AED`

---

## Legend box (bottom-right corner)

Draw a small legend box with:
- Solid arrow → request / action
- Dashed arrow → response / return value
- Lightning / open-head dashed → fire-and-forget async
- Cylinder shape → data store or cache
- Hexagon shape → HTTP endpoint
- Rounded rectangle → service / model
- Dashed-border rectangle → background async task

---

## Key annotation boxes (floating callouts)

Place three floating callout boxes connected to the relevant nodes with a thin
dotted line:

1. **Near BGTASK:**
   > "Captioning is async — the UI never waits for the LLM. The client polls
   > /caption_lookup every 2.5 s until the caption lands in the cache."

2. **Near SVC (Rocchio):**
   > "Rocchio weights: α = 0.8 (query), β = 0.5 (positives), γ = 0.15
   > (negatives). Per-round decay prevents drift. Image:text fusion = 40:60
   > when a caption is available."

3. **Near the loop-back arrow from CARD to U:**
   > "The loop repeats from Phase ③ until the user is satisfied or
   > abandons (hard cap = 8 rounds in the user study)."

---

## Suggested tools to generate from this description

- **Eraser.io** — paste the description into the AI diagram prompt, choose
  "flowchart" style.
- **Whimsical AI** — use the "create a flowchart" AI command and paste
  this file.
- **Miro AI** — use the diagram generation feature.
- **ChatGPT / Claude** — ask it to produce a Mermaid diagram from this
  description, then render in VS Code or GitHub.
- **draw.io / diagrams.net** — use this as a manual reference to build
  the diagram with their drag-and-drop editor.
