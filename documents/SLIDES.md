# VisualRef v2 — 10-Slide Presentation Spec

Academic-style thesis proposal deck. Each slide: **layout**, **content** (what goes on the slide), **visual suggestions**, and **speaker notes** (what to say).

Target total talk length: ~10 minutes + 5 min Q&A.

---

## Slide 1 — Title

**Layout:** Full-bleed title slide (one central title, one subtitle line).

**Content**
- **Title:** VisualRef v2 — Interactive Visual Search with Segmentation-Grounded Relevance Feedback and an LLM Dialogue Agent
- **Subtitle:** Rareș Boghean · Maastricht University · Thesis proposal · 2026

**Visual suggestions**
- Hero screenshot of the app: a result with the SAM mask overlay and a Llama caption chip visible.
- If available, small logo strip: SigLIP · SAM 3 · Llama 3.2-Vision · FAISS.

**Speaker notes (~30 s)**
- Introduce yourself and the project in one sentence: *"I'm building an interactive visual search system that lets users refine results by clicking on image regions and talking to a vision-language model."*
- Promise the structure: motivation → system → what's new → user study → future work.

---

## Slide 2 — Motivation & Research Questions

**Layout:** Title + two columns. Left column = motivation. Right column = research questions.

**Content — Left column (Motivation: the Semantic Gap)**
- Free-text queries rarely capture a user's exact visual intent.
- Classical retrieval returns neighbours of the *query* embedding, not of the *intent*.
- Users see a near-miss and have no precise way to say *"this bit, not that bit."*
- The previous prototype (RecSys '25) only supported whole-image feedback — still coarse.

**Content — Right column (Research Questions)**
- **RQ1** — To what extent does fine-grained visual **segmentation** influence retrieval precision and interaction efficiency in asset search tasks?
- **RQ2** — How effectively can a **proactive LLM agent** (multi-turn dialogue + intent clarification) resolve the semantic gap and reduce user effort?
- **RQ3** — To what extent does a **conversational LLM-based agent** improve search efficiency and result relevance compared to non-agent interaction?

**Visual suggestions**
- Small diagram on the left: two circles labelled *"what the user typed"* and *"what the user meant"*, with a gap arrow between them.
- Right column numbered boxes (RQ1/2/3) in the template's accent colour.

**Speaker notes (~60 s)**
- Define *semantic gap* plainly: the distance between what users *type* and what they *mean*.
- Retrieval engines optimise for embedding similarity, not intent satisfaction — that is the gap.
- Two levers under investigation: **region-level segmentation** (RQ1) and an **LLM dialogue agent** (RQ2, RQ3 are two framings of the agent's value — effort reduction vs effectiveness vs a no-agent baseline).

---

## Slide 3 — The VisualRef v2 Pipeline

**Layout:** Title + horizontal 6-step flow across the slide; thin stack-strip along the bottom.

**Content — Six-step interaction loop (left → right)**
1. **Query** — user types text; SigLIP encodes; FAISS returns top-K images.
2. **Segment** — user clicks a region; SAM 3 masks it; Visual Genome phrases attach.
3. **Caption** — Llama 3.2-Vision describes the masked crop, conditioned on the query.
4. **Refine** — user confirms or adds a free-text hint.
5. **Fuse** — four signals combine: region embedding + hint + VG phrase + VLM caption.
6. **Update** — Rocchio step `q ← α·q + β·pos − γ·neg` → new FAISS search → new results.

**Content — Stack strip (bottom)**
- **Backend:** FastAPI · SigLIP-large 256 · SAM 3 · Ollama / Llama 3.2-Vision 11B · FAISS IndexFlatIP · 108 k Visual Genome images.
- **Frontend:** Next.js 16 · React 19 · Zustand · Tailwind · shadcn/ui · Canvas overlay.

**Visual suggestions**
- Horizontal flow with 6 boxes, arrows between them, a loop-back arrow from step 6 to step 1.
- Use 1 icon per step (keyboard, pointer, speech bubble, pencil, merge, cycle).

**Speaker notes (~75 s)**
- Walk the 6 steps linearly once — this is the whole system in one sentence per step.
- Emphasise two design decisions:
  - Captioning is **asynchronous** — the UI stays responsive, and the client polls a cache endpoint until the caption is ready.
  - The loop is **closed** — every refinement updates the query vector and re-searches. No page reloads, no restart.

---

## Slide 4 — Core Innovations: Segmentation + Agent + Fusion

**Layout:** Title + three equal-width boxes across the slide.

**Content — Box 1: Region-level feedback (SAM 3)**
- One click → precise mask → crop isolates the intent.
- Logit cache (50 FIFO) makes follow-up clicks feel instant.

**Content — Box 2: Query-aware VLM agent (Llama 3.2-Vision)**
- Six prompt templates (query-aware, hint-aware, VG-grounded) — captions emphasise the axes the user cares about.
- `asyncio.create_task` + client polling → no UI freeze; captions appear live.

**Content — Box 3: Multi-signal Rocchio fusion**
- Each positive / negative carries four signals: region image embedding, hint, VG phrase, VLM caption.
- When a caption is present: 40 % image / 60 % text weighting.
- α = 0.8, β = 0.5, γ = 0.15 with per-round decay — prevents drift, stabilises the loop.

**Visual suggestions**
- Three vertical cards. Bottom of slide: the fusion formula rendered as a clean equation.

**Speaker notes (~60 s)**
- "This is the technical heart of v2. **Segmentation** says *where*, the **VLM** says *what*, and **Rocchio** says *how much*."
- The Rocchio weights are not arbitrary — they were chosen so one bad click does not derail the whole session.

---

## Slide 5 — Differences from the Previous Prototype

**Layout:** Title + two-column comparison table.

**Content — Left column: Prototype (RecSys '25)**
- CLIP-base encoder.
- Whole-image feedback only.
- No VLM — single-label feedback.
- Generic prompts.
- No region grounding.
- Synchronous — UI blocked during captioning.
- No caching — repeated work per click.
- Thin evaluation on a small custom set.

**Content — Right column: VisualRef v2**
- SigLIP-large 256 joint encoder.
- Region-level feedback via SAM 3.
- Async Llama 3.2-Vision captioning.
- Context-aware, query-conditioned prompts.
- Visual Genome phrase grounding per region.
- `asyncio` background tasks + client polling.
- Bounded caption + SAM logit caches; AbortController cancellation.
- 108 k VG corpus + instrumented user study.

**Visual suggestions**
- Two-column comparison, with v2 column in the template's accent colour to signal *"this is the contribution."*

**Speaker notes (~45 s)**
- v1 was a proof-of-concept; v2 is a system you can actually run a controlled study on.
- The two biggest deltas: **segmentation as the primary feedback signal** (replaces clicking whole thumbnails) and the **async VLM agent** (it *talks*, rather than simply labels).

---

## Slide 6 — Limitations & Open Problems

**Layout:** Title + single bulleted body.

**Content**
- **Latency** — Llama 3.2-Vision 11B on a laptop is 2–4 s per caption; async hides most of it, but cold starts show.
- **Memory** — SigLIP + SAM 3 + Ollama all compete for MPS memory on an M-series laptop.
- **Corpus scope** — Visual Genome + COCO only; generalisation to creative and brand assets still to be proven.
- **SAM 3 quality** — occasional over-segmentation on texture-heavy scenes.
- **Ollama fragility** — no graceful failover if the local model process crashes; errors surface only in the dashboard.
- **Study bias** — small sample (friends, colleagues) → directional findings, not statistically inferential claims.

**Visual suggestions**
- Keep this slide text-only and dense; it earns credibility by being honest.

**Speaker notes (~45 s)**
- Be explicit: *"None of these are fatal; each has a planned mitigation in the future-work section."*
- Emphasise that surfacing limitations is what distinguishes a research prototype from a demo.

---

## Slide 7 — User Testing: Study Design

**Layout:** Title + body; small visual on the right (mock of a story card).

**Content**
- **Participants** — ~12–18 friends and colleagues from Maastricht University; informal recruitment; no compensation.
- **Setting** — my laptop, one-to-one, think-aloud protocol, ~25 min per session.
- **Stimuli — "story cards" (×8–10)** — each card describes a target scene and shows one reference photo, for example:
  > *"A red-haired child on a wooden swing in a sunny backyard."*
- **Task per card** — find an image that satisfies the card using VisualRef v2. No time limit. **The participant can abandon whenever they wish.**
- **Conditions (within-subject, counterbalanced via Latin-square):**
  - **C1** — text search only.
  - **C2** — text search + whole-image relevance feedback (prototype-equivalent).
  - **C3** — full VisualRef v2 (segmentation + VLM agent).

**Visual suggestions**
- Mock-up of a printed story card on the right — target description + reference photo — so the audience *sees* what a participant is handed.

**Speaker notes (~60 s)**
- Story cards give every participant the **same target**, so effort is comparable across conditions.
- Within-subject + Latin-square counterbalancing controls for order and learning effects.
- Asking *"can you find this?"* instead of *"rate this result"* keeps the interaction natural and ecologically valid.

---

## Slide 8 — User Testing: Metrics, Hypotheses & Abandonment

**Layout:** Title + two columns: left = metrics, right = hypotheses. Tiny logging footer at the bottom.

**Content — Left column: Metrics**
- **Primary 1 — Segmentations-to-satisfaction.** Number of region clicks + hint edits before the participant declares *"this one is good enough."*
- **Primary 2 — Abandonment.** After how many **unsuccessful query rounds** does the participant give up? Hard cap = 8 rounds.
- **Primary 3 — Time-to-satisfaction.** Wall-clock seconds per card.
- **Secondary** — feedback clicks per card · query reformulations · SUS (usability) · NASA-TLX (effort).

**Content — Right column: Hypotheses**
- **H1 (RQ1)** — C3 reduces segmentations-to-satisfaction vs C2 by **≥ 25 %**.
- **H2 (RQ2)** — C3 has **fewer query reformulations** than C1/C2 (the LLM captures intent users cannot phrase).
- **H3 (RQ3)** — Abandonment is **highest in C1** and drops monotonically through **C2 → C3**.

**Content — Footer**
- Per-session JSON trace (client + server side) → CSV → analysis in R / Python.

**Visual suggestions**
- Small bar-chart mock-up illustrating the predicted H3 direction (C1 > C2 > C3).

**Speaker notes (~60 s)**
- *"Satisfaction"* and *"abandonment"* are the two cornerstones — one measures success, the other measures failure.
- H1 captures **efficiency**, H3 captures **giving up** — the two failure modes of visual search.
- With n ≥ 12, a consistent direction across participants is a thesis-scale empirical claim; we are not reaching for significance on tiny effects.

---

## Slide 9 — Future Work: A Staged, Concrete Plan

**Layout:** Title + three horizontal time-band columns (Short / Mid / Long-term). This slide must feel like a **roadmap**, not a wish list.

**Content — Short-term (next 2–3 months, before thesis deadline)**
- Run and analyse the user study (n ≥ 12); report per-RQ effect sizes with confidence intervals.
- Host Llama 3.2-Vision on a remote endpoint → study reproducible on any hardware.
- Add a **structured dialogue mode**: when query confidence is low, the LLM asks *one* clarifying question instead of only captioning.
- Ship a 15 k creative-asset corpus (open stock photos) → generalisation test beyond Visual Genome.
- Automate per-session trace export (CSV + JSON schema) — analysis pipeline ready before data collection ends.

**Content — Mid-term (thesis completion + publication)**
- Baseline comparison against a **PicHunter-style Bayesian feedback** engine → positions v2 against published work.
- **Ablations** — remove segmentation · remove VLM · remove Rocchio — isolate each component's contribution to H1/H2/H3.
- Submit a short paper to **CHIIR 2027** or the **SIGIR demo track**.
- Release anonymised session traces + MIT-licensed code — a reference dataset for interactive visual retrieval research.

**Content — Long-term (research vision beyond the thesis)**
- **Learned Rocchio weights** — α/β/γ learned online per user, not fixed.
- **Cross-session dialogue memory** — *"like the last one but bluer"* carries intent across queries.
- **Personalised taste model** — a lightweight per-user style profile replaces rigid queries.
- **Generative fallback** — when retrieval genuinely fails, hand the accumulated multi-signal evidence to a text-to-image model.
- **End goal** — a reusable, privacy-preserving framework for *any* visual search task: stock photography, medical imaging, brand compliance.

**Visual suggestions**
- Three vertical columns with a small timeline ribbon across the top (Now → 2026 Q3 → 2027 → beyond).
- Use iconography: hourglass (short) · book (mid) · telescope (long).

**Speaker notes (~90 s)**
- This is the pitch for the next year of work.
- *Short-term* = concrete deliverables with dates → proves there is a plan, not just hopes.
- *Mid-term* = the publishable contribution (ablations are non-negotiable — they make the empirical claims credible).
- *Long-term* = vision that outlasts the thesis and makes the framework genuinely reusable.
- Close this slide with: *"I know what the next year looks like, and every step has a measurable output."*

---

## Slide 10 — Conclusion & Thanks

**Layout:** Title slide style, matching Slide 1.

**Content**
- **Headline:** THANK YOU
- **One-line summary:** VisualRef v2 closes the semantic gap with **region-level feedback** + a **proactive VLM agent**, evaluated on a controlled within-subject study.
- **What's next:** user study · ablations · publication.
- **Contact:** r.boghean@student.maastrichtuniversity.nl · Maastricht University · 2026
- **Questions?**

**Visual suggestions**
- Re-use the hero screenshot from Slide 1 at reduced opacity as a background.

**Speaker notes (~30 s)**
- One-sentence recap: *"Gap → segmentation + agent → study → roadmap."*
- Open for questions.

---

## Section-to-slide mapping (sanity check)

| Requested section                | Slide(s)          |
|----------------------------------|-------------------|
| Research questions               | 2                 |
| Process                          | 3                 |
| What we have (system)            | 3, 4              |
| Differences from the old version | 5                 |
| Limitations / problems           | 6                 |
| User testing                     | 7, 8              |
| **Future work (emphasised)**     | **9**             |
| Title + thanks                   | 1, 10             |

All requested sections are covered; slide 9 is the dedicated, three-horizon roadmap slide you asked to emphasise.
