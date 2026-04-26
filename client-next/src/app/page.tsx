"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useAppStore } from "@/lib/store";
import { searchImages, applyFeedback, healthCheck } from "@/lib/api";
import {
  setDashboardMetricsActive,
  useMetricsStore,
} from "@/lib/metrics-store";

import { Header } from "@/components/header";
import { SearchBar } from "@/components/search-bar";
import { ImageGallery } from "@/components/image-gallery";
import { FeedbackPanel } from "@/components/feedback-panel";
import { CaptionPanel } from "@/components/caption-panel";
import { ServerDashboard } from "@/components/server-dashboard";
import { ErrorBanner } from "@/components/error-banner";

function createSessionId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function Home() {
  const [activeTab, setActiveTab] = useState("search");
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Poll /health until the backend responds (models can take 2-5 min to load)
    const check = () => {
      healthCheck()
        .then(() => {
          setBackendReady(true);
          if (pollRef.current) clearInterval(pollRef.current);
        })
        .catch(() => setBackendReady(false));
    };
    check();
    pollRef.current = setInterval(check, 5000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    const onDashboard = activeTab === "dashboard";
    setDashboardMetricsActive(onDashboard);
    if (onDashboard) void useMetricsStore.getState().poll();
  }, [activeTab]);

  const store = useAppStore();

  const handleSearch = useCallback(async () => {
    const { query, topK, setIsSearching, setSearchResults, setError } =
      useAppStore.getState();

    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);

    try {
      const sessionId = createSessionId();
      const data = await searchImages(query, topK, sessionId);
      if (data.success) {
        setSearchResults(
          data.images,
          data.image_paths,
          data.scores,
          data.preview_width ?? 224,
          data.preview_height ?? 224,
          data.session_id
        );
      } else {
        setError(data.message || "Search failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search request failed");
    } finally {
      setIsSearching(false);
    }
  }, []);

  const handleApplyFeedback = useCallback(async () => {
    const {
      query,
      sessionId,
      topK,
      images,
      imagePaths,
      relevantCaptions,
      irrelevantCaptions,
      fuseInitialQuery,
      samAnnotations,
      setIsApplyingFeedback,
      setFeedbackResults,
      setError,
    } = useAppStore.getState();

    setIsApplyingFeedback(true);
    setError(null);

    try {
      const boxesList = images.map((img) =>
        img.boxes.length > 0 ? img.boxes : null
      );

      const samList = images.map((img, i) => {
        const annot = samAnnotations.get(i);
        if (!annot?.mask_rle) return null;
        const hasRelevant = annot.points.some((p) => p.label === 1);
        return {
          mask_rle: annot.mask_rle,
          label: hasRelevant ? ("Relevant" as const) : ("Irrelevant" as const),
          image_path: img.path,
          vg_phrases: annot.vg_phrases ?? [],
        };
      });

      const hasSam = samList.some((s) => s !== null);

      const data = await applyFeedback({
        query,
        top_k: topK,
        relevant_image_paths: imagePaths,
        relevant_captions: relevantCaptions,
        irrelevant_captions: irrelevantCaptions,
        annotator_json_boxes_list: boxesList,
        ...(hasSam ? { sam_annotations: samList } : {}),
        fuse_initial_query: fuseInitialQuery,
        ...(sessionId ? { session_id: sessionId } : {}),
      });

      if (data.success) {
        setFeedbackResults(
          data.images,
          data.image_paths,
          data.scores,
          data.preview_width ?? 224,
          data.preview_height ?? 224,
          data.session_id
        );
      } else {
        setError(data.message || "Feedback failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Feedback request failed");
    } finally {
      setIsApplyingFeedback(false);
    }
  }, []);

  const empty = store.images.length === 0 && !store.isSearching;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="mx-auto max-w-6xl px-6 py-10 lg:px-8 lg:py-14">
        {activeTab === "search" && (
          <div className="space-y-8">
            {empty && (
              <section className="space-y-6 pb-2">
                <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                  Interactive Image Retrieval
                </p>
                <h1 className="max-w-3xl text-4xl font-medium tracking-tight text-foreground leading-[1.1]">
                  Search by sentence. Refine by pointing.
                </h1>
                <p className="max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
                  Type a natural-language query, then click on the regions of the
                  retrieved images you want more — or less — of. SAM&nbsp;3 segments,
                  Llama&nbsp;3.2-Vision describes, and SigLIP&nbsp;+&nbsp;Rocchio
                  refines the search vector each round.
                </p>
                <dl className="grid grid-cols-2 gap-x-8 gap-y-3 pt-2 text-xs sm:grid-cols-4 max-w-3xl">
                  {[
                    { k: "Embedding", v: "SigLIP large/256" },
                    { k: "Segmentation", v: "SAM 3" },
                    { k: "VLM", v: "Llama 3.2-Vision" },
                    { k: "Index", v: "FAISS · 108k" },
                  ].map(({ k, v }) => (
                    <div key={k} className="border-l border-border pl-3">
                      <dt className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                        {k}
                      </dt>
                      <dd className="mt-0.5 text-foreground">{v}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            )}

            <SearchBar onSearch={handleSearch} />

            <ErrorBanner />

            {backendReady === false && (
              <div className="flex items-center gap-3 rounded-md border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
                <span className="relative flex h-2 w-2 shrink-0">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400/60 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400" />
                </span>
                <span>
                  Backend is loading models (SigLIP + SAM 3). Search will
                  enable when ready — typically 2–5 minutes on first run.
                </span>
              </div>
            )}

            <ImageGallery />

            <FeedbackPanel onApply={handleApplyFeedback} />

            <CaptionPanel />
          </div>
        )}

        {activeTab === "dashboard" && <ServerDashboard />}
      </main>

      <footer className="border-t border-border mt-16">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5 text-[11px] text-muted-foreground lg:px-8">
          <span className="font-mono">VisualReF · v2 · Maastricht University</span>
          <span className="font-mono">SAM 3 · SigLIP · Llama 3.2-Vision</span>
        </div>
      </footer>
    </div>
  );
}
