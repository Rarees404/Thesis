"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useAppStore } from "@/lib/store";
import { searchImages, applyFeedback, healthCheck } from "@/lib/api";
import {
  setDashboardMetricsActive,
  useMetricsStore,
} from "@/lib/metrics-store";

import { NeuralBackground } from "@/components/neural-background";
import { Header } from "@/components/header";
import { SearchBar } from "@/components/search-bar";
import { ImageGallery } from "@/components/image-gallery";
import { FeedbackPanel } from "@/components/feedback-panel";
import { CaptionPanel } from "@/components/caption-panel";
import { ServerDashboard } from "@/components/server-dashboard";
import { ErrorBanner } from "@/components/error-banner";

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
      const data = await searchImages(query, topK);
      if (data.success) {
        setSearchResults(
          data.images,
          data.image_paths,
          data.scores,
          data.preview_width ?? 224,
          data.preview_height ?? 224
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
          // Include image path so server can do VG region lookup by image
          image_path: img.path,
          // Pre-computed IoU-matched VG phrases from /segment — reuse to skip re-querying
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
      });

      if (data.success) {
        setFeedbackResults(
          data.images,
          data.image_paths,
          data.scores,
          data.preview_width ?? 224,
          data.preview_height ?? 224
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

  return (
    <div className="min-h-screen bg-black">
      <NeuralBackground />
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="relative z-10 mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {activeTab === "search" && (
          <div className="space-y-6">
            {store.images.length === 0 && !store.isSearching && (
              <div className="flex flex-col items-center gap-4 pb-6 pt-16 text-center">
                <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-400 bg-clip-text text-transparent">
                  Visual Relevance Feedback
                </h2>
                <p className="max-w-2xl text-white/40 text-lg leading-relaxed">
                  Search for images using natural language. Then click on objects
                  in the results to tell the system what you want more or less of.
                  The AI refines your search in real time.
                </p>
                <div className="flex flex-wrap gap-3 justify-center mt-2">
                  <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/30 ring-1 ring-white/10">
                    SAM 3 Segmentation
                  </span>
                  <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/30 ring-1 ring-white/10">
                    Llama 3.2 Vision
                  </span>
                  <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/30 ring-1 ring-white/10">
                    SigLIP Embeddings
                  </span>
                  <span className="rounded-full bg-white/5 px-3 py-1 text-xs text-white/30 ring-1 ring-white/10">
                    Rocchio Feedback
                  </span>
                </div>
              </div>
            )}

            <SearchBar onSearch={handleSearch} />

            <ErrorBanner />

            {backendReady === false && (
              <div className="flex items-center gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-400">
                <svg className="h-4 w-4 animate-spin shrink-0" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                </svg>
                <span>
                  Backend is loading models (SigLIP + SAM 3) — this can take a few minutes on first run.
                  Search will be enabled once ready.
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
    </div>
  );
}
