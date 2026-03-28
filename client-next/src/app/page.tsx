"use client";

import { useState, useCallback, useEffect } from "react";
import { useAppStore } from "@/lib/store";
import { searchImages, applyFeedback } from "@/lib/api";
import { startMetricsPolling } from "@/lib/metrics-store";

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

  useEffect(() => {
    startMetricsPolling();
  }, []);
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

      const samList = images.map((_, i) => {
        const annot = samAnnotations.get(i);
        if (!annot?.mask_rle) return null;
        const hasRelevant = annot.points.some((p) => p.label === 1);
        return {
          mask_rle: annot.mask_rle,
          label: hasRelevant ? ("Relevant" as const) : ("Irrelevant" as const),
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
