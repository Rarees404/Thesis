"use client";

import { useState, useCallback, useEffect } from "react";
import { useAppStore } from "@/lib/store";
import { searchImages, applyFeedback } from "@/lib/api";
import { startMetricsPolling } from "@/lib/metrics-store";

import { MissionBriefing } from "@/components/mission-briefing";
import { NeuralBackground } from "@/components/neural-background";
import { Header } from "@/components/header";
import { SearchBar } from "@/components/search-bar";
import { ImageGallery } from "@/components/image-gallery";
import { FeedbackPanel } from "@/components/feedback-panel";
import { SAM3Panel } from "@/components/sam3-panel";
import { CaptionPanel } from "@/components/caption-panel";
import { HistoryPanel } from "@/components/history-panel";
import { ServerDashboard } from "@/components/server-dashboard";
import { ErrorBanner } from "@/components/error-banner";

export default function Home() {
  const [activeTab, setActiveTab] = useState("search");
  const [missionReady, setMissionReady] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem("mission-briefed") === "true") {
      setMissionReady(true);
    }
  }, []);

  useEffect(() => {
    startMetricsPolling();
  }, []);
  const store = useAppStore();

  const handleMissionComplete = useCallback(() => {
    setMissionReady(true);
    sessionStorage.setItem("mission-briefed", "true");
  }, []);

  const handleSearch = useCallback(async () => {
    const { query, topK, setIsSearching, setSearchResults, setError } =
      useAppStore.getState();

    if (!query.trim()) return;

    setIsSearching(true);
    setError(null);

    try {
      const data = await searchImages(query, topK);
      if (data.success) {
        setSearchResults(data.images, data.image_paths, data.scores);
      } else {
        setError(data.message || "Intel acquisition failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Intel request failed");
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
        setFeedbackResults(data.images, data.image_paths, data.scores);
      } else {
        setError(data.message || "Feedback processing failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Feedback request failed");
    } finally {
      setIsApplyingFeedback(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {!missionReady && <MissionBriefing onComplete={handleMissionComplete} />}

      <NeuralBackground />

      {/* Scan lines overlay */}
      <div className="scan-lines" />

      {/* Classified watermark */}
      <div className="classified-watermark" />

      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="relative z-10 mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {activeTab === "search" && (
          <div className="space-y-6">
            {store.images.length === 0 && !store.isSearching && (
              <div className="flex flex-col items-center gap-4 pb-6 pt-16 text-center">
                <div className="mb-2">
                  <p className="font-mono text-[10px] tracking-[0.4em] text-red-600/40 uppercase mb-3">
                    CLASSIFICATION: TOP SECRET // SI
                  </p>
                  <h2 className="font-rajdhani text-4xl font-bold tracking-[0.1em] text-red-500 sm:text-5xl uppercase">
                    MISSION BRIEFING
                  </h2>
                </div>
                <p className="max-w-xl text-neutral-500 text-sm font-mono leading-relaxed tracking-wider">
                  Search visual intelligence assets using natural language descriptors.
                  Click to segment targets with SAM. Iteratively refine results via
                  Rocchio feedback protocol. All operations encrypted. Mission data
                  is classified.
                </p>
                <div className="flex items-center gap-4 mt-2">
                  <div className="h-[1px] w-16 bg-red-600/20" />
                  <span className="text-[10px] font-mono text-red-600/30 tracking-[0.3em] uppercase">
                    Begin operations below
                  </span>
                  <div className="h-[1px] w-16 bg-red-600/20" />
                </div>
              </div>
            )}

            <SearchBar onSearch={handleSearch} />

            <ErrorBanner />

            <ImageGallery />

            <SAM3Panel />

            <FeedbackPanel onApply={handleApplyFeedback} />

            <CaptionPanel />

            <HistoryPanel />
          </div>
        )}

        {activeTab === "dashboard" && <ServerDashboard />}
      </main>
    </div>
  );
}
