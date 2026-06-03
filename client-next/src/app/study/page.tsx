"use client";

import { useCallback } from "react";
import Link from "next/link";
import { Loader2, FlaskConical, ArrowLeft } from "lucide-react";

import { useAppStore } from "@/lib/store";
import { useStudyStore } from "@/lib/study/store";
import { searchImages, applyFeedback } from "@/lib/api";
import { buildBackup, downloadBackup } from "@/lib/study/logger";

import { SearchBar } from "@/components/search-bar";
import { ImageGallery } from "@/components/image-gallery";
import { FeedbackPanel } from "@/components/feedback-panel";
import { ErrorBanner } from "@/components/error-banner";
import { Button } from "@/components/ui/button";

import { TaskSelector } from "@/components/study/task-selector";
import { TaskBriefing } from "@/components/study/task-briefing";
import { TaskBanner } from "@/components/study/task-banner";
import { Questionnaire } from "@/components/study/questionnaire";
import { StudyComplete } from "@/components/study/study-complete";

// Number of search tasks each participant completes in one study session.
const TASK_COUNT = 3;
// Results shown per search during the study. Overrides the store default (5) via
// setTopK when a task starts, so every participant sees the same grid size.
const STUDY_TOP_K = 5;

function createSessionId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function StudyPage() {
  const phase = useStudyStore((s) => s.phase);
  const tasks = useStudyStore((s) => s.tasks);
  const taskIndex = useStudyStore((s) => s.taskIndex);
  const taskStartedAt = useStudyStore((s) => s.taskStartedAt);
  const loadingTasks = useStudyStore((s) => s.loadingTasks);
  const studyError = useStudyStore((s) => s.error);
  const answers = useStudyStore((s) => s.answers);
  const setAnswer = useStudyStore((s) => s.setAnswer);

  const round = useAppStore((s) => s.round);
  const currentTask = tasks[taskIndex];

  // ── Instrumented search ─────────────────────────────────────────────────────
  const handleSearch = useCallback(async () => {
    const { query, topK, setIsSearching, setSearchResults, setError } =
      useAppStore.getState();
    if (!query.trim()) return;
    setIsSearching(true);
    setError(null);
    const t0 = performance.now();
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
          data.session_id,
        );
        useStudyStore.getState().logEvent("search_submitted", {
          query,
          top_k: topK,
          latency_ms: Math.round(performance.now() - t0),
          result_paths: data.image_paths,
          scores: data.scores,
        });
      } else {
        setError(data.message || "Search failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search request failed");
    } finally {
      setIsSearching(false);
    }
  }, []);

  // ── Instrumented feedback ───────────────────────────────────────────────────
  const handleApplyFeedback = useCallback(async () => {
    const st = useAppStore.getState();
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
      imageLabels,
      setIsApplyingFeedback,
      setFeedbackResults,
      setError,
    } = st;

    setIsApplyingFeedback(true);
    setError(null);
    const t0 = performance.now();
    try {
      const boxesList = images.map((img) =>
        img.boxes.length > 0 ? img.boxes : null,
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
      const labelsList = images.map((_, i) => {
        if (samAnnotations.get(i)?.mask_rle) return null;
        return imageLabels.get(i) ?? null;
      });
      const hasLabels = labelsList.some((l) => l !== null);

      const data = await applyFeedback({
        query,
        top_k: topK,
        relevant_image_paths: imagePaths,
        relevant_captions: relevantCaptions,
        irrelevant_captions: irrelevantCaptions,
        annotator_json_boxes_list: boxesList,
        ...(hasSam ? { sam_annotations: samList } : {}),
        ...(hasLabels ? { image_labels: labelsList } : {}),
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
          data.session_id,
          data.hard_filter ?? null,
        );
        useStudyStore.getState().logEvent("feedback_applied", {
          round: useAppStore.getState().round,
          latency_ms: Math.round(performance.now() - t0),
          relevant_hint: relevantCaptions,
          irrelevant_hint: irrelevantCaptions,
          sam_regions: samList.filter((s) => s !== null).length,
          full_labels: labelsList.filter((l) => l !== null).length,
          result_paths: data.image_paths,
          scores: data.scores,
          hard_filter: data.hard_filter ?? null,
        });
      } else {
        setError(data.message || "Feedback failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Feedback request failed");
    } finally {
      setIsApplyingFeedback(false);
    }
  }, []);

  // ── Phase transitions ───────────────────────────────────────────────────────
  const handleSelect = useCallback((index: number, participantId: string) => {
    useAppStore.getState().reset();
    useAppStore.getState().setTopK(STUDY_TOP_K);
    useStudyStore.getState().selectTask(index, participantId);
  }, []);

  const handleStart = useCallback(() => {
    useStudyStore.getState().startTask();
  }, []);

  const handleFinish = useCallback((outcome: "found" | "gave_up") => {
    const { round: r, imagePaths } = useAppStore.getState();
    useStudyStore.getState().finishTask(outcome, r, imagePaths);
    useAppStore.getState().reset();
  }, []);

  const handleDownload = useCallback(() => {
    const s = useStudyStore.getState();
    downloadBackup(
      s.participantId,
      buildBackup({
        participantId: s.participantId,
        sessionId: s.sessionId,
        eventLog: s.eventLog,
        results: s.results,
        answers: s.answers,
      }),
    );
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Slim study chrome */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/95">
        <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-6 lg:px-8">
          <span className="inline-flex items-center gap-2 text-[13px] font-medium tracking-tight">
            <FlaskConical className="h-4 w-4 text-amber-400" />
            VisualReF Study
          </span>
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="h-3 w-3" />
            exit
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10 lg:px-8 lg:py-12">

        {/* ── Idle: landing / intro ───────────────────────────────────────── */}
        {phase === "idle" && (
          <div className="mx-auto max-w-md space-y-6 py-16 text-center">
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              User study
            </p>
            <h1 className="text-3xl font-medium tracking-tight">The Archivist</h1>
            <p className="text-[15px] leading-relaxed text-muted-foreground">
              Three cases await. You&apos;ll choose one mission: find a
              photograph that matches a target, using search, region clicks,
              and text hints. Then a quick questionnaire — about 10 minutes total.
            </p>
            {studyError && (
              <p className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-[13px] text-destructive">
                {studyError}
              </p>
            )}
            <Button
              onClick={() => useStudyStore.getState().loadTasks(TASK_COUNT)}
              disabled={loadingTasks}
              className="h-10 gap-2 px-5"
            >
              {loadingTasks && <Loader2 className="h-4 w-4 animate-spin" />}
              View missions
            </Button>
          </div>
        )}

        {/* ── Selecting: pick one mission ─────────────────────────────────── */}
        {phase === "selecting" && (
          <TaskSelector
            tasks={tasks}
            loading={loadingTasks}
            onSelect={handleSelect}
          />
        )}

        {/* ── Briefing: full task details before logging starts ───────────── */}
        {phase === "briefing" && currentTask && (
          <TaskBriefing
            task={currentTask}
            taskIndex={taskIndex}
            total={tasks.length}
            loading={loadingTasks}
            onStart={handleStart}
          />
        )}

        {/* ── Running: active search session ──────────────────────────────── */}
        {phase === "running" && currentTask && (
          <div className="space-y-6">
            <TaskBanner
              task={currentTask}
              taskIndex={taskIndex}
              total={tasks.length}
              round={round}
              startedAt={taskStartedAt}
              onFinish={handleFinish}
            />
            <SearchBar onSearch={handleSearch} />
            <ErrorBanner />
            <ImageGallery />
            <FeedbackPanel onApply={handleApplyFeedback} />
          </div>
        )}

        {/* ── Form: post-task questionnaire ───────────────────────────────── */}
        {phase === "form" && (
          <Questionnaire
            answers={answers}
            setAnswer={setAnswer}
            onSubmit={() => void useStudyStore.getState().submitForm()}
          />
        )}

        {/* ── Done: thank you + download ──────────────────────────────────── */}
        {phase === "done" && (
          <StudyComplete
            participantId={useStudyStore.getState().participantId}
            taskCount={useStudyStore.getState().results.length}
            onDownload={handleDownload}
            onRestart={() => useStudyStore.getState().reset()}
          />
        )}
      </main>
    </div>
  );
}
