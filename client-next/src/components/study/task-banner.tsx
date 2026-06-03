"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Clock, Flag, Target, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { StudyTask, TaskOutcome } from "@/lib/study/types";

interface TaskBannerProps {
  task: StudyTask;
  taskIndex: number;
  total: number;
  round: number;
  startedAt: number;
  onFinish: (outcome: TaskOutcome) => void;
}

function fmt(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

/** Full-detail overlay shown when the participant clicks the task thumbnail/story. */
function TaskDetailModal({
  task,
  taskIndex,
  total,
  onClose,
}: {
  task: StudyTask;
  taskIndex: number;
  total: number;
  onClose: () => void;
}) {
  // Close on backdrop click or Escape key
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg overflow-hidden rounded-lg border border-border bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <span className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            <Target className="h-3.5 w-3.5" />
            The Archivist &mdash; Task {taskIndex + 1} / {total}
          </span>
          <button
            onClick={onClose}
            className="rounded p-1 text-muted-foreground transition-colors hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-5 px-5 py-5">
          {/* Large target image + story */}
          <div className="flex gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${task.target_b64}`}
              alt="Target to find"
              className="h-36 w-36 shrink-0 rounded-md border border-border object-cover"
              draggable={false}
            />
            <div className="space-y-2">
              <p className="font-mono text-[10px] uppercase tracking-wider text-amber-400/80">
                Your target
              </p>
              <p className="text-[13.5px] leading-relaxed text-foreground/90">
                {task.story}
              </p>
              <p className="font-mono text-[11px] text-muted-foreground/70">
                Hint: <span className="text-foreground/70">{task.query}</span>
              </p>
            </div>
          </div>

          {/* Instructions reminder */}
          <div className="rounded-md border border-border bg-background/60 px-4 py-3 text-[12.5px] leading-relaxed text-muted-foreground">
            Search for an image that matches the target above. Start with a
            sentence, then{" "}
            <span className="text-foreground/90">
              click on objects inside the results
            </span>{" "}
            and add{" "}
            <span className="text-foreground/90">text hints</span> to refine.
            When you have found a good match, press{" "}
            <span className="text-foreground/90">&ldquo;I found it&rdquo;</span>.
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end border-t border-border px-5 py-3">
          <Button onClick={onClose} className="h-9 gap-1.5 px-4">
            Got it — back to search
          </Button>
        </div>
      </div>
    </div>
  );
}

export function TaskBanner({
  task,
  taskIndex,
  total,
  round,
  startedAt,
  onFinish,
}: TaskBannerProps) {
  const [elapsed, setElapsed] = useState(0);
  const [detailOpen, setDetailOpen] = useState(false);

  useEffect(() => {
    const tick = () => setElapsed(Date.now() - startedAt);
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [startedAt]);

  return (
    <>
      <div className="sticky top-12 z-30 mb-6 flex items-center gap-4 rounded-md border border-amber-500/30 bg-amber-500/[0.06] px-4 py-2.5">
        {/* Clickable left section: thumbnail + story */}
        <button
          onClick={() => setDetailOpen(true)}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 text-left"
          title="Click to review the task details"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${task.target_b64}`}
            alt="Target"
            className="h-11 w-11 shrink-0 rounded border border-border object-cover transition-opacity hover:opacity-80"
            draggable={false}
          />
          <div className="min-w-0">
            <p className="font-mono text-[10px] uppercase tracking-wider text-amber-400/80">
              Find this &middot; task {taskIndex + 1}/{total}{" "}
              <span className="normal-case text-muted-foreground/60">(click to review)</span>
            </p>
            <p className="truncate text-[12.5px] text-foreground/85">{task.story}</p>
          </div>
        </button>

        {/* Right side: timer, round, action buttons */}
        <div className="flex shrink-0 items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {fmt(elapsed)}
          </span>
          <span className="hidden sm:inline">round {Math.max(round, 0)}</span>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onFinish("gave_up")}
            className="h-8 gap-1.5 px-2 text-[12px] text-muted-foreground"
          >
            <Flag className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Give up</span>
          </Button>
          <Button
            size="sm"
            onClick={() => onFinish("found")}
            className="h-8 gap-1.5 bg-emerald-600 px-3 text-[12px] text-white hover:bg-emerald-500"
          >
            <CheckCircle2 className="h-3.5 w-3.5" />
            I found it
          </Button>
        </div>
      </div>

      {detailOpen && (
        <TaskDetailModal
          task={task}
          taskIndex={taskIndex}
          total={total}
          onClose={() => setDetailOpen(false)}
        />
      )}
    </>
  );
}
