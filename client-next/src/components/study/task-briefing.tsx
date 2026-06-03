"use client";

import { Loader2, Play, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { StudyTask } from "@/lib/study/types";

interface TaskBriefingProps {
  task: StudyTask;
  taskIndex: number;
  total: number;
  loading?: boolean;
  onStart: () => void;
}

export function TaskBriefing({
  task,
  taskIndex,
  total,
  loading,
  onStart,
}: TaskBriefingProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg overflow-hidden rounded-lg border border-border bg-card shadow-2xl">
        {/* Header strip */}
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <span className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            <Target className="h-3.5 w-3.5" />
            The Archivist
          </span>
          <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
            Case {taskIndex + 1} / {total}
          </span>
        </div>

        <div className="space-y-5 px-5 py-5">
          {/* Target image + story */}
          <div className="flex gap-4">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${task.target_b64}`}
              alt="Target to find"
              className="h-28 w-28 shrink-0 rounded-md border border-border object-cover"
              draggable={false}
            />
            <div className="space-y-2">
              <p className="font-mono text-[10px] uppercase tracking-wider text-amber-400/80">
                Your target
              </p>
              <p className="text-[13.5px] leading-relaxed text-foreground/90">
                {task.story}
              </p>
              <p className="font-mono text-[11px] text-muted-foreground/60">
                Hint:{" "}
                <span className="text-foreground/60">{task.query}</span>
              </p>
            </div>
          </div>

          {/* Instructions */}
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
          <Button
            onClick={onStart}
            disabled={loading}
            className="h-9 gap-1.5 px-4"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            Start — begin logging
          </Button>
        </div>
      </div>
    </div>
  );
}
