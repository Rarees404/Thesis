"use client";

import { useState } from "react";
import { BookOpen, ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { StudyTask } from "@/lib/study/types";

interface TaskSelectorProps {
  tasks: StudyTask[];
  loading?: boolean;
  onSelect: (index: number, participantId: string) => void;
}

export function TaskSelector({ tasks, loading, onSelect }: TaskSelectorProps) {
  const [pid, setPid] = useState("");
  const [chosen, setChosen] = useState<number | null>(null);

  const canConfirm = pid.trim().length > 0 && chosen !== null;

  function handleConfirm() {
    if (chosen !== null && pid.trim()) {
      onSelect(chosen, pid.trim());
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-8 py-10">
      {/* Header */}
      <div className="space-y-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          The Archivist — User study
        </p>
        <h1 className="text-2xl font-medium tracking-tight text-foreground">
          Choose your mission
        </h1>
        <p className="text-[14px] leading-relaxed text-muted-foreground">
          Three cases have arrived in the archive. Pick the one you&apos;d like
          to investigate. You&apos;ll use the image search system to find a
          matching photograph. After that, a short questionnaire.
        </p>
      </div>

      {/* Participant ID */}
      <div className="space-y-1.5">
        <label className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Participant ID
        </label>
        <input
          type="text"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && canConfirm && handleConfirm()}
          placeholder="e.g. P01"
          className="h-10 w-full max-w-xs rounded-md border border-input bg-background px-3 text-[14px] text-foreground outline-none transition-colors focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
        />
        <p className="font-mono text-[10.5px] text-muted-foreground/60">
          Used to label your logs. Pick anything you&apos;ll remember.
        </p>
      </div>

      {/* Mission cards */}
      <div className="space-y-3">
        <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Select one mission
        </p>
        {tasks.map((task, i) => {
          const active = chosen === i;
          return (
            <button
              key={task.task_id}
              onClick={() => setChosen(i)}
              className={[
                "w-full overflow-hidden rounded-lg border text-left transition-colors duration-100",
                active
                  ? "border-amber-500/60 bg-amber-500/[0.06]"
                  : "border-border bg-card hover:border-border/80 hover:bg-card/80",
              ].join(" ")}
            >
              <div className="flex items-start gap-4 px-4 py-4">
                {/* Target thumbnail */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/png;base64,${task.target_b64}`}
                  alt={`Mission ${i + 1} target`}
                  className="h-20 w-20 shrink-0 rounded border border-border object-cover"
                  draggable={false}
                />

                <div className="min-w-0 flex-1 space-y-1.5 pt-0.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={[
                        "inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[11px] font-medium tabular-nums",
                        active
                          ? "border-amber-500/60 bg-amber-500/[0.12] text-amber-400"
                          : "border-border text-muted-foreground",
                      ].join(" ")}
                    >
                      {i + 1}
                    </span>
                    <span
                      className={[
                        "font-mono text-[10px] uppercase tracking-wider",
                        active ? "text-amber-400/80" : "text-muted-foreground/60",
                      ].join(" ")}
                    >
                      Case {i + 1}
                    </span>
                  </div>

                  {/* Story */}
                  <p className="text-[13px] leading-relaxed text-foreground/85">
                    {task.story}
                  </p>

                  {/* Hint query */}
                  <p className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground/60">
                    <BookOpen className="h-3 w-3" />
                    <span className="text-foreground/50">{task.query}</span>
                  </p>
                </div>

                {/* Selection indicator */}
                <div
                  className={[
                    "mt-1 shrink-0 transition-opacity duration-100",
                    active ? "opacity-100" : "opacity-0",
                  ].join(" ")}
                >
                  <ChevronRight className="h-4 w-4 text-amber-400" />
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Confirm button */}
      <div className="flex items-center justify-between border-t border-border pt-5">
        <span className="font-mono text-[11px] text-muted-foreground">
          {!pid.trim()
            ? "Enter your participant ID above"
            : chosen === null
            ? "Select a mission above"
            : `Ready — Case ${chosen + 1} selected`}
        </span>
        <Button
          onClick={handleConfirm}
          disabled={!canConfirm || loading}
          className="h-9 gap-1.5 px-4"
        >
          {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          Accept mission
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}
