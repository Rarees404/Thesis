"use client";

import { Check, Download, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface StudyCompleteProps {
  participantId: string;
  taskCount: number;
  onDownload: () => void;
  onRestart: () => void;
}

export function StudyComplete({
  participantId,
  taskCount,
  onDownload,
  onRestart,
}: StudyCompleteProps) {
  return (
    <div className="mx-auto max-w-md space-y-6 py-16 text-center">
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-emerald-500/40 bg-emerald-500/[0.08]">
        <Check className="h-6 w-6 text-emerald-400" />
      </div>
      <div className="space-y-2">
        <h1 className="text-2xl font-medium tracking-tight text-foreground">
          Thank you!
        </h1>
        <p className="text-[14px] leading-relaxed text-muted-foreground">
          Your {taskCount} task{taskCount === 1 ? "" : "s"} and questionnaire are
          recorded under participant{" "}
          <span className="font-mono text-foreground/90">
            {participantId || "anon"}
          </span>
          . You can download a personal backup below.
        </p>
      </div>
      <div className="flex items-center justify-center gap-2">
        <Button onClick={onDownload} variant="outline" className="h-9 gap-1.5 px-4">
          <Download className="h-3.5 w-3.5" />
          Download my data
        </Button>
        <Button onClick={onRestart} variant="ghost" className="h-9 gap-1.5 px-4">
          <RotateCcw className="h-3.5 w-3.5" />
          New session
        </Button>
      </div>
    </div>
  );
}
