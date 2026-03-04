"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  Loader2,
} from "lucide-react";
import { useAppStore } from "@/lib/store";

interface FeedbackPanelProps {
  onApply: () => void;
}

export function FeedbackPanel({ onApply }: FeedbackPanelProps) {
  const images = useAppStore((s) => s.images);
  const relevantCaptions = useAppStore((s) => s.relevantCaptions);
  const irrelevantCaptions = useAppStore((s) => s.irrelevantCaptions);
  const setRelevantCaptions = useAppStore((s) => s.setRelevantCaptions);
  const setIrrelevantCaptions = useAppStore((s) => s.setIrrelevantCaptions);
  const isApplyingFeedback = useAppStore((s) => s.isApplyingFeedback);
  const fuseInitialQuery = useAppStore((s) => s.fuseInitialQuery);
  const setFuseInitialQuery = useAppStore((s) => s.setFuseInitialQuery);

  if (images.length === 0) return null;

  const totalBoxes = images.reduce((sum, img) => sum + img.boxes.length, 0);
  const hasAnyFeedback = totalBoxes > 0 || relevantCaptions.trim() || irrelevantCaptions.trim();

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="h-4 w-4 text-indigo-400" />
          Relevance Feedback
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-white/40">
          Draw bounding boxes on the images above to annotate relevant (green) and
          irrelevant (red) regions. Optionally provide text descriptions below.
        </p>

        <Separator className="bg-white/[0.06]" />

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-green-400">
              <ThumbsUp className="h-3.5 w-3.5" />
              Relevant details
            </label>
            <input
              placeholder="e.g., person on bicycle, red jacket..."
              value={relevantCaptions}
              onChange={(e) => setRelevantCaptions(e.target.value)}
              disabled={isApplyingFeedback}
              className="h-9 w-full rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 text-sm text-white placeholder:text-white/25 backdrop-blur-xl focus:border-green-500/40 focus:outline-none focus:ring-1 focus:ring-green-500/20 disabled:opacity-50"
            />
          </div>
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-red-400">
              <ThumbsDown className="h-3.5 w-3.5" />
              Irrelevant details
            </label>
            <input
              placeholder="e.g., cars, buildings, background..."
              value={irrelevantCaptions}
              onChange={(e) => setIrrelevantCaptions(e.target.value)}
              disabled={isApplyingFeedback}
              className="h-9 w-full rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 text-sm text-white placeholder:text-white/25 backdrop-blur-xl focus:border-red-500/40 focus:outline-none focus:ring-1 focus:ring-red-500/20 disabled:opacity-50"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="fuse-query"
            checked={fuseInitialQuery}
            onChange={(e) => setFuseInitialQuery(e.target.checked)}
            className="h-4 w-4 rounded border-white/20 bg-white/5 accent-indigo-500"
          />
          <label htmlFor="fuse-query" className="text-sm text-white/40">
            Fuse with initial query (prevents drift across multiple rounds)
          </label>
        </div>

        <div className="flex items-center justify-between pt-2">
          <div className="text-xs text-white/30">
            {totalBoxes > 0 && (
              <span>{totalBoxes} bounding box{totalBoxes !== 1 && "es"} drawn</span>
            )}
          </div>
          <Button
            onClick={onApply}
            disabled={!hasAnyFeedback || isApplyingFeedback}
            className="gap-2 bg-indigo-600 hover:bg-indigo-500 border border-indigo-500/30"
          >
            {isApplyingFeedback ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Apply Feedback
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
