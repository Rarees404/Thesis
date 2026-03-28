"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  Loader2,
  Eye,
  EyeOff,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { ollamaStatus } from "@/lib/api";
import { VanishInput } from "@/components/ui/vanish-input";

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
  const samAnnotations = useAppStore((s) => s.samAnnotations);

  const [ollamaAvailable, setOllamaAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    ollamaStatus()
      .then((s) => { if (!cancelled) setOllamaAvailable(s.available); })
      .catch(() => { if (!cancelled) setOllamaAvailable(false); });
    return () => { cancelled = true; };
  }, []);

  if (images.length === 0) return null;

  const totalBoxes = images.reduce((sum, img) => sum + img.boxes.length, 0);
  const samCount = samAnnotations.size;
  const hasAnyFeedback =
    totalBoxes > 0 ||
    samCount > 0 ||
    relevantCaptions.trim() ||
    irrelevantCaptions.trim();

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Sparkles className="h-4 w-4 text-indigo-400" />
          Relevance Feedback
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-white/40">
            Click results to segment with SAM (green = relevant, red = irrelevant).
            {ollamaAvailable
              ? " Vision analysis will auto-caption your selections for stronger retrieval."
              : " Optional text hints below also steer retrieval."}
          </p>
          {ollamaAvailable !== null && (
            <span
              className={`ml-3 flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                ollamaAvailable
                  ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
                  : "bg-white/5 text-white/30 ring-1 ring-white/10"
              }`}
              title={ollamaAvailable ? "Ollama Vision is running — SAM crops are auto-captioned" : "Ollama Vision unavailable — using image-only feedback"}
            >
              {ollamaAvailable ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
              {ollamaAvailable ? "Vision ON" : "Vision OFF"}
            </span>
          )}
        </div>

        <Separator className="bg-white/[0.06]" />

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-green-400">
              <ThumbsUp className="h-3.5 w-3.5" />
              Relevant details
            </label>
            <VanishInput
              placeholders={[
                "e.g. person on bicycle, red jacket...",
                "e.g. golden hour lighting, warm tones...",
                "e.g. close-up, sharp focus, vivid colors...",
              ]}
              value={relevantCaptions}
              onChange={(e) => setRelevantCaptions(e.target.value)}
              disabled={isApplyingFeedback}
              accentColor="green"
              hideSubmitButton
              className="h-11"
            />
          </div>
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-red-400">
              <ThumbsDown className="h-3.5 w-3.5" />
              Irrelevant details
            </label>
            <VanishInput
              placeholders={[
                "e.g. cars, buildings, background...",
                "e.g. indoor scenes, dark lighting...",
                "e.g. crowds, blurry, low quality...",
              ]}
              value={irrelevantCaptions}
              onChange={(e) => setIrrelevantCaptions(e.target.value)}
              disabled={isApplyingFeedback}
              accentColor="red"
              hideSubmitButton
              className="h-11"
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
          <div className="flex gap-3 text-xs text-white/30">
            {samCount > 0 && (
              <span className="text-indigo-400">
                {samCount} SAM mask{samCount !== 1 && "s"}
              </span>
            )}
            {totalBoxes > 0 && (
              <span>{totalBoxes} bounding box{totalBoxes !== 1 && "es"}</span>
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
