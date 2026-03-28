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
  Info,
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
          Refine Your Results
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-2">
            <div className="flex items-start gap-2 text-sm text-white/50">
              <Info className="h-4 w-4 mt-0.5 shrink-0 text-indigo-400" />
              <div className="space-y-1">
                <p className="font-medium text-white/70">How to give feedback:</p>
                <ol className="list-decimal list-inside space-y-0.5 text-white/40">
                  <li><span className="text-green-400">Click on objects</span> in the images you want <span className="text-green-400">more</span> of</li>
                  <li><span className="text-red-400">Switch to irrelevant mode</span> and click things you want <span className="text-red-400">less</span> of</li>
                  <li>Optionally type text hints below to guide the search</li>
                  <li>Hit <strong className="text-white/70">Apply Feedback</strong> to see improved results</li>
                </ol>
              </div>
            </div>
          </div>
          {ollamaAvailable !== null && (
            <span
              className={`flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                ollamaAvailable
                  ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
                  : "bg-white/5 text-white/30 ring-1 ring-white/10"
              }`}
              title={ollamaAvailable
                ? "Ollama Vision is running — your selections are auto-captioned for better results"
                : "Ollama Vision not running — using image embeddings only. Run: ollama serve && ollama pull llama3.2-vision"}
            >
              {ollamaAvailable ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
              {ollamaAvailable ? "AI Vision ON" : "AI Vision OFF"}
            </span>
          )}
        </div>

        <Separator className="bg-white/[0.06]" />

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-green-400">
              <ThumbsUp className="h-3.5 w-3.5" />
              What do you want more of?
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
              What do you want less of?
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
            Keep original search in mind (prevents drift across multiple rounds)
          </label>
        </div>

        <div className="flex items-center justify-between pt-2">
          <div className="flex gap-3 text-xs text-white/30">
            {samCount > 0 && (
              <span className="text-indigo-400">
                {samCount} region{samCount !== 1 && "s"} selected
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
