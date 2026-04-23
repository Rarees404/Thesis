"use client";

import { useEffect, useState, useMemo } from "react";
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
  Lightbulb,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { ollamaStatus, samStatus } from "@/lib/api";
import { VanishInput } from "@/components/ui/vanish-input";

interface FeedbackPanelProps {
  onApply: () => void;
}

export function FeedbackPanel({ onApply }: FeedbackPanelProps) {
  const images               = useAppStore((s) => s.images);
  const relevantCaptions     = useAppStore((s) => s.relevantCaptions);
  const irrelevantCaptions   = useAppStore((s) => s.irrelevantCaptions);
  const setRelevantCaptions  = useAppStore((s) => s.setRelevantCaptions);
  const setIrrelevantCaptions= useAppStore((s) => s.setIrrelevantCaptions);
  const isApplyingFeedback   = useAppStore((s) => s.isApplyingFeedback);
  const fuseInitialQuery     = useAppStore((s) => s.fuseInitialQuery);
  const setFuseInitialQuery  = useAppStore((s) => s.setFuseInitialQuery);
  const samAnnotations       = useAppStore((s) => s.samAnnotations);

  const [ollamaAvailable, setOllamaAvailable] = useState<boolean | null>(null);
  const [samLoaded, setSamLoaded]             = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    ollamaStatus()
      .then((s) => { if (!cancelled) setOllamaAvailable(s.available); })
      .catch(() => { if (!cancelled) setOllamaAvailable(false); });
    samStatus()
      .then((s) => { if (!cancelled) setSamLoaded(s.loaded); })
      .catch(() => { if (!cancelled) setSamLoaded(false); });
    return () => { cancelled = true; };
  }, []);

  // Collect cached Ollama captions from all annotated regions
  const aiSuggestions = useMemo(() => {
    const relevant: string[]   = [];
    const irrelevant: string[] = [];
    samAnnotations.forEach((ann) => {
      if (!ann.cached_caption) return;
      const isRel = ann.points.some((p) => p.label === 1);
      if (isRel) relevant.push(ann.cached_caption);
      else       irrelevant.push(ann.cached_caption);
    });
    return { relevant, irrelevant };
  }, [samAnnotations]);

  if (images.length === 0) return null;

  const totalBoxes  = images.reduce((sum, img) => sum + img.boxes.length, 0);
  const samWithMask = Array.from(samAnnotations.values()).filter((a) => a.mask_rle).length;
  const hasAnyFeedback =
    totalBoxes > 0 ||
    samWithMask > 0 ||
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

        {/* ── How-to + status badges ── */}
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-2">
            <div className="flex items-start gap-2 text-sm text-white/50">
              <Info className="h-4 w-4 mt-0.5 shrink-0 text-indigo-400" />
              <div className="space-y-1">
                <p className="font-medium text-white/70">How to give feedback:</p>
                <ol className="list-decimal list-inside space-y-0.5 text-white/40">
                  <li><span className="text-green-400">Click objects</span> you want <span className="text-green-400">more</span> of (green mode)</li>
                  <li><span className="text-red-400">Switch to irrelevant</span> and click things you want <span className="text-red-400">less</span> of</li>
                  <li>Optionally type text hints — AI Vision auto-captions your selections</li>
                  <li>Hit <strong className="text-white/70">Apply Feedback</strong></li>
                </ol>
              </div>
            </div>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1.5">
            {samLoaded !== null && (
              <span
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                  samLoaded
                    ? "bg-violet-500/10 text-violet-400 ring-1 ring-violet-500/20"
                    : "bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20"
                }`}
                title={samLoaded ? "SAM 3 ready" : "SAM 3 loading…"}
              >
                {samLoaded ? "SAM Ready" : "SAM Loading…"}
              </span>
            )}
            {ollamaAvailable !== null && (
              <span
                className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                  ollamaAvailable
                    ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
                    : "bg-white/5 text-white/30 ring-1 ring-white/10"
                }`}
                title={
                  ollamaAvailable
                    ? "Ollama Vision running — regions auto-captioned in background after each click"
                    : "Ollama Vision not running — run: ollama serve && ollama pull llama3.2-vision"
                }
              >
                {ollamaAvailable ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                {ollamaAvailable ? "AI Vision ON" : "AI Vision OFF"}
              </span>
            )}
          </div>
        </div>

        <Separator className="bg-white/[0.06]" />

        {/* ── AI caption suggestions (appears when Ollama has described regions) ── */}
        {(aiSuggestions.relevant.length > 0 || aiSuggestions.irrelevant.length > 0) && (
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 space-y-2">
            <p className="flex items-center gap-1.5 text-xs font-medium text-amber-400">
              <Lightbulb className="h-3.5 w-3.5" />
              AI Vision described your selections
            </p>
            {aiSuggestions.relevant.length > 0 && (
              <div className="space-y-1">
                <p className="text-[10px] text-green-400/70 uppercase tracking-wider">Wants more of</p>
                {aiSuggestions.relevant.map((cap, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <p className="flex-1 text-[11px] text-white/60 leading-snug italic">&ldquo;{cap}&rdquo;</p>
                    {!relevantCaptions && (
                      <button
                        className="shrink-0 text-[10px] text-green-400/70 hover:text-green-400 transition-colors"
                        onClick={() => setRelevantCaptions(cap)}
                        title="Use this as your positive hint"
                      >
                        Use
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
            {aiSuggestions.irrelevant.length > 0 && (
              <div className="space-y-1">
                <p className="text-[10px] text-red-400/70 uppercase tracking-wider">Wants less of</p>
                {aiSuggestions.irrelevant.map((cap, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <p className="flex-1 text-[11px] text-white/60 leading-snug italic">&ldquo;{cap}&rdquo;</p>
                    {!irrelevantCaptions && (
                      <button
                        className="shrink-0 text-[10px] text-red-400/70 hover:text-red-400 transition-colors"
                        onClick={() => setIrrelevantCaptions(cap)}
                        title="Use this as your negative hint"
                      >
                        Use
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Text hint inputs ── */}
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-sm font-medium text-green-400">
              <ThumbsUp className="h-3.5 w-3.5" />
              What do you want more of?
              {ollamaAvailable && (
                <span className="ml-auto text-[10px] font-normal text-emerald-500/70 bg-emerald-500/10 px-1.5 py-0.5 rounded-full ring-1 ring-emerald-500/20">
                  AI Vision
                </span>
              )}
            </label>
            <VanishInput
              placeholders={[
                "e.g. person on bicycle, red jacket…",
                "e.g. golden hour lighting, warm tones…",
                "e.g. close-up, sharp focus, vivid colors…",
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
              {ollamaAvailable && (
                <span className="ml-auto text-[10px] font-normal text-red-500/70 bg-red-500/10 px-1.5 py-0.5 rounded-full ring-1 ring-red-500/20">
                  AI Vision
                </span>
              )}
            </label>
            <VanishInput
              placeholders={[
                "e.g. cars, buildings, background…",
                "e.g. indoor scenes, dark lighting…",
                "e.g. crowds, blurry, low quality…",
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

        {/* ── Fuse initial query checkbox ── */}
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

        {/* ── Stats + Apply button ── */}
        <div className="flex items-center justify-between pt-2">
          <div className="flex gap-3 text-xs text-white/30">
            {samWithMask > 0 && (
              <span className="text-indigo-400">
                {samWithMask} region{samWithMask !== 1 && "s"} selected
              </span>
            )}
            {totalBoxes > 0 && (
              <span>{totalBoxes} bounding box{totalBoxes !== 1 && "es"}</span>
            )}
            {aiSuggestions.relevant.length + aiSuggestions.irrelevant.length > 0 && (
              <span className="text-amber-400/70">
                {aiSuggestions.relevant.length + aiSuggestions.irrelevant.length} AI caption
                {aiSuggestions.relevant.length + aiSuggestions.irrelevant.length !== 1 && "s"} ready
              </span>
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
