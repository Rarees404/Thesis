"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  ThumbsUp,
  ThumbsDown,
  Crosshair,
  Loader2,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
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

  if (images.length === 0) return null;

  const totalBoxes = images.reduce((sum, img) => sum + img.boxes.length, 0);
  const samCount = samAnnotations.size;
  const hasAnyFeedback =
    totalBoxes > 0 ||
    samCount > 0 ||
    relevantCaptions.trim() ||
    irrelevantCaptions.trim();

  return (
    <Card className="border-red-600/15">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Crosshair className="h-4 w-4 text-red-500" />
          INTEL ASSESSMENT
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-neutral-500">
          Click on intel assets above to mark targets (green) and exclusions (red)
          using SAM segmentation. Add text descriptions below to refine mission parameters.
        </p>

        <Separator className="bg-red-600/10" />

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <label className="flex items-center gap-1.5 text-xs font-rajdhani font-semibold tracking-wider text-green-400 uppercase">
              <ThumbsUp className="h-3.5 w-3.5" />
              Target details
            </label>
            <VanishInput
              placeholders={[
                "e.g. person on bicycle, red jacket...",
                "e.g. warm lighting, daylight conditions...",
                "e.g. close-up, high resolution target...",
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
            <label className="flex items-center gap-1.5 text-xs font-rajdhani font-semibold tracking-wider text-red-400 uppercase">
              <ThumbsDown className="h-3.5 w-3.5" />
              Exclude details
            </label>
            <VanishInput
              placeholders={[
                "e.g. vehicles, structures, terrain...",
                "e.g. indoor environments, low-vis...",
                "e.g. crowds, degraded imagery...",
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
            className="h-4 w-4 border-red-600/30 bg-black/40 accent-red-600"
          />
          <label htmlFor="fuse-query" className="text-sm text-neutral-500 font-mono text-xs tracking-wider uppercase">
            Fuse with initial query (prevents mission drift)
          </label>
        </div>

        <div className="flex items-center justify-between pt-2">
          <div className="flex gap-3 text-xs font-mono text-neutral-600 uppercase tracking-wider">
            {samCount > 0 && (
              <span className="text-red-400">
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
            className="gap-2 bg-red-600 hover:bg-red-500 border border-red-500/30 font-rajdhani tracking-[0.15em] uppercase text-xs font-semibold"
          >
            {isApplyingFeedback ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Crosshair className="h-4 w-4" />
            )}
            EXECUTE FEEDBACK
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
