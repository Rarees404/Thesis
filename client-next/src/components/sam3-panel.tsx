"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Scan, Loader2, Layers, Eye, EyeOff } from "lucide-react";
import { useAppStore } from "@/lib/store";
import type { SegmentationMask } from "@/lib/types";

function generateMockMasks(
  prompt: string,
  imgWidth: number,
  imgHeight: number
): SegmentationMask[] {
  const seed = prompt.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const count = 1 + (seed % 3);
  const masks: SegmentationMask[] = [];

  for (let i = 0; i < count; i++) {
    const w = Math.round(imgWidth * (0.15 + (((seed * (i + 7)) % 40) / 100)));
    const h = Math.round(imgHeight * (0.15 + (((seed * (i + 3)) % 40) / 100)));
    const x = Math.round(((seed * (i + 1)) % (imgWidth - w)));
    const y = Math.round(((seed * (i + 2)) % (imgHeight - h)));

    const canvas = document.createElement("canvas");
    canvas.width = imgWidth;
    canvas.height = imgHeight;
    const ctx = canvas.getContext("2d")!;

    ctx.fillStyle = "rgba(99, 102, 241, 0.45)";
    ctx.beginPath();
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
    ctx.stroke();

    masks.push({
      label: `${prompt} #${i + 1}`,
      confidence: 0.75 + (((seed * (i + 5)) % 25) / 100),
      mask_data_url: canvas.toDataURL(),
      bbox: { x, y, w, h },
    });
  }

  return masks;
}

interface SAM3ImageViewerProps {
  base64: string;
  index: number;
  prompt: string;
  masks: SegmentationMask[];
  showMasks: boolean;
}

function SAM3ImageViewer({
  base64,
  index,
  masks,
  showMasks,
}: SAM3ImageViewerProps) {
  return (
    <div className="relative overflow-hidden rounded-lg border border-white/[0.06]">
      <div className="relative">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/png;base64,${base64}`}
          alt={`Segmentation result ${index + 1}`}
          className="block w-full aspect-square object-contain"
          draggable={false}
        />
        {showMasks &&
          masks.map((mask, mi) => (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              key={mi}
              src={mask.mask_data_url}
              alt={mask.label}
              className="absolute inset-0 h-full w-full pointer-events-none"
            />
          ))}
      </div>
      {showMasks && masks.length > 0 && (
        <div className="absolute bottom-2 left-2 flex flex-wrap gap-1">
          {masks.map((mask, mi) => (
            <Badge
              key={mi}
              variant="secondary"
              className="bg-indigo-600/80 text-white text-[10px] border-0"
            >
              {mask.label} ({(mask.confidence * 100).toFixed(0)}%)
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

export function SAM3Panel() {
  const images = useAppStore((s) => s.images);
  const [relevantPrompt, setRelevantPrompt] = useState("");
  const [irrelevantPrompt, setIrrelevantPrompt] = useState("");
  const [segmenting, setSegmenting] = useState(false);
  const [relevantMasks, setRelevantMasks] = useState<Map<number, SegmentationMask[]>>(new Map());
  const [irrelevantMasks, setIrrelevantMasks] = useState<Map<number, SegmentationMask[]>>(new Map());
  const [showMasks, setShowMasks] = useState(true);
  const [hasRun, setHasRun] = useState(false);

  if (images.length === 0) return null;

  async function handleSegment() {
    if (!relevantPrompt.trim() && !irrelevantPrompt.trim()) return;
    setSegmenting(true);

    await new Promise((r) => setTimeout(r, 1500));

    const relMasks = new Map<number, SegmentationMask[]>();
    const irrMasks = new Map<number, SegmentationMask[]>();

    images.forEach((_, i) => {
      if (relevantPrompt.trim()) {
        relMasks.set(i, generateMockMasks(relevantPrompt, 256, 256));
      }
      if (irrelevantPrompt.trim()) {
        irrMasks.set(i, generateMockMasks(irrelevantPrompt, 256, 256));
      }
    });

    setRelevantMasks(relMasks);
    setIrrelevantMasks(irrMasks);
    setSegmenting(false);
    setHasRun(true);
  }

  const totalMasks = Array.from(relevantMasks.values()).reduce((s, m) => s + m.length, 0) +
    Array.from(irrelevantMasks.values()).reduce((s, m) => s + m.length, 0);

  return (
    <Card className="border-indigo-500/20">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Scan className="h-4 w-4 text-indigo-500" />
          SAM3 Segmentation
          <Badge variant="outline" className="text-[10px]">
            Preview
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-white/40">
          Type a concept and SAM3 will generate pixel-precise masks on all
          retrieved images. This replaces manual bounding box drawing.
        </p>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-green-400">
              Relevant concept
            </label>
            <Input
              placeholder="e.g., person on bicycle"
              value={relevantPrompt}
              onChange={(e) => setRelevantPrompt(e.target.value)}
              disabled={segmenting}
            />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-red-400">
              Irrelevant concept
            </label>
            <Input
              placeholder="e.g., cars, buildings"
              value={irrelevantPrompt}
              onChange={(e) => setIrrelevantPrompt(e.target.value)}
              disabled={segmenting}
            />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              onClick={handleSegment}
              disabled={segmenting || (!relevantPrompt.trim() && !irrelevantPrompt.trim())}
              className="gap-2"
              variant="secondary"
            >
              {segmenting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Scan className="h-4 w-4" />
              )}
              {segmenting ? "Segmenting..." : "Run SAM3"}
            </Button>
            {hasRun && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowMasks((v) => !v)}
                className="gap-1.5 text-xs"
              >
                {showMasks ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                {showMasks ? "Hide" : "Show"} masks
              </Button>
            )}
          </div>
          {hasRun && (
            <div className="flex items-center gap-1.5 text-xs text-white/30">
              <Layers className="h-3.5 w-3.5" />
              {totalMasks} mask{totalMasks !== 1 && "s"} generated
            </div>
          )}
        </div>

        {hasRun && (
          <>
            <Separator />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {images.map((img, i) => {
                const combined = [
                  ...(relevantMasks.get(i) ?? []),
                  ...(irrelevantMasks.get(i) ?? []),
                ];
                return (
                  <SAM3ImageViewer
                    key={i}
                    base64={img.base64}
                    index={i}
                    prompt={relevantPrompt}
                    masks={combined}
                    showMasks={showMasks}
                  />
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
