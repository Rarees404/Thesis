"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Scan,
  AlertTriangle,
  CheckCircle2,
  Trash2,
  Loader2,
  Layers,
  Eye,
  EyeOff,
  MousePointer2,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { samStatus as fetchSamStatus, segmentText } from "@/lib/api";
import { VanishInput } from "@/components/ui/vanish-input";
import type { SegmentTextInstance } from "@/lib/types";

interface TextMaskResult {
  imageIndex: number;
  instances: SegmentTextInstance[];
  label: "Relevant" | "Irrelevant";
}

export function SAM3Panel() {
  const images = useAppStore((s) => s.images);
  const samAnnotations = useAppStore((s) => s.samAnnotations);
  const clearAllSamAnnotations = useAppStore((s) => s.clearAllSamAnnotations);
  const setSamAnnotation = useAppStore((s) => s.setSamAnnotation);

  const [samLoaded, setSamLoaded] = useState<boolean | null>(null);
  const [samType, setSamType] = useState<string | null>(null);
  const [relevantPrompt, setRelevantPrompt] = useState("");
  const [irrelevantPrompt, setIrrelevantPrompt] = useState("");
  const [segmenting, setSegmenting] = useState(false);
  const [textResults, setTextResults] = useState<TextMaskResult[]>([]);
  const [showMasks, setShowMasks] = useState(true);

  useEffect(() => {
    fetchSamStatus()
      .then((s) => {
        setSamLoaded(s.loaded);
        setSamType(s.model_type);
      })
      .catch(() => setSamLoaded(false));
  }, []);

  const handleSegmentText = useCallback(async () => {
    if (!relevantPrompt.trim() && !irrelevantPrompt.trim()) return;
    setSegmenting(true);
    setTextResults([]);

    try {
      const results: TextMaskResult[] = [];

      for (let i = 0; i < images.length; i++) {
        if (relevantPrompt.trim()) {
          const resp = await segmentText(images[i].path, relevantPrompt.trim());
          if (resp.instances.length > 0) {
            results.push({ imageIndex: i, instances: resp.instances, label: "Relevant" });
            const bestInstance = resp.instances.reduce((a, b) => (a.score > b.score ? a : b));
            setSamAnnotation(i, {
              points: [{ x: 0, y: 0, label: 1 }],
              mask_rle: bestInstance.mask_rle,
              region_b64: bestInstance.region_b64,
              score: bestInstance.score,
            });
          }
        }

        if (irrelevantPrompt.trim()) {
          const resp = await segmentText(images[i].path, irrelevantPrompt.trim());
          if (resp.instances.length > 0) {
            results.push({ imageIndex: i, instances: resp.instances, label: "Irrelevant" });
          }
        }
      }

      setTextResults(results);
    } catch (e) {
      console.error("SAM3 text segmentation failed:", e);
    } finally {
      setSegmenting(false);
    }
  }, [images, relevantPrompt, irrelevantPrompt, setSamAnnotation]);

  if (images.length === 0) return null;

  const clickAnnotatedCount = samAnnotations.size;
  const totalClickPoints = Array.from(samAnnotations.values()).reduce(
    (sum, a) => sum + a.points.length,
    0
  );
  const totalTextInstances = textResults.reduce((sum, r) => sum + r.instances.length, 0);

  return (
    <Card className="border-red-600/15">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Scan className="h-4 w-4 text-red-500" />
          SEGMENTATION MODULE
          {samLoaded === true && samType === "sam3" && (
            <Badge variant="outline" className="text-[10px] gap-1 text-green-400 border-green-400/30">
              <CheckCircle2 className="h-3 w-3" /> SAM 3
            </Badge>
          )}
          {samLoaded === true && samType === "sam2" && (
            <Badge variant="outline" className="text-[10px] gap-1 text-amber-400 border-amber-400/30">
              <CheckCircle2 className="h-3 w-3" /> SAM 2
            </Badge>
          )}
          {samLoaded === false && (
            <Badge variant="outline" className="text-[10px] gap-1 text-red-400 border-red-400/30">
              <AlertTriangle className="h-3 w-3" /> OFFLINE
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {samLoaded === false && (
          <p className="text-sm text-amber-400/80 font-mono">
            SEGMENTATION MODULE OFFLINE. Deploy checkpoint to{" "}
            <code className="bg-red-600/10 px-1.5 py-0.5 text-xs border border-red-600/20">server/checkpoints/</code>{" "}
            and restart operations.
          </p>
        )}

        <div className="space-y-3">
          <p className="text-sm text-neutral-500">
            <span className="text-red-400 font-rajdhani font-semibold tracking-wider uppercase">Text prompts</span> — Describe a target
            and SAM 3 will identify all matching instances across all intel assets.
            {samType === "sam2" && (
              <span className="text-amber-400/80 ml-1 font-mono text-xs">
                (REQUIRES SAM 3 — PENDING ACCESS CLEARANCE)
              </span>
            )}
          </p>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1.5">
              <label className="text-xs font-rajdhani font-semibold tracking-wider text-green-400 uppercase">Target concept</label>
              <VanishInput
                placeholders={[
                  "e.g. person on bicycle",
                  "e.g. red vehicle",
                  "e.g. hostile equipment",
                ]}
                value={relevantPrompt}
                onChange={(e) => setRelevantPrompt(e.target.value)}
                disabled={segmenting}
                accentColor="green"
                hideSubmitButton
                className="h-10"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-rajdhani font-semibold tracking-wider text-red-400 uppercase">Exclude concept</label>
              <VanishInput
                placeholders={[
                  "e.g. buildings, terrain",
                  "e.g. civilian crowd",
                  "e.g. stationary vehicles",
                ]}
                value={irrelevantPrompt}
                onChange={(e) => setIrrelevantPrompt(e.target.value)}
                disabled={segmenting}
                accentColor="red"
                hideSubmitButton
                className="h-10"
              />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button
                onClick={handleSegmentText}
                disabled={segmenting || samType !== "sam3" || (!relevantPrompt.trim() && !irrelevantPrompt.trim())}
                className="gap-2 bg-red-600/80 hover:bg-red-600 border border-red-500/30 font-rajdhani tracking-wider uppercase text-xs"
                variant="secondary"
              >
                {segmenting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Scan className="h-4 w-4" />
                )}
                {segmenting ? "SCANNING..." : "EXECUTE SAM 3"}
              </Button>
              {textResults.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowMasks((v) => !v)}
                  className="gap-1.5 text-xs font-mono"
                >
                  {showMasks ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  {showMasks ? "HIDE" : "SHOW"} MASKS
                </Button>
              )}
            </div>
            {totalTextInstances > 0 && (
              <div className="flex items-center gap-1.5 text-xs font-mono text-neutral-500">
                <Layers className="h-3.5 w-3.5" />
                {totalTextInstances} INSTANCE{totalTextInstances !== 1 && "S"} DETECTED
              </div>
            )}
          </div>
        </div>

        {textResults.length > 0 && showMasks && (
          <>
            <Separator className="bg-red-600/10" />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {images.map((img, i) => {
                const results = textResults.filter((r) => r.imageIndex === i);
                const allInstances = results.flatMap((r) =>
                  r.instances.map((inst) => ({ ...inst, resultLabel: r.label }))
                );
                if (allInstances.length === 0) return null;

                return (
                  <div key={i} className="relative overflow-hidden border border-red-600/15">
                    <div className="relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${img.base64}`}
                        alt={`SAM result ${i + 1}`}
                        className="block w-full aspect-square object-contain"
                        draggable={false}
                      />
                      {allInstances.map((inst, mi) => (
                        <div
                          key={mi}
                          className="absolute inset-0 pointer-events-none"
                          style={{
                            border: `2px solid ${inst.resultLabel === "Relevant" ? "#22c55e" : "#ef4444"}`,
                          }}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`data:image/png;base64,${inst.region_b64}`}
                            alt={`mask ${mi}`}
                            className="absolute inset-0 h-full w-full opacity-40"
                          />
                        </div>
                      ))}
                    </div>
                    <div className="absolute bottom-1 left-1 flex flex-wrap gap-1">
                      {allInstances.map((inst, mi) => (
                        <Badge
                          key={mi}
                          variant="secondary"
                          className={`text-[9px] border-0 ${
                            inst.resultLabel === "Relevant"
                              ? "bg-green-600/80 text-white"
                              : "bg-red-600/80 text-white"
                          }`}
                        >
                          {(inst.score * 100).toFixed(0)}%
                        </Badge>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}

        <Separator className="bg-red-600/10" />

        <div className="space-y-2">
          <p className="text-sm text-neutral-500">
            <MousePointer2 className="inline h-3.5 w-3.5 mr-1 text-red-400" />
            <span className="text-red-400 font-rajdhani font-semibold tracking-wider uppercase">Click prompts</span> — Click directly
            on intel assets above for precision targeting.{" "}
            <span className="text-green-400 font-mono text-xs">GREEN</span> = target,{" "}
            <span className="text-red-400 font-mono text-xs">RED</span> = exclude.
          </p>

          {clickAnnotatedCount > 0 && (
            <div className="flex items-center justify-between">
              <div className="flex gap-3 text-xs font-mono text-neutral-500 uppercase tracking-wider">
                <span>
                  <span className="text-amber-400 font-semibold">{clickAnnotatedCount}</span> asset{clickAnnotatedCount !== 1 && "s"} annotated
                </span>
                <span className="text-neutral-600">{totalClickPoints} click{totalClickPoints !== 1 && "s"} total</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearAllSamAnnotations}
                className="gap-1.5 text-xs text-neutral-500 hover:text-red-400 font-mono uppercase tracking-wider"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Clear all
              </Button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
