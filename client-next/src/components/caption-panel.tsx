"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { MessageSquareText, Loader2, Zap, Clock, AlertCircle } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { captionImage, ollamaStatus } from "@/lib/api";
import type { CaptionResult } from "@/lib/types";

export function CaptionPanel() {
  const images = useAppStore((s) => s.images);
  const query = useAppStore((s) => s.query);
  const samAnnotations = useAppStore((s) => s.samAnnotations);
  const [generating, setGenerating] = useState(false);
  const [captions, setCaptions] = useState<Map<number, CaptionResult>>(new Map());
  const [error, setError] = useState<string | null>(null);

  if (images.length === 0) return null;

  const annotatedIndices = Array.from(samAnnotations.entries())
    .filter(([, ann]) => ann.region_b64)
    .map(([idx]) => idx);

  async function checkAndGenerate() {
    setGenerating(true);
    setError(null);

    try {
      const status = await ollamaStatus();
      if (!status.available) {
        setError(
          "Ollama is not running. Start it with: ollama serve, then pull the model: ollama pull llama3.2-vision"
        );
        setGenerating(false);
        return;
      }

      const results = new Map<number, CaptionResult>();

      const targets = annotatedIndices.length > 0
        ? annotatedIndices
        : images.map((_, i) => i);

      for (const i of targets) {
        const ann = samAnnotations.get(i);
        const imageB64 = ann?.region_b64 ?? images[i].base64;
        const hasRelevant = ann?.points?.some((p) => p.label === 1) ?? true;
        const label = hasRelevant ? "Relevant" : "Irrelevant";

        try {
          const result = await captionImage(imageB64, query, label);
          results.set(i, {
            caption: result.caption ?? "No caption generated",
            model: result.model,
            latency_ms: result.latency_ms,
          });
          setCaptions(new Map(results));
        } catch {
          results.set(i, {
            caption: "Caption failed — check Ollama connection",
            model: "error",
            latency_ms: 0,
          });
          setCaptions(new Map(results));
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to generate captions");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <Card className="border-amber-500/20">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquareText className="h-4 w-4 text-amber-500" />
          Ollama Vision Captions
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-white/40">
          Generates descriptions for each image region using Ollama&apos;s Llama
          3.2 Vision. Captions help the retrieval system understand what you
          marked as relevant or irrelevant.
        </p>

        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="secondary" className="gap-1 text-xs">
            <Zap className="h-3 w-3" />
            llama3.2-vision
          </Badge>
          {annotatedIndices.length > 0 && (
            <Badge variant="outline" className="text-xs">
              {annotatedIndices.length} segmented region{annotatedIndices.length !== 1 && "s"}
            </Badge>
          )}
          {query && (
            <Badge variant="outline" className="text-xs text-white/40">
              Query: &ldquo;{query}&rdquo;
            </Badge>
          )}
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/20 p-3 text-sm text-red-300">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            {error}
          </div>
        )}

        <Button
          onClick={checkAndGenerate}
          disabled={generating}
          variant="secondary"
          className="gap-2"
        >
          {generating ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <MessageSquareText className="h-4 w-4" />
          )}
          {generating ? "Generating captions..." : "Generate Captions"}
        </Button>

        {captions.size > 0 && (
          <>
            <Separator />
            <div className="space-y-3">
              {images.map((img, i) => {
                const cap = captions.get(i);
                if (!cap) return null;
                const ann = samAnnotations.get(i);
                const thumbSrc = ann?.region_b64
                  ? `data:image/png;base64,${ann.region_b64}`
                  : `data:image/png;base64,${img.base64}`;
                return (
                  <div
                    key={i}
                    className="flex gap-3 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={thumbSrc}
                      alt={`Region ${i + 1}`}
                      className="h-16 w-16 rounded object-cover flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0 space-y-1">
                      <p className="text-sm font-medium leading-snug">
                        &ldquo;{cap.caption}&rdquo;
                      </p>
                      <div className="flex items-center gap-2 text-xs text-white/30">
                        <Badge variant="secondary" className="text-[10px]">
                          {cap.model}
                        </Badge>
                        {cap.latency_ms > 0 && (
                          <span className="flex items-center gap-0.5">
                            <Clock className="h-3 w-3" />
                            {cap.latency_ms}ms
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
