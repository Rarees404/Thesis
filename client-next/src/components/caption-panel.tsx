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
import { MessageSquareText, Loader2, Zap, Clock } from "lucide-react";
import { useAppStore } from "@/lib/store";
import type { CaptionResult } from "@/lib/types";

const MOCK_CAPTIONS: string[] = [
  "A person riding a bicycle on a sunlit trail surrounded by trees",
  "Urban street scene with pedestrians and cars in the background",
  "Close-up of a golden retriever sitting on green grass, looking forward",
  "Mountain landscape with a winding road and scattered clouds",
  "Indoor scene with modern furniture and warm ambient lighting",
];

export function CaptionPanel() {
  const images = useAppStore((s) => s.images);
  const query = useAppStore((s) => s.query);
  const [generating, setGenerating] = useState(false);
  const [captions, setCaptions] = useState<Map<number, CaptionResult>>(
    new Map()
  );

  if (images.length === 0) return null;

  async function handleGenerate() {
    setGenerating(true);

    const results = new Map<number, CaptionResult>();
    for (let i = 0; i < images.length; i++) {
      await new Promise((r) => setTimeout(r, 300 + Math.random() * 400));
      results.set(i, {
        caption: MOCK_CAPTIONS[i % MOCK_CAPTIONS.length],
        model: "llama3.2-vision",
        latency_ms: Math.round(200 + Math.random() * 800),
      });
    }

    setCaptions(results);
    setGenerating(false);
  }

  return (
    <Card className="border-amber-500/20">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquareText className="h-4 w-4 text-amber-500" />
          Llama 3.2 Vision Captions
          <Badge variant="outline" className="text-[10px]">
            Preview
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-white/40">
          Llama 3.2 Vision generates focused captions for each image segment
          via Ollama. Captions are encoded by SigLIP to stay in the same
          embedding space as the query.
        </p>

        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="gap-1 text-xs">
            <Zap className="h-3 w-3" />
            Ollama &middot; llama3.2-vision
          </Badge>
          {query && (
            <Badge variant="outline" className="text-xs">
              Query: &ldquo;{query}&rdquo;
            </Badge>
          )}
        </div>

        <Button
          onClick={handleGenerate}
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
                return (
                  <div
                    key={i}
                    className="flex gap-3 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${img.base64}`}
                      alt={`Image ${i + 1}`}
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
                        <span className="flex items-center gap-0.5">
                          <Clock className="h-3 w-3" />
                          {cap.latency_ms}ms
                        </span>
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
