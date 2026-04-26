"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { MessageSquareText, Loader2 } from "lucide-react";
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
          "Ollama not running. Start it: `ollama serve` then `ollama pull llama3.2-vision`",
        );
        setGenerating(false);
        return;
      }

      const results = new Map<number, CaptionResult>();
      const targets =
        annotatedIndices.length > 0
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
            caption: "Caption failed — check Ollama",
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
    <section className="rounded-md border border-border bg-card">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium tracking-tight text-foreground">
            Captions
          </h3>
          <span className="font-mono text-[11px] text-muted-foreground">
            llama3.2-vision · synchronous
          </span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
          {annotatedIndices.length > 0 && (
            <span>
              <span className="text-foreground">{annotatedIndices.length}</span>{" "}
              regions
            </span>
          )}
          {captions.size > 0 && (
            <span>
              <span className="text-foreground">{captions.size}</span> captioned
            </span>
          )}
        </div>
      </header>

      {/* Body */}
      <div className="space-y-4 px-5 py-4">
        <p className="text-[13px] leading-relaxed text-muted-foreground">
          Generate explicit Llama&nbsp;3.2-Vision captions for each region. The
          background pipeline already does this when you click — this panel is
          for manual re-runs and inspection.
        </p>

        {error && (
          <p className="rounded-md border border-destructive/30 bg-destructive/[0.06] px-3 py-2 font-mono text-[11.5px] text-destructive">
            {error}
          </p>
        )}

        <Button
          onClick={checkAndGenerate}
          disabled={generating}
          variant="outline"
          size="sm"
          className="h-8"
        >
          {generating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <MessageSquareText className="h-3.5 w-3.5" />
          )}
          {generating ? "Generating" : "Generate captions"}
        </Button>

        {captions.size > 0 && (
          <ul className="divide-y divide-border border-t border-border">
            {images.map((img, i) => {
              const cap = captions.get(i);
              if (!cap) return null;
              const ann = samAnnotations.get(i);
              const thumbSrc = ann?.region_b64
                ? `data:image/png;base64,${ann.region_b64}`
                : `data:image/png;base64,${img.base64}`;
              return (
                <li key={i} className="flex items-start gap-3 py-3">
                  <span className="mt-0.5 font-mono text-[10.5px] tabular-nums text-muted-foreground">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={thumbSrc}
                    alt={`Region ${i + 1}`}
                    className="h-12 w-12 shrink-0 rounded-sm border border-border object-cover"
                  />
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="text-[13px] leading-snug text-foreground/85">
                      {cap.caption}
                    </p>
                    <div className="flex items-center gap-3 font-mono text-[10.5px] text-muted-foreground">
                      <span>{cap.model}</span>
                      {cap.latency_ms > 0 && (
                        <span className="tabular-nums">
                          {cap.latency_ms} ms
                        </span>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
