"use client";

import { useEffect, useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import {
  Sparkles,
  Loader2,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { ollamaStatus, samStatus } from "@/lib/api";
import type { HardFilterInfo } from "@/lib/types";

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
  const imageLabels = useAppStore((s) => s.imageLabels);
  const hardFilter = useAppStore((s) => s.hardFilter);

  const [ollamaAvailable, setOllamaAvailable] = useState<boolean | null>(null);
  const [samLoaded, setSamLoaded] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    ollamaStatus()
      .then((s) => {
        if (!cancelled) setOllamaAvailable(s.available);
      })
      .catch(() => {
        if (!cancelled) setOllamaAvailable(false);
      });
    samStatus()
      .then((s) => {
        if (!cancelled) setSamLoaded(s.loaded);
      })
      .catch(() => {
        if (!cancelled) setSamLoaded(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const aiSuggestions = useMemo(() => {
    const relevant: string[] = [];
    const irrelevant: string[] = [];
    samAnnotations.forEach((ann) => {
      if (!ann.cached_caption) return;
      const isRel = ann.points.some((p) => p.label === 1);
      if (isRel) relevant.push(ann.cached_caption);
      else irrelevant.push(ann.cached_caption);
    });
    return { relevant, irrelevant };
  }, [samAnnotations]);

  if (images.length === 0) return null;

  const totalBoxes = images.reduce((sum, img) => sum + img.boxes.length, 0);
  const samWithMask = Array.from(samAnnotations.values()).filter(
    (a) => a.mask_rle,
  ).length;
  const fullImagePos = Array.from(imageLabels.entries()).filter(
    ([i, l]) => l === "Relevant" && !samAnnotations.get(i)?.mask_rle,
  ).length;
  const fullImageNeg = Array.from(imageLabels.entries()).filter(
    ([i, l]) => l === "Irrelevant" && !samAnnotations.get(i)?.mask_rle,
  ).length;
  const totalFullLabels = fullImagePos + fullImageNeg;
  const totalAi =
    aiSuggestions.relevant.length + aiSuggestions.irrelevant.length;
  const hasAnyFeedback =
    totalBoxes > 0 ||
    samWithMask > 0 ||
    totalFullLabels > 0 ||
    relevantCaptions.trim() ||
    irrelevantCaptions.trim();

  return (
    <section className="rounded-md border border-border bg-card">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium tracking-tight text-foreground">
            Refine
          </h3>
          <span className="font-mono text-[11px] text-muted-foreground">
            relevance feedback
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10.5px] font-mono">
          <StatusPill label="SAM 3" online={samLoaded} />
          <StatusPill label="AI Vision" online={ollamaAvailable} />
        </div>
      </header>

      {/* Body */}
      <div className="space-y-5 px-5 py-4">
        {/* Hard-filter telemetry from the previous round */}
        {hardFilter &&
          (hardFilter.dropped > 0 ||
            hardFilter.boosted > 0 ||
            hardFilter.blacklist_size > 0 ||
            hardFilter.boostlist_size > 0) && (
            <HardFilterBadge info={hardFilter} />
          )}

        {/* AI caption suggestions, if any */}
        {totalAi > 0 && (
          <div className="space-y-2 border-l-2 border-amber-500/40 pl-3">
            <p className="font-mono text-[10px] uppercase tracking-wider text-amber-400/80">
              AI Vision described your selections
            </p>
            <div className="grid gap-1">
              {aiSuggestions.relevant.map((cap, i) => (
                <CaptionRow
                  key={`r-${i}`}
                  caption={cap}
                  tone="relevant"
                  onUse={() =>
                    !relevantCaptions && setRelevantCaptions(cap)
                  }
                  used={Boolean(relevantCaptions)}
                />
              ))}
              {aiSuggestions.irrelevant.map((cap, i) => (
                <CaptionRow
                  key={`i-${i}`}
                  caption={cap}
                  tone="irrelevant"
                  onUse={() =>
                    !irrelevantCaptions && setIrrelevantCaptions(cap)
                  }
                  used={Boolean(irrelevantCaptions)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Hint inputs */}
        <div className="grid gap-4 sm:grid-cols-2">
          <HintField
            tone="relevant"
            label="More of"
            placeholder="e.g. golden hour lighting"
            value={relevantCaptions}
            onChange={setRelevantCaptions}
            disabled={isApplyingFeedback}
          />
          <HintField
            tone="irrelevant"
            label="Less of"
            placeholder="e.g. crowds, blurry"
            value={irrelevantCaptions}
            onChange={setIrrelevantCaptions}
            disabled={isApplyingFeedback}
          />
        </div>

        {/* Drift toggle */}
        <label
          htmlFor="fuse-query"
          className="flex cursor-pointer select-none items-center gap-2.5 text-[12.5px] text-muted-foreground transition-colors duration-100 hover:text-foreground"
        >
          <input
            type="checkbox"
            id="fuse-query"
            checked={fuseInitialQuery}
            onChange={(e) => setFuseInitialQuery(e.target.checked)}
            className="h-3.5 w-3.5 rounded-sm border border-input bg-card accent-foreground"
          />
          Anchor to original query
          <span className="font-mono text-[10.5px] text-muted-foreground/70">
            prevents drift across rounds
          </span>
        </label>
      </div>

      {/* Footer */}
      <footer className="flex items-center justify-between gap-4 border-t border-border bg-card px-5 py-3">
        <div className="flex items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
          {samWithMask > 0 && (
            <Stat label="regions" value={samWithMask} accent />
          )}
          {totalFullLabels > 0 && (
            <Stat label="images" value={totalFullLabels} accent />
          )}
          {totalBoxes > 0 && <Stat label="boxes" value={totalBoxes} />}
          {totalAi > 0 && (
            <Stat label="captions" value={totalAi} amber />
          )}
          {!samWithMask && !totalFullLabels && !totalBoxes && !totalAi && (
            <span className="text-muted-foreground/60">
              Click an image region or mark whole images
            </span>
          )}
        </div>
        <Button
          onClick={onApply}
          disabled={!hasAnyFeedback || isApplyingFeedback}
          size="default"
          className="h-9 px-4"
        >
          {isApplyingFeedback ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Sparkles className="h-3.5 w-3.5" />
          )}
          Apply Feedback
        </Button>
      </footer>
    </section>
  );
}

function StatusPill({
  label,
  online,
}: {
  label: string;
  online: boolean | null;
}) {
  if (online === null) return null;
  return (
    <span className="inline-flex items-center gap-1.5 text-muted-foreground">
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          online ? "bg-emerald-400" : "bg-muted-foreground/40"
        }`}
      />
      <span className={online ? "text-foreground/80" : "text-muted-foreground/60"}>
        {label}
      </span>
    </span>
  );
}

function Stat({
  label,
  value,
  accent,
  amber,
}: {
  label: string;
  value: number;
  accent?: boolean;
  amber?: boolean;
}) {
  return (
    <span className="inline-flex items-baseline gap-1">
      <span
        className={
          accent
            ? "text-foreground"
            : amber
              ? "text-amber-400/80"
              : "text-foreground"
        }
      >
        {value}
      </span>
      <span className="text-muted-foreground/70">{label}</span>
    </span>
  );
}

function CaptionRow({
  caption,
  tone,
  onUse,
  used,
}: {
  caption: string;
  tone: "relevant" | "irrelevant";
  onUse: () => void;
  used: boolean;
}) {
  const accent =
    tone === "relevant" ? "text-emerald-400/80" : "text-rose-400/80";
  return (
    <div className="flex items-start gap-2">
      <span className={`mt-1 font-mono text-[9px] uppercase tracking-wider ${accent}`}>
        {tone === "relevant" ? "+" : "−"}
      </span>
      <p className="flex-1 text-[12px] leading-snug text-foreground/85">
        {caption}
      </p>
      {!used && (
        <button
          onClick={onUse}
          className="shrink-0 font-mono text-[10px] text-muted-foreground underline-offset-2 transition-colors duration-100 hover:text-foreground hover:underline"
          title="Use as hint"
        >
          use
        </button>
      )}
    </div>
  );
}

function HardFilterBadge({ info }: { info: HardFilterInfo }) {
  const negPhrases = info.top_negative_phrases.slice(0, 3);
  const posPhrases = info.top_positive_phrases.slice(0, 3);
  const renderPhrases = (phrases: { phrase: string; count: number }[]) =>
    phrases.map((p, i) => (
      <span key={p.phrase}>
        <em className="not-italic text-foreground/85">{p.phrase}</em>
        {i < phrases.length - 1 && (
          <span className="text-muted-foreground/40">, </span>
        )}
      </span>
    ));

  return (
    <div className="space-y-1.5 border-l-2 border-blue-500/40 pl-3">
      <p className="font-mono text-[10px] uppercase tracking-wider text-blue-400/80">
        Region filter narrowed your search
      </p>
      <div className="space-y-0.5 text-[12px] leading-snug text-muted-foreground">
        {info.dropped > 0 && (
          <p>
            <span className="text-rose-400/90">−{info.dropped}</span>{" "}
            images removed this round
            {negPhrases.length > 0 && (
              <>
                {" "}
                <span className="text-muted-foreground/60">·</span>{" "}
                <span className="text-muted-foreground/85">
                  matched {renderPhrases(negPhrases)}
                </span>
              </>
            )}
          </p>
        )}
        {info.boosted > 0 && (
          <p>
            <span className="text-emerald-400/90">+{info.boosted}</span>{" "}
            images boosted
            {posPhrases.length > 0 && (
              <>
                {" "}
                <span className="text-muted-foreground/60">·</span>{" "}
                <span className="text-muted-foreground/85">
                  matched {renderPhrases(posPhrases)}
                </span>
              </>
            )}
          </p>
        )}
        {(info.blacklist_size > 0 || info.boostlist_size > 0) && (
          <p className="font-mono text-[10.5px] text-muted-foreground/60">
            session totals: {info.blacklist_size.toLocaleString()} blacklisted ·{" "}
            {info.boostlist_size.toLocaleString()} boosted ·{" "}
            over-fetched {info.overfetch.toLocaleString()}
          </p>
        )}
      </div>
    </div>
  );
}

function HintField({
  tone,
  label,
  placeholder,
  value,
  onChange,
  disabled,
}: {
  tone: "relevant" | "irrelevant";
  label: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  const symbol = tone === "relevant" ? "+" : "−";
  const accent =
    tone === "relevant" ? "text-emerald-400/80" : "text-rose-400/80";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-1.5 text-[11.5px] font-mono uppercase tracking-wider text-muted-foreground">
          <span className={accent}>{symbol}</span>
          {label}
        </label>
      </div>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="h-9 w-full rounded-md border border-input bg-background px-3 text-[13px] tracking-tight text-foreground outline-none transition-colors duration-100 placeholder:text-muted-foreground/60 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50"
      />
    </div>
  );
}
