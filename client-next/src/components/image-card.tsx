"use client";

import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { Trash2, ZoomIn, Loader2, MousePointer2 } from "lucide-react";
import type { ClickPoint, RetrievedImage } from "@/lib/types";
import { useAppStore } from "@/lib/store";
import { segmentImage, lookupCachedCaption } from "@/lib/api";

interface ImageCardProps {
  image: RetrievedImage;
  index: number;
  showAnnotations?: boolean;
}

const POINT_COLORS = { 1: "#22c55e", 0: "#ef4444" } as const;
const MASK_COLORS = {
  relevant: [34, 197, 94],
  irrelevant: [239, 68, 68],
} as const;

export function ImageCard({
  image,
  index,
  showAnnotations = true,
}: ImageCardProps) {
  const setSamAnnotation = useAppStore((s) => s.setSamAnnotation);
  const clearSamAnnotation = useAppStore((s) => s.clearSamAnnotation);
  const samAnnotation = useAppStore((s) => s.samAnnotations.get(index));
  const query = useAppStore((s) => s.query);
  const relevantCaptions = useAppStore((s) => s.relevantCaptions);
  const irrelevantCaptions = useAppStore((s) => s.irrelevantCaptions);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const segmentAbortRef = useRef<AbortController | null>(null);
  const segmentGenRef = useRef(0);
  const captionPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [activeLabel, setActiveLabel] = useState<1 | 0>(1);
  const [expanded, setExpanded] = useState(false);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const points = useMemo(
    () => samAnnotation?.points ?? [],
    [samAnnotation?.points],
  );

  const getImageDisplayRect = useCallback(
    (canvasW: number, canvasH: number) => {
      if (naturalSize.w === 0 || naturalSize.h === 0) {
        return {
          offsetX: 0,
          offsetY: 0,
          renderedW: canvasW,
          renderedH: canvasH,
        };
      }
      const imgAspect = naturalSize.w / naturalSize.h;
      const cntAspect = canvasW / canvasH;
      let renderedW: number, renderedH: number, offsetX: number, offsetY: number;
      if (imgAspect > cntAspect) {
        renderedW = canvasW;
        renderedH = canvasW / imgAspect;
        offsetX = 0;
        offsetY = (canvasH - renderedH) / 2;
      } else {
        renderedH = canvasH;
        renderedW = canvasH * imgAspect;
        offsetX = (canvasW - renderedW) / 2;
        offsetY = 0;
      }
      return { offsetX, offsetY, renderedW, renderedH };
    },
    [naturalSize],
  );

  const drawPoints = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      if (naturalSize.w === 0) return;
      ctx.clearRect(0, 0, w, h);
      const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(w, h);
      for (const pt of points) {
        const dx = offsetX + (pt.x / naturalSize.w) * renderedW;
        const dy = offsetY + (pt.y / naturalSize.h) * renderedH;
        const color = POINT_COLORS[pt.label];

        ctx.beginPath();
        ctx.arc(dx, dy, 6, 0, Math.PI * 2);
        ctx.fillStyle = color + "22";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(dx, dy, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#0a0a0a";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    },
    [points, naturalSize, getImageDisplayRect],
  );

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() =>
      setImgSize({ w: img.clientWidth, h: img.clientHeight }),
    );
    ro.observe(img);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imgSize.w === 0) return;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (ctx) drawPoints(ctx, imgSize.w, imgSize.h);
  }, [points, imgSize, drawPoints]);

  // ---------------------------------------------------------------------------
  // Mask rendering — decode RLE and draw a coloured overlay on the mask canvas
  // ---------------------------------------------------------------------------
  const drawMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas) return;
    if (!samAnnotation?.mask_rle || imgSize.w === 0) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const rle = samAnnotation.mask_rle;
    const [rh, rw] = rle.size;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const offscreen = document.createElement("canvas");
    offscreen.width = rw;
    offscreen.height = rh;
    const offCtx = offscreen.getContext("2d")!;
    const imgData = offCtx.createImageData(rw, rh);
    const data = imgData.data;

    const isRelevant = points.some((p) => p.label === 1);
    const [mR, mG, mB] = isRelevant
      ? MASK_COLORS.relevant
      : MASK_COLORS.irrelevant;
    const FILL_ALPHA = 130;

    let pos = 0;
    let val = 0;
    for (const length of rle.counts) {
      if (val === 1) {
        const end = pos + length;
        for (let j = pos; j < end; j++) {
          const idx = j * 4;
          data[idx] = mR;
          data[idx + 1] = mG;
          data[idx + 2] = mB;
          data[idx + 3] = FILL_ALPHA;
        }
      }
      pos += length;
      val = 1 - val;
    }
    offCtx.putImageData(imgData, 0, 0);

    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(
      imgSize.w,
      imgSize.h,
    );
    ctx.drawImage(offscreen, 0, 0, rw, rh, offsetX, offsetY, renderedW, renderedH);
  }, [samAnnotation, imgSize, getImageDisplayRect, points]);

  useEffect(() => {
    drawMask();
  }, [drawMask]);

  function getOriginalCoords(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || naturalSize.w === 0) return null;

    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(
      rect.width,
      rect.height,
    );
    const imageX = e.clientX - rect.left - offsetX;
    const imageY = e.clientY - rect.top - offsetY;

    if (imageX < 0 || imageY < 0 || imageX > renderedW || imageY > renderedH)
      return null;

    return {
      x: Math.round((imageX / renderedW) * naturalSize.w),
      y: Math.round((imageY / renderedH) * naturalSize.h),
    };
  }

  async function handleClick(e: React.MouseEvent) {
    if (!showAnnotations) return;
    const coords = getOriginalCoords(e);
    if (!coords) return;

    segmentAbortRef.current?.abort();
    const controller = new AbortController();
    segmentAbortRef.current = controller;
    const gen = ++segmentGenRef.current;

    const newPoint: ClickPoint = {
      x: coords.x,
      y: coords.y,
      label: activeLabel,
    };
    const switchedLabel =
      points.length > 0 && points.some((p) => p.label !== activeLabel);
    const updatedPoints = switchedLabel ? [newPoint] : [...points, newPoint];

    setSamAnnotation(
      index,
      switchedLabel
        ? { points: updatedPoints }
        : { ...(samAnnotation ?? {}), points: updatedPoints },
    );
    setSegmentError(null);
    setLoading(true);

    const hint = activeLabel === 1 ? relevantCaptions : irrelevantCaptions;
    const label = activeLabel === 1 ? "Relevant" : "Irrelevant";

    try {
      const samPoints = updatedPoints.map((p) => ({ ...p, label: 1 as const }));
      const result = await segmentImage(
        image.path,
        samPoints,
        naturalSize.w,
        naturalSize.h,
        controller.signal,
        query || undefined,
        hint || undefined,
        label,
      );

      if (gen !== segmentGenRef.current) return;

      setSamAnnotation(index, {
        points: updatedPoints,
        mask_rle: result.mask_rle,
        region_b64: result.region_b64,
        score: result.score,
        vg_phrases: result.vg_phrases,
        cached_caption: result.cached_caption ?? null,
      });

      if (
        !result.cached_caption &&
        result.captioning_available &&
        query &&
        query.trim()
      ) {
        if (captionPollRef.current) clearInterval(captionPollRef.current);
        const startedAt = Date.now();
        const pollGen = gen;
        captionPollRef.current = setInterval(async () => {
          if (pollGen !== segmentGenRef.current) {
            if (captionPollRef.current) clearInterval(captionPollRef.current);
            return;
          }
          if (Date.now() - startedAt > 120_000) {
            if (captionPollRef.current) clearInterval(captionPollRef.current);
            return;
          }
          try {
            const lookup = await lookupCachedCaption(
              image.path,
              query,
              label,
              hint || "",
            );
            if (pollGen !== segmentGenRef.current) return;
            if (lookup.ready && lookup.caption) {
              setSamAnnotation(index, {
                points: updatedPoints,
                mask_rle: result.mask_rle,
                region_b64: result.region_b64,
                score: result.score,
                vg_phrases: result.vg_phrases,
                cached_caption: lookup.caption,
              });
              if (captionPollRef.current) clearInterval(captionPollRef.current);
            } else if (!lookup.in_flight && Date.now() - startedAt > 8_000) {
              if (captionPollRef.current) clearInterval(captionPollRef.current);
            }
          } catch {
            // Swallow transient errors — keep polling until timeout.
          }
        }, 2_500);
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      if (gen !== segmentGenRef.current) return;

      setSamAnnotation(index, { points: updatedPoints });
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("503")) setSegmentError("SAM not loaded");
      else if (msg.includes("504")) setSegmentError("Timed out");
      else if (msg.includes("403")) setSegmentError("Path not in corpus");
      else setSegmentError("Segment failed");
    } finally {
      if (gen === segmentGenRef.current) setLoading(false);
    }
  }

  function handleClear() {
    if (captionPollRef.current) {
      clearInterval(captionPollRef.current);
      captionPollRef.current = null;
    }
    segmentGenRef.current += 1;
    clearSamAnnotation(index);
    setSegmentError(null);
  }

  useEffect(() => {
    return () => {
      if (captionPollRef.current) clearInterval(captionPollRef.current);
    };
  }, []);

  const hasMask = Boolean(samAnnotation?.mask_rle);
  const hasCaption = Boolean(samAnnotation?.cached_caption);
  const isRelevant = points.some((p) => p.label === 1);

  return (
    <article className="group flex flex-col overflow-hidden rounded-md border border-border bg-card transition-colors duration-100 hover:border-foreground/20">
      {/* ── Image canvas region ── */}
      <div className="relative">
        {/* Top-left: index + status */}
        <div className="absolute left-2 top-2 z-20 flex items-center gap-1">
          <span className="rounded border border-border bg-card/95 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-foreground">
            {String(index + 1).padStart(2, "0")}
          </span>
          {loading && (
            <span className="inline-flex items-center gap-1 rounded border border-border bg-card/95 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              <Loader2 className="h-2.5 w-2.5 animate-spin" />
              segmenting
            </span>
          )}
          {hasMask && !loading && samAnnotation?.score != null && (
            <span className="rounded border border-border bg-card/95 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-muted-foreground">
              {(samAnnotation.score * 100).toFixed(0)}%
            </span>
          )}
          {segmentError && !loading && (
            <span
              className="rounded border border-destructive/30 bg-destructive/15 px-1.5 py-0.5 font-mono text-[10px] text-destructive"
              title={segmentError}
            >
              {segmentError}
            </span>
          )}
        </div>

        {/* Top-right: zoom (hover only) */}
        {!expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="absolute right-2 top-2 z-20 rounded border border-border bg-card/95 p-1 text-muted-foreground opacity-0 transition-opacity duration-100 group-hover:opacity-100 hover:text-foreground"
            aria-label="Expand"
          >
            <ZoomIn className="h-3 w-3" />
          </button>
        )}

        {/* Click hint when empty */}
        {showAnnotations && points.length === 0 && (
          <div className="pointer-events-none absolute inset-0 z-10 flex items-end justify-center pb-3 opacity-0 transition-opacity duration-100 group-hover:opacity-100">
            <span className="inline-flex items-center gap-1.5 rounded border border-border bg-card/95 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              <MousePointer2 className="h-2.5 w-2.5" />
              click to select
            </span>
          </div>
        )}

        {/* Image + canvases */}
        <div
          ref={containerRef}
          className="relative cursor-crosshair select-none"
          onClick={handleClick}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imgRef}
            src={`data:image/png;base64,${image.base64}`}
            alt={`Result ${index + 1}`}
            className={`block w-full object-contain ${
              expanded ? "max-h-[600px]" : "aspect-square"
            }`}
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              setImgSize({ w: el.clientWidth, h: el.clientHeight });
              if (el.naturalWidth > 0 && el.naturalHeight > 0) {
                setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
              }
            }}
          />
          <canvas
            ref={maskCanvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        </div>
      </div>

      {/* ── Controls + meta ── */}
      {showAnnotations && (
        <div className="border-t border-border p-2.5">
          {/* Label toggle */}
          <div className="flex items-stretch gap-1.5">
            <button
              onClick={() => setActiveLabel(1)}
              className={`flex-1 rounded border px-2 py-1.5 font-mono text-[10.5px] uppercase tracking-wider transition-colors duration-100 ${
                activeLabel === 1
                  ? "border-emerald-500/40 bg-emerald-500/[0.08] text-emerald-400"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              relevant
            </button>
            <button
              onClick={() => setActiveLabel(0)}
              className={`flex-1 rounded border px-2 py-1.5 font-mono text-[10.5px] uppercase tracking-wider transition-colors duration-100 ${
                activeLabel === 0
                  ? "border-rose-500/40 bg-rose-500/[0.08] text-rose-400"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              irrelevant
            </button>
            <button
              onClick={handleClear}
              disabled={points.length === 0}
              className="rounded border border-border px-2 text-muted-foreground transition-colors duration-100 hover:text-foreground disabled:opacity-40"
              title="Clear selection"
              aria-label="Clear"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Click counts — single quiet line */}
          {points.length > 0 && (
            <p className="mt-2 font-mono text-[10px] tabular-nums text-muted-foreground">
              <span
                className={isRelevant ? "text-emerald-400" : "text-muted-foreground"}
              >
                {points.filter((p) => p.label === 1).length}
              </span>
              <span className="mx-1 text-muted-foreground">/</span>
              <span
                className={!isRelevant ? "text-rose-400" : "text-muted-foreground"}
              >
                {points.filter((p) => p.label === 0).length}
              </span>
              <span className="ml-1.5 text-muted-foreground/60">
                {points.length === 1 ? "click" : "clicks"}
              </span>
            </p>
          )}

          {/* VG phrases */}
          {samAnnotation?.vg_phrases && samAnnotation.vg_phrases.length > 0 && (
            <div className="mt-2 border-l border-border pl-2">
              <p className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                VG phrases
              </p>
              {samAnnotation.vg_phrases.map((phrase, i) => (
                <p
                  key={i}
                  className="mt-0.5 text-[11px] leading-snug text-foreground/80"
                >
                  {phrase}
                </p>
              ))}
            </div>
          )}

          {/* Cached AI caption */}
          {hasCaption && (
            <div className="mt-2 border-l-2 border-amber-500/40 pl-2">
              <p className="font-mono text-[9px] uppercase tracking-wider text-amber-400/80">
                AI vision
              </p>
              <p className="mt-0.5 text-[11px] leading-snug text-foreground/80">
                {samAnnotation!.cached_caption}
              </p>
            </div>
          )}
        </div>
      )}
    </article>
  );
}
