"use client";

import { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, ZoomIn, Loader2, MousePointer2, CheckCircle2, XCircle, MessageSquareText } from "lucide-react";
import type { ClickPoint, RetrievedImage } from "@/lib/types";
import { useAppStore } from "@/lib/store";
import { segmentImage, lookupCachedCaption } from "@/lib/api";

interface ImageCardProps {
  image: RetrievedImage;
  index: number;
  showAnnotations?: boolean;
}

const POINT_COLORS = { 1: "#22c55e", 0: "#ef4444" } as const;
const MASK_COLORS  = { relevant: [34, 197, 94], irrelevant: [239, 68, 68] } as const;

export function ImageCard({ image, index, showAnnotations = true }: ImageCardProps) {
  const setSamAnnotation   = useAppStore((s) => s.setSamAnnotation);
  const clearSamAnnotation = useAppStore((s) => s.clearSamAnnotation);
  const samAnnotation      = useAppStore((s) => s.samAnnotations.get(index));
  const query              = useAppStore((s) => s.query);
  const relevantCaptions   = useAppStore((s) => s.relevantCaptions);
  const irrelevantCaptions = useAppStore((s) => s.irrelevantCaptions);

  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef  = useRef<HTMLDivElement>(null);
  const imgRef        = useRef<HTMLImageElement>(null);
  const segmentAbortRef = useRef<AbortController | null>(null);
  const segmentGenRef   = useRef(0);
  const captionPollRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  const [activeLabel, setActiveLabel] = useState<1 | 0>(1);
  const [expanded,    setExpanded]    = useState(false);
  const [imgSize,     setImgSize]     = useState({ w: 0, h: 0 });
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [loading,     setLoading]     = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const points = useMemo(() => samAnnotation?.points ?? [], [samAnnotation?.points]);

  const getImageDisplayRect = useCallback(
    (canvasW: number, canvasH: number) => {
      if (naturalSize.w === 0 || naturalSize.h === 0) {
        return { offsetX: 0, offsetY: 0, renderedW: canvasW, renderedH: canvasH };
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

        // Outer glow ring
        ctx.beginPath();
        ctx.arc(dx, dy, 7, 0, Math.PI * 2);
        ctx.fillStyle = color + "22";
        ctx.fill();

        // Filled circle
        ctx.beginPath();
        ctx.arc(dx, dy, 5, 0, Math.PI * 2);
        ctx.fillStyle = color + "99";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // White center dot
        ctx.beginPath();
        ctx.arc(dx, dy, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
      }
    },
    [points, naturalSize, getImageDisplayRect],
  );

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const ro = new ResizeObserver(() => setImgSize({ w: img.clientWidth, h: img.clientHeight }));
    ro.observe(img);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imgSize.w === 0) return;
    canvas.width  = imgSize.w;
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

    const rle     = samAnnotation.mask_rle;
    const [rh, rw] = rle.size;      // mask pixel dimensions
    canvas.width  = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Decode RLE into a pixel buffer
    const offscreen = document.createElement("canvas");
    offscreen.width  = rw;
    offscreen.height = rh;
    const offCtx = offscreen.getContext("2d")!;
    const imgData = offCtx.createImageData(rw, rh);
    const data    = imgData.data;

    const isRelevant = points.some((p) => p.label === 1);
    const [mR, mG, mB] = isRelevant ? MASK_COLORS.relevant : MASK_COLORS.irrelevant;
    const FILL_ALPHA = 140;

    let pos = 0;
    let val = 0; // starts at background (COCO convention)
    for (const length of rle.counts) {
      if (val === 1) {
        const end = pos + length;
        for (let j = pos; j < end; j++) {
          const idx = j * 4;
          data[idx]     = mR;
          data[idx + 1] = mG;
          data[idx + 2] = mB;
          data[idx + 3] = FILL_ALPHA;
        }
      }
      pos += length;
      val = 1 - val;
    }
    offCtx.putImageData(imgData, 0, 0);

    // Scale the mask to the rendered image area (not the full canvas)
    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(imgSize.w, imgSize.h);
    ctx.drawImage(offscreen, 0, 0, rw, rh, offsetX, offsetY, renderedW, renderedH);
  }, [samAnnotation, imgSize, getImageDisplayRect, points]);

  useEffect(() => { drawMask(); }, [drawMask]);

  function getOriginalCoords(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || naturalSize.w === 0) return null;

    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(rect.width, rect.height);
    const imageX = e.clientX - rect.left - offsetX;
    const imageY = e.clientY - rect.top  - offsetY;

    if (imageX < 0 || imageY < 0 || imageX > renderedW || imageY > renderedH) return null;

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

    const newPoint: ClickPoint = { x: coords.x, y: coords.y, label: activeLabel };
    const switchedLabel = points.length > 0 && points.some((p) => p.label !== activeLabel);
    const updatedPoints = switchedLabel ? [newPoint] : [...points, newPoint];

    setSamAnnotation(index, switchedLabel ? { points: updatedPoints } : { ...(samAnnotation ?? {}), points: updatedPoints });
    setSegmentError(null);
    setLoading(true);

    // Pass the current query and hint so the server can start Ollama in the background
    const hint = activeLabel === 1 ? relevantCaptions : irrelevantCaptions;
    const label = activeLabel === 1 ? "Relevant" : "Irrelevant";

    try {
      // SAM needs label=1 (foreground) to segment accurately regardless of
      // whether the user is marking the region relevant or irrelevant.
      // Stored points keep their real labels for color/relevance tracking.
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
        mask_rle:       result.mask_rle,
        region_b64:     result.region_b64,
        score:          result.score,
        vg_phrases:     result.vg_phrases,
        cached_caption: result.cached_caption ?? null,
      });

      // If the segment endpoint didn't return a ready caption, the server kicked
      // off background Ollama captioning — poll /caption_lookup so the UI can
      // display the caption the moment it's ready (typically 3–40s on MPS).
      if (!result.cached_caption && result.captioning_available && query && query.trim()) {
        if (captionPollRef.current) clearInterval(captionPollRef.current);
        const startedAt = Date.now();
        const pollGen = gen;
        captionPollRef.current = setInterval(async () => {
          if (pollGen !== segmentGenRef.current) {
            if (captionPollRef.current) clearInterval(captionPollRef.current);
            return;
          }
          // Give up after 120s to avoid polling forever when Ollama is offline
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
                mask_rle:       result.mask_rle,
                region_b64:     result.region_b64,
                score:          result.score,
                vg_phrases:     result.vg_phrases,
                cached_caption: lookup.caption,
              });
              if (captionPollRef.current) clearInterval(captionPollRef.current);
            } else if (!lookup.in_flight && Date.now() - startedAt > 8_000) {
              // No caption, not in flight, gave it a few seconds — Ollama likely unavailable.
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
      if (msg.includes("503"))      setSegmentError("SAM not loaded");
      else if (msg.includes("504")) setSegmentError("Timed out — retry");
      else if (msg.includes("403")) setSegmentError("Path not in corpus");
      else                          setSegmentError("Segment failed");
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

  const hasMask    = Boolean(samAnnotation?.mask_rle);
  const hasCaption = Boolean(samAnnotation?.cached_caption);

  return (
    <Card className="group overflow-hidden gap-0 py-0">
      <div className="relative border-b">

        {/* ── Top-left badges ── */}
        <div className="absolute left-2 top-2 z-20 flex items-center gap-1.5 flex-wrap max-w-[80%]">
          <Badge
            variant="secondary"
            className="text-[10px] font-mono bg-black/70 text-white/80 border border-white/10 backdrop-blur-md"
          >
            #{index + 1}
          </Badge>

          {loading && (
            <Badge
              variant="secondary"
              className="text-[10px] font-mono bg-violet-950/80 text-violet-200 border border-violet-600/30 backdrop-blur-md gap-1"
            >
              <Loader2 className="h-3 w-3 animate-spin" />
              Segmenting…
            </Badge>
          )}

          {hasMask && !loading && samAnnotation?.score != null && (
            <Badge
              variant="secondary"
              className="text-[10px] font-mono bg-black/60 text-white/60 border border-white/10 backdrop-blur-md"
            >
              {(samAnnotation.score * 100).toFixed(0)}% conf
            </Badge>
          )}

          {segmentError && !loading && (
            <Badge
              variant="secondary"
              className="text-[10px] font-mono bg-red-950/80 text-red-300 border border-red-600/30 backdrop-blur-md"
              title={segmentError}
            >
              ⚠ {segmentError}
            </Badge>
          )}
        </div>

        {/* ── Zoom button ── */}
        {!expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="absolute right-2 top-2 z-20 rounded-md bg-black/50 p-1.5 text-white/70 border border-white/10 opacity-0 transition-opacity group-hover:opacity-100 backdrop-blur-md"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        )}

        {/* ── Click hint ── */}
        {showAnnotations && points.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
            <span className="rounded-lg bg-black/70 px-3 py-1.5 text-xs text-white/70 backdrop-blur-md border border-white/10 opacity-0 group-hover:opacity-100 transition-opacity">
              <MousePointer2 className="inline h-3 w-3 mr-1" />
              Click to select a region
            </span>
          </div>
        )}

        {/* ── Image + canvas overlays ── */}
        <div
          ref={containerRef}
          className="relative z-[1] cursor-crosshair select-none"
          onClick={handleClick}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            ref={imgRef}
            src={`data:image/png;base64,${image.base64}`}
            alt={`Result ${index + 1}`}
            className={`block w-full object-contain ${expanded ? "max-h-[600px]" : "aspect-square"}`}
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              setImgSize({ w: el.clientWidth, h: el.clientHeight });
              if (el.naturalWidth > 0 && el.naturalHeight > 0) {
                setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
              }
            }}
          />
          {/* Mask overlay (bottom layer) */}
          <canvas
            ref={maskCanvasRef}
            className="absolute inset-0 h-full w-full pointer-events-none"
          />
          {/* Points overlay (top layer) */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full pointer-events-none"
          />
        </div>
      </div>

      {/* ── Controls & info ── */}
      {showAnnotations && (
        <div className="space-y-2 p-3">
          {/* Relevant / Irrelevant toggle + clear */}
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={activeLabel === 1 ? "default" : "outline"}
              onClick={() => setActiveLabel(1)}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === 1 ? { backgroundColor: "#22c55e", borderColor: "#16a34a" } : {}}
            >
              <CheckCircle2 className="h-3 w-3" />
              I want this
            </Button>
            <Button
              size="sm"
              variant={activeLabel === 0 ? "default" : "outline"}
              onClick={() => setActiveLabel(0)}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === 0 ? { backgroundColor: "#ef4444", borderColor: "#dc2626" } : {}}
            >
              <XCircle className="h-3 w-3" />
              Not this
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleClear}
              className="h-7 w-7 p-0"
              disabled={points.length === 0}
              title="Clear all selections"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>

          {/* Click counts */}
          {points.length > 0 && (
            <p className="text-[10px] text-white/30 text-center">
              {points.filter((p) => p.label === 1).length} relevant
              {" · "}
              {points.filter((p) => p.label === 0).length} irrelevant click
              {points.length !== 1 && "s"}
            </p>
          )}

          {/* VG region descriptions */}
          {samAnnotation?.vg_phrases && samAnnotation.vg_phrases.length > 0 && (
            <div className="mt-1 space-y-0.5">
              <p className="text-[9px] text-white/40 uppercase tracking-wider">VG Descriptions</p>
              {samAnnotation.vg_phrases.map((phrase, i) => (
                <p key={i} className="text-[10px] text-white/60 leading-tight">{phrase}</p>
              ))}
            </div>
          )}

          {/* Ollama cached caption — shows what the AI understood about this region */}
          {hasCaption && (
            <div className="mt-1 rounded-md border border-amber-500/20 bg-amber-500/5 px-2 py-1.5">
              <p className="text-[9px] text-amber-400/70 uppercase tracking-wider flex items-center gap-1 mb-0.5">
                <MessageSquareText className="h-2.5 w-2.5" />
                AI Vision says
              </p>
              <p className="text-[11px] text-amber-100/80 leading-snug">
                &ldquo;{samAnnotation!.cached_caption}&rdquo;
              </p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
