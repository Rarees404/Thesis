"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, ZoomIn, Loader2, MousePointer2 } from "lucide-react";
import type { ClickPoint, RetrievedImage } from "@/lib/types";
import { useAppStore } from "@/lib/store";
import { segmentImage } from "@/lib/api";

interface ImageCardProps {
  image: RetrievedImage;
  index: number;
  showAnnotations?: boolean;
}

const POINT_COLORS = { 1: "#22c55e", 0: "#ef4444" } as const;

export function ImageCard({ image, index, showAnnotations = true }: ImageCardProps) {
  const setSamAnnotation = useAppStore((s) => s.setSamAnnotation);
  const clearSamAnnotation = useAppStore((s) => s.clearSamAnnotation);
  const samAnnotation = useAppStore((s) => s.samAnnotations.get(index));

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [activeLabel, setActiveLabel] = useState<1 | 0>(1);
  const [expanded, setExpanded] = useState(false);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  const [loading, setLoading] = useState(false);

  const points = samAnnotation?.points ?? [];

  /**
   * Compute the actual rendered image rectangle inside the container when
   * `object-contain` is applied. Returns the letterbox offset and rendered
   * size so that canvas drawing and click mapping stay in sync.
   */
  const getImageDisplayRect = useCallback(
    (canvasW: number, canvasH: number) => {
      if (naturalSize.w === 0 || naturalSize.h === 0) {
        return { offsetX: 0, offsetY: 0, renderedW: canvasW, renderedH: canvasH };
      }
      const imgAspect = naturalSize.w / naturalSize.h;
      const containerAspect = canvasW / canvasH;
      let renderedW: number, renderedH: number, offsetX: number, offsetY: number;
      if (imgAspect > containerAspect) {
        // wider image → fills width, letterbox top/bottom
        renderedW = canvasW;
        renderedH = canvasW / imgAspect;
        offsetX = 0;
        offsetY = (canvasH - renderedH) / 2;
      } else {
        // taller image → fills height, letterbox left/right
        renderedH = canvasH;
        renderedW = canvasH * imgAspect;
        offsetX = (canvasW - renderedW) / 2;
        offsetY = 0;
      }
      return { offsetX, offsetY, renderedW, renderedH };
    },
    [naturalSize]
  );

  const drawPoints = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      if (naturalSize.w === 0 || naturalSize.h === 0) return;
      ctx.clearRect(0, 0, w, h);
      const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(w, h);
      for (const pt of points) {
        const displayX = offsetX + (pt.x / naturalSize.w) * renderedW;
        const displayY = offsetY + (pt.y / naturalSize.h) * renderedH;

        ctx.beginPath();
        ctx.arc(displayX, displayY, 8, 0, Math.PI * 2);
        ctx.fillStyle = POINT_COLORS[pt.label] + "80";
        ctx.fill();
        ctx.strokeStyle = POINT_COLORS[pt.label];
        ctx.lineWidth = 2.5;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(displayX, displayY, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
      }
    },
    [points, naturalSize, getImageDisplayRect]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imgSize.w === 0) return;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (ctx) drawPoints(ctx, imgSize.w, imgSize.h);
  }, [points, imgSize, drawPoints]);

  const drawMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !samAnnotation?.mask_rle || imgSize.w === 0) {
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const rle = samAnnotation.mask_rle;
    const [h, w] = rle.size;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const offscreen = document.createElement("canvas");
    offscreen.width = w;
    offscreen.height = h;
    const offCtx = offscreen.getContext("2d")!;
    const imgData = offCtx.createImageData(w, h);
    const data = imgData.data;

    // Mask color reflects semantic label:
    //   green  (34, 197, 94)  → at least one relevant (foreground) click
    //   red    (220, 38, 38)  → all clicks are irrelevant (background)
    const isRelevant = (samAnnotation.points ?? []).some((p) => p.label === 1);
    const [mR, mG, mB] = isRelevant ? [34, 197, 94] : [220, 38, 38];

    // COCO-convention RLE: first count is background, then alternates fg/bg.
    // val=0 means background (skip), val=1 means foreground (draw).
    let pos = 0;
    let val = 0;
    for (const length of rle.counts) {
      if (val === 1) {
        for (let j = 0; j < length; j++) {
          const idx = (pos + j) * 4;
          data[idx]     = mR;
          data[idx + 1] = mG;
          data[idx + 2] = mB;
          data[idx + 3] = 150;
        }
      }
      pos += length;
      val = 1 - val;
    }
    offCtx.putImageData(imgData, 0, 0);

    // Draw mask aligned to the letterboxed image area, not the full container
    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(imgSize.w, imgSize.h);
    ctx.drawImage(offscreen, 0, 0, w, h, offsetX, offsetY, renderedW, renderedH);
  }, [samAnnotation, imgSize, getImageDisplayRect]);

  useEffect(() => {
    drawMask();
  }, [drawMask]);

  function getOriginalCoords(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || naturalSize.w === 0 || naturalSize.h === 0) return null;

    // Compute the letterbox-corrected rendered image rectangle.
    // The canvas / container may be larger than the displayed image when
    // object-contain is applied; we must map only within the image area.
    const { offsetX, offsetY, renderedW, renderedH } = getImageDisplayRect(
      rect.width,
      rect.height
    );

    const imageX = e.clientX - rect.left - offsetX;
    const imageY = e.clientY - rect.top - offsetY;

    // Reject clicks that land in the letterbox area outside the image
    if (imageX < 0 || imageY < 0 || imageX > renderedW || imageY > renderedH) {
      return null;
    }

    return {
      x: Math.round((imageX / renderedW) * naturalSize.w),
      y: Math.round((imageY / renderedH) * naturalSize.h),
    };
  }

  async function handleClick(e: React.MouseEvent) {
    if (!showAnnotations) return;
    const coords = getOriginalCoords(e);
    if (!coords) return;

    const newPoint: ClickPoint = { x: coords.x, y: coords.y, label: activeLabel };
    const updatedPoints = [...points, newPoint];

    setSamAnnotation(index, { points: updatedPoints });

    setLoading(true);
    try {
      const result = await segmentImage(
        image.path,
        updatedPoints,
        naturalSize.w,
        naturalSize.h
      );
      setSamAnnotation(index, {
        points: updatedPoints,
        mask_rle: result.mask_rle,
        region_b64: result.region_b64,
        score: result.score,
      });
    } catch {
      setSamAnnotation(index, { points: updatedPoints });
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    clearSamAnnotation(index);
  }

  return (
    <Card className="group overflow-hidden gap-0 py-0">
      <div className="relative border-b">
        <div className="absolute left-2 top-2 z-20 flex items-center gap-1.5">
          <Badge
            variant="secondary"
            className="text-[10px] font-mono bg-black/70 text-white/80 border border-white/10 backdrop-blur-md"
          >
            #{index + 1}
          </Badge>
          {loading && (
            <Badge
              variant="secondary"
              className="text-[10px] font-mono bg-red-950/80 text-red-200 border border-red-600/30 backdrop-blur-md gap-1"
            >
              <Loader2 className="h-3 w-3 animate-spin" />
              SEG
            </Badge>
          )}
        </div>

        {!expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="absolute right-2 top-2 z-20 rounded-md bg-black/50 p-1.5 text-white/70 border border-white/10 opacity-0 transition-opacity group-hover:opacity-100 backdrop-blur-md"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        )}

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
            className={`block w-full object-contain ${expanded ? "max-h-[600px]" : "aspect-square"}`}
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              setImgSize({ w: el.clientWidth, h: el.clientHeight });
              setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
            }}
          />
          <canvas
            ref={maskCanvasRef}
            className="absolute inset-0 h-full w-full pointer-events-none"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full pointer-events-none"
          />
        </div>
      </div>

      {showAnnotations && (
        <div className="space-y-2 p-3">
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={activeLabel === 1 ? "default" : "outline"}
              onClick={() => setActiveLabel(1)}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === 1 ? { backgroundColor: "#22c55e" } : {}}
            >
              <MousePointer2 className="h-3 w-3" />
              Relevant
            </Button>
            <Button
              size="sm"
              variant={activeLabel === 0 ? "default" : "outline"}
              onClick={() => setActiveLabel(0)}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === 0 ? { backgroundColor: "#ef4444" } : {}}
            >
              <MousePointer2 className="h-3 w-3" />
              Irrelevant
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleClear}
              className="h-7 w-7 p-0"
              disabled={points.length === 0}
              title="Clear clicks"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
