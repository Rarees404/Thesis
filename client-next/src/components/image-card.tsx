"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, ZoomIn } from "lucide-react";
import type { BoundingBox, RetrievedImage } from "@/lib/types";
import { useAppStore } from "@/lib/store";

interface ImageCardProps {
  image: RetrievedImage;
  index: number;
  showAnnotations?: boolean;
}

type DrawingState = {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
} | null;

const LABEL_COLORS: Record<string, string> = {
  Relevant: "rgba(34,197,94,0.35)",
  Irrelevant: "rgba(239,68,68,0.35)",
};

const LABEL_BORDER: Record<string, string> = {
  Relevant: "#22c55e",
  Irrelevant: "#ef4444",
};

export function ImageCard({ image, index, showAnnotations = true }: ImageCardProps) {
  const updateBoxes = useAppStore((s) => s.updateBoxes);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState<DrawingState>(null);
  const [activeLabel, setActiveLabel] = useState<"Relevant" | "Irrelevant">("Relevant");
  const [expanded, setExpanded] = useState(false);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const boxes = image.boxes;

  const drawBoxes = useCallback(
    (ctx: CanvasRenderingContext2D, w: number, h: number) => {
      ctx.clearRect(0, 0, w, h);
      for (const box of boxes) {
        const fill = LABEL_COLORS[box.label] ?? "rgba(100,100,255,0.3)";
        const stroke = LABEL_BORDER[box.label] ?? "#6466f1";
        ctx.fillStyle = fill;
        ctx.fillRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 2;
        ctx.strokeRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);

        ctx.fillStyle = stroke;
        ctx.font = "bold 11px sans-serif";
        const textY = box.ymin > 14 ? box.ymin - 4 : box.ymin + 14;
        ctx.fillText(box.label, box.xmin + 3, textY);
      }
    },
    [boxes]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || imgSize.w === 0) return;
    canvas.width = imgSize.w;
    canvas.height = imgSize.h;
    const ctx = canvas.getContext("2d");
    if (ctx) drawBoxes(ctx, imgSize.w, imgSize.h);
  }, [boxes, imgSize, drawBoxes]);

  function getPos(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.round(e.clientX - rect.left),
      y: Math.round(e.clientY - rect.top),
    };
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (!showAnnotations) return;
    const pos = getPos(e);
    setDrawing({ startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y });
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!drawing) return;
    const pos = getPos(e);
    setDrawing((prev) => prev && { ...prev, currentX: pos.x, currentY: pos.y });

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawBoxes(ctx, canvas.width, canvas.height);
    const fill = LABEL_COLORS[activeLabel];
    const stroke = LABEL_BORDER[activeLabel];
    ctx.fillStyle = fill;
    ctx.fillRect(
      drawing.startX, drawing.startY,
      pos.x - drawing.startX, pos.y - drawing.startY
    );
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      drawing.startX, drawing.startY,
      pos.x - drawing.startX, pos.y - drawing.startY
    );
  }

  function handleMouseUp() {
    if (!drawing) return;
    const { startX, startY, currentX, currentY } = drawing;
    setDrawing(null);

    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    if (w < 5 || h < 5) return;

    const newBox: BoundingBox = {
      xmin: Math.min(startX, currentX),
      ymin: Math.min(startY, currentY),
      xmax: Math.max(startX, currentX),
      ymax: Math.max(startY, currentY),
      label: activeLabel,
      color: LABEL_BORDER[activeLabel],
    };
    updateBoxes(index, [...boxes, newBox]);
  }

  function clearBoxes() {
    updateBoxes(index, []);
  }

  const relevantCount = boxes.filter((b) => b.label === "Relevant").length;
  const irrelevantCount = boxes.filter((b) => b.label === "Irrelevant").length;

  return (
    <Card className="group overflow-hidden gap-0 py-0">
      <div className="relative border-b">
        <div className="absolute left-2 top-2 z-20 flex items-center gap-1.5">
          <Badge variant="secondary" className="text-[10px] font-mono bg-black/70 text-white/80 border border-white/10 backdrop-blur-md">
            #{index + 1}
          </Badge>
          <Badge variant="secondary" className="text-[10px] font-mono bg-black/70 text-white/80 border border-white/10 backdrop-blur-md">
            {image.score.toFixed(4)}
          </Badge>
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
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => drawing && handleMouseUp()}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${image.base64}`}
            alt={`Result ${index + 1}`}
            className={`block w-full object-contain ${
              expanded ? "max-h-[600px]" : "aspect-square"
            }`}
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              setImgSize({ w: el.clientWidth, h: el.clientHeight });
            }}
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full"
            style={{ pointerEvents: "none" }}
          />
        </div>
      </div>

      {showAnnotations && (
        <div className="space-y-2 p-3">
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant={activeLabel === "Relevant" ? "default" : "outline"}
              onClick={() => setActiveLabel("Relevant")}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === "Relevant" ? { backgroundColor: "#22c55e" } : {}}
            >
              <span className="inline-block h-2 w-2 rounded-full bg-green-500" />
              Relevant
            </Button>
            <Button
              size="sm"
              variant={activeLabel === "Irrelevant" ? "default" : "outline"}
              onClick={() => setActiveLabel("Irrelevant")}
              className="h-7 gap-1 text-xs flex-1"
              style={activeLabel === "Irrelevant" ? { backgroundColor: "#ef4444" } : {}}
            >
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
              Irrelevant
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={clearBoxes}
              className="h-7 w-7 p-0"
              disabled={boxes.length === 0}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>

          {boxes.length > 0 && (
            <div className="flex gap-2 text-[10px] text-muted-foreground">
              {relevantCount > 0 && (
                <span className="text-green-600">{relevantCount} relevant</span>
              )}
              {irrelevantCount > 0 && (
                <span className="text-red-500">{irrelevantCount} irrelevant</span>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
