"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ArrowRight, X } from "lucide-react";
import { SLIDES } from "@/components/present/slides";

const stepsFor = (i: number) => SLIDES[i]?.steps ?? 0;

export default function PresentPage() {
  const [pos, setPos] = useState({ i: 0, s: 0 });
  const { i: index, s: sub } = pos;
  const total = SLIDES.length;

  const go = useCallback((dir: number) => {
    setPos(({ i, s }) => {
      if (dir > 0) {
        if (s < stepsFor(i)) return { i, s: s + 1 };
        if (i < total - 1) return { i: i + 1, s: 0 };
        return { i, s };
      }
      if (s > 0) return { i, s: s - 1 };
      if (i > 0) return { i: i - 1, s: stepsFor(i - 1) };
      return { i, s };
    });
  }, [total]);

  const jumpTo = useCallback((i: number) => setPos({ i, s: 0 }), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        e.preventDefault();
        go(-1);
      } else if (e.key === "Home") {
        jumpTo(0);
      } else if (e.key === "End") {
        setPos({ i: total - 1, s: stepsFor(total - 1) });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, jumpTo, total]);

  const Slide = SLIDES[index].component;
  const slideSteps = stepsFor(index);
  const fullBleed = SLIDES[index].fullBleed ?? false;

  return (
    <div className="present-root fixed inset-0 flex flex-col overflow-hidden text-foreground">
      <div className="present-bg" />
      <div className="present-grid" />

      <div className="absolute inset-x-0 top-0 z-30 h-1 bg-white/[0.06]">
        <div
          className="h-full bg-primary transition-[width] duration-300 ease-out"
          style={{ width: `${((index + 1) / total) * 100}%` }}
        />
      </div>

      <Link
        href="/"
        aria-label="Exit presentation"
        className="absolute right-6 top-5 z-30 text-muted-foreground transition-colors hover:text-foreground"
      >
        <X className="h-5 w-5" />
      </Link>

      <main className="relative z-10 flex flex-1 items-center justify-center overflow-y-auto">
        {fullBleed ? (
          <div
            key={index}
            className="h-full w-full animate-in fade-in duration-300"
          >
            <Slide step={sub} />
          </div>
        ) : (
          <div
            key={index}
            className="w-full max-w-6xl animate-in fade-in slide-in-from-bottom-2 px-10 py-16 duration-300 lg:px-24"
          >
            <Slide step={sub} />
          </div>
        )}
      </main>

      <footer className="relative z-30 flex items-center justify-between gap-6 border-t border-white/10 bg-black/20 px-8 py-4 backdrop-blur-sm lg:px-12">
        <div className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="block h-4 w-4 rounded-sm bg-foreground"
            style={{
              boxShadow:
                "inset 0 0 0 1.5px var(--background), 0 0 0 1px var(--foreground)",
            }}
          />
          <span className="text-[15px] font-medium tracking-tight text-foreground">
            VisualReF
          </span>
          <span className="font-mono text-xs text-muted-foreground">v2</span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => go(-1)}
            disabled={index === 0 && sub === 0}
            aria-label="Previous"
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/15 text-muted-foreground transition-colors hover:border-white/30 hover:text-foreground disabled:pointer-events-none disabled:opacity-25"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>

          <div className="flex items-center gap-2 px-1">
            {SLIDES.map((s, i) => (
              <button
                key={s.id}
                onClick={() => jumpTo(i)}
                aria-label={`Go to slide ${i + 1}`}
                className={`h-2 rounded-full transition-all duration-200 ${
                  i === index
                    ? "w-7 bg-primary"
                    : "w-2 bg-white/15 hover:bg-white/35"
                }`}
              />
            ))}
          </div>

          {slideSteps > 0 && (
            <div className="ml-1 flex items-center gap-1.5 border-l border-white/10 pl-3">
              {Array.from({ length: slideSteps + 1 }).map((_, i) => (
                <span
                  key={i}
                  className={`h-1.5 w-1.5 rounded-full transition-colors duration-200 ${
                    i <= sub ? "bg-primary" : "bg-white/15"
                  }`}
                />
              ))}
            </div>
          )}

          <button
            onClick={() => go(1)}
            disabled={index === total - 1 && sub === slideSteps}
            aria-label="Next"
            className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/15 text-muted-foreground transition-colors hover:border-white/30 hover:text-foreground disabled:pointer-events-none disabled:opacity-25"
          >
            <ArrowRight className="h-5 w-5" />
          </button>
        </div>

        <div className="flex items-baseline gap-1 font-mono text-sm tabular-nums">
          <span className="text-foreground">{String(index + 1).padStart(2, "0")}</span>
          <span className="text-muted-foreground">/ {String(total).padStart(2, "0")}</span>
        </div>
      </footer>
    </div>
  );
}
