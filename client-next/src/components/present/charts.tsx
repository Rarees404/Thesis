"use client";

import { useEffect, useState, type ReactNode } from "react";

export function StatCard({
  value,
  unit,
  label,
  sub,
  accent,
}: {
  value: string;
  unit?: string;
  label: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`present-card flex flex-col gap-1.5 rounded-2xl border p-6 ${
        accent
          ? "border-primary/40 bg-primary/[0.10]"
          : "border-white/10 bg-white/[0.03]"
      }`}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-5xl font-semibold tracking-tight tabular-nums text-foreground lg:text-7xl">
          {value}
        </span>
        {unit && (
          <span className="text-2xl font-medium text-muted-foreground">
            {unit}
          </span>
        )}
      </div>
      <span className="text-xl font-medium text-foreground">{label}</span>
      {sub && <span className="text-base text-foreground/65">{sub}</span>}
    </div>
  );
}

export function LikertBar({
  label,
  value,
  max = 5,
  tag,
}: {
  label: ReactNode;
  value: number;
  max?: number;
  tag?: string;
}) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setW((value / max) * 100));
    return () => cancelAnimationFrame(id);
  }, [value, max]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {tag && (
            <span className="inline-flex items-center rounded border border-primary/40 bg-primary/10 px-2 py-0.5 font-mono text-sm font-medium text-primary">
              {tag}
            </span>
          )}
          <span className="text-xl text-foreground">{label}</span>
        </div>
        <span className="font-mono text-xl font-semibold tabular-nums text-foreground">
          {value.toFixed(1)}
        </span>
      </div>
      <div className="h-3.5 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-700 ease-out"
          style={{ width: `${w}%` }}
        />
      </div>
    </div>
  );
}
