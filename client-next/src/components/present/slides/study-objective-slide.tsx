"use client";

import { Eyebrow, SlideTitle } from "../primitives";
import { StatCard } from "../charts";

interface Card {
  group: number;
  value: string;
  unit?: string;
  label: string;
  sub?: string;
  accent?: boolean;
}

const CARDS: Card[] = [
  { group: 0, value: "9/10", label: "Found a satisfactory image", sub: "90% — one give-up (lid task)", accent: true },
  { group: 1, value: "2.8", unit: "rounds", label: "Feedback rounds", sub: "median 3 · range 2–4" },
  { group: 1, value: "180", unit: "s", label: "Median completion time", sub: "mean 199 ± 110 s" },
  { group: 2, value: "1.1", unit: "s", label: "Initial search latency", sub: "median" },
  { group: 2, value: "15.6", unit: "s", label: "Feedback-round latency", sub: "median · mostly neural encoding" },
  { group: 2, value: "281", label: "Peak session blacklist", sub: "grew monotonically · 1–16 dropped/round", accent: true },
];

export function StudyObjectiveSlide({ step }: { step: number }) {
  return (
    <div className="space-y-7">
      <div className="space-y-3">
        <Eyebrow>User Study · Objective results</Eyebrow>
        <SlideTitle>It works — fast first search, a few quick rounds.</SlideTitle>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CARDS.map((c, i) => (
          <div
            key={i}
            className={`transition-all duration-500 ${
              c.group <= step
                ? "translate-y-0 opacity-100"
                : "pointer-events-none translate-y-2 opacity-0"
            }`}
          >
            <StatCard
              value={c.value}
              unit={c.unit}
              label={c.label}
              sub={c.sub}
              accent={c.accent}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
