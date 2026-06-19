"use client";

import { Eyebrow, SlideTitle } from "../primitives";
import { LikertBar } from "../charts";

const FEATURES = [
  { tag: "RQ1", label: "Region + text combo", value: 4.3 },
  { tag: "RQ2", label: "Region selection", value: 4.2 },
  { tag: "RQ3", label: "Few repeat irrelevants", value: 4.2 },
  { tag: "RQ1", label: "Text hints alone", value: 3.8 },
];

export function StudySubjectiveSlide({ step }: { step: number }) {
  return (
    <div className="space-y-7">
      <div className="space-y-3">
        <Eyebrow>User Study · Subjective results</Eyebrow>
        <SlideTitle>Participants rated every dimension highly.</SlideTitle>
        <p className="text-xl text-foreground/70">
          Five-point Likert ratings of each feature (n = 10).
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1.1fr_1fr] lg:items-start">
        <div className="present-card space-y-6 rounded-2xl border border-white/10 bg-white/[0.03] p-7">
          {FEATURES.map((b) => (
            <LikertBar key={b.label} tag={b.tag} label={b.label} value={b.value} />
          ))}
        </div>

        <div
          className={`space-y-4 transition-all duration-500 ${
            step >= 1 ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"
          }`}
        >
          <div className="rounded-2xl border border-primary/30 bg-primary/[0.10] p-6">
            <p className="text-xl leading-relaxed text-foreground/90">
              <span className="font-semibold text-foreground">The combination won.</span>{" "}
              Region + text together scored <span className="font-mono font-semibold text-foreground">4.3</span> —
              the highest feature item, above text alone (<span className="font-mono">3.8</span>).
              That reinforces the fusion premise behind RQ1.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
            <p className="text-xl leading-relaxed text-foreground/90">
              <span className="font-semibold text-foreground">Captioning praised.</span>{" "}
              The 3 who ran the full vision model singled out the auto-description
              as what most helped them turn vague intent into a precise next query.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
