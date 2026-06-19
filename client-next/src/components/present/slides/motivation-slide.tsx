import { ArrowRight } from "lucide-react";
import { Eyebrow, SlideTitle, RQTag } from "../primitives";

const LINKS = [
  {
    problem: "The granularity problem",
    detail:
      "Marking a whole image feeds everything back into the query — background included. Liking a photo for its dog also reinforces the beach umbrella and the food stall beside it.",
    tag: "RQ2",
    theme: "Granularity",
    question:
      "Do precise region clicks help more than whole-image or box annotation?",
  },
  {
    problem: "The expressiveness problem",
    detail:
      "A clean crop carries only visual information. A vision-language model can say what is in the region in words — often a sharper feedback signal than the crop's embedding alone.",
    tag: "RQ1",
    theme: "Fusion",
    question:
      "Does combining visual, caption, and typed-text feedback in one update beat single-signal feedback?",
  },
  {
    problem: "The persistence problem",
    detail:
      "Once a user rejects something, near-duplicates keep resurfacing round after round, spending effort re-rejecting content the system was already told to avoid.",
    tag: "RQ3",
    theme: "Hard filtering",
    question:
      "Does permanently dropping images similar to rejected content stop irrelevant results from coming back?",
  },
];

export function MotivationSlide() {
  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <Eyebrow>Motivation & Research Questions</Eyebrow>
        <SlideTitle>You recognise the right image — but can&apos;t describe it.</SlideTitle>
        <p className="max-w-4xl text-2xl leading-relaxed text-foreground/80">
          This is the{" "}
          <span className="font-medium text-foreground">semantic gap</span>.
          Relevance feedback closes it by iterating — the user marks results good
          or bad and the query shifts toward what they liked. But the classic,
          whole-image version of that loop leaves three problems open. Each one
          motivates a research question.
        </p>
      </div>

      <div className="space-y-3.5">
        {LINKS.map((row) => (
          <div
            key={row.tag}
            className="present-card grid grid-cols-1 items-stretch gap-3 rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 md:grid-cols-[1fr_auto_1fr] md:items-center md:gap-4"
          >
            <div className="space-y-2.5 p-2.5">
              <p className="text-2xl font-semibold tracking-tight text-foreground">
                {row.problem}
              </p>
              <p className="text-xl leading-relaxed text-foreground/90">
                {row.detail}
              </p>
            </div>

            <div className="hidden md:flex md:items-center md:justify-center">
              <div className="flex h-11 w-11 items-center justify-center rounded-full border border-primary/30 bg-primary/10">
                <ArrowRight className="h-6 w-6 text-primary" />
              </div>
            </div>

            <div className="space-y-2.5 rounded-xl border border-primary/25 bg-primary/[0.10] p-5">
              <div className="flex items-center gap-2.5">
                <RQTag>{row.tag}</RQTag>
                <span className="font-mono text-base uppercase tracking-wider text-primary/80">
                  {row.theme}
                </span>
              </div>
              <p className="text-xl font-medium leading-relaxed text-foreground">
                {row.question}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
