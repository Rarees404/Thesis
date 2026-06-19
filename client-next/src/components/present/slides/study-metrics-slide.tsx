import { CheckCircle2, Crosshair, ListChecks, Timer } from "lucide-react";
import { Eyebrow, SlideTitle, RQTag } from "../primitives";

const FAMILIES = [
  {
    icon: CheckCircle2,
    name: "Task success",
    body: "Did they find a satisfactory image? Did the exact target appear, and at what rank? How many rounds, how long?",
  },
  {
    icon: Crosshair,
    name: "Retrieval precision",
    body: "Precision@5 and @10 at the stopping round, scored against Visual Genome scene-graph labels as ground truth — independent of when the user chose to stop.",
  },
  {
    icon: ListChecks,
    name: "Subjective (8 Likert items)",
    body: "Five-point ratings on region help, text help, region+text combo, repeat suppression, ease, responsiveness, confidence, and would-use-again.",
  },
  {
    icon: Timer,
    name: "Latency telemetry",
    body: "Initial search vs feedback-round time, plus per-round blacklist / boostlist sizes and images dropped — logged automatically every round.",
  },
];

export function StudyMetricsSlide() {
  return (
    <div className="space-y-7">
      <div className="space-y-3">
        <Eyebrow>User Study · What we measured</Eyebrow>
        <SlideTitle>Four metric families, each tied to a question.</SlideTitle>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {FAMILIES.map((f) => {
          const Icon = f.icon;
          return (
            <div
              key={f.name}
              className="present-card space-y-2.5 rounded-2xl border border-white/10 bg-white/[0.03] p-6"
            >
              <div className="flex items-center gap-3">
                <Icon className="h-7 w-7 text-primary" />
                <span className="text-2xl font-semibold text-foreground">
                  {f.name}
                </span>
              </div>
              <p className="text-lg leading-relaxed text-foreground/80">
                {f.body}
              </p>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-white/10 bg-white/[0.03] px-6 py-5">
        <span className="text-lg font-medium text-foreground">Mapped to:</span>
        <span className="flex items-center gap-2.5 text-lg text-foreground/85">
          <RQTag>RQ1</RQTag> fusion — text_help, combo_help
        </span>
        <span className="flex items-center gap-2.5 text-lg text-foreground/85">
          <RQTag>RQ2</RQTag> granularity — region_help
        </span>
        <span className="flex items-center gap-2.5 text-lg text-foreground/85">
          <RQTag>RQ3</RQTag> hard filter — no_repeats + telemetry
        </span>
      </div>
    </div>
  );
}
