import { FlaskConical, Cpu, Server } from "lucide-react";
import { Eyebrow, SlideTitle } from "../primitives";

const LIMITS = [
  {
    icon: FlaskConical,
    title: "Single full-system condition",
    body: "Ten participants, one task each — no image-only or text-only baseline, so no component (RQ1 fusion, RQ2 granularity) can be statistically isolated.",
  },
  {
    icon: Cpu,
    title: "Captioning evidence is qualitative",
    body: "Live Llama 3.2-Vision ran for only 3 of 10 — the hardware was briefly available — so the caption's benefit is reported qualitatively, not measured.",
  },
  {
    icon: Server,
    title: "Single-process backend",
    body: "SAM is serialised behind a lock and query state lives in memory — lost on restart. Not yet a multi-user service.",
  },
];

export function LimitationsSlide() {
  return (
    <div className="space-y-7">
      <div className="space-y-3">
        <Eyebrow>Limitations</Eyebrow>
        <SlideTitle>What this study does — and doesn&apos;t — show.</SlideTitle>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {LIMITS.map((l) => {
          const Icon = l.icon;
          return (
            <div
              key={l.title}
              className="present-card space-y-2.5 rounded-2xl border border-white/10 bg-white/[0.03] p-6"
            >
              <Icon className="h-7 w-7 text-chart-3" />
              <p className="text-xl font-semibold leading-tight text-foreground">
                {l.title}
              </p>
              <p className="text-lg leading-relaxed text-foreground/80">
                {l.body}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
