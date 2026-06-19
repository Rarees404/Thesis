import { Check } from "lucide-react";
import { Eyebrow } from "../primitives";

const TAKEAWAYS = [
  "Region clicks via SAM 3 replace whole-image feedback — only the object you point at feeds back.",
  "Three signals fused in SigLIP space: the crop, a caption or Visual Genome phrase, and your typed hint.",
  "Hard filtering adds a guarantee Rocchio can't: rejected look-alikes cannot return within a session.",
  "10 participants — 9 found a satisfactory image in ~2.8 rounds; region + text rated most helpful.",
];

export function ClosingSlide() {
  return (
    <div className="space-y-9">
      <div className="space-y-4">
        <Eyebrow>Conclusion</Eyebrow>
        <h2 className="max-w-4xl text-5xl font-semibold leading-[1.08] tracking-tight text-foreground lg:text-6xl">
          Search by sentence, refine by pointing —{" "}
          <span className="text-muted-foreground">
            precise, multimodal relevance feedback.
          </span>
        </h2>
      </div>

      <ul className="grid gap-3.5 md:grid-cols-2">
        {TAKEAWAYS.map((t) => (
          <li
            key={t}
            className="present-card flex gap-3.5 rounded-2xl border border-white/10 bg-white/[0.03] p-5"
          >
            <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary/40 bg-primary/10">
              <Check className="h-4 w-4 text-primary" />
            </span>
            <span className="text-xl leading-relaxed text-foreground/85">
              {t}
            </span>
          </li>
        ))}
      </ul>

      <div className="flex flex-col items-center gap-1.5 border-t border-white/10 pt-8 text-center">
        <p className="text-4xl font-semibold tracking-tight text-foreground">
          Thank you for your attention
        </p>
        <p className="text-xl text-muted-foreground">
          Rares Boghean · VisualReF v2 · Maastricht University
        </p>
        <p className="font-mono text-base text-muted-foreground">
          r.boghean@student.maastrichtuniversity.nl
        </p>
      </div>
    </div>
  );
}
