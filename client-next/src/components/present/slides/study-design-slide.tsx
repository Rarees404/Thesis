import { Users, Target, FlaskConical } from "lucide-react";
import { Eyebrow, SlideTitle } from "../primitives";

const TARGETS = [
  { name: "Zebra in a dry landscape", who: "P01 · P04 · P08", hit: "2 / 3 found" },
  { name: "Computer tower in an office", who: "P02 · P05 · P07", hit: "3 / 3 found" },
  { name: "Container-lid scene", who: "P03 · P06 · P09 · P10", hit: "1 / 4 found" },
];

export function StudyDesignSlide() {
  return (
    <div className="space-y-7">
      <div className="space-y-3">
        <Eyebrow>User Study · Design</Eyebrow>
        <SlideTitle>How we tested it — real people, real targets.</SlideTitle>
        <p className="max-w-4xl text-2xl leading-relaxed text-foreground/80">
          Ten participants (P01–P10), each given <span className="font-medium text-foreground">one
          target-finding task</span>: shown an example image, they retrieved a
          matching Visual Genome image — starting from their own free-text query
          and refining over rounds until satisfied, or giving up.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="present-card space-y-2.5 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Users className="h-7 w-7 text-primary" />
          <p className="text-3xl font-semibold text-foreground">10 participants</p>
          <p className="text-lg text-foreground/75">
            All familiar with image search. One task each.
          </p>
        </div>
        <div className="present-card space-y-2.5 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <Target className="h-7 w-7 text-primary" />
          <p className="text-3xl font-semibold text-foreground">3 categories</p>
          <p className="text-lg text-foreground/75">
            Each attempted by several participants.
          </p>
        </div>
        <div className="present-card space-y-2.5 rounded-2xl border border-white/10 bg-white/[0.03] p-6">
          <FlaskConical className="h-7 w-7 text-primary" />
          <p className="text-3xl font-semibold text-foreground">Full pipeline</p>
          <p className="text-lg text-foreground/75">
            SAM 3 clicks · labels · text hints · phrase index · hard filter.
          </p>
        </div>
      </div>

      <div className="grid gap-3.5 md:grid-cols-3">
        {TARGETS.map((t) => (
          <div
            key={t.name}
            className="flex flex-col gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-5"
          >
            <span className="text-xl font-medium text-foreground">{t.name}</span>
            <span className="font-mono text-base text-foreground/65">{t.who}</span>
            <span className="mt-0.5 text-lg font-semibold text-primary">{t.hit}</span>
          </div>
        ))}
      </div>

      <div className="rounded-xl border border-chart-3/30 bg-chart-3/[0.08] p-5">
        <p className="text-lg leading-relaxed text-foreground/85">
          <span className="font-semibold text-foreground">Honest scope:</span>{" "}
          live Llama 3.2-Vision captioning needs GPU hardware that was only
          briefly available, so the full vision model ran for{" "}
          <span className="font-medium text-foreground">3 of 10</span> participants.
          The other 7 used Visual Genome phrases — exactly the zero-latency
          fallback the design intends.
        </p>
      </div>
    </div>
  );
}
