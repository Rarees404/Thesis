import type { ComponentType } from "react";

interface SectionProps {
  part: string;
  title: string;
  subtitle: string;
}

function SectionDivider({ part, title, subtitle }: SectionProps) {
  return (
    <div className="relative flex h-full w-full items-center overflow-hidden px-14 py-20 lg:px-28">
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(70% 90% at 10% 50%, rgba(72,118,255,0.32), transparent 62%), linear-gradient(120deg, rgba(72,118,255,0.18), transparent 58%)",
        }}
      />
      <div className="relative max-w-4xl space-y-7">
        <div className="flex items-center gap-4">
          <span className="h-px w-16 bg-primary" />
          <span className="font-mono text-lg uppercase tracking-[0.32em] text-primary">
            {part}
          </span>
        </div>
        <h2 className="text-6xl font-semibold leading-[1.02] tracking-tight text-foreground lg:text-8xl">
          {title}
        </h2>
        <p className="max-w-2xl text-2xl leading-relaxed text-foreground/75">
          {subtitle}
        </p>
      </div>
    </div>
  );
}

function makeSection(props: SectionProps): ComponentType<{ step: number }> {
  const Comp = () => <SectionDivider {...props} />;
  Comp.displayName = `Section(${props.title})`;
  return Comp;
}

export const SectionMotivation = makeSection({
  part: "Part 01",
  title: "Motivation",
  subtitle: "Why this problem matters — and the questions it raises.",
});

export const SectionHowItWorks = makeSection({
  part: "Part 02",
  title: "How it works",
  subtitle: "From a sentence to the right image, end to end.",
});

export const SectionUserStudy = makeSection({
  part: "Part 03",
  title: "User study",
  subtitle: "Ten people, real targets — what we measured and found.",
});

export const SectionReflections = makeSection({
  part: "Part 04",
  title: "Reflections",
  subtitle: "What the study does and doesn't show — and the takeaways.",
});
