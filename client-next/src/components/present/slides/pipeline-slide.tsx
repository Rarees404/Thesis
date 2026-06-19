import type { ComponentType, ReactNode } from "react";
import {
  Search,
  Database,
  LayoutGrid,
  MousePointerClick,
  Scissors,
  MessageSquareText,
  SlidersHorizontal,
  Sparkles,
  ChevronRight,
  RotateCcw,
} from "lucide-react";
import { Eyebrow, SlideTitle } from "../primitives";

type Actor = "You" | "System" | "You + System";

interface Phase {
  label: string;
  icon: ComponentType<{ className?: string }>;
  actor: Actor;
  title: string;
  summary: string;
  bullets: string[];
  visual?: ReactNode;
}

function BatchVisual() {
  return (
    <div className="flex gap-2.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <div
          key={n}
          className="flex h-14 w-14 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] font-mono text-sm text-muted-foreground"
        >
          {String(n).padStart(2, "0")}
        </div>
      ))}
    </div>
  );
}

function SegmentVisual() {
  return (
    <div className="flex flex-wrap items-center gap-4 text-lg">
      <span className="flex items-center gap-2 text-emerald-400">
        <span className="h-3.5 w-3.5 rounded-full bg-emerald-500" /> green = more like this
      </span>
      <span className="flex items-center gap-2 text-rose-400">
        <span className="h-3.5 w-3.5 rounded-full bg-rose-500" /> red = not this
      </span>
    </div>
  );
}

function BackendVisual() {
  const steps = [
    "Combine feedback",
    "Blend image + words",
    "Update search",
    "Find & filter",
    "New 5",
  ];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((s, i) => (
        <div key={s} className="flex items-center gap-2">
          {i > 0 && <ChevronRight className="h-5 w-5 text-primary" />}
          <span
            className={`rounded-lg border px-3 py-2 font-mono text-[15px] ${
              i === steps.length - 1
                ? "border-white/10 bg-white/[0.04] text-foreground/85"
                : "border-primary/30 bg-primary/[0.10] text-foreground"
            }`}
          >
            {s}
          </span>
        </div>
      ))}
    </div>
  );
}

const PHASES: Phase[] = [
  {
    label: "Search",
    icon: Search,
    actor: "You",
    title: "1 · You search in plain words",
    summary: "Just describe what you're looking for.",
    bullets: [
      "Type it like a sentence — “a dog playing on a beach at sunset.”",
      "No keywords or filters to learn.",
      "Can't phrase it perfectly? That's fine — this is only round one.",
    ],
  },
  {
    label: "Find matches",
    icon: Database,
    actor: "System",
    title: "2 · The computer finds close matches",
    summary: "Your words are compared to every image.",
    bullets: [
      "Your sentence is turned into numbers that capture its meaning.",
      "Every one of the 108,000 images was turned into numbers the same way — so they can be compared.",
      "The closest matches come back in about one second.",
    ],
  },
  {
    label: "5 results",
    icon: LayoutGrid,
    actor: "System",
    title: "3 · You get the 5 best matches",
    summary: "Shown as cards you can react to.",
    bullets: [
      "The top 5 images appear side by side.",
      "Each shows its rank — and, once you click, how sure the selection is.",
      "You react to what you see instead of rewriting your search.",
    ],
    visual: <BatchVisual />,
  },
  {
    label: "React",
    icon: MousePointerClick,
    actor: "You",
    title: "4 · On any image, tell it what you think",
    summary: "Several easy ways to give feedback.",
    bullets: [
      "Tap once to mark a whole image “yes, like this” or “no, not this.”",
      "Or click a single object inside it to be more precise.",
      "Zoom in to look closer; clear to undo.",
    ],
  },
  {
    label: "Point at it",
    icon: Scissors,
    actor: "You + System",
    title: "5 · Click an object and the AI outlines it",
    summary: "So only that object counts, not the background.",
    bullets: [
      "Click an object and the AI (SAM 3) traces its exact shape for you.",
      "Green clicks mean “more like this”; red clicks mean “not this.”",
      "Only the object you picked is used — the background is ignored.",
    ],
    visual: <SegmentVisual />,
  },
  {
    label: "Describe it",
    icon: MessageSquareText,
    actor: "System",
    title: "6 · It puts your selection into words",
    summary: "A short description makes the hint sharper.",
    bullets: [
      "The moment you select something, the system quietly writes a short description of it in the background.",
      "By the time you continue, that description is ready and shown on the card.",
      "If that model isn't running, ready-made labels from the dataset do the same job instantly.",
    ],
  },
  {
    label: "Steer",
    icon: SlidersHorizontal,
    actor: "You",
    title: "7 · Optionally nudge it with words",
    summary: "Add a little guidance before continuing.",
    bullets: [
      "Add “more of…” and “less of…” to steer the next search.",
      "Reuse the AI's description with a single tap.",
      "An “anchor” option keeps later rounds from drifting off your original idea.",
    ],
  },
  {
    label: "Improve",
    icon: Sparkles,
    actor: "System",
    title: "8 · It builds a better search — behind the scenes",
    summary: "Everything you marked becomes one improved search.",
    bullets: [
      "What you liked pulls the search toward it; what you disliked pushes it away.",
      "Your picks, the descriptions, and your typed words are all blended together.",
      "Rejected images are also blocked completely, so they can't come back.",
      "You get a fresh set of 5 — repeat until you find it (about 3 rounds on average).",
    ],
    visual: <BackendVisual />,
  },
];

export function PipelineSlide({ step }: { step: number }) {
  const active = Math.min(step, PHASES.length - 1);
  const phase = PHASES[active];
  const Icon = phase.icon;
  const isLast = active === PHASES.length - 1;

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Eyebrow>How the App Works · End to End</Eyebrow>
        <SlideTitle>From a sentence to the right image.</SlideTitle>
      </div>

      <div className="grid grid-cols-1 gap-7 lg:grid-cols-[280px_1fr]">
        <div className="flex flex-col gap-1.5">
          {PHASES.map((p, i) => {
            const PIcon = p.icon;
            const done = i < active;
            const isActive = i === active;
            return (
              <div
                key={p.label}
                className={`flex items-center gap-3 rounded-xl border px-3.5 py-2.5 transition-all duration-300 ${
                  isActive
                    ? "border-primary bg-primary/10"
                    : done
                      ? "border-white/10 bg-white/[0.03]"
                      : "border-dashed border-white/10 opacity-40"
                }`}
              >
                <span
                  className={`font-mono text-xs tabular-nums ${
                    isActive ? "text-primary" : "text-muted-foreground"
                  }`}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <PIcon
                  className={`h-5 w-5 ${
                    isActive
                      ? "text-primary"
                      : done
                        ? "text-foreground"
                        : "text-muted-foreground"
                  }`}
                />
                <span
                  className={`text-lg font-medium ${
                    isActive || done ? "text-foreground" : "text-muted-foreground"
                  }`}
                >
                  {p.label}
                </span>
              </div>
            );
          })}
          <div className="mt-1 flex items-center gap-2.5 pl-3.5 text-muted-foreground">
            <RotateCcw className={`h-5 w-5 ${isLast ? "text-primary" : ""}`} />
            <span className="text-lg">repeat until found</span>
          </div>
        </div>

        <div
          key={active}
          className="present-card animate-in fade-in slide-in-from-bottom-1 space-y-5 rounded-2xl border border-white/10 bg-white/[0.03] p-7 duration-300"
        >
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl border border-primary/30 bg-primary/10">
                <Icon className="h-7 w-7 text-primary" />
              </div>
              <div>
                <h3 className="text-3xl font-semibold tracking-tight text-foreground">
                  {phase.title}
                </h3>
                <p className="text-xl text-foreground/70">{phase.summary}</p>
              </div>
            </div>
            <ActorBadge actor={phase.actor} />
          </div>

          {phase.visual && <div className="pt-1">{phase.visual}</div>}

          <ul className="space-y-3.5">
            {phase.bullets.map((b, i) => (
              <li key={i} className="flex gap-3.5">
                <span className="mt-2.5 h-2 w-2 shrink-0 rounded-full bg-primary" />
                <span className="text-2xl leading-relaxed text-foreground/90">
                  {b}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function ActorBadge({ actor }: { actor: Actor }) {
  const system = actor !== "You";
  return (
    <span
      className={`shrink-0 rounded-full border px-4 py-1.5 font-mono text-sm font-medium uppercase tracking-wider ${
        system
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-white/15 bg-white/[0.04] text-foreground/80"
      }`}
    >
      {actor}
    </span>
  );
}
