import type { ReactNode } from "react";

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-base font-medium uppercase tracking-[0.24em] text-primary lg:text-lg">
      {children}
    </p>
  );
}

export function SlideTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-4xl font-semibold leading-[1.08] tracking-tight text-foreground lg:text-6xl">
      {children}
    </h2>
  );
}

export function SlideHeader({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
}) {
  return (
    <div className="space-y-4">
      <Eyebrow>{eyebrow}</Eyebrow>
      <SlideTitle>{title}</SlideTitle>
      {subtitle && (
        <p className="max-w-3xl text-xl leading-relaxed text-foreground/80 lg:text-2xl">
          {subtitle}
        </p>
      )}
    </div>
  );
}

export function RQTag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex shrink-0 items-center rounded-md border border-primary/40 bg-primary/10 px-3 py-1 font-mono text-base font-medium tracking-wide text-primary">
      {children}
    </span>
  );
}
