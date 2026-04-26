"use client";

import { useAppStore } from "@/lib/store";
import { ImageCard } from "./image-card";

export function ImageGallery() {
  const images = useAppStore((s) => s.images);
  const isSearching = useAppStore((s) => s.isSearching);
  const isApplyingFeedback = useAppStore((s) => s.isApplyingFeedback);
  const round = useAppStore((s) => s.round);

  const loading = isSearching || isApplyingFeedback;

  if (loading) {
    return (
      <section className="space-y-4">
        <SectionHeading
          label={isSearching ? "Searching" : "Refining"}
          count={null}
        />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div
              key={i}
              className="aspect-square w-full animate-pulse rounded-md border border-border bg-card"
            />
          ))}
        </div>
      </section>
    );
  }

  if (images.length === 0) return null;

  return (
    <section className="space-y-4">
      <SectionHeading label="Results" count={images.length} round={round} />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {images.map((img, i) => (
          <ImageCard key={`${img.path}-${i}`} image={img} index={i} />
        ))}
      </div>
    </section>
  );
}

function SectionHeading({
  label,
  count,
  round,
}: {
  label: string;
  count: number | null;
  round?: number;
}) {
  return (
    <div className="flex items-baseline justify-between border-b border-border pb-2">
      <h2 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        {label}
        {count !== null && (
          <span className="ml-2 font-sans text-[12px] tracking-tight text-foreground tabular-nums">
            {count}
          </span>
        )}
      </h2>
      {round !== undefined && round > 1 && (
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
          round {round}
        </span>
      )}
    </div>
  );
}
