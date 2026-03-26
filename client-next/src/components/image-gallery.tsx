"use client";

import { useAppStore } from "@/lib/store";
import { ImageCard } from "./image-card";
import { Skeleton } from "@/components/ui/skeleton";

export function ImageGallery() {
  const images = useAppStore((s) => s.images);
  const isSearching = useAppStore((s) => s.isSearching);
  const isApplyingFeedback = useAppStore((s) => s.isApplyingFeedback);
  const round = useAppStore((s) => s.round);

  const loading = isSearching || isApplyingFeedback;

  if (loading) {
    return (
      <div className="space-y-4">
        <h2 className="text-xs font-rajdhani font-semibold tracking-[0.2em] text-red-400/60 uppercase">
          {isSearching ? "ACQUIRING INTEL..." : "PROCESSING FEEDBACK..."}
        </h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="aspect-square w-full" />
              <Skeleton className="h-4 w-3/4" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (images.length === 0) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-rajdhani font-semibold tracking-[0.2em] text-neutral-500 uppercase">
          INTEL ASSETS
          <span className="ml-3 text-amber-500 font-mono">{images.length} RESULTS</span>
          {round > 1 && (
            <span className="ml-3 text-red-400 font-mono">
              (REFINEMENT PHASE {round - 1})
            </span>
          )}
        </h2>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        {images.map((img, i) => (
          <ImageCard key={`${img.path}-${i}`} image={img} index={i} />
        ))}
      </div>
    </div>
  );
}
