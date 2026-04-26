"use client";

import { useAppStore } from "@/lib/store";
import { X } from "lucide-react";

export function ErrorBanner() {
  const error = useAppStore((s) => s.error);
  const setError = useAppStore((s) => s.setError);

  if (!error) return null;

  return (
    <div className="flex items-start gap-3 rounded-md border border-destructive/30 bg-destructive/[0.06] px-4 py-3">
      <span
        aria-hidden
        className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full bg-destructive"
      />
      <p className="flex-1 text-sm leading-snug text-foreground">{error}</p>
      <button
        onClick={() => setError(null)}
        className="-mr-1 -mt-1 rounded p-1 text-muted-foreground transition-colors duration-100 hover:bg-foreground/5 hover:text-foreground"
        aria-label="Dismiss"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
