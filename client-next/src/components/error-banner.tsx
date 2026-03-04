"use client";

import { useAppStore } from "@/lib/store";
import { XCircle, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ErrorBanner() {
  const error = useAppStore((s) => s.error);
  const setError = useAppStore((s) => s.setError);

  if (!error) return null;

  return (
    <div className="rounded-xl border border-red-500/20 bg-red-500/[0.08] p-4 backdrop-blur-xl">
      <div className="flex items-start gap-3">
        <XCircle className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 text-sm text-red-300">{error}</div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setError(null)}
          className="h-6 w-6 p-0 text-red-400 hover:text-red-300 hover:bg-red-500/10"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
