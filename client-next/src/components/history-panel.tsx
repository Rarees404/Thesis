"use client";

import { useAppStore } from "@/lib/store";

export function HistoryPanel() {
  const history = useAppStore((s) => s.history);

  if (history.length === 0) return null;

  return (
    <section className="rounded-md border border-border bg-card">
      <header className="flex items-center justify-between border-b border-border px-5 py-3">
        <h3 className="text-sm font-medium tracking-tight text-foreground">
          History
        </h3>
        <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
          {history.length} {history.length === 1 ? "round" : "rounds"}
        </span>
      </header>

      <div className="max-h-[260px] space-y-3 overflow-y-auto px-5 py-3">
        {history.map((round) => (
          <div key={round.round} className="space-y-1.5">
            <div className="flex items-baseline justify-between font-mono text-[10.5px] tabular-nums text-muted-foreground">
              <span>round {round.round}</span>
              <span>{new Date(round.timestamp).toLocaleTimeString()}</span>
            </div>
            <div className="flex gap-1 overflow-x-auto pb-1">
              {round.images.slice(0, 5).map((img, i) => (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  key={i}
                  src={`data:image/png;base64,${img.base64}`}
                  alt={`Round ${round.round} result ${i + 1}`}
                  className="h-12 w-12 shrink-0 rounded-sm border border-border object-cover"
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
