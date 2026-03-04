"use client";

import { useAppStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { History } from "lucide-react";

export function HistoryPanel() {
  const history = useAppStore((s) => s.history);

  if (history.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base text-white/80">
          <History className="h-4 w-4 text-white/40" />
          Feedback History
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px] pr-4">
          <div className="space-y-4">
            {history.map((round) => (
              <div key={round.round} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-[10px] border-white/10 bg-white/5 text-white/60">
                    Round {round.round}
                  </Badge>
                  <span className="text-[10px] text-white/30">
                    {new Date(round.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="flex gap-1 overflow-x-auto pb-1">
                  {round.images.slice(0, 5).map((img, i) => (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      key={i}
                      src={`data:image/png;base64,${img.base64}`}
                      alt={`Round ${round.round} result ${i + 1}`}
                      className="h-14 w-14 rounded-lg border border-white/[0.06] object-cover flex-shrink-0"
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
