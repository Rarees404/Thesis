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
    <Card className="border-red-600/10">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base text-neutral-400">
          <History className="h-4 w-4 text-red-500/60" />
          MISSION LOG
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px] pr-4">
          <div className="space-y-4">
            {history.map((round) => (
              <div key={round.round} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-[10px] border-red-600/20 bg-red-600/5 text-red-400 font-mono tracking-widest">
                    PHASE {String(round.round).padStart(2, "0")}
                  </Badge>
                  <span className="text-[10px] font-mono text-neutral-600 tracking-wider">
                    {new Date(round.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="flex gap-1 overflow-x-auto pb-1">
                  {round.images.slice(0, 5).map((img, i) => (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      key={i}
                      src={`data:image/png;base64,${img.base64}`}
                      alt={`Phase ${round.round} asset ${i + 1}`}
                      className="h-14 w-14 border border-red-600/10 object-cover flex-shrink-0"
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
