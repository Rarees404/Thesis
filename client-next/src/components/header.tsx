"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { XOctagon } from "lucide-react";
import { useAppStore } from "@/lib/store";

interface HeaderProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: "search", label: "INTEL" },
  { id: "dashboard", label: "COMMAND CENTER" },
];

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const round = useAppStore((s) => s.round);
  const reset = useAppStore((s) => s.reset);

  return (
    <header className="sticky top-0 z-40 border-b border-red-600/10 bg-black/60 backdrop-blur-2xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-4">
          <h1 className="font-rajdhani text-lg font-bold tracking-[0.15em] text-red-500 uppercase">
            OPERATION: <span className="text-red-400">VISUALREF</span>
          </h1>
          {round > 0 && (
            <Badge variant="outline" className="text-[10px] font-mono border-red-500/20 bg-red-500/5 text-red-400 tracking-widest">
              PHASE {String(round).padStart(2, "0")}
            </Badge>
          )}

          <nav className="ml-4 flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`px-3 py-1.5 text-xs font-rajdhani font-semibold tracking-[0.15em] transition-all duration-200 border ${
                  activeTab === tab.id
                    ? "bg-red-600/15 text-red-400 border-red-600/30 shadow-inner shadow-red-600/5"
                    : "text-neutral-500 border-transparent hover:bg-red-600/5 hover:text-neutral-300 hover:border-red-600/10"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={reset}
          className="gap-1.5 text-neutral-500 hover:text-red-400 hover:bg-red-600/10 font-rajdhani tracking-wider text-xs font-semibold"
        >
          <XOctagon className="h-4 w-4" />
          <span className="hidden sm:inline">ABORT</span>
        </Button>
      </div>
    </header>
  );
}
