"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import { useAppStore } from "@/lib/store";

interface HeaderProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: "search", label: "Search" },
  { id: "dashboard", label: "Dashboard" },
];

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const round = useAppStore((s) => s.round);
  const reset = useAppStore((s) => s.reset);

  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-white/[0.03] backdrop-blur-2xl">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
            VisualReF
          </h1>
          {round > 0 && (
            <Badge variant="outline" className="text-[10px] border-white/10 bg-white/5">
              Round {round}
            </Badge>
          )}

          <nav className="ml-4 flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-white/10 text-white shadow-inner shadow-white/5"
                    : "text-white/50 hover:bg-white/5 hover:text-white/80"
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
          className="gap-1.5 text-white/60 hover:text-white hover:bg-white/5"
        >
          <RotateCcw className="h-4 w-4" />
          <span className="hidden sm:inline">Reset</span>
        </Button>
      </div>
    </header>
  );
}
