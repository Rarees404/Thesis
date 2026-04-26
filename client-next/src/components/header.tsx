"use client";

import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import { useAppStore } from "@/lib/store";

interface HeaderProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: "search", label: "Search" },
  { id: "dashboard", label: "System" },
];

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const round = useAppStore((s) => s.round);
  const reset = useAppStore((s) => s.reset);

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/95">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-6 px-6 lg:px-8">
        {/* Brand */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span
              aria-hidden
              className="block h-3.5 w-3.5 rounded-sm bg-foreground"
              style={{
                boxShadow:
                  "inset 0 0 0 1.5px var(--background), 0 0 0 1px var(--foreground)",
              }}
            />
            <span className="text-[13px] font-medium tracking-tight text-foreground">
              VisualReF
            </span>
            <span className="hidden font-mono text-[10px] text-muted-foreground sm:inline">
              v2
            </span>
          </div>

          {/* Tabs */}
          <nav className="flex items-center gap-1">
            {TABS.map((tab) => {
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`relative h-12 px-2.5 text-[13px] tracking-tight transition-colors duration-100 ${
                    active
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab.label}
                  {active && (
                    <span className="absolute inset-x-2.5 -bottom-px h-px bg-foreground" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Right side: round indicator + reset */}
        <div className="flex items-center gap-3">
          {round > 0 && (
            <span className="hidden font-mono text-[11px] text-muted-foreground sm:inline tabular-nums">
              round&nbsp;{round}
            </span>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={reset}
            className="-mr-2 h-7 gap-1.5 px-2 text-[12px]"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Reset</span>
          </Button>
        </div>
      </div>
    </header>
  );
}
