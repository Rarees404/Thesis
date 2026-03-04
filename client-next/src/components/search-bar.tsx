"use client";

import { Button } from "@/components/ui/button";
import { Search, Loader2 } from "lucide-react";
import { useAppStore } from "@/lib/store";

interface SearchBarProps {
  onSearch: () => void;
}

export function SearchBar({ onSearch }: SearchBarProps) {
  const query = useAppStore((s) => s.query);
  const setQuery = useAppStore((s) => s.setQuery);
  const isSearching = useAppStore((s) => s.isSearching);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && query.trim() && !isSearching) {
      onSearch();
    }
  }

  return (
    <div className="flex gap-2">
      <div className="relative flex-1">
        <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-white/30" />
        <input
          placeholder="Describe the image you would like to find..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSearching}
          className="h-11 w-full rounded-xl border border-white/[0.08] bg-white/[0.04] pl-10 pr-4 text-sm text-white placeholder:text-white/30 backdrop-blur-xl transition-colors focus:border-indigo-500/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/30 disabled:opacity-50"
        />
      </div>
      <Button
        onClick={onSearch}
        disabled={!query.trim() || isSearching}
        size="lg"
        className="gap-2 px-6 bg-indigo-600 hover:bg-indigo-500 border border-indigo-500/30"
      >
        {isSearching ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Search className="h-4 w-4" />
        )}
        Search
      </Button>
    </div>
  );
}
