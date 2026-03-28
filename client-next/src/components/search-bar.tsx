"use client";

import { Loader2, Search } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { VanishInput } from "@/components/ui/vanish-input";

const SEARCH_PLACEHOLDERS = [
  "Type what you're looking for — e.g. dog on a beach...",
  "e.g. person riding a bicycle in a city",
  "e.g. red car on a mountain road",
  "e.g. group of people at a dinner table",
  "e.g. aerial view of a coastline at dawn",
];

interface SearchBarProps {
  onSearch: () => void;
}

export function SearchBar({ onSearch }: SearchBarProps) {
  const query = useAppStore((s) => s.query);
  const setQuery = useAppStore((s) => s.setQuery);
  const isSearching = useAppStore((s) => s.isSearching);

  return (
    <div className="flex gap-3 items-center">
      {/* vanish input takes up full width */}
      <div className="flex-1">
        <VanishInput
          placeholders={SEARCH_PLACEHOLDERS}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSubmit={(e) => { e.preventDefault(); if (query.trim() && !isSearching) onSearch(); }}
          onKeyDown={(e) => { if (e.key === "Enter" && query.trim() && !isSearching) onSearch(); }}
          disabled={isSearching}
          accentColor="indigo"
          className="h-14 text-base"
          submitIcon={
            isSearching ? (
              <Loader2 className="h-4 w-4 text-white/60 animate-spin" />
            ) : (
              <Search className="h-4 w-4 text-white/60" />
            )
          }
        />
      </div>
    </div>
  );
}
