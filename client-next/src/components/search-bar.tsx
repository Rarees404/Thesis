"use client";

import { Loader2, Search } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { VanishInput } from "@/components/ui/vanish-input";

const SEARCH_PLACEHOLDERS = [
  "ENTER TARGET DESCRIPTION...",
  "e.g. HOSTILE VEHICLE ON MOUNTAIN ROAD",
  "e.g. PERSON OF INTEREST NEAR COASTLINE",
  "e.g. AERIAL SURVEILLANCE OF COMPOUND",
  "e.g. UNIDENTIFIED OBJECT IN URBAN AREA",
  "e.g. TACTICAL FORMATION IN OPEN FIELD",
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
      <div className="flex-1">
        <VanishInput
          placeholders={SEARCH_PLACEHOLDERS}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSubmit={(e) => { e.preventDefault(); if (query.trim() && !isSearching) onSearch(); }}
          onKeyDown={(e) => { if (e.key === "Enter" && query.trim() && !isSearching) onSearch(); }}
          disabled={isSearching}
          accentColor="red"
          className="h-14 text-base"
          submitIcon={
            isSearching ? (
              <Loader2 className="h-4 w-4 text-red-400/60 animate-spin" />
            ) : (
              <Search className="h-4 w-4 text-red-400/60" />
            )
          }
        />
      </div>
    </div>
  );
}
