"use client";

import { Loader2, Search, CornerDownLeft } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";

const PLACEHOLDER = "dog playing fetch on a beach at sunset";

interface SearchBarProps {
  onSearch: () => void;
}

export function SearchBar({ onSearch }: SearchBarProps) {
  const query = useAppStore((s) => s.query);
  const setQuery = useAppStore((s) => s.setQuery);
  const isSearching = useAppStore((s) => s.isSearching);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (query.trim() && !isSearching) onSearch();
      }}
      className="flex items-center gap-2"
    >
      <label className="relative flex-1">
        <span className="sr-only">Search query</span>
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={PLACEHOLDER}
          disabled={isSearching}
          className="h-11 w-full rounded-md border border-input bg-card pl-9 pr-20 text-[14px] tracking-tight text-foreground outline-none transition-colors duration-100 placeholder:text-muted-foreground/60 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <kbd className="pointer-events-none absolute right-3 top-1/2 hidden h-5 -translate-y-1/2 items-center gap-1 rounded border border-border bg-background px-1.5 font-mono text-[10px] text-muted-foreground sm:inline-flex">
          <CornerDownLeft className="h-3 w-3" />
        </kbd>
      </label>

      <Button
        type="submit"
        disabled={!query.trim() || isSearching}
        size="default"
        className="h-11 px-4"
      >
        {isSearching ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Searching</span>
          </>
        ) : (
          <span>Search</span>
        )}
      </Button>
    </form>
  );
}
