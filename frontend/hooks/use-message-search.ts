/**
 * useMessageSearch — debounced search over the IndexedDB message archive.
 *
 * Debounced because the query is a cursor scan, not an index lookup: cost
 * grows with archive size, so firing one per keystroke would queue scans
 * faster than they complete. 200ms is below the threshold where typing feels
 * laggy and above the interval between keystrokes.
 *
 * `cancelled` guards the async gap. Two searches started in order can finish
 * out of order, and without the flag a slow scan for "inv" could land after
 * the fast one for "invoice" and replace the right results with stale ones.
 */
"use client";

import { useEffect, useState } from "react";
import { MIN_QUERY_LENGTH, searchMessages, type StoredMessage } from "@/lib/message-db";

const DEBOUNCE_MS = 200;

interface UseMessageSearchResult {
  results: StoredMessage[];
  /** True between a query changing and its results landing. */
  isSearching: boolean;
  /** True when the query is long enough to have run at all. */
  isActive: boolean;
}

export function useMessageSearch(query: string): UseMessageSearchResult {
  const [results, setResults] = useState<StoredMessage[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const isActive = query.trim().length >= MIN_QUERY_LENGTH;

  useEffect(() => {
    if (!isActive) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    let cancelled = false;

    const timer = window.setTimeout(() => {
      void searchMessages(query).then((found) => {
        if (cancelled) return;
        setResults(found);
        setIsSearching(false);
      });
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, isActive]);

  return { results, isSearching, isActive };
}
