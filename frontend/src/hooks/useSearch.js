import { useCallback, useState } from "react";
import { fetchTopPapers, saveSession } from "../api/papersApi";

export function useSearch() {
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const runSearch = useCallback(async (q) => {
    const trimmed = q.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      const data = await fetchTopPapers(trimmed);
      setPapers(data.papers || []);
      setQuery(trimmed);

      // best-effort session save; failure here shouldn't block the results
      saveSession(
        trimmed,
        (data.papers || []).map((p) => p.id)
      ).catch(() => {});
    } catch (err) {
      setError(err.message || "Search failed");
      setPapers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  return { query, papers, loading, error, hasSearched, runSearch };
}
