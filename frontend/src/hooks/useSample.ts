import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { SampleData } from "../types";

const MAX_CACHED = 6;

/** Loads (and caches) the ground truth, FNO prediction and stats of one sample. */
export function useSample(key: string | null) {
  const cache = useRef(new Map<string, SampleData>());
  const [data, setData] = useState<SampleData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!key) return;
    const hit = cache.current.get(key);
    if (hit) {
      setData(hit);
      setError(null);
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    api
      .sample(key, ctrl.signal)
      .then((d) => {
        cache.current.set(key, d);
        while (cache.current.size > MAX_CACHED) {
          const oldest = cache.current.keys().next().value;
          if (oldest === undefined) break;
          cache.current.delete(oldest);
        }
        setData(d);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => ctrl.abort();
  }, [key]);

  return { data, loading, error };
}
