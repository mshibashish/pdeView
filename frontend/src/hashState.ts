// View state in the URL hash (#sample=0905&t=0.42&view=3d), so a link reopens the same view.
// The hash is rewritten with replaceState: it never adds history entries or fires hashchange.
import { useEffect, useRef } from "react";

/** Known keys, in the order they are written. Defaults are left out to keep links short. */
const ORDER = ["tab", "sample", "t", "view", "scale", "surface"];

export function readHashParams(): URLSearchParams {
  return new URLSearchParams(window.location.hash.slice(1));
}

/** Merge `patch` into the hash; null or undefined removes a key. */
export function patchHash(patch: Record<string, string | null | undefined>): void {
  const cur = readHashParams();
  for (const [k, v] of Object.entries(patch)) {
    if (v === null || v === undefined) cur.delete(k);
    else cur.set(k, v);
  }
  const next = new URLSearchParams();
  for (const k of ORDER) if (cur.has(k)) next.set(k, cur.get(k)!);
  for (const [k, v] of cur) if (!ORDER.includes(k)) next.set(k, v);
  const hash = next.toString();
  if (hash !== window.location.hash.slice(1)) {
    window.history.replaceState(window.history.state, "", hash ? `#${hash}` : window.location.pathname + window.location.search);
  }
}

/** Run `fn` when the user edits the hash or follows a #link (not on our own replaceState writes). */
export function useHashChange(fn: (p: URLSearchParams) => void): void {
  const ref = useRef(fn);
  useEffect(() => {
    ref.current = fn;
  });
  useEffect(() => {
    const on = () => ref.current(readHashParams());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
}

/** Index of the time closest to `value` (a "t" parameter); null if it does not parse. */
export function frameAt(times: number[], value: string | null): number | null {
  const v = value === null ? NaN : Number(value);
  if (!Number.isFinite(v) || times.length === 0) return null;
  let best = 0;
  for (let i = 1; i < times.length; i++) if (Math.abs(times[i] - v) < Math.abs(times[best] - v)) best = i;
  return best;
}
