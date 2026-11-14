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

/** The viewer state a link carries besides the tab and sample. */
export interface ViewState {
  frame: number;
  mode: "2d" | "3d";
  errScale: "frame" | "rollout";
  surfaceSource: "gt" | "pred";
}

/** Hash parameters for a view (#t=0.42&view=3d&scale=frame&surface=gt); defaults are left out. */
export function viewToHash(times: number[], v: ViewState): Record<string, string | null> {
  return {
    t: v.frame > 0 ? times[v.frame].toFixed(2) : null,
    view: v.mode === "3d" ? "3d" : null,
    scale: v.errScale === "frame" ? "frame" : null,
    surface: v.surfaceSource === "gt" ? "gt" : null,
  };
}

/** The inverse of viewToHash; missing or unknown values fall back to the defaults. */
export function viewFromHash(times: number[], p: URLSearchParams): ViewState {
  return {
    frame: frameAt(times, p.get("t")) ?? 0,
    mode: p.get("view") === "3d" ? "3d" : "2d",
    errScale: p.get("scale") === "frame" ? "frame" : "rollout",
    surfaceSource: p.get("surface") === "gt" ? "gt" : "pred",
  };
}
