import type {
  CustomRun, DamParams, Field, Generalization, Heldout, Meta, SampleData, SampleStats,
  SampleSummary, Verification,
} from "./types";
import type { SimProgress } from "./static/protocol";

const BASE = "/api";

/** Static build for GitHub Pages (VITE_STATIC=1): precomputed files, custom runs in the browser. */
export const STATIC = import.meta.env.VITE_STATIC === "1";

/** Loaded only in static builds, so the server build carries no static-mode code. */
const staticApi = () => import("./static/staticApi");

async function json<T>(path: string, signal?: AbortSignal): Promise<T> {
  if (STATIC) return (await staticApi()).staticJson<T>(path, signal);
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

async function field(key: string, kind: "gt" | "pred", signal?: AbortSignal): Promise<Field> {
  if (STATIC) return (await staticApi()).staticField(key, kind, signal);
  const res = await fetch(`${BASE}/samples/${encodeURIComponent(key)}/field?kind=${kind}`, {
    signal,
  });
  if (!res.ok) throw new Error(`field ${key}/${kind}: HTTP ${res.status}`);
  const shape = (res.headers.get("X-Shape") ?? "").split(",").map(Number);
  if (shape.length !== 3 || shape.some((n) => !Number.isFinite(n) || n <= 0)) {
    throw new Error(`field ${key}/${kind}: missing or invalid X-Shape header`);
  }
  const [nt, nx, ny] = shape;
  const data = new Float32Array(await res.arrayBuffer());
  if (data.length !== nt * nx * ny) throw new Error(`field ${key}/${kind}: size mismatch`);
  return { data, nt, nx, ny };
}

export type { SimProgress };

export const api = {
  meta: (signal?: AbortSignal) => json<Meta>("/meta", signal),
  samples: (signal?: AbortSignal) => json<SampleSummary[]>("/samples", signal),
  verification: (signal?: AbortSignal) => json<Verification>("/verification", signal),
  generalization: (signal?: AbortSignal) => json<Generalization>("/generalization", signal),
  heldout: (signal?: AbortSignal) => json<Heldout>("/heldout", signal),
  async simulate(p: DamParams, signal?: AbortSignal, onProgress?: (p: SimProgress) => void): Promise<CustomRun> {
    if (STATIC) return (await staticApi()).staticSimulate(p, signal, onProgress);
    const res = await fetch(`${BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(p),
      signal,
    });
    if (!res.ok) throw new Error(`simulate: HTTP ${res.status}`);
    return (await res.json()) as CustomRun;
  },
  figureUrl: (name: string) =>
    STATIC
      ? `${import.meta.env.BASE_URL}static-api/figures/${encodeURIComponent(name)}`
      : `${BASE}/report/figures/${encodeURIComponent(name)}`,
  async sample(key: string, signal?: AbortSignal): Promise<SampleData> {
    if (STATIC) {
      const hit = (await staticApi()).customSample(key);
      if (hit) return hit;
    }
    // The prediction may need to be computed server-side (~1 s); fetch all three in parallel.
    const [gt, pred, stats] = await Promise.all([
      field(key, "gt", signal),
      field(key, "pred", signal),
      json<SampleStats>(`/samples/${encodeURIComponent(key)}/stats`, signal),
    ]);
    return { key, gt, pred, stats };
  },
};

/** Value of a [T, X, Y] field at (t, ix, iy). */
export function at(f: Field, t: number, ix: number, iy: number): number {
  return f.data[(t * f.nx + ix) * f.ny + iy];
}

/** View of one frame (no copy). */
export function frame(f: Field, t: number): Float32Array {
  const n = f.nx * f.ny;
  return f.data.subarray(t * n, (t + 1) * n);
}
