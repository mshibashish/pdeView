import type { Field, Meta, SampleData, SampleStats, SampleSummary, Verification } from "./types";

const BASE = "/api";

async function json<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

async function field(key: string, kind: "gt" | "pred", signal?: AbortSignal): Promise<Field> {
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

export const api = {
  meta: (signal?: AbortSignal) => json<Meta>("/meta", signal),
  samples: (signal?: AbortSignal) => json<SampleSummary[]>("/samples", signal),
  verification: (signal?: AbortSignal) => json<Verification>("/verification", signal),
  figureUrl: (name: string) => `${BASE}/report/figures/${encodeURIComponent(name)}`,
  async sample(key: string, signal?: AbortSignal): Promise<SampleData> {
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
