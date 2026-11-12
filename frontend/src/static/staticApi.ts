// Static-site implementation of the API (VITE_STATIC=1): precomputed files under static-api/
// plus in-browser custom runs. Layout and formats: docs/static-site.md.
import type { CustomRun, DamParams, Field, Meta, SampleData } from "../types";
import { decodeField } from "./codec";
import type { SimProgress, SimRequest, SimResponse } from "./protocol";

const ROOT = `${import.meta.env.BASE_URL}static-api`;

export async function staticJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${ROOT}${path}.json`, { signal });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

export async function staticField(key: string, kind: "gt" | "pred", signal?: AbortSignal): Promise<Field> {
  const res = await fetch(`${ROOT}/samples/${encodeURIComponent(key)}/${kind}.bin`, { signal });
  if (!res.ok) throw new Error(`field ${key}/${kind}: HTTP ${res.status}`);
  return decodeField(await res.arrayBuffer(), `field ${key}/${kind}`);
}

// ---------------------------------------------------------------- custom runs (in the browser)
const MAX_CUSTOM = 8;
const custom = new Map<string, { data: SampleData; run: CustomRun }>();
let metaPromise: Promise<Meta> | null = null;
let worker: Worker | null = null;
let nextId = 1;

export function customSample(key: string): SampleData | undefined {
  return custom.get(key)?.data;
}

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;
const fmtG = (v: number) => String(Number(v.toPrecision(6)));

/** Same reasons, same wording as DamParams.ood_reasons() on the server. */
function oodReasons(p: DamParams): string[] {
  const out: string[] = [];
  if (!(p.radius >= 0.3 && p.radius <= 0.7)) out.push(`radius ${fmtG(p.radius)} outside training range 0.3-0.7`);
  if (Math.abs(p.cx) > 1e-9 || Math.abs(p.cy) > 1e-9) out.push("dam not centred at the origin");
  if (Math.abs(p.h_in - 2) > 1e-9) out.push(`inner depth ${fmtG(p.h_in)} (training: 2)`);
  return out;
}

async function runKey(p: DamParams): Promise<string> {
  const blob = JSON.stringify({ cx: p.cx, cy: p.cy, h_in: p.h_in, h_out: 1, radius: p.radius });
  if (crypto?.subtle) {
    const d = new Uint8Array(await crypto.subtle.digest("SHA-1", new TextEncoder().encode(blob)));
    return "c-" + Array.from(d.slice(0, 5), (b) => b.toString(16).padStart(2, "0")).join("");
  }
  let h = 0;
  for (let i = 0; i < blob.length; i++) h = (Math.imul(h, 31) + blob.charCodeAt(i)) >>> 0;
  return "c-" + h.toString(16).padStart(10, "0");
}

function getWorker(): Worker {
  worker ??= new Worker(new URL("./simWorker.ts", import.meta.url), { type: "module" });
  return worker;
}

export async function staticSimulate(
  raw: DamParams, signal?: AbortSignal, onProgress?: (p: SimProgress) => void,
): Promise<CustomRun> {
  const p: DamParams = { radius: round4(raw.radius), cx: round4(raw.cx), cy: round4(raw.cy), h_in: round4(raw.h_in) };
  const key = await runKey(p);
  const hit = custom.get(key);
  if (hit) return hit.run;
  metaPromise ??= staticJson<Meta>("/meta");
  const meta = await metaPromise;
  const req: SimRequest = {
    id: nextId++, key, params: p, t: meta.t, initialStep: meta.initial_step,
    modelUrl: new URL(`${ROOT}/model/fno_step.onnx`, window.location.href).href,
  };
  const w = getWorker();
  const res = await new Promise<Extract<SimResponse, { type: "done" }>>((resolve, reject) => {
    const onMsg = (ev: MessageEvent<SimResponse>) => {
      const m = ev.data;
      if (m.id !== req.id) return;
      if (m.type === "progress") onProgress?.(m);
      else {
        w.removeEventListener("message", onMsg);
        if (m.type === "done") resolve(m);
        else reject(new Error(`custom run failed: ${m.message}`));
      }
    };
    w.addEventListener("message", onMsg);
    w.postMessage(req);
  });
  if (signal?.aborted) throw new DOMException("aborted", "AbortError");
  const [nt, nx, ny] = meta.shape;
  const run: CustomRun = {
    key, params: { ...p, h_out: 1 }, ood: oodReasons(p),
    solver_seconds: +res.solverSeconds.toFixed(3), rollout_seconds: +res.rolloutSeconds.toFixed(3),
  };
  const data: SampleData = {
    key, stats: res.stats,
    gt: { data: res.gt, nt, nx, ny },
    pred: { data: res.pred, nt, nx, ny },
  };
  custom.set(key, { data, run });
  while (custom.size > MAX_CUSTOM) custom.delete(custom.keys().next().value as string);
  return run;
}
