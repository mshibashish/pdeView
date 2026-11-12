/// <reference lib="webworker" />
// Custom dam breaks for the static site: the solver port, then the FNO step model in
// onnxruntime-web (wasm, single thread: GitHub Pages cannot send the headers threads need).
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { cellCentres, damBreakIC, outputTimes, simulate } from "./swe";
import { sampleStats } from "./stats";
import type { SimRequest, SimResponse } from "./protocol";

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1;

let session: Promise<ort.InferenceSession> | null = null;
let sessionUrl = "";

function getSession(url: string): Promise<ort.InferenceSession> {
  if (!session || sessionUrl !== url) {
    sessionUrl = url;
    session = ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    session.catch(() => { session = null; });
  }
  return session;
}

const post = (msg: SimResponse, transfer: Transferable[] = []) =>
  (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

self.onmessage = async (ev: MessageEvent<SimRequest>) => {
  const { id, key, params, t, initialStep: k0, modelUrl } = ev.data;
  try {
    const sess = getSession(modelUrl); // download the model while the solver runs
    post({ id, type: "progress", stage: "solver" });
    const x = cellCentres();
    const nx = x.length, ny = x.length, n = nx * ny;
    const dx = x[1] - x[0];
    const t0 = performance.now();
    const { h: gt } = simulate(
      damBreakIC(x, x, params.radius, params.cx, params.cy, params.h_in, 1), nx, ny, outputTimes(t.length), dx, dx,
    );
    const solverSeconds = (performance.now() - t0) / 1000;

    post({ id, type: "progress", stage: "model" });
    const s = await sess;
    const inName = s.inputNames[0];
    const outName = s.outputNames[0];
    const nt = t.length;
    const pred = new Float32Array(nt * n);
    pred.set(gt.subarray(0, k0 * n));
    let window = gt.slice(0, k0 * n);
    const t1 = performance.now();
    for (let f = k0; f < nt; f++) {
      const out = await s.run({ [inName]: new ort.Tensor("float32", window, [1, k0, nx, ny]) });
      const next = out[outName].data as Float32Array;
      pred.set(next, f * n);
      const w = new Float32Array(k0 * n);
      w.set(window.subarray(n));
      w.set(next, (k0 - 1) * n);
      window = w;
      if ((f - k0) % 7 === 0) post({ id, type: "progress", stage: "rollout", step: f - k0 + 1, total: nt - k0 });
    }
    const rolloutSeconds = (performance.now() - t1) / 1000;
    const stats = sampleStats(key, gt, pred, t, k0, nx, ny, dx * dx);
    post({ id, type: "done", gt, pred, stats, solverSeconds, rolloutSeconds }, [gt.buffer, pred.buffer]);
  } catch (e) {
    post({ id, type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
