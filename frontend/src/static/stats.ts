// Port of ViewerService.stats (backend/pdeview/service.py) for in-browser custom runs.
import type { SampleStats } from "../types";

/** Per-frame RMSE, nRMSE and max |error| of p against r over frames k0..T-1. */
function frameErrors(p: Float32Array, r: Float32Array, nt: number, n: number, k0: number) {
  const rmse: number[] = [], nrmse: number[] = [], maxAbs: number[] = [];
  for (let t = k0; t < nt; t++) {
    let se = 0, sr = 0, mx = 0;
    for (let i = t * n, end = (t + 1) * n; i < end; i++) {
      const e = p[i] - r[i];
      se += e * e;
      sr += r[i] * r[i];
      mx = Math.max(mx, Math.abs(e));
    }
    const v = Math.sqrt(se / n);
    rmse.push(v);
    nrmse.push(v / Math.sqrt(sr / n));
    maxAbs.push(mx);
  }
  return { rmse, nrmse, maxAbs };
}

const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

export function sampleStats(
  key: string, gt: Float32Array, pred: Float32Array, t: number[], k0: number,
  nx: number, ny: number, cellArea: number,
): SampleStats {
  const nt = t.length, n = nx * ny;
  const fno = frameErrors(pred, gt, nt, n, k0);
  // persistence baseline: hold the last input frame
  const persist = new Float32Array(gt.length);
  persist.set(gt.subarray(0, k0 * n));
  const last = gt.subarray((k0 - 1) * n, k0 * n);
  for (let f = k0; f < nt; f++) persist.set(last, f * n);
  const base = frameErrors(persist, gt, nt, n, k0);

  const volume = (a: Float32Array) => Array.from({ length: nt }, (_, f) => {
    let s = 0;
    for (let i = f * n, end = (f + 1) * n; i < end; i++) s += a[i];
    return s * cellArea;
  });
  let lo = Infinity, hi = -Infinity, errMax = 0;
  for (let i = 0; i < gt.length; i++) {
    lo = Math.min(lo, gt[i], pred[i]);
    hi = Math.max(hi, gt[i], pred[i]);
    if (i >= k0 * n) errMax = Math.max(errMax, Math.abs(pred[i] - gt[i]));
  }
  const pad = (a: number[]): (number | null)[] => [...Array<null>(k0).fill(null), ...a];
  return {
    key,
    t,
    initial_step: k0,
    rmse: pad(fno.rmse),
    nrmse: pad(fno.nrmse),
    max_abs: pad(fno.maxAbs),
    persistence_nrmse: pad(base.nrmse),
    volume_gt: volume(gt),
    volume_pred: volume(pred),
    range: { min: lo, max: hi, err_abs_max: errMax },
    summary: {
      nRMSE: mean(fno.nrmse),
      RMSE: mean(fno.rmse),
      max_error: Math.max(...fno.maxAbs),
      persistence_nRMSE: mean(base.nrmse),
    },
  };
}
