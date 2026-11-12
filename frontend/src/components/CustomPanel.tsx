import { useState } from "react";
import { STATIC, api, type SimProgress } from "../api";
import type { CustomRun, DamParams } from "../types";

interface Props {
  onRun: (run: CustomRun) => void;
}

const PRESETS: { label: string; p: DamParams }[] = [
  { label: "Training-like", p: { radius: 0.5, cx: 0, cy: 0, h_in: 2 } },
  { label: "Off-centre", p: { radius: 0.5, cx: 1, cy: 0, h_in: 2 } },
  { label: "Large dam", p: { radius: 1.0, cx: 0, cy: 0, h_in: 2 } },
  { label: "Deep dam", p: { radius: 0.5, cx: 0, cy: 0, h_in: 3 } },
];

function oodReasons(p: DamParams): string[] {
  const r: string[] = [];
  if (p.radius < 0.3 || p.radius > 0.7) r.push("radius outside 0.3–0.7");
  if (Math.abs(p.cx) > 1e-9 || Math.abs(p.cy) > 1e-9) r.push("off-centre");
  if (Math.abs(p.h_in - 2) > 1e-9) r.push("inner depth ≠ 2");
  return r;
}

function Slider({ label, value, min, max, step, onChange, unit = "" }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; unit?: string;
}) {
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{label}</span>
        <strong>{value.toFixed(2)}{unit}</strong>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function progressLabel(p: SimProgress | null): string {
  if (!p || p.stage === "solver") return "Solving…";
  if (p.stage === "model") return "Loading the FNO…";
  return `FNO step ${p.step} / ${p.total}…`;
}

/** Define a dam break, solve it with the reference solver, and roll the FNO out on it. */
export function CustomPanel({ onRun }: Props) {
  const [p, setP] = useState<DamParams>(PRESETS[1].p);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [progress, setProgress] = useState<SimProgress | null>(null);
  const ood = oodReasons(p);

  const run = async () => {
    setBusy(true);
    setErr(null);
    setProgress(null);
    try {
      onRun(await api.simulate(p, undefined, setProgress));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  return (
    <details className="custom-panel">
      <summary>
        <span>Custom dam break</span>
        <span className="custom-sub">solver + FNO on unseen initial conditions</span>
      </summary>
      <div className="custom-body">
        <div className="presets">
          {PRESETS.map((pr) => (
            <button key={pr.label} type="button" onClick={() => setP(pr.p)}>{pr.label}</button>
          ))}
        </div>
        <Slider label="Radius r" value={p.radius} min={0.1} max={1.2} step={0.01} onChange={(radius) => setP({ ...p, radius })} />
        <Slider label="Centre x" value={p.cx} min={-1.5} max={1.5} step={0.05} onChange={(cx) => setP({ ...p, cx })} />
        <Slider label="Centre y" value={p.cy} min={-1.5} max={1.5} step={0.05} onChange={(cy) => setP({ ...p, cy })} />
        <Slider label="Inner depth h_in" value={p.h_in} min={1.25} max={3} step={0.05} onChange={(h_in) => setP({ ...p, h_in })} />
        <div className={`ood ${ood.length ? "out" : "in"}`} role="status">
          {ood.length ? (
            <>
              <svg viewBox="0 0 16 16" aria-hidden><path d="M8 2.5 14.5 13.5h-13z" /><path d="M8 6.5v3.2M8 11.6v.2" /></svg>
              <span>Outside the training distribution: {ood.join(", ")}</span>
            </>
          ) : (
            <>
              <svg viewBox="0 0 16 16" aria-hidden><path d="m3.5 8.5 3 3 6-7" /></svg>
              <span>Inside the training distribution (centred, r 0.3–0.7, h_in = 2)</span>
            </>
          )}
        </div>
        <button type="button" className="run-btn" onClick={run} disabled={busy}>
          {busy ? progressLabel(progress) : "Solve and run FNO"}
        </button>
        {err && <p className="custom-err" role="alert">{err}</p>}
        <p className="custom-note">
          The reference comes from pdeView&apos;s finite-volume solver, which reproduces PyClaw&apos;s
          ground truth to 6e-8. The outer depth stays 1 and g = 1.
          {STATIC && (
            <>
              {" "}Here both run in your browser: the solver as a JavaScript port and the FNO with onnxruntime-web.
              A run takes a few seconds. The first one also downloads the model and runtime (about 8 MB).
            </>
          )}
        </p>
      </div>
    </details>
  );
}
