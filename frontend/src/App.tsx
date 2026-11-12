import { useEffect, useMemo, useState } from "react";
import { api, at, frame as frameOf } from "./api";
import { lut, type Theme } from "./colormap";
import { Colorbar } from "./components/Colorbar";
import { CustomPanel } from "./components/CustomPanel";
import { FieldPanel, type Hover } from "./components/FieldPanel";
import { LineChart } from "./components/LineChart";
import { SampleList } from "./components/SampleList";
import { Surface3D } from "./components/Surface3D";
import { TimeControls } from "./components/TimeControls";
import { VerificationView } from "./components/VerificationView";
import { fmtNum, fmtSci, fmtSigned } from "./format";
import { usePlayback } from "./hooks/usePlayback";
import { useSample } from "./hooks/useSample";
import { useTheme } from "./hooks/useTheme";
import type { CustomRun, Meta, SampleData, SampleSummary } from "./types";

type Tab = "viewer" | "verification" | "about";

/** Entity colours (categorical slots 1-3), fixed across every chart and both themes. */
const SERIES: Record<Theme, { gt: string; fno: string; persist: string }> = {
  light: { gt: "#2a78d6", fno: "#eb6834", persist: "#1baf7a" },
  dark: { gt: "#3987e5", fno: "#d95926", persist: "#199e70" },
};

function readHash(): { tab: Tab; sample: string | null } {
  const h = new URLSearchParams(window.location.hash.slice(1));
  const tab = (h.get("tab") as Tab) || "viewer";
  return { tab: ["viewer", "verification", "about"].includes(tab) ? tab : "viewer", sample: h.get("sample") };
}

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme();
  const initial = useMemo(readHash, []);
  const [tab, setTab] = useState<Tab>(initial.tab);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(initial.sample);
  const [bootError, setBootError] = useState<string | null>(null);
  const [customRuns, setCustomRuns] = useState<Record<string, CustomRun>>({});
  const onCustomRun = (run: CustomRun) => {
    setCustomRuns((c) => ({ ...c, [run.key]: run }));
    setSelected(run.key);
  };

  useEffect(() => {
    const ctrl = new AbortController();
    Promise.all([api.meta(ctrl.signal), api.samples(ctrl.signal)])
      .then(([m, s]) => {
        setMeta(m);
        setSamples(s);
        setSelected((cur) => (cur && s.some((x) => x.key === cur) ? cur : s.find((x) => x.split === "test")?.key ?? null));
      })
      .catch((e: unknown) => {
        if (!ctrl.signal.aborted) setBootError(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, []);

  useEffect(() => {
    const p = new URLSearchParams();
    if (tab !== "viewer") p.set("tab", tab);
    if (selected) p.set("sample", selected);
    window.history.replaceState(null, "", `#${p.toString()}`);
  }, [tab, selected]);

  const verdict = meta?.verification;
  const nFindings = verdict?.criteria.filter((c) => c.kind === "finding" && !c.pass).length ?? 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" className="logo" aria-hidden>
            <circle cx="16" cy="16" r="14" />
            <circle cx="16" cy="16" r="8" className="ring" />
          </svg>
          <div>
            <h1>pdeView</h1>
            <p>2D shallow-water equations · PDEBench radial dam break · pretrained Fourier Neural Operator</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Sections">
          {(["viewer", "verification", "about"] as Tab[]).map((t) => (
            <button key={t} type="button" className={tab === t ? "on" : ""} aria-current={tab === t ? "page" : undefined} onClick={() => setTab(t)}>
              {t === "viewer" ? "Viewer" : t === "verification" ? "Model verification" : "About"}
              {t === "verification" && verdict?.verdict && (
                <span className={`pill ${verdict.verdict === "PASS" ? "good" : "critical"}`}>
                  {verdict.verdict === "PASS" ? "gates pass" : "gates fail"}
                  {nFindings > 0 && ` · ${nFindings} findings`}
                </span>
              )}
            </button>
          ))}
        </nav>
        <button type="button" className="icon-btn theme" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Toggle theme">
          {theme === "dark" ? (
            <svg viewBox="0 0 16 16" aria-hidden><circle cx="8" cy="8" r="3" /><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M3 13l1.4-1.4M11.6 4.4 13 3" /></svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden><path d="M13 9.5A5.5 5.5 0 1 1 6.5 3a4.5 4.5 0 0 0 6.5 6.5z" /></svg>
          )}
        </button>
      </header>

      {bootError && (
        <div className="panel error-box" role="alert">
          Could not reach the backend ({bootError}). Start it with <code>uv run uvicorn pdeview.api.app:app</code>.
        </div>
      )}

      {tab === "viewer" && meta && (
        <div className="viewer-layout">
          <div className="sidebar">
            <CustomPanel onRun={onCustomRun} />
            <SampleList samples={samples} selected={selected} onSelect={setSelected} />
          </div>
          <Viewer meta={meta} sampleKey={selected} summary={samples.find((s) => s.key === selected)}
            custom={selected ? customRuns[selected] : undefined} theme={theme} />
        </div>
      )}
      {tab === "verification" && <VerificationView />}
      {tab === "about" && <About meta={meta} />}
    </div>
  );
}

function Viewer({ meta, sampleKey, summary, custom, theme }: {
  meta: Meta; sampleKey: string | null; summary?: SampleSummary; custom?: CustomRun; theme: Theme;
}) {
  const { data, loading, error } = useSample(sampleKey);
  const [nt, nx, ny] = meta.shape;
  const pb = usePlayback(nt);
  const [hover, setHover] = useState<Hover | null>(null);
  const [mode, setMode] = useState<"2d" | "3d">("2d");
  const [surfaceSource, setSurfaceSource] = useState<"gt" | "pred">("pred");
  const [errScale, setErrScale] = useState<"frame" | "rollout">("rollout");

  // keyboard: space = play/pause, arrows = step (ignored while typing in a control)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName) && el.getAttribute("type") !== "range") return;
      if (e.key === " " && el.tagName !== "BUTTON") {
        e.preventDefault();
        pb.toggle();
      } else if (e.key === "ArrowRight" && el.getAttribute("type") !== "range") pb.step(e.shiftKey ? 10 : 1);
      else if (e.key === "ArrowLeft" && el.getAttribute("type") !== "range") pb.step(e.shiftKey ? -10 : -1);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [pb]);

  if (!sampleKey) return <div className="panel">Select a sample.</div>;
  if (error) return <div className="panel error-box">Could not load sample {sampleKey}: {error}</div>;
  if (!data) return <div className="panel loading">Loading sample {sampleKey} (running the FNO rollout)…</div>;

  return (
    <ViewerBody
      meta={meta} data={data} summary={summary} custom={custom} theme={theme} loading={loading}
      pb={pb} hover={hover} setHover={setHover} mode={mode} setMode={setMode}
      surfaceSource={surfaceSource} setSurfaceSource={setSurfaceSource}
      errScale={errScale} setErrScale={setErrScale} nx={nx} ny={ny}
    />
  );
}

interface BodyProps {
  meta: Meta;
  data: SampleData;
  summary?: SampleSummary;
  custom?: CustomRun;
  theme: Theme;
  loading: boolean;
  pb: ReturnType<typeof usePlayback>;
  hover: Hover | null;
  setHover: (h: Hover | null) => void;
  mode: "2d" | "3d";
  setMode: (m: "2d" | "3d") => void;
  surfaceSource: "gt" | "pred";
  setSurfaceSource: (s: "gt" | "pred") => void;
  errScale: "frame" | "rollout";
  setErrScale: (s: "frame" | "rollout") => void;
  nx: number;
  ny: number;
}

function ViewerBody({
  meta, data, summary, custom, theme, loading, pb, hover, setHover, mode, setMode,
  surfaceSource, setSurfaceSource, errScale, setErrScale, nx, ny,
}: BodyProps) {
  const { gt, pred, stats } = data;
  const t = meta.t;
  const f = pb.frame;
  const k0 = meta.initial_step;
  const colors = SERIES[theme];
  const hLut = lut("height", theme);
  const eLut = lut("error", theme);
  const gtFrame = frameOf(gt, f);
  const prFrame = frameOf(pred, f);

  const frameRange = useMemo(() => {
    let e = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < gtFrame.length; i++) {
      const a = gtFrame[i];
      const b = prFrame[i];
      e = Math.max(e, Math.abs(b - a));
      lo = Math.min(lo, a, b);
      hi = Math.max(hi, a, b);
    }
    return { e, lo, hi };
  }, [gtFrame, prFrame]);
  const perFrame = errScale === "frame";
  const lo = perFrame ? frameRange.lo : stats.range.min;
  const hi = perFrame ? Math.max(frameRange.hi, frameRange.lo + 1e-3) : stats.range.max;
  const eMax = Math.max(1e-6, perFrame ? frameRange.e : stats.range.err_abs_max);
  const errAt = (ix: number, iy: number) => prFrame[ix * ny + iy] - gtFrame[ix * ny + iy];
  const version = `${data.key}-${f}`;

  const iyCut = hover?.iy ?? Math.floor(ny / 2);
  const x = meta.grid.x;
  const y = meta.grid.y;

  const cross = useMemo(() => ({
    gt: x.map((_, ix) => at(gt, f, ix, iyCut)),
    pred: x.map((_, ix) => at(pred, f, ix, iyCut)),
  }), [gt, pred, f, iyCut, x]);

  const v0 = stats.volume_gt[0];
  const volGt = stats.volume_gt.map((v) => (v / v0 - 1) * 100);
  const volPr = stats.volume_pred.map((v) => (v / v0 - 1) * 100);

  const isInput = f < k0;
  const twins = summary?.n_twins_total ?? 0;
  const refLabel = custom ? "Reference" : "Ground truth";

  return (
    <section className={`viewer ${loading ? "refreshing" : ""}`} aria-label={`Sample ${data.key}`}>
      <div className="sample-header">
        {custom ? (
          <div>
            <h2>
              Custom dam break
              <span className="tag">reference solver</span>
            </h2>
            <p>
              r = {fmtNum(custom.params.radius, 2)} · centre ({fmtNum(custom.params.cx, 2)}, {fmtNum(custom.params.cy, 2)}) ·
              h = {fmtNum(custom.params.h_in, 2)} inside, 1 outside · solved in {custom.solver_seconds.toFixed(2)} s
            </p>
            <p className={`ood-line ${custom.ood.length ? "out" : "in"}`}>
              {custom.ood.length ? `Outside the training distribution: ${custom.ood.join("; ")}` : "Inside the training distribution"}
            </p>
          </div>
        ) : summary?.split === "heldout" ? (
          <div>
            <h2>
              Held-out case {data.key}
              <span className="tag test">{summary.family === "subcell" ? "sub-cell dam position" : "cell-averaged dam wall"}</span>
            </h2>
            <p>
              r = {fmtNum(summary.dam_radius)} ·{" "}
              {summary.family === "subcell"
                ? `centre (${fmtSigned(summary.cx ?? 0, 4)}, ${fmtSigned(summary.cy ?? 0, 4)}), within half a cell of the origin`
                : "centred; cells cut by the dam wall hold the fraction of water inside it"}{" "}
              · h = 2 inside, 1 outside · g = 1
            </p>
            <p className="ood-line in">
              In distribution, but the initial condition never occurs in PDEBench&apos;s training split · reference from PyClaw
            </p>
          </div>
        ) : (
        <div>
          <h2>
            Sample {data.key}
            <span className={`tag ${summary?.split === "train" ? "" : "test"}`}>{summary?.split === "train" ? "train probe" : "PDEBench test split"}</span>
          </h2>
          <p>
            Dam radius r = {fmtNum(summary?.dam_radius)} · h = 2 inside, 1 outside · g = 1 · {twins > 0 && (
              <span className="twin-note" title="These samples have the same discrete initial condition, so the PyClaw solution is identical bit for bit.">
                bit-identical to {twins} other sample{twins === 1 ? "" : "s"} in the dataset
              </span>
            )}
          </p>
        </div>
        )}
        <dl className="stats">
          <div><dt>FNO nRMSE</dt><dd>{fmtSci(stats.summary.nRMSE)}</dd></div>
          <div><dt>Persistence nRMSE</dt><dd>{fmtSci(stats.summary.persistence_nRMSE)}</dd></div>
          <div><dt>Max |error|</dt><dd>{fmtNum(stats.summary.max_error, 3)}</dd></div>
        </dl>
      </div>

      <div className="view-bar">
        <div className="segmented" role="radiogroup" aria-label="View">
          <button type="button" role="radio" aria-checked={mode === "2d"} className={mode === "2d" ? "on" : ""} onClick={() => setMode("2d")}>2D maps</button>
          <button type="button" role="radio" aria-checked={mode === "3d"} className={mode === "3d" ? "on" : ""} onClick={() => setMode("3d")}>3D surface</button>
        </div>
        {mode === "2d" ? (
          <div className="segmented" role="radiogroup" aria-label="Colour scale">
            <span className="seg-label">Colour scale</span>
            <button type="button" role="radio" aria-checked={errScale === "rollout"} className={errScale === "rollout" ? "on" : ""} onClick={() => setErrScale("rollout")}>Whole rollout</button>
            <button type="button" role="radio" aria-checked={errScale === "frame"} className={errScale === "frame" ? "on" : ""} onClick={() => setErrScale("frame")}>This frame</button>
          </div>
        ) : (
          <div className="segmented" role="radiogroup" aria-label="Surface source">
            <button type="button" role="radio" aria-checked={surfaceSource === "gt"} className={surfaceSource === "gt" ? "on" : ""} onClick={() => setSurfaceSource("gt")}>Ground truth</button>
            <button type="button" role="radio" aria-checked={surfaceSource === "pred"} className={surfaceSource === "pred" ? "on" : ""} onClick={() => setSurfaceSource("pred")}>FNO</button>
          </div>
        )}
      </div>

      {mode === "2d" ? (
        <div className="fields">
          <FieldPanel title={custom ? "Reference solver" : "Ground truth"}
            subtitle={custom ? "pdeView FV (PyClaw scheme)" : "PyClaw finite volume"} nx={nx} ny={ny} values={gtFrame}
            lo={lo} hi={hi} table={hLut} hover={hover} onHover={setHover} version={`${version}-${theme}`} />
          <FieldPanel title="FNO prediction" subtitle={isInput ? "input frame = ground truth" : `${f - k0 + 1} autoregressive steps`}
            badge={isInput ? "input" : undefined} nx={nx} ny={ny} values={prFrame}
            lo={lo} hi={hi} table={hLut} hover={hover} onHover={setHover} version={`${version}-${theme}`} />
          <FieldPanel title="Error" subtitle={`FNO − ${refLabel.toLowerCase()}`} nx={nx} ny={ny} values={errAt}
            lo={-eMax} hi={eMax} table={eLut} hover={hover} onHover={setHover} version={`${version}-${theme}-${eMax}`} />
        </div>
      ) : (
        <div className="surface-wrap">
          <Surface3D values={surfaceSource === "gt" ? gtFrame : prFrame} nx={nx} ny={ny} lo={lo} hi={hi}
            table={hLut} theme={theme} version={`${version}-${surfaceSource}-${theme}`} />
          <p className="hint">Drag to rotate · scroll to zoom · height exaggerated 1.5×</p>
        </div>
      )}

      <div className="legend-row">
        <Colorbar kind="height" theme={theme} lo={lo} hi={hi} label={`Water height h (${perFrame ? "this frame" : "whole rollout"})`} />
        {mode === "2d" && (
          <Colorbar kind="error" theme={theme} lo={-eMax} hi={eMax}
            label={perFrame && isInput ? "Error (input frames are exact)" : `Error FNO − ${custom ? "ref" : "GT"} (${perFrame ? "this frame" : "whole rollout"})`}
            format={(v) => (Math.abs(v) < 1e-5 ? "0" : fmtSigned(v, 3))} />
        )}
        <div className="readout" aria-live="polite">
          {hover ? (
            <>
              <span>x = {fmtNum(x[hover.ix], 3)}, y = {fmtNum(y[hover.iy], 3)}</span>
              <span><i style={{ background: colors.gt }} />{custom ? "Ref" : "GT"} <strong>{fmtNum(at(gt, f, hover.ix, hover.iy), 4)}</strong></span>
              <span><i style={{ background: colors.fno }} />FNO <strong>{fmtNum(at(pred, f, hover.ix, hover.iy), 4)}</strong></span>
              <span>error <strong>{fmtSigned(at(pred, f, hover.ix, hover.iy) - at(gt, f, hover.ix, hover.iy), 4)}</strong></span>
            </>
          ) : (
            <span className="muted">Hover a map to read values. The cross-section follows the hovered row.</span>
          )}
        </div>
      </div>

      <TimeControls frame={f} nFrames={t.length} t={t} initialStep={k0} playing={pb.playing} speed={pb.speed}
        onFrame={pb.setFrame} onToggle={pb.toggle} onStep={pb.step} onSpeed={pb.setSpeed} />

      <div className="charts">
        <LineChart
          title={`Cross-section at y = ${fmtNum(y[iyCut], 2)}, t = ${t[f].toFixed(2)}`}
          x={x}
          series={[
            { id: "gt", label: refLabel, color: colors.gt, values: cross.gt },
            { id: "fno", label: "FNO", color: colors.fno, values: cross.pred, dash: "5 3" },
          ]}
          xLabel="x"
          yLabel="h"
          formatY={(v) => v.toFixed(2)}
          formatX={(v) => v.toFixed(1)}
          formatTip={(v) => v.toFixed(4)}
          formatTipX={(v) => v.toFixed(3)}
          marker={hover ? x[hover.ix] : undefined}
        />
        <LineChart
          title="Error over time (click to seek)"
          x={t}
          series={[
            { id: "fno", label: "FNO", color: colors.fno, values: stats.nrmse },
            { id: "persist", label: "Persistence baseline", color: colors.persist, values: stats.persistence_nrmse },
          ]}
          yScale="log"
          xLabel="t"
          yLabel="nRMSE"
          formatY={(v) => fmtSci(v, 0)}
          formatX={(v) => v.toFixed(1)}
          formatTip={(v) => fmtSci(v, 2)}
          formatTipX={(v) => v.toFixed(2)}
          marker={t[f]}
          onPick={pb.setFrame}
          shade={[t[0], t[k0 - 1]]}
          shadeLabel="input"
        />
        <LineChart
          title="Water volume change vs. t = 0"
          x={t}
          series={[
            { id: "gt", label: refLabel, color: colors.gt, values: volGt },
            { id: "fno", label: "FNO", color: colors.fno, values: volPr, dash: "5 3" },
          ]}
          xLabel="t"
          yLabel="ΔV (%)"
          formatY={(v) => (Math.abs(v) < 1e-9 ? "0" : fmtSigned(v, 3))}
          formatX={(v) => v.toFixed(1)}
          formatTip={(v) => `${fmtSigned(v, 4)} %`}
          formatTipX={(v) => v.toFixed(2)}
          leftMargin={62}
          marker={t[f]}
          onPick={pb.setFrame}
        />
      </div>
    </section>
  );
}

function About({ meta }: { meta: Meta | null }) {
  return (
    <article className="panel about">
      <h2>What you are looking at</h2>
      <p>
        The <strong>2D shallow-water equations</strong> model a thin layer of fluid with depth h and depth-averaged
        velocity (u, v): ∂h/∂t + ∇·(h<b>u</b>) = 0 and ∂(h<b>u</b>)/∂t + ∇·(h<b>u</b>⊗<b>u</b> + ½gh²<b>I</b>) = 0.
        In PDEBench&apos;s <em>radial dam break</em>, a column of water (h = 2) of radius r ∈ [0.3, 0.7] stands in still
        water (h = 1) on [−2.5, 2.5]², with g = 1 and outflow boundaries. It collapses into an outward bore and an
        inward rarefaction. PyClaw solves it on a 128 × 128 grid and saves 101 frames for t ∈ [0, 1].
      </p>
      <p>
        The <strong>Fourier Neural Operator</strong> (Li et al., 2021) is PDEBench&apos;s pretrained{" "}
        <code>2D_rdb_NA_NA_FNO.pt</code>
        {meta ? ` (${meta.model.n_parameters.toLocaleString()} parameters, ${meta.model.modes} Fourier modes, width ${meta.model.width})` : ""}.
        It takes the last {meta?.initial_step ?? 10} frames and predicts the next one. Each prediction is fed back in,
        so frames 10–100 are a 91-step autoregressive rollout from ground-truth frames 0–9.
      </p>
      <p>
        <strong>Before use the checkpoint was verified independently</strong> (see <em>Model verification</em>). We
        checked its integrity, reproduced the paper&apos;s metrics, compared it with a persistence baseline and audited
        the dataset. The audit showed that PDEBench&apos;s SWE test split repeats training trajectories, which bounds
        what its test accuracy can tell us.
      </p>
      <p>
        The model was therefore also tested on 200 new in-distribution dam breaks whose initial conditions never occur in
        training. They were generated with PyClaw exactly as PDEBench generated its data (<em>Viewer → Held-out</em>).
        To probe beyond the training distribution, pdeView includes its own finite-volume solver, which matches PyClaw
        to float32 precision. Use <em>Custom dam break</em> in the viewer to move, resize or deepen the dam and see how
        the FNO responds.
      </p>
      <p className="note">
        Data and checkpoint: PDEBench (Takamoto et al., NeurIPS 2022), DaRUS doi:10.18419/darus-2986 and
        doi:10.18419/darus-2987, CC BY 4.0. Source: <a href="https://github.com/mshibashish/pdeView" target="_blank" rel="noreferrer">github.com/mshibashish/pdeView</a>.
      </p>
    </article>
  );
}
