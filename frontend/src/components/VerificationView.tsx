import { useEffect, useState } from "react";
import { api } from "../api";
import { fmtSci, metricLabel } from "../format";
import type {
  Generalization, GeneralizationCase, Heldout, HeldoutFamily, MetricKey, Verification,
} from "../types";

const METRICS: MetricKey[] = [
  "RMSE", "nRMSE", "cRMSE", "max_error", "bRMSE", "fRMSE_low", "fRMSE_mid", "fRMSE_high",
];
function fmtValue(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : fmtSci(v, 3);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (v && typeof v === "object") {
    return Object.entries(v as Record<string, number>).map(([k, x]) => `${k} ${x.toFixed(2)}`).join(" · ");
  }
  return String(v);
}

const REPO = "https://github.com/mshibashish/pdeView/blob/main/reports/VERIFICATION.md";

function StatusIcon({ ok, kind }: { ok: boolean; kind: "gate" | "finding" }) {
  if (ok) {
    return (
      <svg className="status good" viewBox="0 0 16 16" aria-hidden><path d="m3.5 8.5 3 3 6-7" /></svg>
    );
  }
  return kind === "gate" ? (
    <svg className="status critical" viewBox="0 0 16 16" aria-hidden><path d="M4 4l8 8M12 4l-8 8" /></svg>
  ) : (
    <svg className="status warning" viewBox="0 0 16 16" aria-hidden><path d="M8 2.5 14.5 13.5h-13z" /><path d="M8 6.5v3.2M8 11.6v.2" /></svg>
  );
}

export function VerificationView() {
  const [v, setV] = useState<Verification | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [gen, setGen] = useState<Generalization | null>(null);
  const [held, setHeld] = useState<Heldout | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    api.verification(ctrl.signal).then(setV).catch((e: unknown) => {
      if (!ctrl.signal.aborted) setErr(e instanceof Error ? e.message : String(e));
    });
    api.generalization(ctrl.signal).then(setGen).catch(() => {
      /* optional report - the section is hidden if it has not been generated */
    });
    api.heldout(ctrl.signal).then(setHeld).catch(() => {
      /* optional report, as above */
    });
    return () => ctrl.abort();
  }, []);

  if (err) return <div className="panel error-box">Could not load the verification report: {err}</div>;
  if (!v) return <div className="panel loading">Loading verification report…</div>;

  const acc = v.accuracy;
  const au = v.audit;
  const fp = v.fingerprint;
  const gates = v.criteria.filter((c) => c.kind === "gate");
  const findings = v.criteria.filter((c) => c.kind === "finding");

  return (
    <div className="verification">
      <section className="tiles">
        <div className="tile">
          <div className="tile-label">Accuracy gates</div>
          <div className="tile-value">
            {gates.filter((g) => g.pass).length}/{gates.length} passed
          </div>
          <div className="tile-note">Published checkpoint, run as PDEBench runs it</div>
        </div>
        <div className="tile">
          <div className="tile-label">nRMSE vs. paper</div>
          <div className="tile-value">
            {fmtSci(acc.pdebench_compat.nRMSE)} <span className="tile-sub">vs {fmtSci(acc.paper.nRMSE, 1)}</span>
          </div>
          <div className="tile-note">{acc.ratio_to_paper.nRMSE.toFixed(2)}× the paper (PDEBench aggregation)</div>
        </div>
        <div className="tile">
          <div className="tile-label">vs. persistence baseline</div>
          <div className="tile-value">{acc.persistence_over_fno_nRMSE.toFixed(0)}× better</div>
          <div className="tile-note">Holding the last input frame constant</div>
        </div>
        <div className="tile tile-warn">
          <div className="tile-label">Distinct trajectories</div>
          <div className="tile-value">
            {au.n_distinct_trajectories} <span className="tile-sub">of {au.n_samples} samples</span>
          </div>
          <div className="tile-note">
            {au.test_with_identical_train_trajectory}/{au.n_test} test samples duplicate a training sample
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Checks</h2>
        <ul className="checks">
          {[...gates, ...findings].map((c) => (
            <li key={c.name} className={c.pass ? "ok" : c.kind === "gate" ? "fail" : "issue"}>
              <StatusIcon ok={c.pass} kind={c.kind} />
              <div>
                <div className="check-name">
                  {c.name}
                  <span className="check-kind">{c.kind === "gate" ? (c.pass ? "Pass" : "Fail") : c.pass ? "OK" : "Issue"}</span>
                </div>
                <div className="check-detail">
                  {fmtValue(c.value)}
                  {c.detail ? ` · ${c.detail}` : ""} · criterion: {c.threshold}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <h2>Finding 1 · the test split repeats the training split</h2>
        <p>
          Every sample is a centred dam break whose only parameter is the radius r ~ U(0.3, 0.7). The depth is set per
          cell, so all radii between two consecutive cell-centre distances give the <em>same</em> discrete initial
          condition, and PyClaw is deterministic. With r in [0.3, 0.7] the 128 × 128 grid allows only{" "}
          {au.n_possible_initial_conditions} initial conditions, and all of them occur. The {au.n_samples} samples hold{" "}
          <strong>{au.n_distinct_trajectories} distinct trajectories</strong>. Each one repeats{" "}
          {au.copies_per_trajectory.min}–{au.copies_per_trajectory.max} times, and{" "}
          <strong>
            all {au.test_with_identical_train_trajectory} of the {au.n_test} test samples
          </strong>{" "}
          match a training sample bit for bit. The test metrics therefore measure how well the model reproduces
          trajectories it was trained on, not how well it generalises.
        </p>
      </section>

      <section className="panel">
        <h2>Finding 2 · the stored validation loss does not reproduce</h2>
        <p>
          The checkpoint stores PDEBench&apos;s best validation loss, <code>{fp.stored_val_loss.toExponential(4)}</code>.
          Recomputing it from our rollouts gives <code>{fp.reproduced_val_loss.toExponential(4)}</code>, which is{" "}
          {(fp.reproduced_val_loss / fp.stored_val_loss).toFixed(2)}× higher. Shorter training horizons, other grid
          conventions and the split do not explain the gap. The checkpoint was published before this data release, so it
          was probably trained on an earlier revision of the data or pipeline.
        </p>
      </section>

      {held && <HeldoutSection held={held} />}

      {gen && <GeneralizationSection gen={gen} />}

      <section className="panel">
        <h2>Metrics vs. the PDEBench paper (Table 5)</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Paper</th>
                <th scope="col">Ours (PDEBench aggr.)</th>
                <th scope="col">Ratio</th>
                <th scope="col">Ours (clean)</th>
                <th scope="col">Ours (dedup, n={acc.n_dedup_test})</th>
                <th scope="col">Persistence</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map((k) => (
                <tr key={k}>
                  <th scope="row">{metricLabel[k]}</th>
                  <td>{fmtSci(acc.paper[k], 1)}</td>
                  <td>{fmtSci(acc.pdebench_compat[k])}</td>
                  <td>{acc.ratio_to_paper[k].toFixed(2)}</td>
                  <td>{fmtSci(acc.clean[k])}</td>
                  <td>{fmtSci(acc.dedup_test_clean[k])}</td>
                  <td>{fmtSci(acc.persistence_clean[k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          The PDEBench aggregation repeats the paper&apos;s evaluation loop: metrics per batch of 5, summed, then divided by
          n_batches − 1. The clean column treats all 100 samples as one population.
        </p>
      </section>

      <section className="figures">
        {[
          ["error_vs_time.png", "Error growth along the rollout"],
          ["error_vs_radius.png", "Per-sample error vs. dam radius"],
          ["snapshots_median.png", "Median-error test sample"],
          ["snapshots_worst.png", "Worst test sample"],
        ].map(([f, cap]) => (
          <figure key={f} className="panel">
            <img src={api.figureUrl(f)} alt={cap} loading="lazy" />
            <figcaption>{cap}</figcaption>
          </figure>
        ))}
      </section>
      <p className="note">
        Generated {new Date(v.generated_at).toLocaleString()} by <code>scripts/verify_checkpoint.py</code>.{" "}
        <a href={REPO} target="_blank" rel="noreferrer">Full report on GitHub</a>.
      </p>
    </div>
  );
}

function caseLabel(family: string, c: GeneralizationCase): string {
  const p = c.params;
  if (family === "radius") return `r = ${p.radius}`;
  if (family === "centre") return `centre (${p.cx}, ${p.cy})`;
  if (family === "height") return `h_in = ${p.h_in}`;
  return p.second ? `two dams, r = ${p.radius}` : "";
}

function GeneralizationSection({ gen }: { gen: Generalization }) {
  const sv = gen.solver_validation;
  const pick: [keyof Generalization["families"], number[]][] = [
    ["radius", [0.1, 0.5, 0.8, 1.0, 1.2]],
    ["centre", [0.1, 0.5, 1.0]],
    ["height", [1.5, 2.5, 3.0]],
    ["two_dams", [0.4]],
  ];
  const rows = pick.flatMap(([fam, vals]) =>
    gen.families[fam]
      .filter((c) => {
        const v = fam === "radius" || fam === "two_dams" ? c.params.radius : fam === "centre" ? c.params.cx : c.params.h_in;
        return vals.includes(v) && (fam !== "centre" || c.params.cy === 0);
      })
      .map((c) => ({ fam, c })),
  );
  return (
    <section className="panel">
      <h2>Generalisation to unseen dam breaks</h2>
      <p>
        Because the PDEBench test split repeats the training split, we generated new trajectories with pdeView&apos;s own
        solver. Re-simulating all {sv.n_samples} downloaded samples from their configs, it reproduces PyClaw to a maximum
        |Δh| of {sv.max_abs_error.toExponential(1)}. The FNO is accurate only inside the training distribution. Moving the
        dam off-centre or changing its depth quickly drives the error towards the persistence baseline, and at the
        extremes the error exceeds it. Try it in <em>Viewer → Custom dam break</em>.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Initial condition</th>
              <th scope="col">FNO nRMSE</th>
              <th scope="col">× in-distribution</th>
              <th scope="col">Persistence nRMSE</th>
              <th scope="col">Skill</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ fam, c }) => (
              <tr key={`${fam}-${caseLabel(fam, c)}`}>
                <th scope="row">
                  {caseLabel(fam, c)}
                  {c.ood.length === 0 && <span className="tag test" style={{ marginLeft: 8 }}>in-dist.</span>}
                </th>
                <td>{fmtSci(c.nRMSE)}</td>
                <td>{(c.nRMSE / gen.in_distribution_nRMSE).toFixed(1)}×</td>
                <td>{fmtSci(c.persistence_nRMSE)}</td>
                <td className={c.skill < 0 ? "neg" : undefined}>{c.skill.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Skill = 1 − nRMSE<sub>FNO</sub> / nRMSE<sub>persistence</sub>. At 0 the FNO is no better than holding the last
        input frame; below 0 it is worse.
      </p>
      <figure className="gen-figure">
        <img src={api.figureUrl("generalization_families.png")} alt="FNO error vs. radius, dam position and inner depth" loading="lazy" />
      </figure>
    </section>
  );
}

const HELDOUT_ROWS: [HeldoutFamily | "seen", string][] = [
  ["subcell", "Sub-cell dam position"],
  ["cellavg", "Cell-averaged dam wall"],
  ["seen", "Control: the 81 training ICs"],
];

function HeldoutSection({ held }: { held: Heldout }) {
  const f = held.families;
  const gc = held.generator_check;
  const gates = held.criteria.filter((c) => c.kind === "gate");
  const findings = held.criteria.filter((c) => c.kind === "finding");
  const famLabel = (fam: string) => (fam === "all" ? "Both sets" : held.design.labels[fam] ?? fam);
  return (
    <section className="panel">
      <h2>
        Held-out in-distribution test
        <span className={`pill ${held.verdict === "PASS" ? "good" : "critical"}`} style={{ marginLeft: 10 }}>
          {held.verdict === "PASS" ? "gates pass" : "gates fail"}
        </span>
      </h2>
      <p>
        PDEBench&apos;s generator can only produce {held.design.n_training_ics} distinct initial conditions on this grid,
        and its training split contains all of them, so no amount of re-sampling yields unseen in-distribution data.
        We kept the physical specification (a centred circular dam, r ~ U(0.3, 0.7), h = 2 inside and 1 outside) and
        discretised it in two ways that never reproduce a training IC: a <strong>sub-cell dam position</strong> (the
        centre drawn within half a cell of the origin, which the grid cannot resolve) and a{" "}
        <strong>cell-averaged dam wall</strong> (cut cells hold the fraction of water). Every reference trajectory comes
        from PyClaw configured as PDEBench, which reproduces {gc.n_bit_identical}/{gc.n_checked} downloaded samples bit
        for bit.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Initial conditions</th>
              <th scope="col">n</th>
              <th scope="col">FNO nRMSE (95 % CI)</th>
              <th scope="col">× paper</th>
              <th scope="col">× training ICs</th>
              <th scope="col">Persistence</th>
              <th scope="col">Nearest-trajectory lookup</th>
            </tr>
          </thead>
          <tbody>
            {HELDOUT_ROWS.map(([fam, label]) => {
              const r = f[fam];
              const ratio = (r.pdebench_compat?.nRMSE ?? r.nRMSE) / held.paper.nRMSE;
              return (
                <tr key={fam}>
                  <th scope="row">{label}</th>
                  <td>{r.n}</td>
                  <td>
                    {fmtSci(r.nRMSE)} <span className="muted">[{fmtSci(r.nRMSE_ci95[0], 1)}, {fmtSci(r.nRMSE_ci95[1], 1)}]</span>
                  </td>
                  <td>{ratio.toFixed(2)}×</td>
                  <td>{(r.nRMSE / f.seen.nRMSE).toFixed(2)}×</td>
                  <td>{fmtSci(r.persistence_nRMSE)}</td>
                  <td>{fam === "seen" ? "exact (0)" : fmtSci(r.lookup_nRMSE)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="note">
        × paper uses PDEBench&apos;s aggregation for the held-out sets (gate: ≤ 2×). The control row is weighted by each
        IC&apos;s probability under r ~ U(0.3, 0.7). The lookup returns the training trajectory whose 10 input frames are
        closest. Because the distribution is so narrow, it does as well as the FNO or better. Browse the cases in{" "}
        <em>Viewer → Held-out</em>.
      </p>
      <ul className="checks">
        {[...gates, ...findings].map((c) => (
          <li key={`${c.family}-${c.name}`} className={c.pass ? "ok" : c.kind === "gate" ? "fail" : "issue"}>
            <StatusIcon ok={c.pass} kind={c.kind} />
            <div>
              <div className="check-name">
                {c.name}
                <span className="check-kind">{famLabel(c.family)} · {c.kind === "gate" ? (c.pass ? "Pass" : "Fail") : c.pass ? "OK" : "Issue"}</span>
              </div>
              <div className="check-detail">
                {fmtValue(c.value)}
                {c.detail ? ` · ${c.detail}` : ""} · criterion: {c.threshold}
              </div>
            </div>
          </li>
        ))}
      </ul>
      <figure className="gen-figure">
        <img src={api.figureUrl("heldout_error_vs_radius.png")} alt="FNO error on held-out and training initial conditions vs. dam radius" loading="lazy" />
      </figure>
      <figure className="gen-figure">
        <img src={api.figureUrl("heldout_offset_sweep.png")} alt="FNO error vs. distance of the dam centre from the origin" loading="lazy" />
      </figure>
    </section>
  );
}
