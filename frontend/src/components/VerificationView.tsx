import { useEffect, useState } from "react";
import { api } from "../api";
import { fmtSci, metricLabel } from "../format";
import type { MetricKey, Verification } from "../types";

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

  useEffect(() => {
    const ctrl = new AbortController();
    api.verification(ctrl.signal).then(setV).catch((e: unknown) => {
      if (!ctrl.signal.aborted) setErr(e instanceof Error ? e.message : String(e));
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
