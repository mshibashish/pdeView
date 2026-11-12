"""Verify the PDEBench pretrained 2D shallow-water FNO before using it in the viewer.

Writes ``reports/verification.json`` (consumed by the API), ``reports/VERIFICATION.md`` and
figures under ``reports/figures/``. Exits non-zero if any check fails.

Usage:  uv run python scripts/verify_checkpoint.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from pdeview.config import FNO_INITIAL_STEP, REPORTS_DIR  # noqa: E402
from pdeview.data.store import SampleStore  # noqa: E402
from pdeview.inference import make_grid, rollout  # noqa: E402
from pdeview.metrics import METRIC_KEYS  # noqa: E402
from pdeview.model.checkpoint import load_fno  # noqa: E402
from pdeview.verification import run_verification  # noqa: E402

# Reference data-viz palette (light mode): entity colours are shared with the web viewer.
SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FNO_C, PERSIST_C = "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "h", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("err", ["#1c5cab", "#f0efec", "#e34948"])

FIG_DIR = REPORTS_DIR / "figures"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def fig_error_vs_time(rep: dict) -> str:
    tc = rep["time_curves"]
    t = np.array(tc["t"])
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    _style(ax)
    series = [
        (tc["persistence_test_nRMSE"], PERSIST_C, "-", "Persistence baseline (test)"),
        (tc["fno_train_nRMSE"], FNO_C, "--", "FNO, training-set probe"),
        (tc["fno_test_nRMSE"], FNO_C, "-", "FNO, PDEBench test split"),
    ]
    for y, c, ls, label in series:
        ax.plot(t, y, color=c, linestyle=ls, linewidth=2, label=label)
    ax.axhline(rep["accuracy"]["paper"]["nRMSE"], color=MUTED, linewidth=1, linestyle=":",
               label="Paper, FNO time-averaged nRMSE")
    ax.set_yscale("log")
    ax.set_xlabel("simulation time t (frames 10-100 are autoregressive predictions)")
    ax.set_ylabel("nRMSE (mean over samples)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              labelcolor=INK2)
    ax.set_title("Error growth along the 91-step rollout", loc="left", color=INK, fontsize=11)
    fig.subplots_adjust(right=0.72)
    path = FIG_DIR / "error_vs_time.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_error_vs_radius(rep: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    _style(ax)
    for split, face, label in (("test", FNO_C, "PDEBench test split"),
                               ("train", "none", "training-set probe")):
        pts = [p for p in rep["per_sample"] if p["split"] == split]
        ax.scatter([p["dam_radius"] for p in pts], [p["nRMSE"] for p in pts], s=34,
                   facecolors=face, edgecolors=FNO_C, linewidths=1.5, label=label,
                   zorder=3)
    ax.set_xlabel("dam radius r (training range 0.3-0.7)")
    ax.set_ylabel("time-averaged nRMSE")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("Per-sample FNO error vs. dam radius", loc="left", color=INK, fontsize=11,
                 pad=16)
    ax.text(0, 1.01, "coinciding points are bit-identical samples", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="bottom")
    fig.tight_layout()
    path = FIG_DIR / "error_vs_radius.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_snapshots(rep: dict, key: str, tag: str) -> str:
    store = SampleStore()
    model, _ = load_fno()
    gt = store.frames(key)
    pred = rollout(model, gt[:FNO_INITIAL_STEP], make_grid(store.x, store.y), len(store.t))
    times = [10, 30, 60, 100]
    ext = [store.y[0], store.y[-1], store.x[0], store.x[-1]]
    vmin, vmax = float(gt.min()), float(gt.max())
    elim = float(np.abs(pred[10:] - gt[10:]).max())
    fig, axes = plt.subplots(3, len(times), figsize=(2.6 * len(times) + 1, 8),
                             facecolor=SURFACE)
    for c, ti in enumerate(times):
        for r, (field, cmap, lo, hi) in enumerate((
            (gt[ti], SEQ, vmin, vmax), (pred[ti], SEQ, vmin, vmax),
            (pred[ti] - gt[ti], DIV, -elim, elim),
        )):
            ax = axes[r, c]
            im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {store.t[ti]:.2f}", color=INK2, fontsize=10)
            if c == 0:
                ax.set_ylabel(("Ground truth h", "FNO prediction h", "FNO - GT")[r],
                              color=INK2, fontsize=10)
            if c == len(times) - 1:
                fig.colorbar(im, ax=axes[r, :], shrink=0.85, pad=0.02).ax.tick_params(
                    labelsize=8, colors=MUTED)
    nr = next(p["nRMSE"] for p in rep["per_sample"] if p["key"] == key)
    fig.suptitle(f"Sample {key} ({tag} test error, r = {store.samples[key].dam_radius:.3f}, "
                 f"nRMSE = {nr:.2e})", color=INK, fontsize=11, x=0.02, ha="left")
    path = FIG_DIR / f"snapshots_{tag}.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, dict):
        return ", ".join(f"{k} {v:.2f}" for k, v in val.items())
    return str(val)


def write_markdown(rep: dict, figs: dict[str, str]) -> None:
    acc = rep["accuracy"]
    ck = rep["checkpoint"]
    fp = rep["fingerprint"]
    au = rep["audit"]
    dev = rep["device"]
    gates = [c for c in rep["criteria"] if c["kind"] == "gate"]
    findings = [c for c in rep["criteria"] if c["kind"] == "finding"]
    lines = [
        "# Pretrained FNO verification - 2D shallow-water equations (PDEBench `2D_rdb_NA_NA`)",
        "",
        f"Generated {rep['generated_at']} by `scripts/verify_checkpoint.py` "
        f"(torch {rep['environment']['torch']}, {rep['environment']['machine']}).",
        "",
        "## Summary",
        "",
        f"* **Gates: {rep['verdict']}.** The DaRUS checkpoint is intact, loads strictly into "
        "PDEBench's FNO architecture and, run as PDEBench runs it, reaches "
        f"nRMSE {acc['pdebench_compat']['nRMSE']:.2e} on PDEBench's test split "
        f"({acc['ratio_to_paper']['nRMSE']:.2f}x the paper's {acc['paper']['nRMSE']:.1e}; "
        f"all metrics {min(acc['ratio_to_paper'].values()):.2f}-"
        f"{max(acc['ratio_to_paper'].values()):.2f}x the paper) and is "
        f"{acc['persistence_over_fno_nRMSE']:.0f}x better than a persistence baseline. "
        "It is used in the viewer.",
        f"* **Finding 1 - the test split does not measure generalisation.** The 1000 samples "
        f"contain only **{au['n_distinct_trajectories']} distinct trajectories**; "
        f"**{au['test_with_identical_train_trajectory']}/{au['n_test']} test samples are "
        "bit-identical to a training sample.** The reported accuracy is accuracy on trajectories"
        " the model was trained on.",
        f"* **Finding 2 - the checkpoint's stored validation loss is not reproducible** "
        f"({fp['reproduced_val_loss'] / fp['stored_val_loss']:.2f}x higher when recomputed), and"
        " the checkpoint is somewhat less accurate than the paper's table, suggesting it was"
        " trained/evaluated on a different revision of the data or pipeline than the one"
        " published.",
        "",
        "## Gates (must pass to use the model)",
        "",
        "| # | Check | Result | Criterion | |",
        "|---|---|---|---|---|",
    ]
    for i, c in enumerate(gates, 1):
        detail = f" ({c['detail']})" if c["detail"] else ""
        lines.append(f"| {i} | {c['name']} | `{_fmt(c['value'])}`{detail} | {c['threshold']} | "
                     f"{'PASS' if c['pass'] else '**FAIL**'} |")
    lines += ["", "## Findings (properties users of the model must know)", "",
              "| # | Check | Result | Expected | |", "|---|---|---|---|---|"]
    for i, c in enumerate(findings, 1):
        detail = f" ({c['detail']})" if c["detail"] else ""
        lines.append(f"| {i} | {c['name']} | {_fmt(c['value'])}{detail} | {c['threshold']} | "
                     f"{'OK' if c['pass'] else '**ISSUE**'} |")
    lines += [
        "",
        "## Metrics vs. the PDEBench paper",
        "",
        "PDEBench test split = the last 10 % of sorted sample keys (`0900`-`0999`, 100"
        " samples), as in PDEBench's default FNO loader. Frames 0-9 are ground-truth input;"
        " metrics cover the 91 autoregressive predictions (frames 10-100).",
        "",
        "* **PDEBench aggregation** reproduces the paper's evaluation loop (per-batch metrics,"
        " batch size 5, summed and divided by `n_batches - 1`) - the like-for-like comparison.",
        "* **Clean** aggregates all 100 samples as one population (no off-by-one; `max error`,"
        " `cRMSE` and `fRMSE` taken over all samples rather than per batch of 5).",
        f"* **Dedup** keeps one sample per distinct trajectory ({acc['n_dedup_test']} of 100).",
        "",
        "| Metric | Paper (Table 5) | Ours, PDEBench aggr. | Ratio | Ours, clean | Ours, dedup |"
        " Persistence (clean) | Train probe (clean) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in METRIC_KEYS:
        lines.append(
            f"| {k} | {acc['paper'][k]:.2e} | {acc['pdebench_compat'][k]:.2e} | "
            f"{acc['ratio_to_paper'][k]:.2f} | {acc['clean'][k]:.2e} | "
            f"{acc['dedup_test_clean'][k]:.2e} | {acc['persistence_clean'][k]:.2e} | "
            f"{acc['train_probe_clean'][k]:.2e} |")
    sweep = ", ".join(f"t_train={k}: {v:.2e}" for k, v in fp["horizon_sweep"].items())
    base = fp["grid_ablation_mse"]["cell centres, ij (used)"]
    abl = ", ".join(f"{k}: {v / base:.2f}x" for k, v in fp["grid_ablation_mse"].items())
    lines += [
        "",
        "## Checkpoint",
        "",
        f"* File `2D_rdb_NA_NA_FNO.pt`, MD5 `{ck['md5']}` (DaRUS: `{ck['md5_expected']}`).",
        f"* Keys {ck['keys']}; saved at epoch **{ck['epoch']}** of 500; optimizer LR at save"
        f" {ck['optimizer_lr']:.3g} (= 1e-3 x 0.5^4, consistent with StepLR(100, 0.5) after"
        " epoch 400).",
        f"* {ck['n_parameters']:,} parameters ({ck['n_real_parameters']:,} real numbers); "
        f"hyper-parameters {ck['hyperparameters']}.",
        "",
        "## Finding 1: duplicated trajectories (train/test leakage)",
        "",
        "Each sample is a centred dam break whose only parameter is the radius r ~ U(0.3, 0.7)."
        " The initial depth is assigned per cell (h = 2 where the cell centre lies within r),"
        " so every radius between two consecutive cell-centre distances yields the *same*"
        " discrete initial condition - and PyClaw is deterministic. On the 128 x 128 grid,"
        f" r in [0.3, 0.7] crosses only {au['n_possible_initial_conditions'] - 1} distance"
        f" shells, i.e. {au['n_possible_initial_conditions']} possible initial conditions, all"
        " of which occur.",
        "",
        f"* `scripts/audit_dataset.py` reads the `config` of all {au['n_samples']} samples"
        f" remotely: {au['n_distinct_trajectories']} distinct trajectories, each repeated"
        f" {au['copies_per_trajectory']['min']}-{au['copies_per_trajectory']['max']} times"
        f" (median {au['copies_per_trajectory']['median']:.0f}). Training covers"
        f" {au['n_distinct_train']}, test {au['n_distinct_test']} - **every** test trajectory"
        " is in the training set.",
        f"* Confirmed on the downloaded data: within the 110 local samples, "
        f"{au['identical_pairs_in_subset']} pairs are bit-identical, and the initial-condition"
        " signature predicts bit-identity for every pair: "
        f"{au['signature_predicts_bit_identity_in_subset']}.",
        "* Consequences: (i) test metrics measure how well the FNO reproduces training"
        " trajectories, not generalisation; (ii) the train-probe vs. test comparison is"
        f" uninformative (ratio {acc['train_over_test_nRMSE']:.2f}); (iii) PDEBench selects"
        " the saved checkpoint on this same test split. Genuine generalisation can only be"
        " probed with *new* simulations - see [GENERALIZATION.md](GENERALIZATION.md), where an"
        " independent solver that reproduces PyClaw to 6e-8 shows the FNO is accurate only"
        " inside the training distribution.",
        "",
        "## Finding 2: stored validation loss",
        "",
        "PDEBench saves the best validation loss in the checkpoint (`loss`: per-batch MSE over"
        " frames 10..100, summed over the 20 batches of the test split).",
        "",
        f"* Stored: `{fp['stored_val_loss']:.6e}`; recomputed from our rollouts:"
        f" `{fp['reproduced_val_loss']:.6e}`"
        f" ({fp['reproduced_val_loss'] / fp['stored_val_loss']:.2f}x).",
        f"* Shorter training horizons do not explain it (the error is largest early in the"
        f" rollout): {sweep}.",
        f"* The coordinate-grid convention is not the cause - every alternative is worse"
        f" (MSE relative to ours, 10 samples): {abl}.",
        "* The split is not the cause (all test trajectories also appear in training).",
        "* The published SWE data also differs from the current PDEBench generator (radii are"
        " not `default_rng(seed).uniform(0.3, 0.7)`), and the checkpoint (DaRUS 2022-06-21)"
        " predates the data release (2022-07-26). The most plausible explanation is that the"
        " checkpoint was trained on an earlier data/pipeline revision; this cannot be verified"
        " from public artefacts, so it is reported as an open discrepancy.",
        "",
        "## Other notes",
        "",
        "* PDEBench's `metrics()` divides summed per-batch errors by the last batch *index*"
        " (19 instead of 20 batches), inflating reported values by ~5 %; the *clean* column"
        " removes this. Its `max error` is the mean over time of the per-batch worst case, so it"
        " depends on batch size.",
        f"* Ground-truth water volume is conserved to {rep['data']['max_gt_volume_drift']:.1e}"
        " (waves never reach the outflow boundary before t = 1); the FNO is not conservative by"
        " construction - its volume error is the `cRMSE` metric.",
        "* The FNO error is concentrated at the moving bore (ringing from the 12-mode spectral"
        " truncation of a near-discontinuity) and is largest early in the rollout, when the"
        " front is sharpest, and for the largest radii.",
        f"* Inference: {dev['cpu_seconds_per_sample']:.2f} s per 91-step rollout on CPU (batched),"
        + (f" {dev['mps_seconds_single_sample']:.2f} s on Apple MPS (single sample); MPS vs. CPU"
           f" max abs difference {dev['mps_max_abs_diff_vs_cpu']:.1e}."
           if dev.get("mps_seconds_single_sample") else ""),
        "* Our metric implementation was cross-checked against PDEBench's `metric_func` on the"
        " same predictions: all eight metrics agree to <= 7e-6 relative (float32 round-off).",
        "",
        "## Figures",
        "",
        f"![Error vs time](figures/{figs['time']})",
        "",
        f"![Error vs radius](figures/{figs['radius']})",
        "",
        f"![Median sample](figures/{figs['median']})",
        "",
        f"![Worst sample](figures/{figs['worst']})",
        "",
    ]
    (REPORTS_DIR / "VERIFICATION.md").write_text("\n".join(lines))


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rep = run_verification()
    (REPORTS_DIR / "verification.json").write_text(json.dumps(rep, indent=1))
    figs = {
        "time": fig_error_vs_time(rep),
        "radius": fig_error_vs_radius(rep),
        "median": fig_snapshots(rep, rep["highlights"]["median"], "median"),
        "worst": fig_snapshots(rep, rep["highlights"]["worst"], "worst"),
    }
    write_markdown(rep, figs)
    for c in rep["criteria"]:
        status = ("PASS" if c["pass"] else "FAIL") if c["kind"] == "gate" else \
            ("OK" if c["pass"] else "ISSUE")
        print(f"[{status:5s}] {c['kind']:7s} {c['name']}")
    print(f"\nGates: {rep['verdict']}  ->  {REPORTS_DIR / 'VERIFICATION.md'}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
