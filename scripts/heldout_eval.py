"""Evaluate the FNO on a held-out, in-distribution test set that shares no trajectory with
PDEBench's training split. Writes ``reports/heldout.json``, ``reports/HELDOUT.md`` and
figures under ``reports/figures/``; exits non-zero if a gate fails.

References are generated with PyClaw (``scripts/pyclaw_reference.py``, run through uv in a
separate Python 3.13 environment); see ``pdeview.heldout`` for the design.

Usage:  uv run python scripts/heldout_eval.py [--render-only]
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
from pdeview.heldout import REF_FILE, ReferenceStore, run  # noqa: E402
from pdeview.inference import make_grid, rollout  # noqa: E402
from pdeview.metrics import METRIC_KEYS  # noqa: E402
from pdeview.model.checkpoint import load_fno  # noqa: E402
from pdeview.solver.swe import cell_centres  # noqa: E402

SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FNO_C, PERSIST_C, BASE_C = "#eb6834", "#1baf7a", "#898781"
SEQ = LinearSegmentedColormap.from_list(
    "h", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("err", ["#1c5cab", "#f0efec", "#e34948"])
FIG_DIR = REPORTS_DIR / "figures"
JSON_FILE = REPORTS_DIR / "heldout.json"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _cases(rep, fam):
    return [p for p in rep["per_case"] if p["family"] == fam]


def _paper_lines(ax, rep):
    paper = rep["paper"]["nRMSE"]
    ax.axhline(paper, color=MUTED, lw=1, ls=":", label="Paper, FNO nRMSE")
    ax.axhline(2 * paper, color=MUTED, lw=1, ls="--", label="2x the paper (gate on the mean)")


def fig_error_vs_radius(rep: dict) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    seen = sorted(_cases(rep, "seen"), key=lambda p: p["radius"])
    ax.plot([p["radius"] for p in seen], [p["nRMSE"] for p in seen], color=BASE_C, lw=1.5,
            marker="o", ms=3, label="81 training ICs (seen)", zorder=2)
    sub, avg = _cases(rep, "subcell"), _cases(rep, "cellavg")
    ax.scatter([p["radius"] for p in sub], [p["nRMSE"] for p in sub], s=26, color=FNO_C,
               edgecolors=SURFACE, linewidths=0.8, label="Held-out: sub-cell dam position",
               zorder=3)
    ax.scatter([p["radius"] for p in avg], [p["nRMSE"] for p in avg], s=26, marker="s",
               facecolors="none", edgecolors=FNO_C, linewidths=1.3,
               label="Held-out: cell-averaged dam wall", zorder=3)
    _paper_lines(ax, rep)
    ax.set_xlabel("dam radius r (training range 0.3-0.7)")
    ax.set_ylabel("FNO time-averaged nRMSE")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_title("FNO error on held-out vs. training initial conditions", loc="left",
                 color=INK, fontsize=11)
    fig.subplots_adjust(right=0.66, bottom=0.13)
    path = FIG_DIR / "heldout_error_vs_radius.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_error_vs_time(rep: dict) -> str:
    fams = rep["families"]
    t = np.array(fams["subcell"]["t"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    for ax, fam in zip(axes, ("subcell", "cellavg"), strict=True):
        _style(ax)
        tc = fams[fam]["time_curves"]
        ax.plot(t, tc["persistence_nRMSE"], color=PERSIST_C, lw=2, label="Persistence baseline")
        ax.plot(t, tc["lookup_nRMSE"], color=BASE_C, lw=2, ls="--",
                label="Nearest training trajectory")
        ax.plot(t, fams["seen"]["time_curves"]["fno_nRMSE"], color=FNO_C, lw=1.5, ls=":",
                label="FNO on the 81 training ICs")
        ax.plot(t, tc["fno_nRMSE"], color=FNO_C, lw=2, label="FNO on held-out ICs")
        ax.set_yscale("log")
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("simulation time t")
        ax.set_title(rep["design"]["labels"][fam], loc="left", color=INK, fontsize=10.5)
    axes[0].set_ylabel("nRMSE (mean over cases)")
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
                   bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(left=0.07, right=0.79, wspace=0.08, bottom=0.14)
    path = FIG_DIR / "heldout_error_vs_time.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_offset_sweep(rep: dict) -> str:
    pts = _cases(rep, "offset")
    mags = np.array(rep["design"]["offset_magnitudes"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    _style(ax)
    half = rep["design"]["half_cell"]
    ax.axvspan(0, half * np.sqrt(2), color="#f0efec", zorder=0)
    ax.text(half * np.sqrt(2) / 2, 0.97, "sub-cell\nfamily", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=MUTED)

    def mean_curve(key):
        return [np.mean([p[key] for p in pts
                         if abs(np.hypot(p["cx"], p["cy"]) - m) < 1e-9]) for m in mags]

    first = True
    for p_r in rep["design"]["offset_radii"]:
        for direction in ("axis", "diag"):
            sel = sorted((p for p in pts if p["key"].startswith(f"off-{direction}-{p_r:.2f}")),
                         key=lambda p: np.hypot(p["cx"], p["cy"]))
            ax.plot([np.hypot(p["cx"], p["cy"]) for p in sel], [p["nRMSE"] for p in sel],
                    color=FNO_C, lw=0.8, alpha=0.35, zorder=2,
                    label="FNO, each radius and direction" if first else None)
            first = False
    ax.plot(mags, mean_curve("nRMSE"), color=FNO_C, lw=2.2, marker="o", ms=5,
            label="FNO error (mean)", zorder=3)
    ax.plot(mags, mean_curve("distance_to_training_set"), color=BASE_C, lw=2, ls="--",
            marker="o", ms=4, label="Nearest training trajectory\n(its distance)", zorder=3)
    _paper_lines(ax, rep)
    ax.set_xlabel("distance of the dam centre from the origin (cell width = 0.039)")
    ax.set_ylabel("time-averaged nRMSE")
    ax.set_xlim(0, mags[-1])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Error grows with the distance from the training ICs", loc="left", color=INK,
                 fontsize=11)
    fig.subplots_adjust(right=0.66, bottom=0.13)
    path = FIG_DIR / "heldout_offset_sweep.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_snapshots(rep: dict, key: str, tag: str) -> str:
    refs = ReferenceStore(REF_FILE)
    gt = refs.frames(key)
    refs.close()
    model, _ = load_fno()
    x = cell_centres().astype(np.float32)
    pred = rollout(model, gt[:FNO_INITIAL_STEP], make_grid(x, x), gt.shape[0])
    case = next(p for p in rep["per_case"] if p["key"] == key)
    times = [0, 10, 50, 100]
    ext = [x[0], x[-1], x[0], x[-1]]
    vmin, vmax = float(gt.min()), float(gt.max())
    elim = float(np.abs(pred[10:] - gt[10:]).max())
    fig, axes = plt.subplots(3, len(times), figsize=(2.6 * len(times) + 1, 8), facecolor=SURFACE)
    for c, ti in enumerate(times):
        for r, (field, cmap, lo, hi) in enumerate((
            (gt[ti], SEQ, vmin, vmax), (pred[ti], SEQ, vmin, vmax),
            (pred[ti] - gt[ti], DIV, -elim, elim),
        )):
            ax = axes[r, c]
            if ti == 0 and r > 0:  # the FNO starts at frame 10; only the IC is shown at t = 0
                ax.set_visible(False)
                continue
            # rows of the [X, Y] array are x: transpose so that x runs horizontally
            im = ax.imshow(field.T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if ti == 0:
                ax.set_xlim(-0.8, 0.8)
                ax.set_ylim(-0.8, 0.8)
                ax.axhline(0, color=MUTED, lw=0.6)
                ax.axvline(0, color=MUTED, lw=0.6)
            if r == 0:
                ax.set_title("initial condition (zoom)" if ti == 0 else f"t = {ti / 100:.2f}",
                             color=INK2, fontsize=10)
            if c == 1:
                ax.set_ylabel(("PyClaw reference h", "FNO prediction h", "FNO - reference")[r],
                              color=INK2, fontsize=10)
            if c == len(times) - 1:
                fig.colorbar(im, ax=axes[r, 1:], shrink=0.85, pad=0.02).ax.tick_params(
                    labelsize=8, colors=MUTED)
    label = rep["design"]["labels"][case["family"]]
    where = (f", centre ({case['cx']:+.4f}, {case['cy']:+.4f})" if case["family"] == "subcell"
             else "")
    fig.suptitle(f"{label}, {tag.split()[0]} case {key} (r = {case['radius']:.3f}{where}, "
                 f"nRMSE = {case['nRMSE']:.2e})", color=INK, fontsize=11, x=0.02, ha="left")
    path = FIG_DIR / f"heldout_{tag.replace(' ', '_')}.png"
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
    fams, d, paper = rep["families"], rep["design"], rep["paper"]
    sub, avg, seen, off = fams["subcell"], fams["cellavg"], fams["seen"], fams["offset"]
    gc, sc = rep["generator_check"], rep["solver_check"]
    labels = d["labels"]
    lines = [
        "# Held-out in-distribution test of the pretrained FNO",
        "",
        f"Generated {rep['generated_at']} by `scripts/heldout_eval.py` "
        f"(torch {rep['environment']['torch']}). Companion to "
        "[VERIFICATION.md](VERIFICATION.md) and [GENERALIZATION.md](GENERALIZATION.md).",
        "",
        "## Summary",
        "",
        f"* **Gates: {rep['verdict']}.** On {sub['n'] + avg['n']} new dam breaks that follow "
        "PDEBench's physical specification but share no trajectory with its training split, "
        f"the FNO reaches nRMSE **{sub['pdebench_compat']['nRMSE']:.2e}** (sub-cell dam "
        f"position, {sub['ratio_to_paper']['nRMSE']:.2f}x the paper) and "
        f"**{avg['pdebench_compat']['nRMSE']:.2e}** (cell-averaged dam wall, "
        f"{avg['ratio_to_paper']['nRMSE']:.2f}x the paper), PDEBench aggregation. That is "
        f"{sub['persistence_over_fno_nRMSE']:.0f}x and {avg['persistence_over_fno_nRMSE']:.0f}x "
        "better than persistence. The model is satisfactorily accurate in distribution.",
        "* **Why new data was needed.** As r sweeps U(0.3, 0.7), the dam crosses only "
        f"{d['n_training_ics'] - 1} cell-centre distance shells on the 128 x 128 grid, so "
        f"PDEBench's generator can only produce **{d['n_training_ics']} distinct initial "
        "conditions**, and its 900-sample training split contains all of them. Every test "
        "sample, and every sample the generator could ever add, is a copy of a training "
        "trajectory.",
        f"* **Generalisation gap.** Cell-averaged ICs: {avg['gap_to_seen']:.2f}x the error on "
        f"the training ICs (none). Sub-cell positions: {sub['gap_to_seen']:.2f}x (moderate). "
        "The error rises steadily as the dam moves away from the exact grid vertex that every "
        "training sample was centred on (see the offset sweep). Both families stay inside "
        "the 2x gate.",
        "* **Caveat: a lookup table does as well or better.** The distribution is so narrow that "
        "returning the nearest of the 81 training trajectories (chosen on the same 10 input "
        f"frames) gives nRMSE {sub['lookup_nRMSE']:.2e} / {avg['lookup_nRMSE']:.2e}. The FNO "
        f"beats it on {sub['fno_beats_lookup_fraction']:.0%} / "
        f"{avg['fno_beats_lookup_fraction']:.0%} of cases. The FNO's own emulation error "
        f"(~{seen['nRMSE']:.1e}, even on training ICs) is larger than the gap between "
        "neighbouring training trajectories. Its advantage is size (11 MB vs. ~540 MB for "
        "the trajectories), not accuracy.",
        "",
        "## Test sets",
        "",
        "All references come from PyClaw configured as PDEBench "
        f"(`scripts/pyclaw_reference.py`, {rep['reference_generator']}). That configuration "
        f"reproduces **{gc['n_bit_identical']}/{gc['n_checked']} downloaded PDEBench samples "
        "bit for bit** "
        f"(max |dh| = {gc['max_abs_difference']:.1e}). Physical setup as in training: g = 1, "
        "h_in = 2, h_out = 1, zero velocity, 128 x 128 cells on [-2.5, 2.5]^2, outputs every "
        "0.01 up to t = 1. Frames 0-9 are given to the FNO; metrics cover frames 10-100.",
        "",
        "| Set | n | Radius | Centre | Dam wall | Differs from training by |",
        "|---|---|---|---|---|---|",
        f"| {labels['subcell']} | {sub['n']} | U(0.3, 0.7) | uniform in the half-cell square "
        f"around the origin (abs(cx), abs(cy) <= {d['half_cell']:.4f}) | sharp (cell centre "
        "inside -> h = 2) | which cells the wall cuts; the IC is not 8-fold symmetric |",
        f"| {labels['cellavg']} | {avg['n']} | U(0.3, 0.7) | origin | cell average "
        f"({d['supersample']}x{d['supersample']} sub-samples) | cut cells hold 1 + the "
        "fraction inside the wall |",
        f"| {labels['seen']} (control) | {seen['n']} | one per distance shell, weighted by its "
        "probability | origin | sharp | nothing: these *are* the training ICs |",
        f"| {labels['offset']} | {off['n']} | {', '.join(map(str, d['offset_radii']))} | "
        f"offset 0-{max(d['offset_magnitudes'])} along x or the diagonal | sharp | bridges to "
        "the off-centre study in GENERALIZATION.md |",
        "",
        f"Random draws use seed {d['seed']}. A centre offset too small to move any cell "
        "centre across the dam wall reproduces a training IC exactly. "
        f"{d['n_rejected_draws']} such draw(s) were rejected and redrawn, so none of the "
        f"{sub['n'] + avg['n']} held-out ICs equals a training IC or another held-out IC "
        "(checked by hash).",
        "",
        "## Results",
        "",
        "| Set | FNO nRMSE (mean, 95 % CI) | PDEBench aggr. | x paper | Median / max | "
        "Persistence | Nearest-trajectory lookup | Distance to training set |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fam in ("subcell", "cellavg", "seen"):
        f = fams[fam]
        compat = f.get("pdebench_compat")
        name = labels[fam] + ("" if compat else " (probability-weighted)")
        agg = f"{compat['nRMSE']:.2e}" if compat else "-"
        ratio = (compat or f)["nRMSE"] / paper["nRMSE"]
        lines.append(
            f"| {name} | {f['nRMSE']:.2e} [{f['nRMSE_ci95'][0]:.2e}, {f['nRMSE_ci95'][1]:.2e}] "
            f"| {agg} | {ratio:.2f} | {f['nRMSE_median']:.2e} / {f['nRMSE_max']:.2e} | "
            f"{f['persistence_nRMSE']:.2e} | {f['lookup_nRMSE']:.2e} | "
            f"{f['distance_to_training_set']:.2e} |")
    over = {fam: sum(p["nRMSE"] > 2 * paper["nRMSE"] for p in _cases(rep, fam))
            for fam in ("subcell", "cellavg")}
    big = {fam: np.mean([p["nRMSE"] for p in _cases(rep, fam) if p["radius"] > 0.6])
           for fam in ("subcell", "cellavg")}
    lines += [
        "",
        f"The 2x gate applies to the mean. Individually, {over['subcell']}/{sub['n']} sub-cell "
        f"cases and {over['cellavg']}/{avg['n']} cell-averaged cases exceed 2x the paper. "
        "Sub-cell errors grow with the radius, because a larger wall cuts more cells "
        f"asymmetrically: the mean is {big['subcell']:.2e} for r > 0.6 vs. "
        f"{big['cellavg']:.2e} for cell-averaged dams of the same size. In the PDEBench "
        "aggregation the sub-cell set's low-frequency error (fRMSE low) is "
        f"{sub['ratio_to_paper']['fRMSE_low']:.1f}x "
        "the paper, the only metric above 3x. The FNO misplaces the slow central "
        "depression when the dam is not symmetric about the grid vertex (see the worst case "
        "below).",
        "",
        "*Distance to training set* is the nRMSE between a case's reference and the closest "
        "training trajectory (frames 10-100); it is 0 for the training ICs. *Lookup* picks a "
        "training trajectory by the input frames only, as a model would have to.",
        "",
        "## Gates (must pass to call the model accurate in distribution)",
        "",
        "Same thresholds as the main verification.",
        "",
        "| # | Set | Check | Result | Criterion | |",
        "|---|---|---|---|---|---|",
    ]
    gates = [c for c in rep["criteria"] if c["kind"] == "gate"]
    findings = [c for c in rep["criteria"] if c["kind"] == "finding"]
    for i, c in enumerate(gates, 1):
        detail = f" ({c['detail']})" if c["detail"] else ""
        fam = labels.get(c["family"], "all")
        lines.append(f"| {i} | {fam} | {c['name']} | `{_fmt(c['value'])}`{detail} | "
                     f"{c['threshold']} | {'PASS' if c['pass'] else '**FAIL**'} |")
    lines += ["", "## Findings", "", "| # | Set | Check | Result | Expected | |",
              "|---|---|---|---|---|---|"]
    for i, c in enumerate(findings, 1):
        detail = f" ({c['detail']})" if c["detail"] else ""
        lines.append(f"| {i} | {labels[c['family']]} | {c['name']} | {_fmt(c['value'])}{detail}"
                     f" | {c['threshold']} | {'OK' if c['pass'] else '**ISSUE**'} |")
    lines += [
        "",
        "## All metrics vs. the paper (PDEBench aggregation)",
        "",
        f"| Metric | Paper (Table 5) | {labels['subcell']} | x paper | {labels['cellavg']} | "
        "x paper |",
        "|---|---|---|---|---|---|",
    ]
    for k in METRIC_KEYS:
        lines.append(f"| {k} | {paper[k]:.2e} | {sub['pdebench_compat'][k]:.2e} | "
                     f"{sub['ratio_to_paper'][k]:.2f} | {avg['pdebench_compat'][k]:.2e} | "
                     f"{avg['ratio_to_paper'][k]:.2f} |")
    sweep = {}
    for p in _cases(rep, "offset"):
        sweep.setdefault(round(float(np.hypot(p["cx"], p["cy"])), 4), []).append(p["nRMSE"])
    sweep_txt = ", ".join(f"{m:g}: {np.mean(v):.2e}" for m, v in sorted(sweep.items()))
    lines += [
        "",
        "## Offset sweep",
        "",
        "Mean FNO nRMSE over 3 radii x 2 directions, by distance of the dam centre from the "
        f"origin: {sweep_txt}. The sub-cell family stays below {d['half_cell'] * 2**0.5:.4f}. "
        "The error tracks the distance to the nearest training trajectory, which is what "
        "you would expect from a model that interpolates between the 81 training "
        "trajectories rather than one that has learned translation-invariant dynamics. The FNO "
        "also receives absolute coordinates as input, so it is not translation-equivariant by "
        "construction.",
        "",
    ]
    if sc:
        lines += [
            "## Side result: pdeview's solver on the held-out ICs",
            "",
            f"The custom dam breaks in the viewer use pdeview's own NumPy solver. On the "
            f"{sc['n_cases']} held-out ICs (asymmetric and fractional, a configuration it was "
            "not validated on before) it "
            + ("reproduces PyClaw's float32 output bit for bit."
               if sc["max_abs_difference"] == 0 else
               f"matches PyClaw to max |dh| = {sc['max_abs_difference']:.1e} "
               f"(mean nRMSE {sc['mean_nRMSE']:.1e})."),
            "",
        ]
    lines += [
        "## Figures",
        "",
        f"![Error vs radius](figures/{figs['radius']})",
        "",
        f"![Error vs time](figures/{figs['time']})",
        "",
        f"![Offset sweep](figures/{figs['offset']})",
        "",
        f"![Worst sub-cell case](figures/{figs['worst']})",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python scripts/fetch_data.py      # PDEBench subset + checkpoint",
        "uv run python scripts/heldout_eval.py    # PyClaw references (uv, Python 3.13) + "
        "evaluation",
        "```",
        "",
        f"References are cached in `data/heldout/references.h5`. Runtime here: "
        f"{rep['environment']['runtime_seconds']:.0f} s.",
        "",
    ]
    (REPORTS_DIR / "HELDOUT.md").write_text("\n".join(lines))


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if "--render-only" in sys.argv and JSON_FILE.exists():
        rep = json.loads(JSON_FILE.read_text())
    else:
        rep = run()
        JSON_FILE.write_text(json.dumps(rep, indent=1))
    figs = {
        "radius": fig_error_vs_radius(rep),
        "time": fig_error_vs_time(rep),
        "offset": fig_offset_sweep(rep),
        "worst": fig_snapshots(rep, rep["highlights"]["subcell_worst"], "worst sub-cell"),
    }
    write_markdown(rep, figs)
    for c in rep["criteria"]:
        status = ("PASS" if c["pass"] else "FAIL") if c["kind"] == "gate" else \
            ("OK" if c["pass"] else "ISSUE")
        print(f"[{status:5s}] {c['kind']:7s} {c['family']:8s} {c['name']}")
    print(f"\nGates: {rep['verdict']}  ->  {REPORTS_DIR / 'HELDOUT.md'}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
