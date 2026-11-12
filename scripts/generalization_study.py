"""Run the FNO generalisation study and write reports/GENERALIZATION.md (+ JSON, figures).

Usage:  uv run python scripts/generalization_study.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from pdeview.config import FNO_INITIAL_STEP, REPORTS_DIR  # noqa: E402
from pdeview.generalization import FAMILIES, DamParams, reference, run_study  # noqa: E402
from pdeview.inference import make_grid, rollout  # noqa: E402
from pdeview.model.checkpoint import load_fno  # noqa: E402
from pdeview.solver.swe import cell_centres  # noqa: E402

SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
GT_C, FNO_C, PERSIST_C = "#2a78d6", "#eb6834", "#1baf7a"
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
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, color=GRIDC, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def fig_families(rep: dict) -> str:
    fam = rep["families"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE, sharey=True)
    panels = [
        ("radius", "dam radius r", lambda p: p["radius"], (0.3, 0.7)),
        ("centre", "distance of dam centre from origin",
         lambda p: float(np.hypot(p["cx"], p["cy"])), (0, 0)),
        ("height", "inner depth h_in", lambda p: p["h_in"], (2.0, 2.0)),
    ]
    for ax, (name, xlabel, xval, train) in zip(axes, panels, strict=True):
        _style(ax)
        pts = sorted(fam[name], key=lambda r: xval(r["params"]))
        xs = [xval(r["params"]) for r in pts]
        ax.plot(xs, [r["persistence_nRMSE"] for r in pts], color=PERSIST_C, lw=2, marker="o", ms=4,
                label="Persistence baseline")
        ax.plot(xs, [r["nRMSE"] for r in pts], color=FNO_C, lw=2, marker="o", ms=4, label="FNO")
        if train[0] != train[1]:
            ax.axvspan(*train, color="#2a78d6", alpha=0.08, lw=0)
            ax.text(np.mean(train), 1.3e-3, "training range", ha="center", fontsize=8, color=MUTED)
        else:
            ax.axvline(train[0], color=MUTED, lw=1, ls=":")
            ax.text(train[0], 1.3e-3, " training value", fontsize=8, color=MUTED)
        ax.axhline(rep["in_distribution_nRMSE"], color=MUTED, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("time-averaged nRMSE vs. reference")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Pretrained FNO on unseen dam breaks (one parameter varied at a time; dashed: "
                 "in-distribution error)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "generalization_families.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def fig_case(p: DamParams, tag: str, title: str) -> str:
    model, _ = load_fno()
    x = cell_centres()
    ref = reference(p)
    pred = rollout(model, ref[:FNO_INITIAL_STEP], make_grid(x, x), ref.shape[0])
    times = [10, 40, 70, 100]
    ext = [x[0], x[-1], x[0], x[-1]]
    lo, hi = float(ref.min()), float(ref.max())
    elim = float(np.abs(pred[10:] - ref[10:]).max())
    fig, axes = plt.subplots(3, len(times), figsize=(2.6 * len(times) + 1, 8), facecolor=SURFACE)
    for c, ti in enumerate(times):
        rows = ((ref[ti].T, SEQ, lo, hi), (pred[ti].T, SEQ, lo, hi),
                ((pred[ti] - ref[ti]).T, DIV, -elim, elim))
        for r, (field, cmap, a, b) in enumerate(rows):
            ax = axes[r, c]
            im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap, vmin=a, vmax=b)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {T_OUT_STR[ti]}", color=INK2, fontsize=10)
            if c == 0:
                ax.set_ylabel(("Reference solver h", "FNO prediction h", "FNO - reference")[r],
                              color=INK2, fontsize=10)
            if c == len(times) - 1:
                fig.colorbar(im, ax=axes[r, :], shrink=0.85, pad=0.02).ax.tick_params(
                    labelsize=8, colors=MUTED)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    path = FIG_DIR / f"generalization_{tag}.png"
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path.name


T_OUT_STR = [f"{t:.2f}" for t in np.linspace(0, 1, 101)]


def _row(r: dict, label: str, in_dist: float) -> str:
    ratio = r["nRMSE"] / in_dist
    return (f"| {label} | {r['nRMSE']:.2e} | {ratio:.1f}x | {r['nRMSE_final']:.2e} | "
            f"{r['persistence_nRMSE']:.2e} | {r['skill']:.2f} |"
            f" {100 * r['volume_drift_final']:+.2f} % |")


def write_markdown(rep: dict, figs: dict[str, str]) -> None:
    in_dist = rep["in_distribution_nRMSE"]
    v = rep["solver_validation"]
    fam = rep["families"]
    head = ("| Initial condition | FNO nRMSE | vs. in-distribution | nRMSE at t = 1 |"
            " Persistence nRMSE | Skill vs. persistence | Volume drift at t = 1 |\n"
            "|---|---|---|---|---|---|---|")
    lines = [
        "# Does the pretrained FNO generalise? (2D shallow water, radial dam break)",
        "",
        "PDEBench's test split cannot answer this: all 100 test samples are bit-identical copies of"
        " training samples (see [VERIFICATION.md](VERIFICATION.md)). We therefore generate *new*"
        " dam breaks with our own solver and roll the FNO out from the solver's first 10 frames,"
        " exactly as on dataset samples. Generated by `scripts/generalization_study.py`.",
        "",
        "## 1. The reference solver reproduces PyClaw",
        "",
        "`backend/pdeview/solver/swe.py` re-implements the scheme PDEBench used (wave-propagation"
        " with Roe solver + Harten-Hyman entropy fix, MC limiter, Godunov dimensional splitting,"
        " PyClaw's CFL control, outflow boundaries). Re-simulating every downloaded sample from its"
        " generator config:",
        "",
        f"* **{v['n_samples']} / {v['n_samples']} samples reproduced**; max |h - h_PyClaw| ="
        f" **{v['max_abs_error']:.1e}** (the float32 spacing at h = 2 is"
        f" {v['float32_spacing_at_h2']:.1e}, i.e. agreement to the precision of the stored data);"
        f" mean nRMSE {v['mean_nRMSE']:.1e}.",
        f"* {v['seconds_per_trajectory']:.2f} s per 101-frame trajectory on one CPU core (NumPy).",
        "",
        "So any FNO error measured against this solver is FNO error, not solver discrepancy.",
        "",
        f"## 2. Results (in-distribution reference: nRMSE {in_dist:.2e} for a centred r = 0.5"
        " dam)",
        "",
        f"![Generalisation families](figures/{figs['families']})",
        "",
        "Skill = 1 - nRMSE_FNO / nRMSE_persistence (1 = perfect, 0 = no better than holding the"
        " last input frame, < 0 = worse).",
        "",
    ]
    titles = {"radius": "### Dam radius (training range 0.3-0.7)",
              "centre": "### Dam position (training: always centred at the origin; r = 0.5)",
              "height": "### Inner depth h_in (training: always 2)",
              "two_dams": "### Two interacting dams (no training analogue)"}
    for name, rows in fam.items():
        lines += [titles[name], "", head]
        for r in rows:
            p = r["params"]
            if name == "radius":
                label = f"r = {p['radius']:g}" + ("" if 0.3 <= p["radius"] <= 0.7 else " (OOD)")
            elif name == "centre":
                label = f"centre ({p['cx']:g}, {p['cy']:g})"
            elif name == "height":
                label = f"h_in = {p['h_in']:g}"
            else:
                c2 = p["second"]
                label = (f"r = {p['radius']:g} at ({p['cx']:g}, {p['cy']:g}) + r = {c2[2]:g} at"
                         f" ({c2[0]:g}, {c2[1]:g})")
            lines.append(_row(r, label, in_dist))
        lines.append("")
    lines += [
        "## Example rollouts",
        "",
        f"![Off-centre dam](figures/{figs['offcentre']})",
        "",
        f"![Large radius](figures/{figs['large']})",
        "",
        f"![Two dams](figures/{figs['two']})",
        "",
    ]
    (REPORTS_DIR / "GENERALIZATION.md").write_text("\n".join(lines))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rep = run_study()
    for fam in rep["families"].values():
        for r in fam:
            r.pop("nRMSE_t", None)
    (REPORTS_DIR / "generalization.json").write_text(json.dumps(rep, indent=1))
    figs = {
        "families": fig_families(rep),
        "offcentre": fig_case(DamParams(cx=1.0, cy=0.0), "offcentre",
                              "Off-centre dam: r = 0.5 at (1, 0)"),
        "large": fig_case(DamParams(radius=1.0), "large_radius",
                          "Large dam: r = 1.0, centred (training range 0.3-0.7)"),
        "two": fig_case(FAMILIES["two_dams"][0], "two_dams",
                        "Two dams: r = 0.4 at (-0.8, 0) and (0.8, 0)"),
    }
    write_markdown(rep, figs)
    print(f"-> {REPORTS_DIR / 'GENERALIZATION.md'}")


if __name__ == "__main__":
    main()
