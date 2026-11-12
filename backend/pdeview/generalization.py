"""Generalisation study: how does the pretrained FNO behave on dam breaks it has never seen?

PDEBench's test split cannot answer this (every test trajectory is a copy of a training
trajectory, see ``pdeview.data.audit``). We therefore generate new reference trajectories with
our own solver (``pdeview.solver.swe``), which first has to reproduce PyClaw's ground truth on
every downloaded sample. Then the FNO is rolled out from the solver's first 10 frames, exactly
as for dataset samples.

Families of initial conditions (one parameter varied at a time, everything else as in the
training distribution: centred dam, h_in = 2, h_out = 1):

* radius     - r from 0.1 to 1.2 (training range 0.3-0.7)
* centre     - the dam moved off the origin (r = 0.5); the FNO receives absolute coordinates
               as input features, so translation equivariance is not guaranteed
* height     - inner depth h_in from 1.25 to 3 (training value 2)
* two dams   - two interacting columns (no training analogue)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from pdeview.config import FNO_INITIAL_STEP
from pdeview.data.store import SampleStore
from pdeview.inference import make_grid, persistence, rollout
from pdeview.metrics import sample_stats
from pdeview.model.checkpoint import load_fno
from pdeview.solver.swe import cell_centres, dam_break_ic, simulate

T_OUT = np.linspace(0.0, 1.0, 101)
TRAIN_RADIUS = (0.3, 0.7)


@dataclass(frozen=True)
class DamParams:
    radius: float = 0.5
    cx: float = 0.0
    cy: float = 0.0
    h_in: float = 2.0
    h_out: float = 1.0
    second: tuple[float, float, float] | None = None  # (cx, cy, radius) of an optional 2nd dam

    def ood_reasons(self) -> list[str]:
        out = []
        if not TRAIN_RADIUS[0] <= self.radius <= TRAIN_RADIUS[1]:
            out.append(f"radius {self.radius:g} outside training range 0.3-0.7")
        if abs(self.cx) > 1e-9 or abs(self.cy) > 1e-9:
            out.append("dam not centred at the origin")
        if abs(self.h_in - 2.0) > 1e-9:
            out.append(f"inner depth {self.h_in:g} (training: 2)")
        if abs(self.h_out - 1.0) > 1e-9:
            out.append(f"outer depth {self.h_out:g} (training: 1)")
        if self.second is not None:
            out.append("two dams")
        return out


def initial_condition(p: DamParams, x: np.ndarray | None = None) -> np.ndarray:
    x = cell_centres() if x is None else x
    h0 = dam_break_ic(x, x, p.radius, (p.cx, p.cy), p.h_in, p.h_out)
    if p.second is not None:
        cx2, cy2, r2 = p.second
        h0 = np.maximum(h0, dam_break_ic(x, x, r2, (cx2, cy2), p.h_in, p.h_out))
    return h0


def reference(p: DamParams) -> np.ndarray:
    """Reference trajectory ``[101, 128, 128]`` (float32) from our PyClaw-equivalent solver."""
    x = cell_centres()
    dx = float(x[1] - x[0])
    h, _ = simulate(initial_condition(p, x), T_OUT, dx, dx)
    return h.astype(np.float32)


def validate_solver(store: SampleStore, log=print) -> dict:
    """Re-simulate every downloaded sample from its config and compare with PyClaw."""
    x = cell_centres()
    dx = float(x[1] - x[0])
    worst_abs, nrmses, seconds = 0.0, [], []
    for key in store.keys():
        gt = store.frames(key).astype(np.float64)
        t0 = time.perf_counter()
        h, _ = simulate(dam_break_ic(x, x, store.samples[key].dam_radius), T_OUT, dx, dx)
        seconds.append(time.perf_counter() - t0)
        err = h - gt
        worst_abs = max(worst_abs, float(np.abs(err).max()))
        nrmses.append(float((np.sqrt((err**2).mean(axis=(1, 2)))
                             / np.sqrt((gt**2).mean(axis=(1, 2))))[1:].mean()))
    log(f"solver vs PyClaw: {len(nrmses)} samples, max |dh| {worst_abs:.2e}")
    return {
        "n_samples": len(nrmses),
        "max_abs_error": worst_abs,
        "mean_nRMSE": float(np.mean(nrmses)),
        "max_nRMSE": float(np.max(nrmses)),
        "seconds_per_trajectory": float(np.mean(seconds)),
        "float32_spacing_at_h2": float(np.spacing(np.float32(2.0))),
    }


def evaluate(model, grid, p: DamParams, ref: np.ndarray | None = None) -> dict:
    ref = reference(p) if ref is None else ref
    k0 = FNO_INITIAL_STEP
    pred = rollout(model, ref[:k0], grid, ref.shape[0])
    s = sample_stats(pred[k0:], ref[k0:])
    b = sample_stats(persistence(ref[:k0], ref.shape[0])[k0:], ref[k0:])
    v_ref = ref.sum(axis=(1, 2), dtype=np.float64)
    v_pred = pred.sum(axis=(1, 2), dtype=np.float64)
    return {
        "params": asdict(p),
        "ood": p.ood_reasons(),
        "nRMSE": float(s.nrmse.mean()),
        "nRMSE_final": float(s.nrmse[-1]),
        "max_error": float(s.max_abs.max()),
        "persistence_nRMSE": float(b.nrmse.mean()),
        "skill": float(1 - s.nrmse.mean() / b.nrmse.mean()),
        "volume_drift_final": float(v_pred[-1] / v_ref[0] - 1),
        "nRMSE_t": s.nrmse.tolist(),
    }


FAMILIES: dict[str, list[DamParams]] = {
    "radius": [DamParams(radius=r) for r in
               (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2)],
    "centre": [DamParams(cx=c, cy=c2) for c, c2 in
               ((0.0, 0.0), (0.1, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
                (0.5, 0.5), (1.0, 1.0), (-1.0, -0.5))],
    "height": [DamParams(h_in=h) for h in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)],
    "two_dams": [DamParams(radius=0.4, cx=-0.8, cy=0.0, second=(0.8, 0.0, 0.4)),
                 DamParams(radius=0.35, cx=-0.6, cy=-0.6, second=(0.6, 0.6, 0.35))],
}


def run_study(log=print) -> dict:
    store = SampleStore()
    model, _ = load_fno()
    grid = make_grid(store.x, store.y)
    validation = validate_solver(store, log=log)
    results = {}
    for fam, plist in FAMILIES.items():
        results[fam] = []
        for p in plist:
            r = evaluate(model, grid, p)
            results[fam].append(r)
            log(f"  {fam:8s} {p}  nRMSE {r['nRMSE']:.2e}"
                f"  (persistence {r['persistence_nRMSE']:.2e})")
    in_dist = results["radius"][[p.radius for p in FAMILIES["radius"]].index(0.5)]["nRMSE"]
    return {"solver_validation": validation, "in_distribution_nRMSE": in_dist,
            "families": results}
