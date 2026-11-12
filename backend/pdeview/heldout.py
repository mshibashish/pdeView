"""Held-out, in-distribution test set for the pretrained FNO.

**Why PDEBench's own split cannot provide one.** A training sample is fully determined by
its discretised initial condition (IC): h = 2 in every cell whose centre lies within the dam
radius r, else 1. As r sweeps PDEBench's range U(0.3, 0.7), the disc crosses only 80
cell-centre distance shells, so the whole data distribution contains just **81** distinct
ICs. That makes 81 distinct trajectories, since PyClaw is deterministic. The 900 training
samples contain all 81 of them (``reports/dataset_audit.json``). Any new sample drawn from
PDEBench's generator is therefore a bit-identical copy of a training trajectory. No
amount of re-sampling yields unseen data.

**What we use instead.** PDEBench's *physical* specification is a circular dam of radius
r ~ U(0.3, 0.7) centred at the origin, h_in = 2, h_out = 1, g = 1. We discretise it in two
ways that never coincide with one of the 81 training ICs:

* ``subcell``: the dam centre is drawn uniformly from the half-cell square around the
  origin, i.e. the set of points that are closer to the origin (a grid vertex) than to any
  other vertex, which a 128 x 128 grid cannot tell apart from the origin. The IC is still
  sharp, with the same values {1, 2} as the training data. Only the cut through the cells
  differs.
* ``cellavg``: the dam is exactly centred, but each cell holds its *average* depth, so
  cells cut by the dam wall hold the fraction of water inside the wall. This is the
  standard finite-volume discretisation of the same disc, and it varies continuously
  with r.

The reference trajectories come from PyClaw, configured exactly as PDEBench was
(``scripts/pyclaw_reference.py``). That configuration first has to reproduce the downloaded
PDEBench samples bit for bit. As controls we also evaluate:

* ``seen``: one IC per distance shell, i.e. all 81 training ICs, each weighted by its
  probability under r ~ U(0.3, 0.7). This gives the exact expected error over PDEBench's
  distribution, which is also the training error.
* ``offset``: a sweep of the centre offset from 0 to 0.1, to show how the error grows as a
  sample moves away from the training ICs. 0.1 is where the generalisation study starts.

Two baselines put the errors in context: persistence (repeat frame 9), and a lookup of the
nearest of the 81 training trajectories, matched on the 10 input frames the FNO also sees.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import h5py
import numpy as np
import torch

from pdeview.config import DATA_DIR, FNO_INITIAL_STEP, PAPER_FNO_SWE, ROOT
from pdeview.data.store import SampleStore
from pdeview.inference import make_grid, persistence, rollout
from pdeview.metrics import METRIC_KEYS, aggregate, aggregate_pdebench, sample_stats
from pdeview.model.checkpoint import load_fno
from pdeview.solver.swe import cell_centres, dam_break_ic, simulate

log = logging.getLogger(__name__)

TRAIN_RADIUS = (0.3, 0.7)
N_CELLS, LO, HI = 128, -2.5, 2.5
DX = (HI - LO) / N_CELLS
HALF_CELL = DX / 2
T_OUT = np.linspace(0.0, 1.0, 101)
N_PER_FAMILY = 100
SEED = 20261112
SUPERSAMPLE = 16  # sub-cell samples per axis for the cell-average IC

HELDOUT_DIR = DATA_DIR / "heldout"
IC_FILE = HELDOUT_DIR / "initial_conditions.npz"
REF_FILE = HELDOUT_DIR / "references.h5"
PYCLAW_SCRIPT = ROOT / "scripts" / "pyclaw_reference.py"

FAMILY_LABELS = {
    "subcell": "Sub-cell dam position",
    "cellavg": "Cell-averaged dam wall",
    "seen": "All 81 training ICs",
    "offset": "Centre-offset sweep",
    "dataset": "PDEBench samples (generator check)",
}
HELDOUT_FAMILIES = ("subcell", "cellavg")


@dataclass(frozen=True)
class Case:
    key: str
    family: str
    radius: float
    cx: float = 0.0
    cy: float = 0.0
    ic: str = "point"  # "point" (PDEBench: cell centre inside the dam) or "cell_average"
    weight: float = 1.0  # probability of a ``seen`` IC under r ~ U(0.3, 0.7)


# ---------------------------------------------------------------------------- cases
def shell_edges(x: np.ndarray | None = None, lo: float = TRAIN_RADIUS[0],
                hi: float = TRAIN_RADIUS[1]) -> np.ndarray:
    """Radii at which the discretised IC changes, bracketed by the range ends. Consecutive
    edges delimit the radius interval of one distinct IC."""
    x = cell_centres() if x is None else x
    xx, yy = np.meshgrid(x, x, indexing="ij")
    d = np.unique(np.sqrt(xx**2 + yy**2))
    return np.concatenate([[lo], d[(d > lo) & (d < hi)], [hi]])


def seen_cases() -> list[Case]:
    e = shell_edges()
    width = np.diff(e)
    return [Case(f"seen-{i:02d}", "seen", float((e[i] + e[i + 1]) / 2),
                 weight=float(width[i] / (e[-1] - e[0]))) for i in range(len(width))]


def heldout_cases(n: int = N_PER_FAMILY, seed: int = SEED) -> tuple[list[Case], int]:
    """``n`` sub-cell and ``n`` cell-average cases whose ICs differ from every training IC and
    from each other. A small centre offset that moves no cell centre across the dam wall
    reproduces a training IC; such draws are rejected and redrawn. Returns the cases and the
    number of rejected draws."""
    rng = np.random.default_rng(seed)
    x = cell_centres()
    taken = {_sha1(initial_condition(c, x)) for c in seen_cases()}
    out, rejected = [], 0
    for fam, prefix in (("subcell", "h-sub"), ("cellavg", "h-avg")):
        made = 0
        while made < n:
            r = float(rng.uniform(*TRAIN_RADIUS))
            if fam == "subcell":
                cx, cy = (float(v) for v in rng.uniform(-HALF_CELL, HALF_CELL, 2))
                c = Case(f"{prefix}-{made:03d}", fam, r, cx, cy)
            else:
                c = Case(f"{prefix}-{made:03d}", fam, r, ic="cell_average")
            sig = _sha1(initial_condition(c, x))
            if sig in taken:
                rejected += 1
                continue
            taken.add(sig)
            out.append(c)
            made += 1
    return out, rejected


OFFSET_RADII = (0.35, 0.5, 0.65)
OFFSET_MAGNITUDES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1)


def offset_cases() -> list[Case]:
    out = []
    for r in OFFSET_RADII:
        for name, (ux, uy) in (("axis", (1.0, 0.0)), ("diag", (2**-0.5, 2**-0.5))):
            for m in OFFSET_MAGNITUDES:
                out.append(Case(f"off-{name}-{r:.2f}-{m:.3f}", "offset", r, m * ux, m * uy))
    return out


def initial_condition(c: Case, x: np.ndarray | None = None) -> np.ndarray:
    """Initial depth ``[128, 128]`` (float64) of a case."""
    x = cell_centres() if x is None else x
    if c.ic == "point":
        return dam_break_ic(x, x, c.radius, (c.cx, c.cy))
    if c.ic == "cell_average":
        dx = float(x[1] - x[0])
        sub = ((np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5) * dx
        xx, yy = np.meshgrid(x, x, indexing="ij")
        frac = np.zeros_like(xx)
        for ox in sub:
            for oy in sub:
                frac += np.hypot(xx + ox - c.cx, yy + oy - c.cy) <= c.radius
        return 1.0 + frac / SUPERSAMPLE**2
    raise ValueError(f"unknown IC type {c.ic!r}")


def _sha1(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------- references
def solve_reference(h0: np.ndarray) -> np.ndarray:
    """Trajectory ``[101, 128, 128]`` (float32) from our NumPy solver."""
    return simulate(h0, T_OUT, DX, DX)[0].astype(np.float32)


def solver_references(h0s: list[np.ndarray], workers: int = 8) -> list[np.ndarray]:
    """Our NumPy solver (``pdeview.solver.swe``), in parallel processes."""
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(solve_reference, h0s, chunksize=4))


def make_references(keys: list[str], h0s: list[np.ndarray], log=print) -> str:
    """Make sure ``REF_FILE`` holds a reference trajectory for every key whose IC matches.

    Uses PyClaw (``scripts/pyclaw_reference.py``, run by uv in its own Python 3.13
    environment). If PyClaw cannot run, falls back to our solver and says so in the result.
    Returns a description of the generator that produced the file."""
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    want = {k: _sha1(h) for k, h in zip(keys, h0s, strict=True)}
    if REF_FILE.exists():  # drop references whose IC changed since they were made
        with h5py.File(REF_FILE, "a") as f:
            stale = [k for k in f if k in want and f[k].attrs.get("h0_sha1") != want[k]]
            for k in stale:
                del f[k]
            if stale:
                log(f"removed {len(stale)} stale reference(s)")
    np.savez(IC_FILE, keys=np.array(keys), h0=np.stack(h0s))
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "run", "-q", "--python", "3.13", str(PYCLAW_SCRIPT), str(IC_FILE),
               str(REF_FILE)]
        log("running PyClaw: " + " ".join(cmd[1:]))
        proc = subprocess.run(cmd, cwd=HELDOUT_DIR, text=True)  # pyclaw.log lands here
        if proc.returncode == 0:
            with h5py.File(REF_FILE, "r") as f:
                return str(f.attrs["generator"])
        log(f"PyClaw failed (exit {proc.returncode}); falling back to pdeview's solver")
    else:
        log("uv not found; falling back to pdeview's solver")
    with h5py.File(REF_FILE, "a") as f:
        todo = [i for i, k in enumerate(keys) if k not in f]
        for i, h in zip(todo, solver_references([h0s[i] for i in todo]), strict=True):
            ds = f.create_dataset(keys[i], data=h, compression="gzip", compression_opts=4,
                                  chunks=(1, N_CELLS, N_CELLS))
            ds.attrs["h0_sha1"] = want[keys[i]]
        f.attrs["generator"] = "pdeview.solver.swe (PyClaw unavailable)"
    return "pdeview.solver.swe (PyClaw unavailable)"


class ReferenceStore:
    """Read-only access to ``REF_FILE``."""

    def __init__(self, path=REF_FILE):
        self._h5 = h5py.File(path, "r")
        self.generator = str(self._h5.attrs.get("generator", "unknown"))

    def __contains__(self, key: str) -> bool:
        return key in self._h5

    def frames(self, key: str) -> np.ndarray:
        return self._h5[key][()]

    def close(self) -> None:
        self._h5.close()


# ---------------------------------------------------------------------------- evaluation
class TrainingSet:
    """The 81 distinct training trajectories, for nearest-neighbour baselines."""

    def __init__(self, cases: list[Case], refs: ReferenceStore):
        self.cases = cases
        self._refs = refs
        # h - 1 (the still-water excess) keeps the float32 dot products well conditioned
        self._dev = np.stack([refs.frames(c.key) for c in cases]) - np.float32(1.0)
        self.ics = {_sha1(initial_condition(c)) for c in cases}
        self.inp = self._dev[:, :FNO_INITIAL_STEP].reshape(len(cases), -1)
        self.out = self._dev[:, FNO_INITIAL_STEP:].reshape(len(cases), -1)
        self._inp_sq = np.einsum("ij,ij->i", self.inp, self.inp, dtype=np.float64)
        self._out_sq = np.einsum("ij,ij->i", self.out, self.out, dtype=np.float64)

    def nearest(self, ref: np.ndarray, on: str) -> int:
        """Index of the training trajectory closest to ``ref`` in L2, on the input frames
        (``on="input"``, a baseline that sees what the FNO sees) or on the predicted frames
        (``on="output"``, the distance of ``ref`` from the training set)."""
        k0 = FNO_INITIAL_STEP
        v = (ref[:k0] if on == "input" else ref[k0:]).reshape(-1) - 1.0
        mat, sq = (self.inp, self._inp_sq) if on == "input" else (self.out, self._out_sq)
        d = sq - 2.0 * (mat @ v).astype(np.float64)
        return int(np.argmin(d))

    def trajectory(self, j: int) -> np.ndarray:
        return self._refs.frames(self.cases[j].key)  # exact (h - 1 + 1 can round)


def _evaluate(model, grid, cases: list[Case], refs: ReferenceStore, train: TrainingSet,
              batch: int = 10) -> tuple[list[dict], dict]:
    k0 = FNO_INITIAL_STEP
    per_case, stats = [], {"fno": [], "persistence": [], "lookup": []}
    for i in range(0, len(cases), batch):
        chunk = cases[i:i + batch]
        ref = np.stack([refs.frames(c.key) for c in chunk])
        pred = rollout(model, ref[:, :k0], grid, ref.shape[1])
        for c, g, p in zip(chunk, ref, pred, strict=True):
            s = sample_stats(p[k0:], g[k0:])
            b = sample_stats(persistence(g[:k0], g.shape[0])[k0:], g[k0:])
            j_in = train.nearest(g, "input")
            lk = sample_stats(train.trajectory(j_in)[k0:], g[k0:])
            j_out = train.nearest(g, "output")
            dist = sample_stats(train.trajectory(j_out)[k0:], g[k0:])
            stats["fno"].append(s)
            stats["persistence"].append(b)
            stats["lookup"].append(lk)
            per_case.append({
                **asdict(c),
                "ic_is_training_ic": _sha1(initial_condition(c)) in train.ics,
                "finite": bool(np.isfinite(p).all()),
                "nRMSE": float(s.nrmse.mean()),
                "RMSE": float(s.rmse.mean()),
                "max_error": float(s.max_abs.max()),
                "nRMSE_final": float(s.nrmse[-1]),
                "persistence_nRMSE": float(b.nrmse.mean()),
                "lookup_nRMSE": float(lk.nrmse.mean()),
                "distance_to_training_set": float(dist.nrmse.mean()),
                "nearest_training_ic": train.cases[j_out].key,
            })
    return per_case, stats


def _bootstrap_ci(values: np.ndarray, weights: np.ndarray | None = None, n: int = 10_000,
                  seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n, len(values)))
    if weights is None:
        means = values[idx].mean(axis=1)
    else:
        means = (values[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _criterion(family: str, name: str, value, threshold: str, ok: bool, detail: str = "",
               kind: str = "gate") -> dict:
    return {"family": family, "name": name, "value": value, "threshold": threshold,
            "pass": bool(ok), "detail": detail, "kind": kind}


def _summarise(per_case: list[dict], stats: dict, weights: np.ndarray | None = None) -> dict:
    nrmse = np.array([p["nRMSE"] for p in per_case])
    w = weights if weights is not None else np.ones_like(nrmse)
    mean = float((nrmse * w).sum() / w.sum())
    k0 = FNO_INITIAL_STEP

    def curve(key):
        a = np.stack([s.nrmse for s in stats[key]])
        return ((a * w[:, None]).sum(axis=0) / w.sum()).tolist()

    out = {
        "n": len(per_case),
        "nRMSE": mean,
        "nRMSE_ci95": _bootstrap_ci(nrmse, None if weights is None else w),
        "nRMSE_median": float(np.median(nrmse)),
        "nRMSE_max": float(nrmse.max()),
        "persistence_nRMSE": float((np.array([p["persistence_nRMSE"] for p in per_case]) * w)
                                   .sum() / w.sum()),
        "lookup_nRMSE": float((np.array([p["lookup_nRMSE"] for p in per_case]) * w).sum()
                              / w.sum()),
        "distance_to_training_set": float(np.mean([p["distance_to_training_set"]
                                                   for p in per_case])),
        "fno_beats_lookup_fraction": float(np.mean([p["nRMSE"] < p["lookup_nRMSE"]
                                                    for p in per_case])),
        "n_training_ic": int(sum(p["ic_is_training_ic"] for p in per_case)),
        "finite": all(p["finite"] for p in per_case),
        "clean": aggregate(stats["fno"]),
        "persistence_clean": aggregate(stats["persistence"]),
        "lookup_clean": aggregate(stats["lookup"]),
        "time_curves": {"fno_nRMSE": curve("fno"), "persistence_nRMSE": curve("persistence"),
                        "lookup_nRMSE": curve("lookup")},
        "t": T_OUT[k0:].tolist(),
    }
    if len(per_case) >= 10 and len(per_case) % 5 == 0:
        out["pdebench_compat"] = aggregate_pdebench(stats["fno"], batch_size=5)
    return out


def _check_generator(store: SampleStore, refs: ReferenceStore) -> dict:
    keys = [k for k in store.keys() if f"ds-{k}" in refs]
    identical, worst = 0, 0.0
    for k in keys:
        d = np.abs(refs.frames(f"ds-{k}") - store.frames(k))
        identical += int(not d.any())
        worst = max(worst, float(d.max()))
    return {"n_checked": len(keys), "n_bit_identical": identical, "max_abs_difference": worst}


def _check_solver(cases: list[Case], refs: ReferenceStore) -> dict:
    """Our solver vs PyClaw on held-out ICs it has never been validated on."""
    ours = solver_references([initial_condition(c) for c in cases])
    worst, nr = 0.0, []
    for c, h in zip(cases, ours, strict=True):
        g = refs.frames(c.key).astype(np.float64)
        err = h.astype(np.float64) - g
        worst = max(worst, float(np.abs(err).max()))
        nr.append(float((np.sqrt((err**2).mean(axis=(1, 2)))
                         / np.sqrt((g**2).mean(axis=(1, 2))))[1:].mean()))
    return {"n_cases": len(cases), "max_abs_difference": worst, "mean_nRMSE": float(np.mean(nr))}


def run(n: int = N_PER_FAMILY, seed: int = SEED, check_solver: bool = True,
        log=print) -> dict:
    t_start = time.time()
    store = SampleStore()
    x = cell_centres()
    seen, off = seen_cases(), offset_cases()
    held, n_rejected = heldout_cases(n, seed)
    dataset = [Case(f"ds-{k}", "dataset", store.samples[k].dam_radius) for k in store.keys()]
    cases = seen + held + off + dataset
    generator = make_references([c.key for c in cases], [initial_condition(c, x) for c in cases],
                                log=log)
    refs = ReferenceStore()
    log(f"references: {generator}")

    gen_check = _check_generator(store, refs)
    log(f"generator reproduces {gen_check['n_bit_identical']}/{gen_check['n_checked']} "
        f"PDEBench samples bit for bit (max |dh| {gen_check['max_abs_difference']:.1e})")

    model, info = load_fno()
    grid = make_grid(store.x, store.y)
    train = TrainingSet(seen, refs)

    families, per_case_all = {}, []
    for fam, subset in (("seen", seen), ("subcell", held[:n]), ("cellavg", held[n:]),
                        ("offset", off)):
        t0 = time.perf_counter()
        pc, st = _evaluate(model, grid, subset, refs, train)
        weights = np.array([c.weight for c in subset]) if fam == "seen" else None
        families[fam] = _summarise(pc, st, weights)
        per_case_all += pc
        log(f"  {fam:8s} n={len(pc):3d}  FNO nRMSE {families[fam]['nRMSE']:.3e}  "
            f"lookup {families[fam]['lookup_nRMSE']:.3e}  "
            f"persistence {families[fam]['persistence_nRMSE']:.3e}  "
            f"({time.perf_counter() - t0:.0f} s)")

    use_pyclaw = generator.startswith("clawpack")
    solver = _check_solver(held, refs) if check_solver and use_pyclaw else None
    if solver:
        log(f"pdeview solver vs PyClaw on held-out ICs: max |dh| "
            f"{solver['max_abs_difference']:.1e}")

    # ------------------------------------------------------------------ criteria
    seen_nrmse = families["seen"]["nRMSE"]
    criteria = [_criterion(
        "all", "Reference generator reproduces PDEBench's data bit for bit",
        f"{gen_check['n_bit_identical']}/{gen_check['n_checked']} samples identical",
        "all identical", gen_check["n_checked"] > 0
        and gen_check["n_bit_identical"] == gen_check["n_checked"],
        f"generator: {generator}")]
    for fam in HELDOUT_FAMILIES:
        f = families[fam]
        compat = f["pdebench_compat"]
        ratio = {k: compat[k] / PAPER_FNO_SWE[k] for k in METRIC_KEYS}
        beat = f["persistence_clean"]["nRMSE"] / f["clean"]["nRMSE"]
        gap = f["nRMSE"] / seen_nrmse
        f["ratio_to_paper"] = ratio
        f["persistence_over_fno_nRMSE"] = beat
        f["gap_to_seen"] = gap
        criteria += [
            _criterion(fam, "Initial conditions differ from all 81 training ICs",
                       f"{f['n_training_ic']}/{f['n']} coincide", "0 coincide",
                       f["n_training_ic"] == 0),
            _criterion(fam, "Predictions are finite", f["finite"], "no NaN/Inf", f["finite"]),
            _criterion(fam, "nRMSE (PDEBench aggregation) within 2x of the paper",
                       compat["nRMSE"], f"{PAPER_FNO_SWE['nRMSE']:.1e} x [0.5, 2]",
                       0.5 <= ratio["nRMSE"] <= 2.0, f"ratio {ratio['nRMSE']:.2f}"),
            _criterion(fam, "All other metrics within an order of magnitude of the paper",
                       {k: round(v, 3) for k, v in ratio.items()}, "ratio in [0.1, 10]",
                       all(0.1 <= v <= 10 for v in ratio.values())),
            _criterion(fam, "Beats the persistence baseline by >= 10x in nRMSE", beat, ">= 10",
                       beat >= 10, f"persistence nRMSE {f['persistence_clean']['nRMSE']:.3e}"),
            _criterion(fam, "Error on unseen ICs matches the error on training ICs",
                       gap, "<= 1.25x", gap <= 1.25,
                       f"{f['nRMSE']:.2e} vs {seen_nrmse:.2e} on the 81 training ICs",
                       kind="finding"),
            _criterion(fam, "Beats a lookup of the nearest training trajectory",
                       f["lookup_nRMSE"], f"> FNO nRMSE {f['nRMSE']:.2e}",
                       f["nRMSE"] < f["lookup_nRMSE"],
                       f"FNO better on {f['fno_beats_lookup_fraction']:.0%} of cases",
                       kind="finding"),
        ]
    verdict = "PASS" if all(c["pass"] for c in criteria if c["kind"] == "gate") else "FAIL"

    sub = sorted((p for p in per_case_all if p["family"] == "subcell"),
                 key=lambda p: p["nRMSE"])
    avg = sorted((p for p in per_case_all if p["family"] == "cellavg"),
                 key=lambda p: p["nRMSE"])
    refs.close()
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": verdict,
        "criteria": criteria,
        "reference_generator": generator,
        "generator_check": gen_check,
        "solver_check": solver,
        "design": {
            "seed": seed,
            "n_per_family": n,
            "n_rejected_draws": n_rejected,
            "radius_range": list(TRAIN_RADIUS),
            "half_cell": HALF_CELL,
            "supersample": SUPERSAMPLE,
            "n_training_ics": len(seen),
            "offset_radii": list(OFFSET_RADII),
            "offset_magnitudes": list(OFFSET_MAGNITUDES),
            "labels": FAMILY_LABELS,
        },
        "paper": PAPER_FNO_SWE,
        "families": families,
        "highlights": {
            "subcell_median": sub[len(sub) // 2]["key"], "subcell_worst": sub[-1]["key"],
            "cellavg_median": avg[len(avg) // 2]["key"], "cellavg_worst": avg[-1]["key"],
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "checkpoint_md5_ok": info.md5_ok,
                        "runtime_seconds": round(time.time() - t_start, 1)},
        "per_case": [p for p in per_case_all if p["family"] != "dataset"],
    }
