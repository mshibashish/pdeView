"""Independent accuracy verification of the PDEBench pretrained shallow-water FNO.

The checks, in order of how strongly they constrain "is this the right model, used the right way":

1. **Integrity** - the checkpoint's MD5 matches the DaRUS metadata and its state dict loads
   with ``strict=True`` into an FNO2d built with PDEBench's published SWE hyper-parameters.
2. **Fingerprint** - PDEBench stores the best validation loss in the checkpoint
   (``"loss"``: the per-batch MSE over frames 10..100, summed over the 20 batches of the
   held-out split). Re-computing that number from our own rollouts pins down architecture,
   input convention, rollout scheme *and* split at once.
3. **Data sanity** - every downloaded sample is a valid dam break consistent with its
   generator config, on the documented grid, and the ground truth conserves volume.
4. **Accuracy** - PDEBench's metrics on the full test split vs. the paper (Table 5),
   aggregated exactly like PDEBench (like-for-like) and as clean population averages.
5. **Usefulness** - the FNO must beat a trivial persistence baseline by a wide margin.
6. **Split integrity** - a dataset audit (``pdeview.data.audit``) maps every sample to its
   discrete initial condition. All test trajectories turn out to be bit-identical copies of
   training trajectories, so the test split cannot measure generalisation.
7. **Numerics** - CPU vs. Apple-MPS agreement and inference timing.

Checks are either *gates* (must pass for the model to be used in the viewer: is this the
published model, run correctly, as accurate as published?) or *findings* (properties of the
checkpoint's provenance / the benchmark that users of the model must know about).
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import torch

from pdeview.config import FNO_INITIAL_STEP, PAPER_FNO_SWE
from pdeview.data.audit import load_or_run_audit
from pdeview.data.remote import check_sample
from pdeview.data.store import SampleStore
from pdeview.inference import make_grid, persistence, rollout
from pdeview.metrics import (
    METRIC_KEYS,
    SampleErrorStats,
    aggregate,
    aggregate_pdebench,
    sample_stats,
)
from pdeview.model.checkpoint import load_fno

PDEBENCH_BATCH_SIZE = 5  # config_rdb.yaml


def _volume(frames: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return frames.sum(axis=(-2, -1), dtype=np.float64) * dx * dy


def _criterion(name: str, value, threshold: str, ok: bool, detail: str = "",
               kind: str = "gate") -> dict:
    return {"name": name, "value": value, "threshold": threshold, "pass": bool(ok),
            "detail": detail, "kind": kind}


def _grid_ablation(model, store, keys, device) -> dict[str, float]:
    """Mean MSE of the rollout under alternative coordinate-grid conventions."""
    gt = np.stack([store.frames(k) for k in keys])
    nodes = np.linspace(-2.5, 2.5, len(store.x))
    variants = {
        "cell centres, ij (used)": make_grid(store.x, store.y),
        "cell nodes linspace(-2.5, 2.5)": make_grid(nodes, nodes),
        "x/y channels swapped": make_grid(store.x, store.y).flip(-1),
        "unit square [0, 1]": make_grid(np.linspace(0, 1, len(store.x)),
                                        np.linspace(0, 1, len(store.y))),
    }
    out = {}
    for name, g in variants.items():
        pred = rollout(model, gt[:, :FNO_INITIAL_STEP], g, gt.shape[1], device=device)
        out[name] = float(np.mean((pred[:, FNO_INITIAL_STEP:] - gt[:, FNO_INITIAL_STEP:]) ** 2))
    return out


def run_verification(batch: int = 10, log=print) -> dict:
    t_start = time.time()
    model, info = load_fno()
    store = SampleStore()
    grid = make_grid(store.x, store.y)
    k0 = FNO_INITIAL_STEP
    nt = len(store.t)

    # ---------------------------------------------------------------- data sanity
    data_issues: list[str] = []
    gt_volume_drift = []
    for key in store.keys():
        f = store.frames(key)
        try:
            check_sample(key, f[..., None], store.x, store.samples[key].dam_radius)
        except ValueError as err:
            data_issues.append(str(err))
        v = _volume(f, store.dx, store.dy)
        gt_volume_drift.append(float(np.max(np.abs(v / v[0] - 1))))
    dx_expected = 5.0 / 128
    grid_ok = (
        np.allclose(np.diff(store.x), dx_expected, atol=1e-6)
        and np.isclose(store.x[0], -2.5 + dx_expected / 2, atol=1e-6)
        and np.allclose(store.t, np.linspace(0, 1, nt), atol=1e-6)
    )
    log(f"data: {len(store.keys())} samples, {len(data_issues)} issues, grid ok={grid_ok}")

    # ---------------------------------------------------------------- split audit
    audit = load_or_run_audit()
    digests = {k: hashlib.sha1(store.frames(k).tobytes()).hexdigest() for k in store.keys()}
    sig = audit["ic_signature"]
    keys_local = store.keys()
    pairs = [(a, b) for i, a in enumerate(keys_local) for b in keys_local[i + 1:]]
    sig_predicts_identity = all((sig[a] == sig[b]) == (digests[a] == digests[b])
                                for a, b in pairs)
    n_identical_pairs = sum(digests[a] == digests[b] for a, b in pairs)
    log(f"audit: {audit['n_distinct_trajectories']} distinct trajectories; "
        f"{n_identical_pairs} bit-identical pairs in the local subset")

    # ---------------------------------------------------------------- rollouts
    results: dict[str, dict[str, list[SampleErrorStats]]] = {}
    per_sample = []
    finite = True
    rollout_seconds = 0.0
    for split in ("test", "train"):
        keys = store.keys(split)
        results[split] = {"fno": [], "persistence": []}
        for i in range(0, len(keys), batch):
            chunk = keys[i:i + batch]
            gt = np.stack([store.frames(k) for k in chunk])
            t0 = time.perf_counter()
            pred = rollout(model, gt[:, :k0], grid, n_steps=nt)
            rollout_seconds += time.perf_counter() - t0
            finite &= bool(np.isfinite(pred).all())
            base = persistence(gt[:, :k0], nt)
            for j, key in enumerate(chunk):
                s_fno = sample_stats(pred[j, k0:], gt[j, k0:])
                s_base = sample_stats(base[j, k0:], gt[j, k0:])
                results[split]["fno"].append(s_fno)
                results[split]["persistence"].append(s_base)
                v_gt = _volume(gt[j], store.dx, store.dy)
                v_pr = _volume(pred[j], store.dx, store.dy)
                per_sample.append({
                    "key": key,
                    "ic_cells": int(sig[key]),
                    "n_twins_total": len(audit["twins"][key]),
                    "split": split,
                    "dam_radius": store.samples[key].dam_radius,
                    "nRMSE": float(s_fno.nrmse.mean()),
                    "RMSE": float(s_fno.rmse.mean()),
                    "max_error": float(s_fno.max_abs.max()),
                    "persistence_nRMSE": float(s_base.nrmse.mean()),
                    "volume_drift_fno": float(np.max(np.abs(v_pr / v_gt[0] - 1))),
                })
            log(f"  {split}: {min(i + batch, len(keys))}/{len(keys)}")
    n_rolled = sum(len(r["fno"]) for r in results.values())

    test_fno = results["test"]["fno"]
    clean = aggregate(test_fno)
    compat = aggregate_pdebench(test_fno, batch_size=PDEBENCH_BATCH_SIZE)
    persist_clean = aggregate(results["test"]["persistence"])
    train_clean = aggregate(results["train"]["fno"])

    # ---------------------------------------------------------------- fingerprint
    # val_l2_full = sum over batches of MSE(pred[..., 10:101], target[..., 10:101]).
    mse = np.array([np.mean(s.rmse**2) for s in test_fno])
    n_batches = int(np.ceil(len(mse) / PDEBENCH_BATCH_SIZE))
    reproduced = float(sum(mse[b * PDEBENCH_BATCH_SIZE:(b + 1) * PDEBENCH_BATCH_SIZE].mean()
                           for b in range(n_batches)))
    stored = info.stored_val_loss
    fp_rel = abs(reproduced - stored) / stored if stored else float("nan")

    mse_t = np.stack([s.rmse**2 for s in test_fno])  # [N, T] per-frame MSE

    def batched(v):
        return float(sum(v[b * PDEBENCH_BATCH_SIZE:(b + 1) * PDEBENCH_BATCH_SIZE].mean()
                         for b in range(n_batches)))

    horizon_sweep = {int(k0 + h): batched(mse_t[:, :h].mean(axis=1))
                     for h in (10, 20, 40, 60, 80, mse_t.shape[1])}

    # deduplicated test metric: one sample per distinct trajectory
    seen, dedup = set(), []
    for key, st in zip(store.keys("test"), test_fno, strict=True):
        if sig[key] not in seen:
            seen.add(sig[key])
            dedup.append(st)
    dedup_clean = aggregate(dedup)

    # ---------------------------------------------------------------- devices
    device_report: dict = {"cpu_seconds_per_sample": rollout_seconds / n_rolled}
    key = store.keys("test")[0]
    gt = store.frames(key)
    t0 = time.perf_counter()
    cpu_pred = rollout(model, gt[:k0], grid, n_steps=nt)
    device_report["cpu_seconds_single_sample"] = time.perf_counter() - t0
    if torch.backends.mps.is_available():
        model_mps, _ = load_fno(device="mps")
        rollout(model_mps, gt[:k0], grid, n_steps=nt, device="mps")  # warm-up
        t0 = time.perf_counter()
        mps_pred = rollout(model_mps, gt[:k0], grid, n_steps=nt, device="mps")
        device_report["mps_seconds_single_sample"] = time.perf_counter() - t0
        device_report["mps_max_abs_diff_vs_cpu"] = float(np.abs(mps_pred - cpu_pred).max())
        device_report["mps_agrees"] = device_report["mps_max_abs_diff_vs_cpu"] < 1e-4
    else:
        device_report["mps_agrees"] = None
    abl_device = "mps" if device_report["mps_agrees"] else "cpu"
    grid_ablation = _grid_ablation(model_mps if abl_device == "mps" else model, store,
                                   store.keys("test")[:10], abl_device)

    # ---------------------------------------------------------------- criteria
    ratio = {k: compat[k] / PAPER_FNO_SWE[k] for k in METRIC_KEYS}
    beat = persist_clean["nRMSE"] / clean["nRMSE"]
    gen_ratio = train_clean["nRMSE"] / clean["nRMSE"]
    n_twin = audit["test_with_identical_train_trajectory"]
    criteria = [
        _criterion("Checkpoint MD5 matches DaRUS metadata", info.md5, info.md5_expected,
                   info.md5_ok),
        _criterion("State dict loads strictly into FNO2d(modes=12, width=20, initial_step=10)",
                   info.n_parameters, "all keys & shapes match", True),
        _criterion("Downloaded samples are valid dam breaks on the documented grid",
                   len(data_issues), "0 issues", not data_issues and grid_ok),
        _criterion("Ground truth conserves volume (waves stay inside the domain)",
                   max(gt_volume_drift), "< 1e-4 relative", max(gt_volume_drift) < 1e-4),
        _criterion("Predictions are finite", finite, "no NaN/Inf", finite),
        _criterion("nRMSE (PDEBench aggregation) within 2x of the paper",
                   compat["nRMSE"], f"{PAPER_FNO_SWE['nRMSE']:.1e} x [0.5, 2]",
                   0.5 <= ratio["nRMSE"] <= 2.0, f"ratio {ratio['nRMSE']:.2f}"),
        _criterion("All other metrics within an order of magnitude of the paper",
                   {k: round(v, 3) for k, v in ratio.items()}, "ratio in [0.1, 10]",
                   all(0.1 <= v <= 10 for v in ratio.values())),
        _criterion("Beats the persistence baseline by >= 10x in nRMSE", beat, ">= 10",
                   beat >= 10, f"persistence nRMSE {persist_clean['nRMSE']:.3e}"),
        _criterion("Reproduces the validation loss stored in the checkpoint",
                   reproduced, f"within 1 % of {stored:.6e}", fp_rel < 0.01,
                   f"{reproduced / stored:.2f}x the stored value", kind="finding"),
        _criterion("Test split is disjoint from the training split",
                   f"{n_twin}/{audit['n_test']} test samples have a bit-identical training "
                   "trajectory", "0 shared trajectories", n_twin == 0,
                   f"{audit['n_samples']} samples contain only "
                   f"{audit['n_distinct_trajectories']} distinct trajectories", kind="finding"),
    ]
    verdict = "PASS" if all(c["pass"] for c in criteria if c["kind"] == "gate") else "FAIL"

    def curve(stats, attr):
        return np.stack([getattr(s, attr) for s in stats]).mean(axis=0).tolist()

    test_ps = [p for p in per_sample if p["split"] == "test"]
    order = sorted(test_ps, key=lambda p: p["nRMSE"])
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verdict": verdict,
        "criteria": criteria,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "machine": f"{platform.system()} {platform.machine()}",
            "runtime_seconds": round(time.time() - t_start, 1),
        },
        "checkpoint": asdict(info),
        "fingerprint": {"stored_val_loss": stored, "reproduced_val_loss": reproduced,
                        "relative_difference": fp_rel, "n_batches": n_batches,
                        "batch_size": PDEBENCH_BATCH_SIZE,
                        "horizon_sweep": horizon_sweep, "grid_ablation_mse": grid_ablation},
        "audit": {k: v for k, v in audit.items() if k not in ("radius", "ic_signature", "twins")}
        | {"signature_predicts_bit_identity_in_subset": sig_predicts_identity,
           "identical_pairs_in_subset": n_identical_pairs},
        "data": {
            "n_test": len(store.keys("test")),
            "n_train_probe": len(store.keys("train")),
            "issues": data_issues,
            "grid_ok": bool(grid_ok),
            "max_gt_volume_drift": max(gt_volume_drift),
            "dx": store.dx,
            "dt": float(store.t[1] - store.t[0]),
        },
        "accuracy": {
            "paper": PAPER_FNO_SWE,
            "pdebench_compat": compat,
            "clean": clean,
            "ratio_to_paper": ratio,
            "persistence_clean": persist_clean,
            "train_probe_clean": train_clean,
            "persistence_over_fno_nRMSE": beat,
            "train_over_test_nRMSE": gen_ratio,
            "dedup_test_clean": dedup_clean,
            "n_dedup_test": len(dedup),
        },
        "time_curves": {
            "t": store.t[k0:].tolist(),
            "fno_test_nRMSE": curve(test_fno, "nrmse"),
            "fno_train_nRMSE": curve(results["train"]["fno"], "nrmse"),
            "persistence_test_nRMSE": curve(results["test"]["persistence"], "nrmse"),
            "fno_test_RMSE": curve(test_fno, "rmse"),
        },
        "highlights": {
            "best": order[0]["key"],
            "median": order[len(order) // 2]["key"],
            "worst": order[-1]["key"],
        },
        "device": device_report,
        "per_sample": per_sample,
    }
