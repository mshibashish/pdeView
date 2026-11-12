"""Viewer service: owns the data store, the model and a cache of FNO rollouts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import asdict

import numpy as np
import torch

from pdeview.config import FNO_INITIAL_STEP, REPORTS_DIR
from pdeview.data.store import SampleStore
from pdeview.generalization import DamParams, reference
from pdeview.heldout import (
    HELDOUT_FAMILIES,
    REF_FILE,
    Case,
    ReferenceStore,
    initial_condition,
    solve_reference,
)
from pdeview.inference import make_grid, persistence, rollout
from pdeview.metrics import sample_stats
from pdeview.model.checkpoint import load_fno

log = logging.getLogger(__name__)


def _load_json(name: str) -> dict | None:
    path = REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


class ViewerService:
    def __init__(self, device: str | None = None, cache_size: int = 32):
        self.device = device or os.environ.get("PDEVIEW_DEVICE", "cpu")
        self.store = SampleStore()
        self.model, self.ckpt_info = load_fno(device=self.device)
        self.grid = make_grid(self.store.x, self.store.y)
        self.verification = _load_json("verification.json")
        self.audit = _load_json("dataset_audit.json")
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._cache_size = cache_size
        per_sample = (self.verification or {}).get("per_sample", [])
        self._sample_metrics = {p["key"]: p for p in per_sample}
        self._custom: OrderedDict[str, dict] = OrderedDict()
        self._custom_size = 8
        # held-out in-distribution cases (reports/heldout.json); references from PyClaw when
        # data/heldout/references.h5 exists, else re-solved with our solver on demand
        self.heldout = _load_json("heldout.json")
        self._heldout = {p["key"]: p for p in (self.heldout or {}).get("per_case", [])
                         if p["family"] in HELDOUT_FAMILIES}
        self._heldout_refs = ReferenceStore(REF_FILE) if self._heldout and REF_FILE.exists() \
            else None
        self._heldout_lock = threading.Lock()
        self._heldout_frames: OrderedDict[str, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ data access
    def has_key(self, key: str) -> bool:
        if key in self.store.samples or key in self._heldout:
            return True
        with self._lock:
            return key in self._custom

    def _heldout_ground_truth(self, key: str) -> np.ndarray:
        with self._heldout_lock:
            if key in self._heldout_frames:
                self._heldout_frames.move_to_end(key)
                return self._heldout_frames[key]
            if self._heldout_refs is not None and key in self._heldout_refs:
                frames = self._heldout_refs.frames(key)
            else:
                p = self._heldout[key]
                case = Case(key, p["family"], p["radius"], p["cx"], p["cy"], p["ic"])
                frames = solve_reference(initial_condition(case))
            frames.setflags(write=False)
            self._heldout_frames[key] = frames
            while len(self._heldout_frames) > 8:
                self._heldout_frames.popitem(last=False)
            return frames

    def ground_truth(self, key: str) -> np.ndarray:
        """Reference trajectory: PyClaw data for dataset samples, our solver for custom runs."""
        if key in self.store.samples:
            return self.store.frames(key)
        if key in self._heldout:
            return self._heldout_ground_truth(key)
        with self._lock:
            run = self._custom.get(key)
        if run is None:
            raise KeyError(key)
        return run["frames"]

    def simulate(self, params: DamParams) -> dict:
        """Run the reference solver for a custom dam break (cached by parameters)."""
        blob = json.dumps(asdict(params), sort_keys=True).encode()
        key = "c-" + hashlib.sha1(blob).hexdigest()[:10]
        with self._lock:
            if key in self._custom:
                self._custom.move_to_end(key)
                return self._custom[key]["info"]
        t0 = time.perf_counter()
        frames = reference(params)
        frames.setflags(write=False)
        info = {"key": key, "params": asdict(params), "ood": params.ood_reasons(),
                "solver_seconds": round(time.perf_counter() - t0, 3)}
        with self._lock:
            self._custom[key] = {"frames": frames, "info": info}
            while len(self._custom) > self._custom_size:
                old, _ = self._custom.popitem(last=False)
                self._cache.pop(old, None)
        t1 = time.perf_counter()
        self.prediction(key)
        info["rollout_seconds"] = round(time.perf_counter() - t1, 3)
        return info

    # ------------------------------------------------------------------ rollouts
    def prediction(self, key: str) -> np.ndarray:
        """FNO rollout ``[T, X, Y]`` for a sample (computed once, then cached; thread-safe)."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            fut = self._pending.get(key)
            owner = fut is None
            if owner:
                fut = self._pending[key] = Future()
        if not owner:
            return fut.result()
        try:
            gt = self.ground_truth(key)
            pred = rollout(self.model, gt[:FNO_INITIAL_STEP], self.grid, gt.shape[0],
                           device=self.device)
            pred.setflags(write=False)
            with self._lock:
                self._cache[key] = pred
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            fut.set_result(pred)
            return pred
        except BaseException as err:
            fut.set_exception(err)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def warm_up(self, keys: list[str]) -> None:
        for key in keys:
            try:
                self.prediction(key)
            except Exception:  # pragma: no cover - warm-up is best effort
                log.exception("warm-up failed for %s", key)

    # ------------------------------------------------------------------ summaries
    def samples(self) -> list[dict]:
        twins = (self.audit or {}).get("twins", {})
        local = set(self.store.keys())
        out = []
        for key, info in self.store.samples.items():
            m = self._sample_metrics.get(key, {})
            out.append({
                "key": key,
                "split": info.split,
                "dam_radius": info.dam_radius,
                "nRMSE": m.get("nRMSE"),
                "RMSE": m.get("RMSE"),
                "max_error": m.get("max_error"),
                "ic_cells": m.get("ic_cells"),
                "n_twins_total": len(twins.get(key, [])),
                "twins_local": [t for t in twins.get(key, []) if t in local],
            })
        for key, p in self._heldout.items():
            out.append({
                "key": key,
                "split": "heldout",
                "family": p["family"],
                "dam_radius": p["radius"],
                "cx": p["cx"],
                "cy": p["cy"],
                "nRMSE": p["nRMSE"],
                "RMSE": p["RMSE"],
                "max_error": p["max_error"],
                "ic_cells": None,
                "n_twins_total": 0,
                "twins_local": [],
            })
        return out

    def heldout_summary(self) -> dict | None:
        """The held-out report without per-case rows and time curves (for the web UI)."""
        if self.heldout is None:
            return None
        fams = {name: {k: v for k, v in f.items() if k not in ("time_curves", "t")}
                for name, f in self.heldout["families"].items()}
        keep = ("generated_at", "verdict", "criteria", "reference_generator", "generator_check",
                "solver_check", "design", "paper", "highlights")
        return {k: self.heldout[k] for k in keep} | {"families": fams}

    def stats(self, key: str) -> dict:
        gt = self.ground_truth(key)
        pred = self.prediction(key)
        k0 = FNO_INITIAL_STEP
        s = sample_stats(pred[k0:], gt[k0:])
        base = sample_stats(persistence(gt[:k0], gt.shape[0])[k0:], gt[k0:])
        cell = self.store.dx * self.store.dy

        def pad(a: np.ndarray) -> list[float | None]:
            return [None] * k0 + [float(v) for v in a]

        return {
            "key": key,
            "t": self.store.t.tolist(),
            "initial_step": k0,
            "rmse": pad(s.rmse),
            "nrmse": pad(s.nrmse),
            "max_abs": pad(s.max_abs),
            "persistence_nrmse": pad(base.nrmse),
            "volume_gt": (gt.sum(axis=(1, 2), dtype=np.float64) * cell).tolist(),
            "volume_pred": (pred.sum(axis=(1, 2), dtype=np.float64) * cell).tolist(),
            "range": {"min": float(min(gt.min(), pred.min())),
                      "max": float(max(gt.max(), pred.max())),
                      "err_abs_max": float(np.abs(pred[k0:] - gt[k0:]).max())},
            "summary": {"nRMSE": float(s.nrmse.mean()), "RMSE": float(s.rmse.mean()),
                        "max_error": float(s.max_abs.max()),
                        "persistence_nRMSE": float(base.nrmse.mean())},
        }

    def meta(self) -> dict:
        v = self.verification or {}
        acc = v.get("accuracy", {})
        a = self.audit or {}
        return {
            "grid": {"x": self.store.x.tolist(), "y": self.store.y.tolist(),
                     "domain": [-2.5, 2.5, -2.5, 2.5]},
            "t": self.store.t.tolist(),
            "shape": [len(self.store.t), len(self.store.x), len(self.store.y)],
            "initial_step": FNO_INITIAL_STEP,
            "physics": {"equation": "2D shallow-water (radial dam break)", "gravity": 1.0,
                        "h_inside": 2.0, "h_outside": 1.0,
                        "radius_range": [0.3, 0.7], "boundary": "outflow (extrapolation)"},
            "model": {
                "name": "FNO2d (PDEBench pretrained, 2D_rdb_NA_NA_FNO.pt)",
                "device": self.device,
                "torch": torch.__version__,
                "n_parameters": self.ckpt_info.n_parameters,
                "epoch": self.ckpt_info.epoch,
                "md5_ok": self.ckpt_info.md5_ok,
                **self.ckpt_info.hyperparameters,
            },
            "verification": {
                "available": bool(v),
                "verdict": v.get("verdict"),
                "generated_at": v.get("generated_at"),
                "criteria": [{k: c[k] for k in ("name", "pass", "kind")}
                             for c in v.get("criteria", [])],
                "paper": acc.get("paper"),
                "pdebench_compat": acc.get("pdebench_compat"),
                "clean": acc.get("clean"),
                "persistence_over_fno_nRMSE": acc.get("persistence_over_fno_nRMSE"),
            },
            "dataset": {k: a.get(k) for k in (
                "n_samples", "n_distinct_trajectories", "n_possible_initial_conditions",
                "test_with_identical_train_trajectory", "n_test")},
        }
