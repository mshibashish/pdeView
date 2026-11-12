"""Dataset audit: how many *distinct* trajectories does the PDEBench SWE dataset contain?

Every sample is a radial dam break centred at the origin whose only parameter is the dam
radius ``r``. The initial depth is set per cell (``h = 2`` where the cell centre lies within
``r``, else ``1``), so on the 128 x 128 grid all radii between two consecutive cell-centre
distances produce the *same* initial condition - and PyClaw is deterministic, so the same
trajectory, bit for bit. The radius is drawn from U(0.3, 0.7), which spans only a few dozen
such distance "shells". Consequently many samples are exact copies of each other, including
across the train / test split.

This module reads every sample's ``config`` attribute from the remote file (metadata only,
no field data), maps each radius to its discrete initial condition and summarises the
overlap between PDEBench's training and test splits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fsspec
import h5py
import numpy as np
import yaml

from pdeview.config import N_SAMPLES_TOTAL, REPORTS_DIR, SWE_DATA, TEST_RATIO
from pdeview.data.remote import USER_AGENT, resolve_url

log = logging.getLogger(__name__)

AUDIT_FILE = REPORTS_DIR / "dataset_audit.json"


def cell_centres(n: int = 128, lo: float = -2.5, hi: float = 2.5) -> np.ndarray:
    """PyClaw cell centres in float64 (the stored grid is float32)."""
    dx = (hi - lo) / n
    return lo + (np.arange(n) + 0.5) * dx


def ic_signature(radius: float, x: np.ndarray | None = None) -> int:
    """Number of cells inside the dam. The dams are nested discs, so this identifies the IC."""
    x = cell_centres() if x is None else x
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return int((np.sqrt(xx**2 + yy**2) <= radius).sum())


def read_all_radii(n_total: int = N_SAMPLES_TOTAL) -> dict[str, float]:
    url = resolve_url(SWE_DATA)
    fs = fsspec.filesystem("http", headers={"User-Agent": USER_AGENT})
    radii = {}
    with fs.open(url, mode="rb", block_size=2**16, cache_type="blockcache",
                 size=SWE_DATA.size) as fobj, h5py.File(fobj, "r") as h5:
        for i in range(n_total):
            key = f"{i:04d}"
            radii[key] = float(yaml.safe_load(h5[key].attrs["config"])["sim"]["dam_radius"])
            if i % 100 == 99:
                log.info("read %d/%d configs", i + 1, n_total)
    return radii


def audit(radii: dict[str, float]) -> dict:
    keys = sorted(radii)
    n_test = int(round(len(keys) * TEST_RATIO))
    train_keys, test_keys = keys[:-n_test], keys[-n_test:]
    x = cell_centres()
    sig = {k: ic_signature(radii[k], x) for k in keys}

    # all ICs reachable with r ~ U(0.3, 0.7): one per distance shell crossed
    xx, yy = np.meshgrid(x, x, indexing="ij")
    d = np.unique(np.sqrt(xx**2 + yy**2))
    shells = d[(d > 0.3) & (d < 0.7)]
    n_possible = len(shells) + 1

    groups: dict[int, list[str]] = {}
    for k in keys:
        groups.setdefault(sig[k], []).append(k)
    train_sigs = {sig[k] for k in train_keys}
    test_with_twin = [k for k in test_keys if sig[k] in train_sigs]
    copies = np.array([len(v) for v in groups.values()])
    return {
        "n_samples": len(keys),
        "n_possible_initial_conditions": n_possible,
        "n_distinct_trajectories": len(groups),
        "n_distinct_train": len(train_sigs),
        "n_distinct_test": len({sig[k] for k in test_keys}),
        "test_with_identical_train_trajectory": len(test_with_twin),
        "n_test": len(test_keys),
        "copies_per_trajectory": {"min": int(copies.min()), "median": float(np.median(copies)),
                                  "max": int(copies.max())},
        "radius": radii,
        "ic_signature": sig,
        "twins": {k: [t for t in groups[sig[k]] if t != k] for k in keys},
    }


def load_or_run_audit(path: Path = AUDIT_FILE) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    result = audit(read_all_radii())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    return result
