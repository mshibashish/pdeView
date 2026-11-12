"""Read access to the local shallow-water data subset (``data/swe_subset.h5``)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from pdeview.config import SUBSET_FILE


@dataclass(frozen=True)
class SampleInfo:
    key: str
    split: str  # "test" (PDEBench held-out split) or "train" (training-set probe)
    dam_radius: float


class SampleStore:
    """Thread-safe, read-only view of the downloaded samples."""

    def __init__(self, path: Path = SUBSET_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `uv run python scripts/fetch_data.py` first"
            )
        self.path = path
        self._lock = threading.Lock()
        self._h5 = h5py.File(path, "r")
        infos = []
        for key in sorted(self._h5.keys()):
            g = self._h5[key]
            if not g.attrs.get("complete", False):
                continue
            infos.append(SampleInfo(key, str(g.attrs["split"]), float(g.attrs["dam_radius"])))
        if not infos:
            raise RuntimeError(f"{path} contains no complete samples")
        self.samples: dict[str, SampleInfo] = {s.key: s for s in infos}
        first = self._h5[infos[0].key]
        self.x = first["grid/x"][()].astype(np.float32)
        self.y = first["grid/y"][()].astype(np.float32)
        self.t = first["grid/t"][()].astype(np.float32)
        self.frames = lru_cache(maxsize=24)(self._frames)

    def keys(self, split: str | None = None) -> list[str]:
        return [k for k, s in self.samples.items() if split is None or s.split == split]

    def _frames(self, key: str) -> np.ndarray:
        """Water height ``h`` of one sample, shape ``[T=101, X=128, Y=128]`` (float32)."""
        if key not in self.samples:
            raise KeyError(key)
        with self._lock:
            data = self._h5[key]["data"][()]
        out = np.ascontiguousarray(data[..., 0], dtype=np.float32)
        out.setflags(write=False)
        return out

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    def close(self) -> None:
        self._h5.close()
