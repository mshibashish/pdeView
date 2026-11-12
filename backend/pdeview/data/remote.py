"""Partial, resumable download of the PDEBench 2D shallow-water dataset.

The full file (``2D_rdb_NA_NA.h5``) is 6.2 GB. We only need the held-out test split (and a
handful of training samples for a leakage check), so instead of downloading everything we:

1. open the remote HDF5 file through ``fsspec`` HTTP range reads (small block cache) and read
   only the metadata of the groups we need: dataset byte offsets, shapes and the ``config``
   attribute. Every dataset in the file is contiguous and uncompressed, so an offset fully
   describes where its bytes live;
2. download each sample's ``data`` block with a single HTTP ``Range`` GET (in parallel);
3. write the samples into a small local HDF5 file with the same per-sample layout.

DaRUS answers ``/api/access/datafile/<id>`` with a 303 redirect to a presigned S3 URL that
expires after 48 h, so the redirect is resolved fresh for every run (and again on HTTP 403).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fsspec
import h5py
import numpy as np
import requests
import yaml

from pdeview.config import (
    N_SAMPLES_TOTAL,
    N_TRAIN_PROBE,
    SUBSET_FILE,
    SWE_DATA,
    TEST_RATIO,
    RemoteFile,
)

log = logging.getLogger(__name__)

USER_AGENT = "pdeView/0.1 (research data client)"
_DATA_DSETS = ("data", "grid/x", "grid/y", "grid/t")


def resolve_url(remote: RemoteFile, timeout: float = 30.0) -> str:
    """Return the current direct (presigned) download URL for a DaRUS file."""
    r = requests.get(
        remote.url, allow_redirects=False, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    if r.status_code in (301, 302, 303, 307, 308):
        return r.headers["Location"]
    r.raise_for_status()
    return remote.url


def subset_keys(n_total: int = N_SAMPLES_TOTAL) -> tuple[list[str], list[str]]:
    """Keys of PDEBench's default test split (last 10 % of sorted keys) and a train probe."""
    keys = [f"{i:04d}" for i in range(n_total)]
    test_start = int(n_total * (1 - TEST_RATIO))
    return keys[test_start:], keys[:N_TRAIN_PROBE]


@dataclass
class SampleLayout:
    key: str
    offsets: dict[str, int]
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]
    config: str

    @property
    def dam_radius(self) -> float:
        return float(yaml.safe_load(self.config)["sim"]["dam_radius"])

    def nbytes(self, name: str) -> int:
        return int(np.prod(self.shapes[name])) * np.dtype(self.dtypes[name]).itemsize


def read_layouts(url: str, keys: Iterable[str], size: int = SWE_DATA.size) -> list[SampleLayout]:
    """Read the byte layout of the requested sample groups from the remote HDF5 file."""
    fs = fsspec.filesystem("http", headers={"User-Agent": USER_AGENT})
    layouts = []
    with fs.open(url, mode="rb", block_size=2**16, cache_type="blockcache", size=size) as fobj:
        with h5py.File(fobj, "r") as h5:
            last = f"{N_SAMPLES_TOTAL - 1:04d}"
            if last not in h5 or f"{N_SAMPLES_TOTAL:04d}" in h5:
                raise RuntimeError(f"Unexpected sample keys in remote file (expected 0000..{last})")
            for key in keys:
                g = h5[key]
                offsets, shapes, dtypes = {}, {}, {}
                for name in _DATA_DSETS:
                    d = g[name]
                    if d.chunks is not None or d.compression is not None:
                        raise RuntimeError(f"{key}/{name} is chunked/compressed; cannot range-read")
                    offsets[name] = int(d.id.get_offset())
                    shapes[name] = tuple(d.shape)
                    dtypes[name] = d.dtype.str
                layouts.append(
                    SampleLayout(key, offsets, shapes, dtypes, str(g.attrs["config"]))
                )
                log.debug("layout %s: data@%d", key, offsets["data"])
    return layouts


class RangeFetcher:
    """Thread-safe HTTP Range reader that transparently refreshes an expired presigned URL."""

    def __init__(self, remote: RemoteFile, url: str | None = None, retries: int = 4):
        self.remote = remote
        self.url = url or resolve_url(remote)
        self.retries = retries
        self._lock = threading.Lock()
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
            self._local.s.headers["User-Agent"] = USER_AGENT
        return self._local.s

    def get(self, offset: int, nbytes: int) -> bytes:
        last_err: Exception | None = None
        for _attempt in range(self.retries):
            url = self.url
            try:
                r = self._session().get(
                    url, headers={"Range": f"bytes={offset}-{offset + nbytes - 1}"}, timeout=120
                )
                if r.status_code == 403:  # presigned URL expired
                    with self._lock:
                        if self.url == url:
                            self.url = resolve_url(self.remote)
                    continue
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206 Partial Content, got {r.status_code}")
                if len(r.content) != nbytes:
                    raise RuntimeError(f"short read: {len(r.content)} != {nbytes}")
                return r.content
            except (requests.RequestException, RuntimeError) as err:
                last_err = err
                log.warning("range read %d+%d failed (%s); retrying", offset, nbytes, err)
        raise RuntimeError(f"range read failed after {self.retries} attempts") from last_err

    def array(self, layout: SampleLayout, name: str) -> np.ndarray:
        raw = self.get(layout.offsets[name], layout.nbytes(name))
        return np.frombuffer(raw, dtype=np.dtype(layout.dtypes[name])).reshape(layout.shapes[name])


def check_sample(key: str, data: np.ndarray, x: np.ndarray, dam_radius: float) -> None:
    """Cheap physical sanity checks that also catch byte-offset mistakes."""
    if data.shape != (101, 128, 128, 1):
        raise ValueError(f"{key}: unexpected data shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{key}: non-finite values")
    h0 = data[0, :, :, 0]
    if not (np.isclose(h0.min(), 1.0) and np.isclose(h0.max(), 2.0)):
        raise ValueError(f"{key}: initial condition is not a 1/2 dam break")
    # Area of the h=2 disc ~= pi r^2: tolerance of 2 cells on the radius.
    dx = float(x[1] - x[0])
    r_est = float(np.sqrt((h0 > 1.5).sum() * dx * dx / np.pi))
    if abs(r_est - dam_radius) > 2 * dx:
        raise ValueError(f"{key}: IC radius {r_est:.3f} != config dam_radius {dam_radius:.3f}")
    if data.min() < 0.0 or data.max() > 2.5:
        raise ValueError(f"{key}: water height out of physical range")


def build_subset(
    keys: list[str],
    split: dict[str, str],
    out_path: Path = SUBSET_FILE,
    workers: int = 8,
) -> Path:
    """Download the given sample keys into ``out_path`` (resumable)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "a") as out:
        done = {k for k in out.keys() if out[k].attrs.get("complete", False)}
    todo = [k for k in keys if k not in done]
    log.info("%d/%d samples already present, fetching %d", len(done & set(keys)), len(keys),
             len(todo))
    if not todo:
        return out_path

    url = resolve_url(SWE_DATA)
    layouts = read_layouts(url, todo)
    fetcher = RangeFetcher(SWE_DATA, url)

    def fetch(layout: SampleLayout):
        arrays = {name: fetcher.array(layout, name) for name in _DATA_DSETS}
        check_sample(layout.key, arrays["data"], arrays["grid/x"], layout.dam_radius)
        return layout, arrays

    with ThreadPoolExecutor(max_workers=workers) as pool, h5py.File(out_path, "a") as out:
        futures = [pool.submit(fetch, lay) for lay in layouts]
        for i, fut in enumerate(as_completed(futures), 1):
            layout, arrays = fut.result()
            if layout.key in out:
                del out[layout.key]
            g = out.create_group(layout.key)
            for name, arr in arrays.items():
                g.create_dataset(name, data=arr)
            g.attrs["config"] = layout.config
            g.attrs["dam_radius"] = layout.dam_radius
            g.attrs["split"] = split[layout.key]
            g.attrs["source_offset"] = layout.offsets["data"]
            g.attrs["complete"] = True
            out.flush()
            log.info("[%3d/%d] %s (r=%.3f, %s)", i, len(layouts), layout.key, layout.dam_radius,
                     split[layout.key])
    return out_path


def md5sum(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(remote: RemoteFile, dest: Path) -> Path:
    """Download a whole DaRUS file and verify its MD5 against the published checksum."""
    if dest.exists() and md5sum(dest) == remote.md5:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(remote.url, stream=True, timeout=120,
                      headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(2**20):
                f.write(block)
    got = md5sum(tmp)
    if got != remote.md5:
        tmp.unlink()
        raise RuntimeError(f"MD5 mismatch for {remote.name}: got {got}, expected {remote.md5}")
    tmp.replace(dest)
    return dest


def ensure_assets(include_train_probe: bool = True, workers: int = 8) -> dict[str, Path]:
    """Make sure the checkpoint and the local data subset exist (idempotent)."""
    from pdeview.config import CKPT_DIR, SWE_FNO_CKPT

    ckpt = download_file(SWE_FNO_CKPT, CKPT_DIR / SWE_FNO_CKPT.name)
    test, train = subset_keys()
    keys = test + (train if include_train_probe else [])
    split = {k: "test" for k in test} | {k: "train" for k in train}
    data = build_subset(keys, split, workers=workers)
    return {"checkpoint": ckpt, "data": data}
