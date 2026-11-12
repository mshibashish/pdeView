"""Export the data and the model of pdeView's *static demo* (GitHub Pages).

This script precomputes everything the static demo needs instead of a server: every JSON
response of the API, every sample's ground-truth and FNO fields (compressed ``.bin`` files),
the report figures, and the FNO step as an ONNX model for custom runs in the browser. The
layout and the formats are fixed by ``docs/static-site.md``.

It is **not** how you run pdeView. The full app -- the live FastAPI server, PyTorch
inference and custom dam breaks solved on the server -- runs locally from the repository
(or with Docker). See the README's quick start:
https://github.com/mshibashish/pdeView#quick-start

Usage (after ``npm run build``, which empties ``frontend/dist``)::

    uv run --group static python scripts/export_static.py --out frontend/dist/static-api
    uv run --group static python scripts/export_static.py --out DIR [--keys K ...] [--limit N] \\
        [--gt-codec f32|u16] [--force]

* The JSON is produced by ``ViewerService``, the same code that serves the API, and is
  serialised the way the API serialises it. ``stats.json`` is computed from the unquantised
  fields.
* Held-out ground truth comes from ``data/heldout/references.h5`` (PyClaw) when it exists,
  otherwise from pdeView's solver, exactly as on the server.
* Resumable: per-sample files and the model are skipped when they exist (``--force``
  rewrites them). Files are written atomically. The measured quantisation error of every
  exported ``pred.bin`` is kept in ``.export-state.json`` so a resumed run can still report
  the maximum. The index files (``manifest.json``, ``meta.json``, ``samples.json``) and the
  report files are cheap and are regenerated on every run.
* Deterministic: the same inputs give the same files, except for the ``generated_at``
  timestamps (set ``SOURCE_DATE_EPOCH`` to fix them).
* ``samples.json`` lists the samples whose files are all present, so a partial export
  (``--keys``/``--limit``) is still self-consistent; a full export lists every sample.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pdeview.config import FNO_INITIAL_STEP, REPORTS_DIR, ROOT, SWE_FNO_CKPT
from pdeview.metrics import sample_stats
from pdeview.static_export import (
    CODEC_F32,
    CODEC_NAMES,
    CODEC_U16,
    ONNX_OPSET,
    api_json,
    decode_field,
    encode_field,
    export_onnx,
    onnx_io,
    onnx_rollout,
    onnx_session,
    read_header,
)

log = logging.getLogger("export_static")

QUICK_START = "https://github.com/mshibashish/pdeView#quick-start"
ABOUT = (
    "This produces the data for pdeView's *static demo* (GitHub Pages, docs/static-site.md).\n"
    "It is not needed to run pdeView: the full app (live FastAPI server, PyTorch inference,\n"
    "custom dam breaks solved on the server) runs locally from the repository, see the\n"
    f"README's quick start: {QUICK_START}"
)
FORMAT = 1
SIZE_BUDGET_MB = 900  # GitHub Pages sites may not exceed 1 GB
PARITY_TOL = 1e-4  # max |ONNX - PyTorch| over the 91-step rollout
STATE_FILE = ".export-state.json"
PARTIAL = ".partial"
MODEL_FILE = "model/fno_step.onnx"
MODEL_INFO = "model/model.json"


# ---------------------------------------------------------------------------- files
def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file + rename, so an interrupted run never leaves a truncated
    file that a resumed run would skip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + PARTIAL)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    write_atomic(path, data)
    return True


def json_bytes(obj) -> bytes:
    """Compact JSON for files that have no API counterpart (manifest, model info)."""
    return api_json(obj)


def valid_bin(path: Path, codec: int) -> bool:
    try:
        return read_header(path).codec == codec
    except (OSError, ValueError, EOFError):
        return False


class State:
    """Measured quantisation error of every exported ``pred.bin`` (for resumed runs)."""

    def __init__(self, out: Path):
        self.path = out / STATE_FILE
        try:
            self.pred_error: dict[str, float] = json.loads(self.path.read_text())["pred_error"]
        except (OSError, ValueError, KeyError):
            self.pred_error = {}

    def save(self) -> None:
        body = {"format": FORMAT, "pred_error": dict(sorted(self.pred_error.items()))}
        write_atomic(self.path, json.dumps(body, indent=0).encode())


# ---------------------------------------------------------------------------- samples
@dataclass
class Need:
    gt: bool
    pred: bool
    stats: bool

    def any(self) -> bool:
        return self.gt or self.pred or self.stats


def sample_files(out: Path, key: str) -> tuple[Path, Path, Path]:
    d = out / "samples" / key
    return d / "gt.bin", d / "pred.bin", d / "stats.json"


def plan(out: Path, key: str, gt_codec: int, state: State, force: bool) -> Need:
    gt, pred, stats = sample_files(out, key)
    if force:
        return Need(True, True, True)
    return Need(gt=not valid_bin(gt, gt_codec),
                pred=not valid_bin(pred, CODEC_U16) or key not in state.pred_error,
                stats=not stats.exists())


def write_sample(out: Path, key: str, need: Need, gt: np.ndarray, pred: np.ndarray | None,
                 stats: dict | None, gt_codec: int) -> dict:
    """Encode, check by decoding, and write one sample's files (runs in a worker thread)."""
    gt_path, pred_path, stats_path = sample_files(out, key)
    res: dict = {"key": key}
    if need.gt:
        blob = encode_field(gt, gt_codec)
        dec, _ = decode_field(blob)
        err = float(np.abs(dec.astype(np.float64) - gt).max())
        if gt_codec == CODEC_F32 and not np.array_equal(dec, gt):
            raise RuntimeError(f"{key}: lossless gt codec did not round-trip")
        write_atomic(gt_path, blob)
        res |= {"gt_bytes": len(blob), "gt_error": err}
    if need.pred:
        blob = encode_field(pred, CODEC_U16)
        dec, hdr = decode_field(blob)
        err = float(np.abs(dec.astype(np.float64) - pred).max())
        # bound: scale / 2, plus the float32 rounding of the decoded value
        bound = hdr.max_abs_error + float(np.abs(dec).max()) * np.finfo(np.float32).eps
        if not err <= bound:
            raise RuntimeError(f"{key}: pred quantisation error {err:.3g} > bound {bound:.3g}")
        write_atomic(pred_path, blob)
        res |= {"pred_bytes": len(blob), "pred_error": err}
    if need.stats:
        write_atomic(stats_path, api_json(stats, dict))
    return res


def export_samples(svc, out: Path, keys: list[str], gt_codec: int, state: State, force: bool,
                   workers: int) -> None:
    """Rollouts (and held-out solver runs) go one after another through the service in this
    thread, with all of PyTorch's threads, as on the server. Encoding, checking by decoding
    and writing run in a thread pool alongside (zlib and NumPy release the GIL)."""
    todo = [(k, plan(out, k, gt_codec, state, force)) for k in keys]
    todo = [(k, need) for k, need in todo if need.any()]
    if len(todo) < len(keys):
        log.info("samples: %d of %d already exported, skipped (use --force to redo)",
                 len(keys) - len(todo), len(keys))
    if not todo:
        return
    log.info("samples: exporting %d (rollouts on %s, %d writer threads)", len(todo),
             svc.device, workers)
    t0 = time.perf_counter()
    done = 0
    sizes = {"gt": 0, "pred": 0}

    def collect(fut: Future) -> None:
        nonlocal done
        res = fut.result()
        done += 1
        key = res["key"]
        if "pred_error" in res:
            state.pred_error[key] = res["pred_error"]
            state.save()
        sizes["gt"] += res.get("gt_bytes", 0)
        sizes["pred"] += res.get("pred_bytes", 0)
        rate = (time.perf_counter() - t0) / done
        parts = [f"[{done:{len(str(len(todo)))}d}/{len(todo)}] {key:<10}"]
        if "gt_bytes" in res:
            parts.append(f"gt {res['gt_bytes'] / 1e6:5.2f} MB"
                         + (f" (err {res['gt_error']:.2e})" if gt_codec != CODEC_F32 else ""))
        if "pred_bytes" in res:
            parts.append(f"pred {res['pred_bytes'] / 1e6:5.2f} MB (err {res['pred_error']:.2e})")
        parts.append(f"{rate:.1f} s/sample, eta {rate * (len(todo) - done) / 60:.1f} min")
        log.info("  ".join(parts))

    with ThreadPoolExecutor(workers, thread_name_prefix="export") as pool:
        inflight: deque[Future] = deque()
        for key, need in todo:
            gt = svc.ground_truth(key)
            pred = svc.prediction(key) if need.pred or need.stats else None
            stats = svc.stats(key) if need.stats else None
            inflight.append(pool.submit(write_sample, out, key, need, gt, pred, stats, gt_codec))
            while inflight and (inflight[0].done() or len(inflight) > 2 * workers):
                collect(inflight.popleft())
        while inflight:
            collect(inflight.popleft())
    log.info("samples: %d exported in %.0f s (gt %.0f MB, pred %.0f MB written)", done,
             time.perf_counter() - t0, sizes["gt"] / 1e6, sizes["pred"] / 1e6)


def complete_keys(svc, out: Path, gt_codec: int, state: State) -> list[str]:
    """Keys (in API order) whose three files are present and in the current codecs."""
    keys = []
    for s in svc.samples():
        gt, pred, stats = sample_files(out, s["key"])
        if (stats.exists() and s["key"] in state.pred_error and valid_bin(gt, gt_codec)
                and valid_bin(pred, CODEC_U16)):
            keys.append(s["key"])
    return keys


# ---------------------------------------------------------------------------- model
def export_model(svc, out: Path, force: bool) -> None:
    onnx_path, info_path = out / MODEL_FILE, out / MODEL_INFO
    if not force and onnx_path.exists() and info_path.exists():
        log.info("model: %s exists, skipped", MODEL_FILE)
        return
    k0 = FNO_INITIAL_STEP
    sample = svc.store.keys("test")[0]
    tmp = onnx_path.with_name(onnx_path.name + PARTIAL)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "pdeview.description": "One autoregressive step of the PDEBench FNO for 2D shallow "
                               "water (radial dam break), real-DFT form for onnxruntime-web",
        "pdeview.checkpoint": f"{SWE_FNO_CKPT.name} (md5 {svc.ckpt_info.md5})",
        "pdeview.license": "PDEBench pretrained models, CC BY 4.0, doi:10.18419/darus-2987",
    }
    model = export_onnx(svc.model, svc.grid, tmp, metadata=meta)
    log.info("model: exported %s (%.1f MB, opset %d, ops %s)", MODEL_FILE,
             tmp.stat().st_size / 1e6, ONNX_OPSET,
             ", ".join(sorted({n.op_type for n in model.graph.node})))

    gt = svc.ground_truth(sample)
    ref = svc.prediction(sample)
    t0 = time.perf_counter()
    pred = onnx_rollout(onnx_session(tmp), gt[:k0], gt.shape[0])
    seconds = time.perf_counter() - t0
    diff = float(np.abs(pred.astype(np.float64) - ref).max())
    parity = {
        "sample": sample,
        "steps": gt.shape[0] - k0,
        "max_abs_diff_vs_torch": diff,
        "nrmse_torch": float(sample_stats(ref[k0:], gt[k0:]).nrmse.mean()),
        "nrmse_onnx": float(sample_stats(pred[k0:], gt[k0:]).nrmse.mean()),
    }
    log.info("model: %d-step rollout on sample %s in onnxruntime (1 thread, %.1f s): "
             "max |onnx - torch| = %.2e (limit %.0e); nRMSE torch %.6f, onnx %.6f",
             parity["steps"], sample, seconds, diff, PARITY_TOL, parity["nrmse_torch"],
             parity["nrmse_onnx"])
    if not diff <= PARITY_TOL:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"error: the ONNX step differs from PyTorch by {diff:.3g} (max abs) "
                         f"over the {parity['steps']}-step rollout of sample {sample}; "
                         f"the limit is {PARITY_TOL:g}. Nothing was written to {MODEL_FILE}.")
    os.replace(tmp, onnx_path)
    info = {**onnx_io(model), "initial_step": k0, "opset": ONNX_OPSET, "parity": parity}
    write_atomic(info_path, json_bytes(info))


# ---------------------------------------------------------------------------- reports
def export_reports(svc, out: Path) -> None:
    """The report endpoints and figures (cheap: regenerated on every run)."""
    gen_path = REPORTS_DIR / "generalization.json"
    docs = {
        "verification.json": svc.verification,  # JSONResponse endpoints
        "heldout.json": svc.heldout_summary(),
        "generalization.json": json.loads(gen_path.read_text()) if gen_path.exists() else None,
    }
    for name, obj in docs.items():
        if obj is None:  # the API answers 404 here too
            log.warning("reports: %s not available, not written (the API returns 404)", name)
            (out / name).unlink(missing_ok=True)
            continue
        write_if_changed(out / name, api_json(obj))
    figures = sorted(p for p in (REPORTS_DIR / "figures").glob("*") if p.is_file())
    for src in figures:
        dst = out / "figures" / src.name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst.with_name(dst.name + PARTIAL))
            os.replace(dst.with_name(dst.name + PARTIAL), dst)
    log.info("reports: %s, %d figures", ", ".join(k for k, v in docs.items() if v is not None),
             len(figures))


# ---------------------------------------------------------------------------- run
def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    now = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def kind_of(rel: Path) -> str:
    parts = rel.parts
    if parts[0] == "samples" and len(parts) == 3:
        return parts[2]
    if parts[0] in ("figures", "model"):
        return parts[0] + "/"
    return "top-level JSON" if len(parts) == 1 and rel.suffix == ".json" else "other"


def report_sizes(out: Path) -> int:
    by_kind: dict[str, list[int]] = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            by_kind.setdefault(kind_of(path.relative_to(out)), []).append(path.stat().st_size)
    total = sum(sum(v) for v in by_kind.values())
    log.info("size of %s by kind:", out)
    for kind, sizes in sorted(by_kind.items(), key=lambda kv: -sum(kv[1])):
        log.info("  %-15s %5d files  %8.1f MB  (mean %.3f MB)", kind, len(sizes),
                 sum(sizes) / 1e6, sum(sizes) / len(sizes) / 1e6)
    log.info("  %-15s %5d files  %8.1f MB", "total", sum(len(v) for v in by_kind.values()),
             total / 1e6)
    return total


def select_keys(all_keys: list[str], keys: list[str] | None, limit: int | None) -> list[str]:
    if keys:
        unknown = sorted(set(keys) - set(all_keys))
        if unknown:
            raise SystemExit(f"error: unknown sample key(s): {', '.join(unknown)}")
        wanted = set(keys)
        all_keys = [k for k in all_keys if k in wanted]
    return all_keys[:limit] if limit is not None else all_keys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="export_static.py",
        description="Export the precomputed API files and the ONNX FNO step for pdeView's\n"
                    "static demo (GitHub Pages), as specified in docs/static-site.md.",
        epilog=ABOUT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", type=Path, required=True,
                   help="output directory, i.e. static-api/ (e.g. frontend/dist/static-api)")
    p.add_argument("--keys", nargs="+", metavar="K", help="export only these samples")
    p.add_argument("--limit", type=int, metavar="N", help="export only the first N samples")
    p.add_argument("--gt-codec", choices=sorted(CODEC_NAMES), default="f32",
                   help="gt.bin codec: f32 = lossless (default), u16 = quantised (smaller)")
    p.add_argument("--force", action="store_true",
                   help="rewrite per-sample files and the model even if they exist")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), metavar="W",
                   help="threads for encoding and writing (default: %(default)s)")
    p.add_argument("--no-model", action="store_true",
                   help="skip model/ (quick data-only exports; the demo needs the model)")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be at least 1")
    if args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def run(args: argparse.Namespace) -> int:
    from pdeview.service import ViewerService  # imports torch: after --help

    missing = [m for m in ("onnx", "onnxruntime") if importlib.util.find_spec(m) is None]
    if missing and not args.no_model:
        raise SystemExit(f"error: {', '.join(missing)} not installed. The export needs the "
                         "`static` dependency group: uv run --group static python "
                         "scripts/export_static.py ...")
    t_start = time.perf_counter()
    out: Path = args.out.resolve()
    log.info("pdeView static export -> %s", out)
    for line in ABOUT.splitlines():
        log.info(line)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.rglob("*" + PARTIAL):  # left by an interrupted run
        stale.unlink()

    svc = ViewerService()
    gt_codec = CODEC_NAMES[args.gt_codec]
    all_keys = [s["key"] for s in svc.samples()]
    keys = select_keys(all_keys, args.keys, args.limit)
    refs = getattr(svc, "_heldout_refs", None)  # what ViewerService found at start-up
    log.info("%d samples in the API, %d selected; held-out ground truth from %s", len(all_keys),
             len(keys), f"{refs.generator} (references.h5)" if refs is not None
             else "pdeview's solver (no data/heldout/references.h5)")
    state = State(out)

    export_reports(svc, out)
    if args.no_model:
        log.info("model: skipped (--no-model)")
    else:
        export_model(svc, out, args.force)
    export_samples(svc, out, keys, gt_codec, state, args.force, args.workers)

    done = complete_keys(svc, out, gt_codec, state)
    if not done:
        raise SystemExit("error: no sample was exported")
    pred_error = max(state.pred_error[k] for k in done)
    generated_at, sha = timestamp(), git_sha()
    listing = [s for s in svc.samples() if s["key"] in set(done)]
    if len(done) < len(all_keys):
        log.info("samples.json lists the %d exported samples of %d (partial export)", len(done),
                 len(all_keys))
    write_if_changed(out / "samples.json", api_json(listing, list[dict]))
    meta = svc.meta() | {"static": {
        "generated_at": generated_at,
        "git_sha": sha,
        "pred_device": svc.device,
        "pred_codec": CODEC_U16,
        "pred_max_abs_error": pred_error,
    }}
    write_if_changed(out / "meta.json", api_json(meta, dict))
    manifest = {
        "format": FORMAT,
        "generated_at": generated_at,
        "git_sha": sha,
        "n_samples": len(done),
        "codecs": {"gt": gt_codec, "pred": CODEC_U16},
        "pred_max_abs_error": pred_error,
        "model": MODEL_FILE if (out / MODEL_FILE).exists() else None,
    }
    write_if_changed(out / "manifest.json", json_bytes(manifest))
    log.info("manifest: %d samples, codecs gt=%d pred=%d, max pred quantisation error %.3e",
             len(done), gt_codec, CODEC_U16, pred_error)

    total = report_sizes(out)
    log.info("done in %.1f min", (time.perf_counter() - t_start) / 60)
    if total > SIZE_BUDGET_MB * 1e6:
        log.error("error: %s is %.0f MB, above the %d MB budget (GitHub Pages sites are limited "
                  "to 1 GB). Try --gt-codec u16.", out, total / 1e6, SIZE_BUDGET_MB)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
