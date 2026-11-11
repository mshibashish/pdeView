"""Error metrics for autoregressive PDE rollouts.

Independent re-implementation of the metric *definitions* used by PDEBench (Takamoto et al.,
NeurIPS 2022, Sec. 3 / App. B). No PDEBench code is copied (its metrics module carries a
non-commercial licence).

All quantities are computed for one channel (water height ``h``) on fields of shape
``[T, X, Y]``, where ``T`` are the *evaluated* time steps (PDEBench excludes the
``initial_step`` ground-truth input frames).

Two aggregation modes are provided:

* :func:`aggregate` — aggregate a set of samples as one population (what we report as the
  "clean" numbers);
* :func:`aggregate_pdebench` — reproduce PDEBench's evaluation loop exactly: metrics are
  computed per mini-batch (``batch_size=5`` for the shallow-water config), summed over
  batches and divided by ``n_batches - 1`` (the loop divides by the last ``enumerate`` index).
  This is what the paper's Table 5 numbers were produced with, so it is the like-for-like
  comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fourier bands used by PDEBench: k in [0, 4) low, [4, 12) mid, [12, kmax) high
I_LOW, I_HIGH = 4, 12
METRIC_KEYS = ("RMSE", "nRMSE", "cRMSE", "max_error", "bRMSE",
               "fRMSE_low", "fRMSE_mid", "fRMSE_high")


@dataclass
class SampleErrorStats:
    """Per-time-step error statistics of one sample (all arrays indexed by time)."""

    rmse: np.ndarray  # [T]   sqrt(mean_xy (p - t)^2)
    nrmse: np.ndarray  # [T]   rmse / sqrt(mean_xy t^2)
    max_abs: np.ndarray  # [T]   max_xy |p - t|
    bd: np.ndarray  # [T]   RMS error over the four domain edges
    csv: np.ndarray  # [T]   (sum_xy p - sum_xy t) / (nx ny)   (signed conserved-quantity error)
    fourier_sq: np.ndarray  # [K, T] |FFT(p - t)|^2 summed in radial wavenumber bins
    nx: int
    ny: int


def _radial_bins(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Index map of the non-negative-frequency quadrant into integer radial bins."""
    kmax = min(nx // 2, ny // 2)
    i, j = np.meshgrid(np.arange(nx // 2), np.arange(ny // 2), indexing="ij")
    k = np.floor(np.sqrt(i**2 + j**2)).astype(int)
    keep = k < kmax
    return np.stack([i[keep], j[keep]]), k[keep], kmax


def sample_stats(pred: np.ndarray, target: np.ndarray) -> SampleErrorStats:
    """Compute per-time statistics for one sample. ``pred``/``target``: ``[T, X, Y]``."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"expected matching [T, X, Y] arrays, got {pred.shape}, {target.shape}")
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    nt, nx, ny = t.shape
    err = p - t

    rmse = np.sqrt(np.mean(err**2, axis=(1, 2)))
    nrmse = rmse / np.sqrt(np.mean(t**2, axis=(1, 2)))
    max_abs = np.max(np.abs(err), axis=(1, 2))

    # Edges x=0, x=-1 (ny cells each) and y=0, y=-1 (nx cells each); corners count twice.
    edge_sq = (
        np.sum(err[:, 0, :] ** 2, axis=1) + np.sum(err[:, -1, :] ** 2, axis=1)
        + np.sum(err[:, :, 0] ** 2, axis=1) + np.sum(err[:, :, -1] ** 2, axis=1)
    )
    bd = np.sqrt(edge_sq / (2 * nx + 2 * ny))

    csv = (p.sum(axis=(1, 2)) - t.sum(axis=(1, 2))) / (nx * ny)

    power = np.abs(np.fft.fft2(err, axes=(1, 2))) ** 2  # [T, X, Y]
    (ii, jj), kbin, kmax = _radial_bins(nx, ny)
    fourier_sq = np.zeros((kmax, nt))
    np.add.at(fourier_sq, kbin, power[:, ii, jj].T)

    return SampleErrorStats(rmse, nrmse, max_abs, bd, csv, fourier_sq, nx, ny)


def aggregate(stats: list[SampleErrorStats]) -> dict[str, float]:
    """Aggregate samples as a single population; returns time-averaged scalar metrics."""
    if not stats:
        raise ValueError("no samples to aggregate")
    nx, ny = stats[0].nx, stats[0].ny
    stack = {name: np.stack([getattr(s, name) for s in stats]) for name in
             ("rmse", "nrmse", "max_abs", "bd", "csv", "fourier_sq")}

    f_rms = np.sqrt(stack["fourier_sq"].mean(axis=0)) / (nx * ny)  # [K, T]
    return {
        "RMSE": float(stack["rmse"].mean()),
        "nRMSE": float(stack["nrmse"].mean()),
        "cRMSE": float(np.sqrt(np.mean(stack["csv"] ** 2, axis=0)).mean()),
        "max_error": float(stack["max_abs"].max(axis=0).mean()),
        "bRMSE": float(stack["bd"].mean()),
        "fRMSE_low": float(f_rms[:I_LOW].mean(axis=0).mean()),
        "fRMSE_mid": float(f_rms[I_LOW:I_HIGH].mean(axis=0).mean()),
        "fRMSE_high": float(f_rms[I_HIGH:].mean(axis=0).mean()),
    }


def aggregate_pdebench(stats: list[SampleErrorStats], batch_size: int = 5) -> dict[str, float]:
    """Reproduce PDEBench's ``metrics()`` aggregation (per-batch, summed, / (n_batches - 1))."""
    batches = [stats[i:i + batch_size] for i in range(0, len(stats), batch_size)]
    if len(batches) < 2:
        raise ValueError("PDEBench aggregation needs at least two batches")
    total = {k: 0.0 for k in METRIC_KEYS}
    for batch in batches:
        for k, v in aggregate(batch).items():
            total[k] += v
    return {k: v / (len(batches) - 1) for k, v in total.items()}


def time_curves(stats: list[SampleErrorStats]) -> dict[str, list[float]]:
    """Error-vs-time curves averaged over samples."""
    return {
        "RMSE": np.stack([s.rmse for s in stats]).mean(axis=0).tolist(),
        "nRMSE": np.stack([s.nrmse for s in stats]).mean(axis=0).tolist(),
        "max_error": np.stack([s.max_abs for s in stats]).max(axis=0).tolist(),
    }
