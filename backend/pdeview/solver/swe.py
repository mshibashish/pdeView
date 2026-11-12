"""Finite-volume solver for the 2D shallow-water equations (NumPy).

    h_t + (hu)_x + (hv)_y = 0
    (hu)_t + (hu^2 + g h^2 / 2)_x + (huv)_y = 0
    (hv)_t + (huv)_x + (hv^2 + g h^2 / 2)_y = 0

The scheme mirrors the configuration PDEBench used to generate its data with PyClaw
(``ClawSolver2D`` + ``riemann.shallow_roe_with_efix_2D``, MC limiter) so that our reference
trajectories are directly comparable with the dataset:

* LeVeque's high-resolution wave-propagation method in 1D (fluctuations + limited
  second-order correction fluxes), applied with Godunov dimensional splitting
  (x-sweep, then y-sweep) - PyClaw's ``dimensional_split=True`` default;
* Roe Riemann solver with the Harten-Hyman entropy fix for transonic rarefactions;
* MC ("monotonized centred") wave limiter;
* zero-order extrapolation (outflow) boundaries with two ghost cells;
* PyClaw's variable time-step control: ``cfl_desired = 0.9``, steps with CFL > 1 are
  rejected and retried, the step is shortened to hit each output time exactly.

This is an independent implementation written from the published algorithm (LeVeque,
*Finite Volume Methods for Hyperbolic Problems*, 2002); it does not use Clawpack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NG = 2  # ghost cells per side


@dataclass
class SolverStats:
    n_steps: int
    n_rejected: int
    max_cfl: float


def _riemann(ql: np.ndarray, qr: np.ndarray, g: float):
    """Roe solver with entropy fix. ``ql``/``qr``: (3, ...) states left/right of each interface
    with the *normal* momentum in component 1. Returns waves (3 waves, 3 eqs, ...), speeds
    (3, ...), and the left-/right-going fluctuations amdq, apdq (3, ...)."""
    hl, hul, hvl = ql
    hr, hur, hvr = qr
    sl, sr = np.sqrt(hl), np.sqrt(hr)
    hbar = 0.5 * (hl + hr)
    u = (hul / sl + hur / sr) / (sl + sr)
    v = (hvl / sl + hvr / sr) / (sl + sr)
    a = np.sqrt(g * hbar)

    d1, d2, d3 = hr - hl, hur - hul, hvr - hvl
    a1 = ((u + a) * d1 - d2) / (2 * a)
    a2 = -v * d1 + d3
    a3 = (d2 - (u - a) * d1) / (2 * a)

    zeros = np.zeros_like(a1)
    waves = np.stack([
        np.stack([a1, a1 * (u - a), a1 * v]),
        np.stack([zeros, zeros, a2]),
        np.stack([a3, a3 * (u + a), a3 * v]),
    ])
    speeds = np.stack([u - a, u, u + a])

    # --- Harten-Hyman entropy fix (as in Clawpack's rpn2 shallow_roe_with_efix) ---
    s0 = hul / hl - np.sqrt(g * hl)  # left state 1-characteristic speed
    h1 = hl + waves[0, 0]
    hu1 = hul + waves[0, 1]
    s1 = hu1 / h1 - np.sqrt(g * h1)  # speed just right of the 1-wave
    with np.errstate(divide="ignore", invalid="ignore"):
        frac1 = np.where((s0 < 0) & (s1 > 0), s0 * (s1 - speeds[0]) / (s1 - s0),
                         np.where(speeds[0] < 0, speeds[0], 0.0))
    s03 = hur / hr + np.sqrt(g * hr)  # right state 3-characteristic speed
    h3 = hr - waves[2, 0]
    hu3 = hur - waves[2, 1]
    s3 = hu3 / h3 + np.sqrt(g * h3)  # speed just left of the 3-wave
    with np.errstate(divide="ignore", invalid="ignore"):
        frac3 = np.where((s3 < 0) & (s03 > 0), s3 * (s03 - speeds[2]) / (s03 - s3),
                         np.where(speeds[2] < 0, speeds[2], 0.0))

    amdq = frac1 * waves[0]
    left_23 = speeds[1] <= 0  # otherwise the 2- and 3-waves are right-going
    amdq = amdq + np.where(left_23, speeds[1] * waves[1] + frac3 * waves[2], 0.0)
    supersonic_right = (s0 > 0) & (speeds[0] > 0)
    amdq = np.where(supersonic_right, 0.0, amdq)
    apdq = (speeds[:, None] * waves).sum(axis=0) - amdq
    return waves, speeds, amdq, apdq


def _mc(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -1e6, 1e6)  # huge ratios come from near-zero waves; phi saturates anyway
    return np.maximum(0.0, np.minimum(np.minimum((1.0 + r) / 2.0, 2.0), 2.0 * r))


def _sweep(q: np.ndarray, dtdx: float, g: float) -> tuple[np.ndarray, float]:
    """One 1D wave-propagation update along axis 1 of ``q`` (3, n + 2 NG, m).

    Component 1 must hold the momentum normal to the sweep direction. Returns the updated
    array (ghost cells untouched) and the CFL number of the sweep."""
    ql, qr = q[:, :-1], q[:, 1:]  # interface k sits between cells k and k+1
    waves, speeds, amdq, apdq = _riemann(ql, qr, g)
    n_int = q.shape[1] - 1
    inner = slice(NG - 1, n_int - NG + 1)  # interfaces adjacent to interior cells

    # MC wave limiter: compare each wave with the same family's wave at the upwind interface
    w = waves[:, :, 1:-1]  # interfaces 1 .. n_int-2
    wl, wr = waves[:, :, :-2], waves[:, :, 2:]
    wnorm2 = (w * w).sum(axis=1)
    dotl = (w * wl).sum(axis=1)
    dotr = (w * wr).sum(axis=1)
    s = speeds[:, 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(s > 0, dotl, dotr) / wnorm2
    phi = np.where(wnorm2 > 0, _mc(np.nan_to_num(r)), 1.0)
    w_lim = w * phi[:, None]

    abs_s = np.abs(s)
    corr = 0.5 * (abs_s * (1.0 - dtdx * abs_s))[:, None] * w_lim
    flux2 = corr.sum(axis=0)  # (3, interfaces 1 .. n_int-2, m)

    out = q.copy()
    c = slice(NG, q.shape[1] - NG)  # interior cells
    # cell i: left interface i-1, right interface i
    out[:, c] -= dtdx * (apdq[:, NG - 1:n_int - NG] + amdq[:, NG:n_int - NG + 1])
    # flux2 index j corresponds to interface j+1
    out[:, c] -= dtdx * (flux2[:, NG - 1:n_int - NG] - flux2[:, NG - 2:n_int - NG - 1])
    cfl = float(dtdx * np.abs(speeds[:, inner]).max())
    return out, cfl


def _extrapolate(q: np.ndarray) -> None:
    q[:, :NG] = q[:, NG:NG + 1]
    q[:, -NG:] = q[:, -NG - 1:-NG]
    q[:, :, :NG] = q[:, :, NG:NG + 1]
    q[:, :, -NG:] = q[:, :, -NG - 1:-NG]


def _step(q: np.ndarray, dt: float, dx: float, dy: float, g: float):
    _extrapolate(q)
    qx, cfl_x = _sweep(q, dt / dx, g)
    _extrapolate(qx)
    # y-sweep: swap axes so y is axis 1, and swap the momentum components
    qt = qx[[0, 2, 1]].transpose(0, 2, 1)
    qy, cfl_y = _sweep(np.ascontiguousarray(qt), dt / dy, g)
    out = qy[[0, 2, 1]].transpose(0, 2, 1)
    return np.ascontiguousarray(out), max(cfl_x, cfl_y)


def simulate(
    h0: np.ndarray,
    t_out: np.ndarray,
    dx: float,
    dy: float,
    g: float = 1.0,
    hu0: np.ndarray | None = None,
    hv0: np.ndarray | None = None,
    cfl_desired: float = 0.9,
    cfl_max: float = 1.0,
    dt_initial: float = 0.1,
    max_steps: int = 100_000,
) -> tuple[np.ndarray, SolverStats]:
    """Integrate from ``t_out[0]`` and return the depth ``h`` at every output time,
    shape ``[len(t_out), nx, ny]`` (float64)."""
    nx, ny = h0.shape
    q = np.zeros((3, nx + 2 * NG, ny + 2 * NG))
    q[0, NG:-NG, NG:-NG] = h0
    if hu0 is not None:
        q[1, NG:-NG, NG:-NG] = hu0
    if hv0 is not None:
        q[2, NG:-NG, NG:-NG] = hv0
    if (h0 <= 0).any():
        raise ValueError("depth must be positive everywhere (no dry states)")

    out = np.empty((len(t_out), nx, ny))
    out[0] = h0
    t = float(t_out[0])
    dt = dt_initial
    n_steps = n_rej = 0
    max_cfl = 0.0
    for k, tend in enumerate(t_out[1:], start=1):
        tend = float(tend)
        while True:
            if t + dt > tend:
                dt = tend - t
            if tend - t - dt < 1e-14 * t:
                dt = tend - t
            q_new, cfl = _step(q, dt, dx, dy, g)
            if cfl <= cfl_max:
                q = q_new
                t += dt
                n_steps += 1
                max_cfl = max(max_cfl, cfl)
            else:
                n_rej += 1
            if cfl > 0:
                dt = dt * cfl_desired / cfl
            if n_steps + n_rej > max_steps:
                raise RuntimeError("too many time steps")
            if t >= tend:
                break
        h = q[0, NG:-NG, NG:-NG]
        if not np.isfinite(h).all() or (h <= 0).any():
            raise FloatingPointError(f"solver became unstable at t = {t:.3f}")
        out[k] = h
    return out, SolverStats(n_steps, n_rej, max_cfl)


def cell_centres(n: int = 128, lo: float = -2.5, hi: float = 2.5) -> np.ndarray:
    d = (hi - lo) / n
    return lo + (np.arange(n) + 0.5) * d


def dam_break_ic(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    centre: tuple[float, float] = (0.0, 0.0),
    h_in: float = 2.0,
    h_out: float = 1.0,
) -> np.ndarray:
    """PDEBench's radial dam break, generalised to any centre / heights: h = h_in where the
    cell centre lies within ``radius`` of ``centre``, else ``h_out`` (velocity zero)."""
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2)
    return np.where(r <= radius, h_in, h_out)
