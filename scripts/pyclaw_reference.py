# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["clawpack==5.14.0", "numpy>=2", "h5py", "scipy"]
# ///
"""Integrate shallow-water initial conditions with PyClaw, configured as PDEBench was.

This is the generator PDEBench used for ``2D_rdb_NA_NA.h5`` (``sim_radial_dam_break.py``):
``ClawSolver2D`` with ``riemann.shallow_roe_with_efix_2D``, MC limiter, 3 waves, zero-order
extrapolation on all sides, g = 1, 128 x 128 cells on [-2.5, 2.5]^2, 100 outputs up to t = 1.
Only the initial depth is taken from the input; velocities start at zero.

It runs in its own environment (uv reads the dependency block above), separate from the
project's, because clawpack ships wheels for Python 3.13 only:

    uv run --python 3.13 scripts/pyclaw_reference.py INPUT.npz OUTPUT.h5

``INPUT.npz`` holds ``keys`` [N] (str) and ``h0`` [N, 128, 128] (float64). ``OUTPUT.h5`` gets
one float32 dataset ``[101, 128, 128]`` per key (existing keys are skipped, so it resumes).

The macOS wheel links libgfortran from Homebrew's gcc. If that is missing, the script
re-executes itself with the gfortran runtime bundled in SciPy's wheel on the library path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _import_pyclaw():
    try:
        from clawpack import pyclaw, riemann
        return pyclaw, riemann
    except ImportError as err:
        if "libgfortran" not in str(err) or os.environ.get("_PDEVIEW_REEXEC"):
            raise
        import scipy
        libdir = Path(scipy.__file__).parent / ".dylibs"
        if not (libdir / "libgfortran.5.dylib").exists():
            raise
        env = dict(os.environ, _PDEVIEW_REEXEC="1",
                   DYLD_LIBRARY_PATH=f"{libdir}:{os.environ.get('DYLD_LIBRARY_PATH', '')}")
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


pyclaw, riemann = _import_pyclaw()

import clawpack  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

N_CELLS, LO, HI = 128, -2.5, 2.5
T_FINAL, N_OUT, GRAV = 1.0, 100, 1.0


def solve(h0: np.ndarray) -> np.ndarray:
    solver = pyclaw.ClawSolver2D(riemann.shallow_roe_with_efix_2D)
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.num_waves = 3
    solver.num_eqn = 3
    solver.bc_lower[0] = solver.bc_upper[0] = pyclaw.BC.extrap
    solver.bc_lower[1] = solver.bc_upper[1] = pyclaw.BC.extrap

    x = pyclaw.Dimension(LO, HI, N_CELLS, name="x")
    y = pyclaw.Dimension(LO, HI, N_CELLS, name="y")
    domain = pyclaw.Domain([x, y])
    state = pyclaw.State(domain, 3)
    state.problem_data["grav"] = GRAV
    state.q[0] = h0
    state.q[1] = 0.0
    state.q[2] = 0.0

    claw = pyclaw.Controller()
    claw.tfinal = T_FINAL
    claw.num_output_times = N_OUT
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.keep_copy = True
    claw.output_format = None
    claw.verbosity = 0
    claw.run()
    return np.stack([f.q[0] for f in claw.frames]).astype(np.float32)


def main(src: str, dst: str) -> None:
    inp = np.load(src)
    keys = [str(k) for k in inp["keys"]]
    h0s = inp["h0"]
    if h0s.shape[1:] != (N_CELLS, N_CELLS):
        raise SystemExit(f"expected h0 of shape [N, {N_CELLS}, {N_CELLS}], got {h0s.shape}")
    t0 = time.perf_counter()
    with h5py.File(dst, "a") as out:
        out.attrs["generator"] = f"clawpack {clawpack.__version__} (PDEBench SWE configuration)"
        todo = [i for i, k in enumerate(keys) if k not in out]
        for n, i in enumerate(todo, 1):
            h = solve(h0s[i])
            ds = out.create_dataset(keys[i], data=h, compression="gzip", compression_opts=4,
                                    chunks=(1, N_CELLS, N_CELLS))
            ds.attrs["h0_sha1"] = __import__("hashlib").sha1(h0s[i].tobytes()).hexdigest()
            if n % 25 == 0 or n == len(todo):
                print(f"pyclaw: {n}/{len(todo)} ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"pyclaw: {len(keys)} trajectories in {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
