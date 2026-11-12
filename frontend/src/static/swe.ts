// TypeScript port of backend/pdeview/solver/swe.py for the static site: PyClaw's 2D shallow-water
// scheme (Roe solver with Harten-Hyman entropy fix, MC limiter, dimensional splitting,
// extrapolation boundaries, PyClaw-style CFL control) in float64. It is a line-by-line port, so
// its float32 output is bit-identical to the NumPy solver's.
const NG = 2;

interface Buffers {
  W: Float64Array; // waves [k][p][c]
  S: Float64Array; // speeds [k][p]
  AM: Float64Array; // left-going fluctuations [k][c]
  AP: Float64Array; // right-going fluctuations [k][c]
  F2: Float64Array; // second-order correction fluxes [k][c]
}

function mc(r: number): number {
  if (Number.isNaN(r)) r = 0; // np.nan_to_num
  r = Math.min(Math.max(r, -1e6), 1e6);
  return Math.max(0, Math.min(Math.min((1 + r) / 2, 2), 2 * r));
}

/** One sweep along an axis. h, qn (normal momentum) and qt (transverse momentum) are the state;
 * results go to oh, on, ot. Cell (a, m) lives at a*sa + m*sm (a along the sweep, m across it).
 * Returns the largest Courant number. */
function sweep(
  h: Float64Array, qn: Float64Array, qt: Float64Array,
  oh: Float64Array, on: Float64Array, ot: Float64Array,
  na: number, nm: number, sa: number, sm: number, dtdx: number, g: number, buf: Buffers,
): number {
  const ni = na - 1;
  const { W, S, AM, AP, F2 } = buf;
  let cfl = 0;
  oh.set(h);
  on.set(qn);
  ot.set(qt);
  for (let m = 0; m < nm; m++) {
    // Riemann problems at every interface k (between cells k and k+1)
    for (let k = 0; k < ni; k++) {
      const L = k * sa + m * sm;
      const R = L + sa;
      const hl = h[L], hul = qn[L], hvl = qt[L], hr = h[R], hur = qn[R], hvr = qt[R];
      const sl = Math.sqrt(hl), sr = Math.sqrt(hr);
      const hbar = 0.5 * (hl + hr);
      const u = (hul / sl + hur / sr) / (sl + sr);
      const v = (hvl / sl + hvr / sr) / (sl + sr);
      const a = Math.sqrt(g * hbar);
      const d1 = hr - hl, d2 = hur - hul, d3 = hvr - hvl;
      const a1 = ((u + a) * d1 - d2) / (2 * a);
      const a2 = -v * d1 + d3;
      const a3 = (d2 - (u - a) * d1) / (2 * a);
      const w = k * 9;
      W[w] = a1; W[w + 1] = a1 * (u - a); W[w + 2] = a1 * v;
      W[w + 3] = 0; W[w + 4] = 0; W[w + 5] = a2;
      W[w + 6] = a3; W[w + 7] = a3 * (u + a); W[w + 8] = a3 * v;
      const s0_ = u - a, s1_ = u, s2_ = u + a;
      S[k * 3] = s0_; S[k * 3 + 1] = s1_; S[k * 3 + 2] = s2_;
      // Harten-Hyman entropy fix
      const s0 = hul / hl - Math.sqrt(g * hl);
      const h1 = hl + W[w], hu1 = hul + W[w + 1];
      const s1 = hu1 / h1 - Math.sqrt(g * h1);
      const frac1 = s0 < 0 && s1 > 0 ? (s0 * (s1 - s0_)) / (s1 - s0) : s0_ < 0 ? s0_ : 0;
      const s03 = hur / hr + Math.sqrt(g * hr);
      const h3 = hr - W[w + 6], hu3 = hur - W[w + 7];
      const s3 = hu3 / h3 + Math.sqrt(g * h3);
      const frac3 = s3 < 0 && s03 > 0 ? (s3 * (s03 - s2_)) / (s03 - s3) : s2_ < 0 ? s2_ : 0;
      const sup = s0 > 0 && s0_ > 0;
      for (let c = 0; c < 3; c++) {
        let am = frac1 * W[w + c];
        if (s1_ <= 0) am = am + (s1_ * W[w + 3 + c] + frac3 * W[w + 6 + c]);
        if (sup) am = 0;
        AM[k * 3 + c] = am;
        AP[k * 3 + c] = s0_ * W[w + c] + s1_ * W[w + 3 + c] + s2_ * W[w + 6 + c] - am;
      }
    }
    // limited second-order correction fluxes at interfaces 1 .. ni-2
    for (let k = 1; k < ni - 1; k++) {
      F2[k * 3] = F2[k * 3 + 1] = F2[k * 3 + 2] = 0;
      for (let p = 0; p < 3; p++) {
        const w = (k * 3 + p) * 3, wl = ((k - 1) * 3 + p) * 3, wr = ((k + 1) * 3 + p) * 3;
        let wn = 0, dl = 0, dr = 0;
        for (let c = 0; c < 3; c++) {
          wn += W[w + c] * W[w + c];
          dl += W[w + c] * W[wl + c];
          dr += W[w + c] * W[wr + c];
        }
        const s = S[k * 3 + p];
        const phi = wn > 0 ? mc((s > 0 ? dl : dr) / wn) : 1;
        const as = Math.abs(s);
        const coef = 0.5 * (as * (1 - dtdx * as)) * phi;
        for (let c = 0; c < 3; c++) F2[k * 3 + c] += coef * W[w + c];
        if (k >= NG - 1 && k <= ni - NG) cfl = Math.max(cfl, dtdx * as);
      }
    }
    // update interior cells: left interface i-1, right interface i
    for (let i = NG; i < na - NG; i++) {
      const o = i * sa + m * sm;
      const lft = (i - 1) * 3, rgt = i * 3;
      oh[o] -= dtdx * (AP[lft] + AM[rgt]);
      on[o] -= dtdx * (AP[lft + 1] + AM[rgt + 1]);
      ot[o] -= dtdx * (AP[lft + 2] + AM[rgt + 2]);
      oh[o] -= dtdx * (F2[rgt] - F2[lft]);
      on[o] -= dtdx * (F2[rgt + 1] - F2[lft + 1]);
      ot[o] -= dtdx * (F2[rgt + 2] - F2[lft + 2]);
    }
  }
  return cfl;
}

function extrapolate(q: Float64Array[], nx: number, ny: number): void {
  for (const a of q) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < NG; i++) a[i * ny + j] = a[NG * ny + j];
      for (let i = nx - NG; i < nx; i++) a[i * ny + j] = a[(nx - NG - 1) * ny + j];
    }
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < NG; j++) a[i * ny + j] = a[i * ny + NG];
      for (let j = ny - NG; j < ny; j++) a[i * ny + j] = a[i * ny + ny - NG - 1];
    }
  }
}

/** Solve from depth h0 (float64, [nx, ny] row-major, zero velocity) and return h at every time in
 * tOut as float32 [T, nx, ny]. */
export function simulate(
  h0: Float64Array, nx: number, ny: number, tOut: Float64Array, dx: number, dy: number,
  g = 1, cflDesired = 0.9, cflMax = 1, dt0 = 0.1,
): { h: Float32Array; nSteps: number; nRejected: number } {
  const X = nx + 2 * NG, Y = ny + 2 * NG, n = X * Y;
  const mk = () => [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  let q = mk(), qy = mk();
  const qx = mk();
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) q[0][(i + NG) * Y + j + NG] = h0[i * ny + j];
  const L = Math.max(X, Y);
  const buf: Buffers = {
    W: new Float64Array(L * 9), S: new Float64Array(L * 3), AM: new Float64Array(L * 3),
    AP: new Float64Array(L * 3), F2: new Float64Array(L * 3),
  };
  const out = new Float32Array(tOut.length * nx * ny);
  out.set(h0);
  let t = tOut[0], dt = dt0, nSteps = 0, nRejected = 0;
  for (let k = 1; k < tOut.length; k++) {
    const tend = tOut[k];
    for (;;) {
      if (t + dt > tend) dt = tend - t;
      if (tend - t - dt < 1e-14 * t) dt = tend - t;
      extrapolate(q, X, Y);
      // x-sweep: along i (stride Y), across j (stride 1); normal momentum hu = q[1]
      const c1 = sweep(q[0], q[1], q[2], qx[0], qx[1], qx[2], X, Y, Y, 1, dt / dx, g, buf);
      extrapolate(qx, X, Y);
      // y-sweep: along j (stride 1), across i (stride Y); normal momentum hv = q[2]
      const c2 = sweep(qx[0], qx[2], qx[1], qy[0], qy[2], qy[1], Y, X, 1, Y, dt / dy, g, buf);
      const cfl = Math.max(c1, c2);
      if (cfl <= cflMax) {
        [q, qy] = [qy, q];
        t += dt;
        nSteps++;
      } else nRejected++;
      if (cfl > 0) dt = (dt * cflDesired) / cfl;
      if (t >= tend) break;
    }
    const base = k * nx * ny;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) out[base + i * ny + j] = q[0][(i + NG) * Y + j + NG];
  }
  return { h: out, nSteps, nRejected };
}

/** PyClaw cell centres on [lo, hi]. */
export function cellCentres(n = 128, lo = -2.5, hi = 2.5): Float64Array {
  const d = (hi - lo) / n;
  return Float64Array.from({ length: n }, (_, i) => lo + (i + 0.5) * d);
}

/** Output times identical to np.linspace(0, 1, 101). */
export function outputTimes(n = 101): Float64Array {
  const step = 1 / (n - 1);
  const t = Float64Array.from({ length: n }, (_, i) => i * step);
  t[n - 1] = 1;
  return t;
}

/** Radial dam break: h_in where the cell centre lies within `radius` of (cx, cy), else h_out. */
export function damBreakIC(
  x: Float64Array, y: Float64Array, radius: number, cx = 0, cy = 0, hIn = 2, hOut = 1,
): Float64Array {
  const h = new Float64Array(x.length * y.length);
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j < y.length; j++) {
      const r = Math.sqrt((x[i] - cx) ** 2 + (y[j] - cy) ** 2);
      h[i * y.length + j] = r <= radius ? hIn : hOut;
    }
  }
  return h;
}
