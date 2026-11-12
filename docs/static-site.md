# Static site contract (`pages` branch)

The `pages` branch builds a **static** version of pdeView for GitHub Pages at
`https://mshibashish.github.io/pdeView/`.

The `main` branch stays the full app: a FastAPI server with live PyTorch inference, run locally or with Docker. The static site replaces the server as follows:

- **Precomputed:** every server endpoint except `POST /api/simulate` is generated in advance as a file, by `scripts/export_static.py`.
- **In the browser:** the custom dam break runs in the visitor's browser, with a JS port of the solver and the FNO in onnxruntime-web.

This document is the interface between three parts:

- the exporter, `scripts/export_static.py`;
- the frontend in static mode, under `frontend/`;
- the Pages workflow, `.github/workflows/pages.yml`.

## Build

```bash
cd frontend && npm ci && VITE_STATIC=1 VITE_BASE=/pdeView/ npm run build    # -> frontend/dist
cd .. && uv run --group static python scripts/export_static.py --out frontend/dist/static-api
```

- `VITE_STATIC=1` switches the frontend from `/api/...` requests to the static files. `VITE_BASE` is Vite's `base`; it defaults to `/`.
- Vite empties `dist/` on every build, so the export runs **after** the build.
- `--group static` is a uv dependency group holding the export-only packages (onnx, onnxruntime, ...). The rest comes from the normal project dependencies.

## Layout of `static-api/`

| File | Content |
|---|---|
| `manifest.json` | `{format: 1, generated_at, git_sha, n_samples, codecs: {gt, pred}, pred_max_abs_error, model: "model/fno_step.onnx"}` |
| `meta.json` | Same JSON as `GET /api/meta`, plus `"static": {generated_at, git_sha, pred_device, pred_codec, pred_max_abs_error}` |
| `samples.json` | Same as `GET /api/samples` |
| `verification.json`, `heldout.json`, `generalization.json` | Same as `GET /api/verification`, `/api/heldout`, `/api/generalization` |
| `figures/<name>` | Same as `GET /api/report/figures/<name>`, for every file in `reports/figures/` |
| `samples/<key>/stats.json` | Same as `GET /api/samples/<key>/stats`. Computed from the **unquantised** fields. |
| `samples/<key>/gt.bin`, `samples/<key>/pred.bin` | The fields of `GET /api/samples/<key>/field?kind=gt\|pred`, encoded as described below |
| `model/fno_step.onnx` | One FNO step, described below |
| `model/model.json` | `{input: {name, shape, dtype}, output: {...}, initial_step: 10, opset, parity: {sample, steps, max_abs_diff_vs_torch, nrmse_torch, nrmse_onnx}}` |

The keys are the ones in `samples.json` (test, train probe and held-out; about 310 in all). All keys are URL-safe as-is.

## Field encoding (`*.bin`)

A `.bin` file is `gzip(header ‖ payload)`. The browser removes the gzip layer with `DecompressionStream('gzip')`. Any transport compression that Pages adds is removed by the browser before that.

**Header** (40 bytes, little-endian):

| Offset | Type | Field |
|---|---|---|
| 0 | char[4] | magic `"PDVF"` |
| 4 | uint8 | version = 1 |
| 5 | uint8 | codec: 1 = float32 lossless, 2 = uint16 quantised + time-delta |
| 6 | uint16 | reserved = 0 |
| 8 | uint32 | T |
| 12 | uint32 | X |
| 16 | uint32 | Y |
| 20 | uint32 | reserved = 0 |
| 24 | float64 | lo (codec 2; 0 for codec 1) |
| 32 | float64 | scale (codec 2; 0 for codec 1) |

**Payload.** Values are `[T, X, Y]` row-major, the same order as the API's float32 body. Let N = T·X·Y.

- **Codec 1:** the N float32 LE values, byte-shuffled into 4 planes of N bytes each. Plane b holds byte b of every value, in order.
- **Codec 2:**
  - Quantise: `q = round((h - lo) / scale)` as uint16, with `lo = min(h)` and `scale = (max(h) - min(h)) / 65535`. Use `scale = 1` if the field is constant.
  - Delta-code along T: `d[0] = q[0]` and `d[t] = (q[t] - q[t-1]) mod 2^16`.
  - Byte-shuffle the uint16 LE values of `d` into 2 planes.
  - To decode, un-shuffle, take the cumulative sum along T mod 2^16, then `h = lo + q * scale` (as float32).
  - Maximum error is `scale / 2`.

**Defaults.** gt uses codec 1, so it stays exactly the PDEBench / reference data. pred uses codec 2. `--gt-codec u16` exists in case the site needs to shrink.

The whole `static-api/` must stay well under Pages' 1 GB site limit. The expected size is about 500 MB with the defaults.

## FNO step model (`model/fno_step.onnx`)

- **Input** `frames`: float32 `[1, 10, 128, 128]`. These are the last 10 frames, oldest first, in the same `[T, X, Y]` layout as the field data.
- **Output** `next`: float32 `[1, 128, 128]`, the next frame.
- The coordinate grid is baked in as a constant.
- **Rollout:** start from gt frames 0–9, then slide the window over predictions until frame 100, exactly like `pdeview.inference.rollout`.
- **Allowed ops:** only those that onnxruntime-web supports on the wasm backend. `torch.fft` is replaced by the equivalent truncated real DFT as dense matmuls.
- **Accuracy:** the exporter fails if the 91-step rollout differs from PyTorch by more than 1e-4 (max abs) on a test sample.

## Frontend static mode

- `api.ts` maps each call to the files above.
- `simulate()` runs the solver and the FNO rollout in a Web Worker. The resulting fields and stats are kept in memory under key `c-<hash>`, so `api.sample(key)` works for custom runs as it does on the server.
- A banner says this is a static demo and links to the repository for the full app (local or Docker).
- A footer gives the CC BY 4.0 attribution and links to `ATTRIBUTION.md`.
