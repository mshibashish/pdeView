"""Building blocks of the static demo export (``scripts/export_static.py``).

The static demo on GitHub Pages replaces the FastAPI server with precomputed files; the
contract is ``docs/static-site.md``. This module holds the parts that are worth testing on
their own:

* the ``.bin`` field codec (encoder and a reference decoder),
* JSON serialisation that produces the same bytes as the API,
* the FNO step rewritten for ONNX (no FFT, grid baked in) and an onnxruntime rollout.

The full app (live server, PyTorch inference, custom runs on the server) does not need any
of this; it runs from the repository as described in the README's quick start.
"""

from __future__ import annotations

import gzip
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pdeview.model.fno import FNO2d

# ---------------------------------------------------------------------------- field codec
MAGIC = b"PDVF"
VERSION = 1
CODEC_F32 = 1  # float32, lossless, byte-shuffled
CODEC_U16 = 2  # uint16 quantised, delta-coded along T, byte-shuffled
CODEC_NAMES = {"f32": CODEC_F32, "u16": CODEC_U16}
HEADER = struct.Struct("<4sBBHIIIIdd")  # magic, version, codec, 0, T, X, Y, 0, lo, scale
assert HEADER.size == 40
U16_LEVELS = 65535
GZIP_LEVEL = 9


@dataclass(frozen=True)
class FieldHeader:
    codec: int
    shape: tuple[int, int, int]
    lo: float
    scale: float

    @property
    def max_abs_error(self) -> float:
        """Quantisation bound of the codec (0 for the lossless codec)."""
        return self.scale / 2 if self.codec == CODEC_U16 else 0.0


def _shuffle(a: np.ndarray) -> bytes:
    """Byte-shuffle: plane ``b`` holds byte ``b`` of every value, in order."""
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.tobytes()


def _unshuffle(buf: bytes | memoryview, dtype: str, n: int) -> np.ndarray:
    size = np.dtype(dtype).itemsize
    planes = np.frombuffer(buf, dtype=np.uint8, count=n * size).reshape(size, n)
    return np.ascontiguousarray(planes.T).view(dtype).reshape(n)


def quantise(h: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``q = round((h - lo) / scale)`` as uint16, with ``lo = min(h)`` and
    ``scale = (max(h) - min(h)) / 65535`` (1 for a constant field)."""
    h64 = np.asarray(h, dtype=np.float64)
    if not np.isfinite(h64).all():
        raise ValueError("cannot quantise a field with non-finite values")
    lo, hi = float(h64.min()), float(h64.max())
    scale = (hi - lo) / U16_LEVELS if hi > lo else 1.0
    q = np.rint((h64 - lo) / scale)
    return np.clip(q, 0, U16_LEVELS).astype("<u2"), lo, scale


def encode_field(h: np.ndarray, codec: int, level: int = GZIP_LEVEL) -> bytes:
    """Encode a ``[T, X, Y]`` field as ``gzip(header || payload)`` (``docs/static-site.md``)."""
    if h.ndim != 3:
        raise ValueError(f"expected a [T, X, Y] field, got shape {h.shape}")
    t, x, y = h.shape
    if codec == CODEC_F32:
        lo = scale = 0.0
        payload = _shuffle(np.asarray(h, dtype="<f4"))
    elif codec == CODEC_U16:
        q, lo, scale = quantise(h)
        d = q.copy()
        d[1:] = q[1:] - q[:-1]  # uint16 arithmetic: mod 2^16
        payload = _shuffle(d)
    else:
        raise ValueError(f"unknown codec {codec}")
    header = HEADER.pack(MAGIC, VERSION, codec, 0, t, x, y, 0, lo, scale)
    return gzip.compress(header + payload, compresslevel=level, mtime=0)


def parse_header(raw: bytes) -> FieldHeader:
    """Parse (and check) the 40-byte header of a decompressed ``.bin`` file."""
    if len(raw) < HEADER.size:
        raise ValueError("truncated field header")
    magic, version, codec, r0, t, x, y, r1, lo, scale = HEADER.unpack_from(raw)
    if magic != MAGIC or version != VERSION or r0 != 0 or r1 != 0:
        raise ValueError(f"not a version-{VERSION} PDVF field (magic {magic!r}, v{version})")
    if codec not in (CODEC_F32, CODEC_U16):
        raise ValueError(f"unknown codec {codec}")
    return FieldHeader(codec, (t, x, y), lo, scale)


def read_header(path) -> FieldHeader:
    """Header of a ``.bin`` file on disk (decompresses only the first 40 bytes)."""
    with gzip.open(path, "rb") as f:
        return parse_header(f.read(HEADER.size))


def decode_field(blob: bytes) -> tuple[np.ndarray, FieldHeader]:
    """Reference decoder: what the browser does after ``DecompressionStream('gzip')``."""
    raw = gzip.decompress(blob)
    hdr = parse_header(raw)
    n = math.prod(hdr.shape)
    body = memoryview(raw)[HEADER.size:]
    if hdr.codec == CODEC_F32:
        if len(body) != 4 * n:
            raise ValueError("payload size does not match the header")
        return _unshuffle(body, "<f4", n).astype(np.float32).reshape(hdr.shape), hdr
    if len(body) != 2 * n:
        raise ValueError("payload size does not match the header")
    d = _unshuffle(body, "<u2", n).reshape(hdr.shape)
    q = np.cumsum(d, axis=0, dtype=np.uint16)  # wraps mod 2^16
    h = (hdr.lo + q.astype(np.float64) * hdr.scale).astype(np.float32)
    return h, hdr


# ---------------------------------------------------------------------------- JSON
def api_json(obj: Any, response_model: Any = None) -> bytes:
    """Serialise ``obj`` exactly as the FastAPI app does.

    Endpoints that return a plain value with a return annotation (``-> dict``,
    ``-> list[dict]``) are serialised by FastAPI through Pydantic's ``dump_json``; endpoints
    that return a ``JSONResponse`` through Starlette's ``json.dumps``. The two differ in how
    they print small floats (``-3.5e-05`` vs ``-0.000035``), not in the values.
    """
    if response_model is None:
        from starlette.responses import JSONResponse

        return bytes(JSONResponse(obj).body)
    from pydantic import TypeAdapter

    ta = TypeAdapter(response_model)
    return ta.dump_json(ta.validate_python(obj))


# ---------------------------------------------------------------------------- ONNX step
ONNX_OPSET = 17
ONNX_INPUT = "frames"
ONNX_OUTPUT = "next"
# ops the real-DFT step exports to; all run on onnxruntime-web's wasm backend
ONNX_ALLOWED_OPS = frozenset({
    "Add", "Concat", "Constant", "Conv", "Div", "Einsum", "Erf", "MatMul", "Mul", "Neg",
    "Slice", "Squeeze", "Sub", "Transpose",
})


class FNOStep(nn.Module):
    """One FNO step for ONNX: ``frames [1, 10, X, Y]`` (oldest first) -> ``next [1, X, Y]``.

    Same maths as :class:`~pdeview.model.fno.FNO2d` with the coordinate grid baked in. The
    truncated ``rfft2`` / ``irfft2`` of each spectral layer become dense real matmuls with
    the ``2 * modes1 x modes2`` kept modes, so the graph has no FFT and no complex tensors.
    """

    def __init__(self, fno: FNO2d, grid: torch.Tensor):
        super().__init__()
        self.fno = fno
        nx, ny = int(grid.shape[0]), int(grid.shape[1])
        p = fno.padding
        big_x, big_y = nx + p, ny + p
        m1, m2 = fno.modes1, fno.modes2
        k1 = np.r_[np.arange(m1), np.arange(big_x - m1, big_x)]  # rows [:m1] and [-m1:]
        k2 = np.arange(m2)
        ay = 2 * np.pi * np.outer(np.arange(big_y), k2) / big_y  # [Ny, m2]
        ax = 2 * np.pi * np.outer(k1, np.arange(big_x)) / big_x  # [2 m1, Nx]
        twice = np.where(k2 == 0, 1.0, 2.0)  # irfft: the non-DC half-plane modes count twice

        def buf(name: str, a: np.ndarray) -> None:
            self.register_buffer(name, torch.tensor(a, dtype=torch.float32))

        buf("cy", np.cos(ay))
        buf("sy", np.sin(ay))
        buf("cx", np.cos(ax))
        buf("sx", np.sin(ax))
        buf("icx", np.cos(ax).T / big_x)  # [Nx, 2 m1]
        buf("isx", np.sin(ax).T / big_x)
        buf("icy", twice[:, None] * np.cos(ay).T / big_y)  # [m2, Ny]
        buf("isy", twice[:, None] * np.sin(ay).T / big_y)
        for i in range(4):
            conv = getattr(fno, f"conv{i}")
            w = torch.cat([conv.weights1, conv.weights2], dim=2).detach()  # [in, out, 2 m1, m2]
            self.register_buffer(f"wr{i}", w.real.float().contiguous())
            self.register_buffer(f"wi{i}", w.imag.float().contiguous())
        self.register_buffer("grid", grid.detach().float().unsqueeze(0))  # [1, X, Y, 2]
        # F.pad(x, [0, p, 0, p]) as two concatenations with zeros: a static graph
        self.register_buffer("pad_y", torch.zeros(1, fno.width, nx, p))
        self.register_buffer("pad_x", torch.zeros(1, fno.width, p, big_y))

    def _spectral(self, x: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        ar = x @ self.cy  # rfft along y, kept modes: [b, c, Nx, m2]
        ai = -(x @ self.sy)
        xr = self.cx @ ar + self.sx @ ai  # fft along x, kept rows: [b, c, 2 m1, m2]
        xi = self.cx @ ai - self.sx @ ar
        yr = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
        yi = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
        br = self.icx @ yr - self.isx @ yi  # inverse fft along x: [b, o, Nx, m2]
        bi = self.icx @ yi + self.isx @ yr
        return br @ self.icy - bi @ self.isy  # inverse rfft along y: [b, o, Nx, Ny]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        f = self.fno
        x = torch.cat((frames.permute(0, 2, 3, 1), self.grid), dim=-1)  # [1, X, Y, 10 + 2]
        x = f.fc0(x).permute(0, 3, 1, 2)  # [1, width, X, Y]
        x = torch.cat((torch.cat((x, self.pad_y), dim=3), self.pad_x), dim=2)
        p = f.padding
        for i in range(4):
            spec = self._spectral(x, getattr(self, f"wr{i}"), getattr(self, f"wi{i}"))
            x = spec + getattr(f, f"w{i}")(x)
            if i < 3:
                x = F.gelu(x)
        x = x[..., :-p, :-p].permute(0, 2, 3, 1)
        x = F.gelu(f.fc1(x))
        return f.fc2(x).squeeze(-1)  # [1, X, Y]


def export_onnx(fno: FNO2d, grid: torch.Tensor, path, metadata: dict[str, str] | None = None):
    """Export :class:`FNOStep` to ``path`` and check it. Returns the ``onnx.ModelProto``."""
    import warnings

    import onnx

    step = FNOStep(fno, grid).eval()
    k = fno.initial_step
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    dummy = torch.zeros(1, k, nx, ny)
    with warnings.catch_warnings(), torch.inference_mode(False), torch.no_grad():
        # the TorchScript-based exporter (deprecated, but in the locked torch): opset 17 and
        # no onnxscript dependency. The real-DFT graph it writes was checked in onnxruntime-web.
        warnings.simplefilter("ignore")
        torch.onnx.export(step, (dummy,), str(path), dynamo=False, opset_version=ONNX_OPSET,
                          input_names=[ONNX_INPUT], output_names=[ONNX_OUTPUT],
                          do_constant_folding=True)
    model = onnx.load(str(path))
    # the exporter names the output dims symbolically; the graph is static, so say so (the
    # full check below runs strict shape inference against it)
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    for dim, size in zip(out_dims, (1, nx, ny), strict=True):
        dim.Clear()
        dim.dim_value = size
    for key, value in sorted((metadata or {}).items()):
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model, full_check=True)
    ops = {n.op_type for n in model.graph.node}
    if ops - ONNX_ALLOWED_OPS:
        extra = sorted(ops - ONNX_ALLOWED_OPS)
        raise RuntimeError(f"ONNX graph uses ops outside the allowed set: {extra}")
    io = onnx_io(model)
    want = {"input": (ONNX_INPUT, [1, k, nx, ny]), "output": (ONNX_OUTPUT, [1, nx, ny])}
    for side, (name, shape) in want.items():
        got = io[side]
        if (got["name"], got["shape"], got["dtype"]) != (name, shape, "float32"):
            raise RuntimeError(f"unexpected ONNX {side}: {got}")
    onnx.save(model, str(path))
    return model


def onnx_io(model) -> dict[str, dict]:
    """``{input: {name, shape, dtype}, output: {...}}`` of a single-input, single-output model."""
    import onnx

    def describe(v) -> dict:
        t = v.type.tensor_type
        return {"name": v.name, "shape": [d.dim_value for d in t.shape.dim],
                "dtype": onnx.helper.tensor_dtype_to_np_dtype(t.elem_type).name}

    (inp,), (out,) = model.graph.input, model.graph.output
    return {"input": describe(inp), "output": describe(out)}


def onnx_session(path, threads: int = 1):
    """onnxruntime CPU session. One thread, like onnxruntime-web on a page that is not
    cross-origin isolated (GitHub Pages), which also keeps the results reproducible."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_rollout(session, init_frames: np.ndarray, n_steps: int) -> np.ndarray:
    """``pdeview.inference.rollout`` with the ONNX step: ``[k, X, Y]`` -> ``[n_steps, X, Y]``."""
    window = np.ascontiguousarray(init_frames[None], dtype=np.float32)  # [1, k, X, Y]
    out = np.empty((n_steps, *init_frames.shape[1:]), dtype=np.float32)
    k = init_frames.shape[0]
    out[:k] = init_frames
    for step in range(k, n_steps):
        nxt = session.run([ONNX_OUTPUT], {ONNX_INPUT: window})[0]  # [1, X, Y]
        out[step] = nxt[0]
        window = np.concatenate([window[:, 1:], nxt[:, None]], axis=1)
    return out
