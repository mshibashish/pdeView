"""Static demo export: the ``.bin`` codec, a small export against ``ViewerService``, and the
ONNX step against PyTorch (``docs/static-site.md``)."""

import gzip
import importlib.util
import json
import logging
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import needs_ckpt, needs_data

from pdeview.inference import make_grid
from pdeview.model.checkpoint import build_model
from pdeview.static_export import (
    CODEC_F32,
    CODEC_U16,
    ONNX_ALLOWED_OPS,
    FNOStep,
    decode_field,
    encode_field,
    quantise,
)

ROOT = Path(__file__).resolve().parents[1]
F32_EPS = float(np.finfo(np.float32).eps)
HAVE_ORT = all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime"))


def raw(blob: bytes) -> bytes:
    return gzip.decompress(blob)


def u16_bound(dec: np.ndarray, scale: float) -> float:
    """scale / 2, plus the float32 rounding of the decoded value."""
    return scale / 2 + float(np.abs(dec).max()) * F32_EPS


# ---------------------------------------------------------------------------- codec
def test_codec1_roundtrip_is_exact():
    rng = np.random.default_rng(0)
    h = rng.normal(1.5, 0.3, (7, 5, 6)).astype(np.float32)
    h[0, 0, :4] = [0.0, -0.0, 1e-40, np.finfo(np.float32).max]  # signed zero, subnormal, max
    dec, hdr = decode_field(encode_field(h, CODEC_F32))
    assert dec.dtype == np.float32 and dec.shape == h.shape
    np.testing.assert_array_equal(dec.view(np.uint32), h.view(np.uint32))  # bit for bit
    assert (hdr.codec, hdr.lo, hdr.scale, hdr.max_abs_error) == (CODEC_F32, 0.0, 0.0, 0.0)


def test_codec2_roundtrip_within_bound():
    rng = np.random.default_rng(1)
    # big jumps between frames, so the time deltas wrap around mod 2^16 in both directions
    h = (1.0 + rng.uniform(0, 2, (9, 1, 1)) + rng.normal(0, 1e-3, (9, 16, 12))).astype(np.float32)
    blob = encode_field(h, CODEC_U16)
    dec, hdr = decode_field(blob)
    q, lo, scale = quantise(h)
    assert (hdr.lo, hdr.scale) == (lo, scale) == (float(h.min()), (h.max() - h.min()) / 65535)
    assert hdr.max_abs_error == scale / 2
    err = np.abs(dec.astype(np.float64) - h)
    assert err.max() <= u16_bound(dec, scale)
    assert err.max() > scale / 4  # it is quantised
    # the decoder recovers q exactly (cumulative sum of the deltas mod 2^16)
    np.testing.assert_array_equal(np.rint((dec.astype(np.float64) - lo) / scale), q)
    assert dec.min() == h.min() and abs(dec.max() - h.max()) <= u16_bound(dec, scale)


def test_codec2_constant_field():
    h = np.full((3, 4, 4), 1.25, dtype=np.float32)
    dec, hdr = decode_field(encode_field(h, CODEC_U16))
    assert (hdr.lo, hdr.scale) == (1.25, 1.0)
    np.testing.assert_array_equal(dec, h)


@pytest.mark.parametrize("codec,size", [(CODEC_F32, 4), (CODEC_U16, 2)])
def test_header_fields(codec, size):
    h = np.linspace(0.5, 2.5, 3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    data = raw(encode_field(h, codec))
    assert len(data) == 40 + size * h.size
    magic, version, c, r0, t, x, y, r1, lo, scale = struct.unpack("<4sBBHIIIIdd", data[:40])
    assert (magic, version, c, r0, r1) == (b"PDVF", 1, codec, 0, 0)
    assert (t, x, y) == (3, 4, 5)
    if codec == CODEC_F32:
        assert (lo, scale) == (0.0, 0.0)
    else:
        assert lo == np.float32(0.5) and scale == pytest.approx(2.0 / 65535)


def test_byte_shuffle_order():
    # float32: 1.0 = 00 00 80 3f, -2.5 = 00 00 20 c0 (little-endian) -> 4 planes of 2 bytes
    h = np.array([1.0, -2.5], dtype=np.float32).reshape(2, 1, 1)
    assert raw(encode_field(h, CODEC_F32))[40:] == bytes.fromhex("0000 0000 8020 3fc0")
    # uint16: q = [0, 65535, 32768] -> d = [0, 65535, 32769 (= -32767 mod 2^16)]
    #         = 0000 ffff 0180 (LE) -> plane 0 = 00 ff 01, plane 1 = 00 ff 80
    h = np.array([0.0, 1.0, 0.5], dtype=np.float32).reshape(3, 1, 1)
    assert raw(encode_field(h, CODEC_U16))[40:] == bytes.fromhex("00ff01 00ff80")


def test_encoding_is_deterministic_and_checked():
    h = np.random.default_rng(2).random((4, 8, 8), dtype=np.float32)
    blob = encode_field(h, CODEC_U16)
    assert blob == encode_field(h, CODEC_U16)
    assert blob[4:8] == b"\0\0\0\0"  # gzip mtime 0
    bad = gzip.compress(b"PDVX" + raw(blob)[4:])
    with pytest.raises(ValueError, match="PDVF"):
        decode_field(bad)
    with pytest.raises(ValueError, match="non-finite"):
        encode_field(np.full((2, 2, 2), np.nan, dtype=np.float32), CODEC_U16)


def test_real_dft_step_matches_fno():
    """The FFT-free rewrite that is exported to ONNX computes the same step as FNO2d."""
    torch.manual_seed(0)
    fno = build_model().eval()
    x = np.linspace(-2.5, 2.5, 40, dtype=np.float32)
    grid = make_grid(x, x[:36])
    frames = 1 + torch.rand(1, 10, 40, 36)
    with torch.no_grad():
        want = fno(frames.permute(0, 2, 3, 1), grid[None])[..., 0, 0]
        got = FNOStep(fno, grid)(frames)
    assert got.shape == (1, 40, 36)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------- export
@pytest.fixture(scope="module")
def export_static():
    spec = importlib.util.spec_from_file_location("export_static",
                                                  ROOT / "scripts" / "export_static.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def svc():
    from pdeview.service import ViewerService

    service = ViewerService()
    yield service
    service.store.close()


@pytest.fixture(scope="module")
def exported(export_static, tmp_path_factory):
    out = tmp_path_factory.mktemp("static-api")
    assert export_static.main(["--out", str(out), "--limit", "2", "--no-model"]) == 0
    return out


def load(path: Path):
    return json.loads(path.read_text())


@needs_ckpt
@needs_data
def test_small_export_matches_service(exported, svc):
    """A second ViewerService, independent of the export's, gives the same JSON and fields
    (exact for gt, within the quantisation bound for pred)."""
    keys = [s["key"] for s in svc.samples()][:2]
    assert load(exported / "samples.json") == [s for s in svc.samples() if s["key"] in keys]
    meta = load(exported / "meta.json")
    static = meta.pop("static")
    assert meta == svc.meta()
    assert load(exported / "verification.json") == svc.verification
    assert load(exported / "heldout.json") == svc.heldout_summary()
    assert load(exported / "generalization.json") == load(ROOT / "reports/generalization.json")
    figures = sorted((ROOT / "reports" / "figures").iterdir())
    assert sorted(p.name for p in (exported / "figures").iterdir()) == [p.name for p in figures]
    assert all((exported / "figures" / p.name).read_bytes() == p.read_bytes() for p in figures)

    errors = []
    for key in keys:
        d = exported / "samples" / key
        assert load(d / "stats.json") == svc.stats(key)
        gt, hdr = decode_field((d / "gt.bin").read_bytes())
        assert hdr.codec == CODEC_F32
        np.testing.assert_array_equal(gt, svc.ground_truth(key))  # exactly the reference data
        pred, hdr = decode_field((d / "pred.bin").read_bytes())
        ref = svc.prediction(key)
        assert hdr.codec == CODEC_U16 and hdr.shape == ref.shape == (101, 128, 128)
        err = float(np.abs(pred.astype(np.float64) - ref).max())
        assert err <= u16_bound(pred, hdr.scale)
        errors.append(err)

    manifest = load(exported / "manifest.json")
    assert set(manifest) == {"format", "generated_at", "git_sha", "n_samples", "codecs",
                             "pred_max_abs_error", "model"}
    assert manifest["format"] == 1 and manifest["n_samples"] == 2
    assert manifest["codecs"] == {"gt": CODEC_F32, "pred": CODEC_U16}
    assert manifest["pred_max_abs_error"] == max(errors) < 1e-4
    assert manifest["model"] is None  # --no-model
    assert static == {"generated_at": manifest["generated_at"], "git_sha": manifest["git_sha"],
                      "pred_device": svc.device, "pred_codec": CODEC_U16,
                      "pred_max_abs_error": manifest["pred_max_abs_error"]}


@needs_ckpt
@needs_data
def test_export_resumes_and_redoes_only_what_changed(exported, export_static, tmp_path,
                                                     monkeypatch, caplog):
    out = tmp_path / "static-api"
    shutil.copytree(exported, out)
    files = sorted((out / "samples").rglob("*.*"))
    before = {p: p.stat().st_mtime_ns for p in files}
    assert export_static.main(["--out", str(out), "--limit", "2", "--no-model"]) == 0
    assert {p: p.stat().st_mtime_ns for p in files} == before  # nothing re-exported

    # another gt codec: only that sample's gt.bin is re-encoded, and the listing follows
    key = load(out / "samples.json")[0]["key"]
    (out / "samples" / key / "stats.json").unlink()
    assert export_static.main(["--out", str(out), "--keys", key, "--gt-codec", "u16",
                               "--no-model"]) == 0
    changed = {p.relative_to(out).as_posix() for p in files if p.stat().st_mtime_ns != before[p]}
    assert changed == {f"samples/{key}/gt.bin", f"samples/{key}/stats.json"}
    assert decode_field((out / "samples" / key / "gt.bin").read_bytes())[1].codec == CODEC_U16
    assert [s["key"] for s in load(out / "samples.json")] == [key]
    assert load(out / "manifest.json")["codecs"]["gt"] == CODEC_U16

    # the size budget fails the run with a clear message
    monkeypatch.setattr(export_static, "SIZE_BUDGET_MB", 1)
    with caplog.at_level(logging.ERROR, logger="export_static"):
        assert export_static.main(["--out", str(out), "--keys", key, "--gt-codec", "u16",
                                   "--no-model"]) == 1
    assert "above the 1 MB budget" in caplog.text


@needs_ckpt
@needs_data
def test_heldout_ground_truth_without_references(export_static, tmp_path, monkeypatch):
    """CI has no data/heldout/references.h5: the export falls back to pdeview's solver."""
    from pdeview.heldout import Case, initial_condition, solve_reference

    monkeypatch.setattr("pdeview.service.REF_FILE", tmp_path / "missing.h5")
    report = load(ROOT / "reports" / "heldout.json")
    p = next(c for c in report["per_case"] if c["family"] == "subcell")
    out = tmp_path / "static-api"
    assert export_static.main(["--out", str(out), "--keys", p["key"], "--no-model"]) == 0
    gt, _ = decode_field((out / "samples" / p["key"] / "gt.bin").read_bytes())
    case = Case(p["key"], p["family"], p["radius"], p["cx"], p["cy"], p["ic"])
    np.testing.assert_array_equal(gt, solve_reference(initial_condition(case)))
    # the report's error was measured against PyClaw; the solver's reference is close to it
    stats = load(out / "samples" / p["key"] / "stats.json")
    assert stats["summary"]["nRMSE"] == pytest.approx(p["nRMSE"], rel=0.05)


@needs_ckpt
@needs_data
@pytest.mark.skipif(not HAVE_ORT, reason="onnx / onnxruntime not installed (uv --group static)")
def test_onnx_step_matches_pytorch(export_static, svc, tmp_path):
    """The exporter's own gate: the 91-step onnxruntime rollout stays within 1e-4 of PyTorch."""
    import onnx

    export_static.export_model(svc, tmp_path, force=True)  # raises SystemExit if not
    info = load(tmp_path / "model" / "model.json")
    assert info["input"] == {"name": "frames", "shape": [1, 10, 128, 128], "dtype": "float32"}
    assert info["output"] == {"name": "next", "shape": [1, 128, 128], "dtype": "float32"}
    assert (info["initial_step"], info["opset"]) == (10, 17)
    par = info["parity"]
    assert par["steps"] == 91 and par["max_abs_diff_vs_torch"] <= 1e-4
    assert par["nrmse_onnx"] == pytest.approx(par["nrmse_torch"], rel=1e-3)
    model = onnx.load(str(tmp_path / "model" / "fno_step.onnx"))
    assert {n.op_type for n in model.graph.node} <= ONNX_ALLOWED_OPS
    assert len(model.graph.input) == 1  # the grid and the weights are constants
