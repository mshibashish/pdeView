import numpy as np
import torch
from conftest import CKPT_PATH, needs_ckpt, needs_data

from pdeview.inference import make_grid, persistence, rollout
from pdeview.model.checkpoint import build_model, load_fno


def test_architecture_matches_pdebench_parameter_count():
    # 465,557 parameters (complex spectral weights counted once), as in PDEBench's FNO2d
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == 465_557


def test_forward_shape():
    model = build_model().eval()
    x = torch.randn(2, 32, 32, 10)
    grid = make_grid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    out = model(x, grid.expand(2, -1, -1, -1))
    assert out.shape == (2, 32, 32, 1, 1)


def test_rollout_keeps_inputs_and_shapes():
    torch.manual_seed(0)
    model = build_model().eval()
    frames = np.random.rand(3, 10, 32, 32).astype(np.float32)
    grid = make_grid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    out = rollout(model, frames, grid, n_steps=14)
    assert out.shape == (3, 14, 32, 32) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :10], frames)
    single = rollout(model, frames[0], grid, n_steps=14)
    np.testing.assert_allclose(single, out[0], atol=1e-6)


def test_persistence_baseline():
    frames = np.random.rand(10, 4, 4).astype(np.float32)
    out = persistence(frames, 15)
    assert out.shape == (15, 4, 4)
    np.testing.assert_array_equal(out[10:], np.repeat(frames[-1:], 5, axis=0))


@needs_ckpt
def test_pretrained_checkpoint_loads_strictly():
    model, info = load_fno(CKPT_PATH)
    assert info.md5_ok
    assert info.n_parameters == 465_557
    assert set(info.keys) >= {"model_state_dict", "epoch", "loss"}


@needs_ckpt
@needs_data
def test_pretrained_rollout_is_accurate_on_a_test_sample():
    from pdeview.data.store import SampleStore
    from pdeview.metrics import aggregate, sample_stats

    store = SampleStore()
    model, _ = load_fno(CKPT_PATH)
    key = store.keys("test")[0]
    gt = store.frames(key)
    pred = rollout(model, gt[:10], make_grid(store.x, store.y), n_steps=gt.shape[0])
    nrmse = aggregate([sample_stats(pred[10:], gt[10:])])["nRMSE"]
    assert nrmse < 1e-2
