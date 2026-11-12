import numpy as np
import pytest
from conftest import needs_data

from pdeview.generalization import DamParams, initial_condition
from pdeview.solver.swe import cell_centres, dam_break_ic, simulate

X = cell_centres(64)
DX = float(X[1] - X[0])
T = np.linspace(0, 0.3, 7)


def test_lake_at_rest_stays_at_rest():
    h, stats = simulate(np.full((64, 64), 1.3), T, DX, DX)
    np.testing.assert_allclose(h, 1.3, atol=1e-14)
    assert stats.n_rejected == 0


def test_volume_is_conserved_before_waves_reach_the_boundary():
    h0 = dam_break_ic(X, X, 0.5)
    h, _ = simulate(h0, T, DX, DX)
    vol = h.sum(axis=(1, 2))
    np.testing.assert_allclose(vol, vol[0], rtol=1e-12)


def test_dam_break_spreads_and_respects_cfl():
    h, stats = simulate(dam_break_ic(X, X, 0.5), T, DX, DX)
    assert h[-1].max() < 2.0 and h[-1].min() < 1.0  # collapse + central rarefaction
    assert stats.max_cfl <= 1.0


def test_dry_states_are_rejected():
    with pytest.raises(ValueError):
        simulate(np.zeros((8, 8)), T, DX, DX)


def test_ood_reasons():
    assert DamParams().ood_reasons() == []
    reasons = DamParams(radius=1.0, cx=0.5, h_in=3.0).ood_reasons()
    assert len(reasons) == 3
    two = initial_condition(DamParams(radius=0.3, cx=-1, second=(1.0, 0.0, 0.3)))
    assert (two == 2).sum() == 2 * (dam_break_ic(cell_centres(), cell_centres(), 0.3) == 2).sum()


@needs_data
def test_reproduces_pyclaw_ground_truth():
    from pdeview.data.store import SampleStore

    store = SampleStore()
    key = store.keys("test")[0]
    gt = store.frames(key)
    x = cell_centres()
    h, _ = simulate(dam_break_ic(x, x, store.samples[key].dam_radius),
                    store.t.astype(np.float64), float(x[1] - x[0]), float(x[1] - x[0]))
    assert np.abs(h - gt).max() < 1e-6
