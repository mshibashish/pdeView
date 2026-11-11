import numpy as np
import pytest

from pdeview.metrics import aggregate, aggregate_pdebench, sample_stats, time_curves


def _target(nt=5, n=32, seed=0):
    rng = np.random.default_rng(seed)
    return 1.0 + rng.random((nt, n, n))


def test_identical_fields_give_zero_error():
    t = _target()
    m = aggregate([sample_stats(t, t)])
    assert all(v == 0.0 for v in m.values())


def test_constant_offset_has_closed_form_metrics():
    c, n = 0.25, 32
    t = _target(n=n)
    m = aggregate([sample_stats(t + c, t)])
    assert m["RMSE"] == pytest.approx(c)
    assert m["max_error"] == pytest.approx(c)
    assert m["bRMSE"] == pytest.approx(c)
    assert m["cRMSE"] == pytest.approx(c)  # (sum p - sum t)/(nx ny) = c
    expected_nrmse = np.mean(c / np.sqrt(np.mean(t**2, axis=(1, 2))))
    assert m["nRMSE"] == pytest.approx(expected_nrmse)
    # a constant only has a DC component: |F(0,0)| / (nx ny) = c in bin 0, low band = mean of 4 bins
    assert m["fRMSE_low"] == pytest.approx(c / 4)
    assert m["fRMSE_mid"] == pytest.approx(0.0, abs=1e-12)
    assert m["fRMSE_high"] == pytest.approx(0.0, abs=1e-12)


def test_pdebench_aggregation_reproduces_divide_by_last_index():
    t = _target()
    stats = [sample_stats(t + 0.1, t) for _ in range(10)]  # 2 batches of 5
    clean = aggregate(stats)
    compat = aggregate_pdebench(stats, batch_size=5)
    # sum over 2 batches divided by (2 - 1) -> exactly twice the clean mean
    assert compat["RMSE"] == pytest.approx(2 * clean["RMSE"])


def test_time_curves_shape():
    t = _target(nt=7)
    curves = time_curves([sample_stats(t * 1.01, t)])
    assert len(curves["nRMSE"]) == 7
    assert curves["nRMSE"][0] == pytest.approx(0.01, rel=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sample_stats(np.zeros((2, 4, 4)), np.zeros((2, 4, 5)))
