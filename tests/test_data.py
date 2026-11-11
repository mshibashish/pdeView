import numpy as np
import pytest
from conftest import needs_data

from pdeview.data.remote import check_sample, subset_keys


def test_split_keys_follow_pdebench_default_loader():
    test, train = subset_keys(1000)
    assert test[0] == "0900" and test[-1] == "0999" and len(test) == 100
    assert train == [f"{i:04d}" for i in range(10)]


def _dam_break(radius, n=128):
    x = (np.arange(n) + 0.5) * 5.0 / n - 2.5
    xx, yy = np.meshgrid(x, x, indexing="ij")
    h0 = np.where(np.hypot(xx, yy) <= radius, 2.0, 1.0)
    data = np.repeat(h0[None, :, :, None], 101, axis=0).astype(np.float32)
    return data, x


def test_check_sample_accepts_consistent_dam_break():
    data, x = _dam_break(0.5)
    check_sample("x", data, x, 0.5)


def test_check_sample_rejects_wrong_radius():
    data, x = _dam_break(0.5)
    with pytest.raises(ValueError, match="radius"):
        check_sample("x", data, x, 0.3)


@needs_data
def test_local_subset_has_expected_layout():
    from pdeview.data.store import SampleStore

    s = SampleStore()
    assert len(s.keys("test")) == 100
    assert s.frames(s.keys("test")[0]).shape == (101, 128, 128)
    assert s.dx == pytest.approx(5 / 128)
    np.testing.assert_allclose(s.t, np.linspace(0, 1, 101), atol=1e-6)
