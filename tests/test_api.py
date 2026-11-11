import numpy as np
import pytest
from conftest import needs_ckpt, needs_data
from fastapi.testclient import TestClient

pytestmark = [needs_ckpt, needs_data]


@pytest.fixture(scope="module")
def client():
    from pdeview.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert meta["shape"] == [101, 128, 128]
    assert meta["initial_step"] == 10
    assert len(meta["grid"]["x"]) == 128


def test_samples_listing(client):
    samples = client.get("/api/samples").json()
    assert sum(s["split"] == "test" for s in samples) == 100
    assert all(0.3 <= s["dam_radius"] <= 0.7 for s in samples)


def test_binary_fields_roundtrip(client):
    key = "0900"
    gt = client.get(f"/api/samples/{key}/field", params={"kind": "gt"})
    assert gt.status_code == 200 and gt.headers["x-shape"] == "101,128,128"
    a = np.frombuffer(gt.content, dtype="<f4").reshape(101, 128, 128)
    assert a[0].min() == pytest.approx(1.0) and a[0].max() == pytest.approx(2.0)
    pred = client.get(f"/api/samples/{key}/field", params={"kind": "pred"})
    p = np.frombuffer(pred.content, dtype="<f4").reshape(101, 128, 128)
    np.testing.assert_array_equal(p[:10], a[:10])  # inputs are ground truth
    assert np.abs(p[10:] - a[10:]).mean() < 0.01


def test_stats(client):
    s = client.get("/api/samples/0900/stats").json()
    assert len(s["t"]) == 101 and s["nrmse"][:10] == [None] * 10
    assert s["summary"]["nRMSE"] < s["summary"]["persistence_nRMSE"]


def test_errors(client):
    assert client.get("/api/samples/9999/stats").status_code == 404
    assert client.get("/api/samples/0900/field", params={"kind": "x"}).status_code == 422
    assert client.get("/api/report/figures/..%2F..%2Fpyproject.toml").status_code == 404
