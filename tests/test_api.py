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


def test_custom_simulation_roundtrip(client):
    res = client.post("/api/simulate", json={"radius": 0.5, "cx": 1.0, "cy": 0.0, "h_in": 2.0})
    assert res.status_code == 200
    info = res.json()
    assert info["key"].startswith("c-") and info["ood"] == ["dam not centred at the origin"]
    gt = client.get(f"/api/samples/{info['key']}/field", params={"kind": "gt"})
    a = np.frombuffer(gt.content, dtype="<f4").reshape(101, 128, 128)
    assert a[0].max() == pytest.approx(2.0)
    stats = client.get(f"/api/samples/{info['key']}/stats").json()
    assert stats["summary"]["nRMSE"] > 0.05  # the FNO does not generalise off-centre
    # identical parameters hit the cache
    again = client.post("/api/simulate", json={"radius": 0.5, "cx": 1.0}).json()
    assert again["key"] == info["key"]


def test_custom_simulation_validates_ranges(client):
    assert client.post("/api/simulate", json={"radius": 5}).status_code == 422
    assert client.post("/api/simulate", json={"h_in": 0.5}).status_code == 422


def test_generalization_report(client):
    res = client.get("/api/generalization")
    if res.status_code == 404:
        pytest.skip("generalization study not run")
    rep = res.json()
    assert rep["solver_validation"]["max_abs_error"] < 1e-6
    assert set(rep["families"]) >= {"radius", "centre", "height"}
