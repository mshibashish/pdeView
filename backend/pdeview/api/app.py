"""FastAPI application: JSON + binary-field API and (in production) the built frontend."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pdeview.config import FRONTEND_DIST, REPORTS_DIR
from pdeview.generalization import DamParams
from pdeview.service import ViewerService

log = logging.getLogger("pdeview")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ViewerService()
    app.state.service = service
    highlights = (service.verification or {}).get("highlights", {})
    first = [highlights.get("median")] + service.store.keys("test")[:4]
    threading.Thread(target=service.warm_up, args=([k for k in first if k],), daemon=True).start()
    log.info("pdeView ready: %d samples, device=%s", len(service.store.samples), service.device)
    yield
    service.store.close()


app = FastAPI(title="pdeView API", version="0.1.0", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)


def _svc(request: Request) -> ViewerService:
    return request.app.state.service


def _check_key(svc: ViewerService, key: str) -> None:
    if not svc.has_key(key):
        raise HTTPException(404, f"unknown sample {key!r}")


class SimulateRequest(BaseModel):
    """A custom radial dam break (the outer depth stays 1, as in the training data)."""

    radius: float = Field(0.5, ge=0.05, le=1.5)
    cx: float = Field(0.0, ge=-2.0, le=2.0)
    cy: float = Field(0.0, ge=-2.0, le=2.0)
    h_in: float = Field(2.0, ge=1.05, le=3.0)


@app.post("/api/simulate")
def simulate(req: SimulateRequest, request: Request) -> dict:
    """Solve a custom dam break with the reference solver and roll the FNO out on it."""
    p = DamParams(radius=round(req.radius, 4), cx=round(req.cx, 4), cy=round(req.cy, 4),
                  h_in=round(req.h_in, 4))
    return _svc(request).simulate(p)


@app.get("/api/health")
def health(request: Request) -> dict:
    svc = _svc(request)
    return {"status": "ok", "samples": len(svc.store.samples), "device": svc.device}


@app.get("/api/meta")
def meta(request: Request) -> dict:
    return _svc(request).meta()


@app.get("/api/samples")
def samples(request: Request) -> list[dict]:
    return _svc(request).samples()


@app.get("/api/samples/{key}/stats")
def sample_stats(key: str, request: Request) -> dict:
    svc = _svc(request)
    _check_key(svc, key)
    return svc.stats(key)


@app.get("/api/samples/{key}/field")
def sample_field(
    key: str, request: Request, kind: str = Query("gt", pattern="^(gt|pred)$")
) -> Response:
    """Raw little-endian float32 array of shape ``[T, X, Y]`` (row-major)."""
    svc = _svc(request)
    _check_key(svc, key)
    arr = svc.ground_truth(key) if kind == "gt" else svc.prediction(key)
    body = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "X-Shape": ",".join(map(str, arr.shape)),
            "X-Dtype": "float32",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Expose-Headers": "X-Shape, X-Dtype",
        },
    )


@app.get("/api/verification")
def verification(request: Request) -> JSONResponse:
    v = _svc(request).verification
    if v is None:
        raise HTTPException(404, "run scripts/verify_checkpoint.py first")
    return JSONResponse(v)


@app.get("/api/generalization")
def generalization() -> JSONResponse:
    path = REPORTS_DIR / "generalization.json"
    if not path.exists():
        raise HTTPException(404, "run scripts/generalization_study.py first")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/report/figures/{name}")
def report_figure(name: str) -> FileResponse:
    path = (REPORTS_DIR / "figures" / name).resolve()
    if path.parent != (REPORTS_DIR / "figures").resolve() or not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path)


class FrontendFiles(StaticFiles):
    """Serve the built SPA: revalidate HTML on every load, cache hashed assets forever."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.startswith("assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


if FRONTEND_DIST.exists():
    app.mount("/", FrontendFiles(directory=FRONTEND_DIST, html=True), name="frontend")
