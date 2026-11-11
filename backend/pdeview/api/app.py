"""FastAPI application: JSON + binary-field API and (in production) the built frontend."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from pdeview.config import FRONTEND_DIST, REPORTS_DIR
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
    if key not in svc.store.samples:
        raise HTTPException(404, f"unknown sample {key!r}")


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
    arr = svc.store.frames(key) if kind == "gt" else svc.prediction(key)
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


@app.get("/api/report/figures/{name}")
def report_figure(name: str) -> FileResponse:
    path = (REPORTS_DIR / "figures" / name).resolve()
    if path.parent != (REPORTS_DIR / "figures").resolve() or not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
