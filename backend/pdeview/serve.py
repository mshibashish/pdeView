"""Production entrypoint: fetch assets if missing (idempotent), then serve API + frontend.

Environment: HOST (default 0.0.0.0), PORT (default 7860), PDEVIEW_DEVICE (default cpu),
PDEVIEW_DATA_DIR / PDEVIEW_CKPT_DIR (asset locations).
"""

from __future__ import annotations

import logging
import os

import uvicorn

from pdeview.data.remote import ensure_assets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    paths = ensure_assets(include_train_probe=True)
    logging.getLogger("pdeview").info("assets ready: %s", paths)
    uvicorn.run(
        "pdeview.api.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
