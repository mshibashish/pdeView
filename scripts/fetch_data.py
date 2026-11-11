"""Fetch the pretrained FNO checkpoint and the shallow-water data subset.

Usage:  uv run python scripts/fetch_data.py [--no-train-probe] [--workers 8]
"""

from __future__ import annotations

import argparse
import logging

import h5py

from pdeview.data.remote import ensure_assets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-train-probe", action="store_true",
                    help="only fetch the test split (skip the 10 training samples)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    paths = ensure_assets(include_train_probe=not args.no_train_probe, workers=args.workers)
    with h5py.File(paths["data"], "r") as f:
        splits = [f[k].attrs["split"] for k in f]
    print(f"checkpoint: {paths['checkpoint']}  (MD5 verified)")
    print(f"data:       {paths['data']}  ({splits.count('test')} test, "
          f"{splits.count('train')} train-probe samples)")


if __name__ == "__main__":
    main()
