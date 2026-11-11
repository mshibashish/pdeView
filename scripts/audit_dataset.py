"""Audit the PDEBench SWE dataset for duplicated trajectories (reads metadata of all 1000
samples remotely, ~5 minutes; result cached in reports/dataset_audit.json).

Usage:  uv run python scripts/audit_dataset.py [--refresh]
"""

from __future__ import annotations

import argparse
import logging

from pdeview.data.audit import AUDIT_FILE, load_or_run_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-read all configs remotely")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    if args.refresh and AUDIT_FILE.exists():
        AUDIT_FILE.unlink()
    a = load_or_run_audit()
    print(f"{a['n_samples']} samples -> {a['n_distinct_trajectories']} distinct trajectories "
          f"(of {a['n_possible_initial_conditions']} possible discrete initial conditions)")
    print(f"train: {a['n_distinct_train']} distinct, test: {a['n_distinct_test']} distinct")
    print(f"test samples with a bit-identical training trajectory: "
          f"{a['test_with_identical_train_trajectory']}/{a['n_test']}")
    print(f"copies per trajectory: {a['copies_per_trajectory']}")


if __name__ == "__main__":
    main()
