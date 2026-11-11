"""Deploy pdeView to a Hugging Face Space (Docker SDK).

Exports the committed tree (``git archive HEAD``), prepends the Space configuration to the
README and uploads it with ``huggingface_hub`` (which stores binary files such as the report
figures the way the Hub requires). The Space builds the repository's Dockerfile and downloads
the PDEBench assets on start-up.

Usage:  uv run --with huggingface_hub python scripts/deploy_hf_space.py <user>/<space-name>
Requires a Hugging Face login (``hf auth login``) with write access.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

FRONT_MATTER = """---
title: pdeView
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Shallow-water equation viewer with a verified PDEBench FNO
---

"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("space", help="<user-or-org>/<space-name>")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=root)
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True)

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private,
                    exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        readme = Path(tmp) / "README.md"
        readme.write_text(FRONT_MATTER + readme.read_text())
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.space,
            repo_type="space",
            commit_message=f"Deploy pdeView {sha.strip()}",
            delete_patterns=["*"],  # mirror the git tree exactly
        )
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
