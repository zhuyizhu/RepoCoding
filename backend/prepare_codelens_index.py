"""Prepare a persistent CodeLens vector index for one real Repograph."""
from __future__ import annotations

import argparse
import re

from app.config import settings
from app.engine.codelens import prepare_codelens_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-key", required=True)
    parser.add_argument("--repograph-path", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.repo_key):
        parser.error("--repo-key must be a repository identifier, not a path")
    path = prepare_codelens_index(
        repo_key=args.repo_key,
        repograph_path=args.repograph_path,
        repograph_root=settings.repograph_root,
        codelens_home=settings.codelens_home,
        vectorstore_root=settings.vectorstore_root,
    )
    print(f"CodeLens index ready: {path}", flush=True)


if __name__ == "__main__":
    main()
