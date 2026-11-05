"""Seed an isolated, persistent Git checkout from the read-only input clone."""
from pathlib import Path
import os
import subprocess


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=True, timeout=300)
    return result.stdout.strip()


source = Path("/seed/apache__dubbo")
target = Path("/workspace/repos/apache__dubbo")
expected = os.environ["EXPECTED_REPO_REVISION"]
if not (source / ".git").exists():
    raise SystemExit("Input Dubbo Git checkout is missing.")
source_revision = git("-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD")
if source_revision != expected:
    raise SystemExit(f"Input revision {source_revision} does not match the configured Repograph revision {expected}.")
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    git("-c", f"safe.directory={source}", "clone", "--no-local", "--no-checkout", str(source), str(target))
    git("-C", str(target), "checkout", "--detach", expected)
if not (target / ".git").is_dir():
    raise SystemExit("Persistent checkout exists but is not a Git repository.")
if git("-C", str(target), "rev-parse", "HEAD") != expected:
    raise SystemExit("Persistent checkout revision changed; refuse to use it with this Repograph.")
if git("-C", str(target), "status", "--porcelain"):
    raise SystemExit("Persistent checkout has uncommitted changes.")
print("Seeded clean Dubbo checkout at", expected)
