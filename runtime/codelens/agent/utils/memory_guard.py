from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    swap_free_bytes: int


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> MemorySnapshot:
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z_()]+):\s+(\d+)\s+kB$", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        pass
    return MemorySnapshot(
        total_bytes=values.get("MemTotal", 0),
        available_bytes=values.get("MemAvailable", values.get("MemFree", 0)),
        swap_free_bytes=values.get("SwapFree", 0),
    )


def resolve_vectorstore_path(template: str, repo_key: str) -> str:
    """Resolve the safe per-repository ``{repo_key}`` path placeholder."""
    safe_repo_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_key).strip("._") or "local_repo"
    return str(template).replace("{repo_key}", safe_repo_key)


def vectorstore_disk_bytes(vectorstore_path: str | Path) -> int:
    root = Path(vectorstore_path)
    total = 0
    for name in ("index.faiss", "index.pkl"):
        try:
            total += (root / name).stat().st_size
        except OSError:
            pass
    return total


def ensure_vectorstore_load_is_safe(
    vectorstore_path: str | Path,
    *,
    snapshot: Optional[MemorySnapshot] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """
    Refuse an existing FAISS load when it would consume WSL's remaining RAM.

    Swap is deliberately excluded from the admission decision.  A previous
    multi-repository index exhausted both 8 GiB RAM and 8 GiB swap, causing the
    whole WSL VM (including VS Code Server) to be OOM-killed.
    """
    env = os.environ if environ is None else environ
    if str(env.get("JLOC_DISABLE_MEMORY_GUARD", "")).strip().lower() in _TRUE_VALUES:
        return

    disk_bytes = vectorstore_disk_bytes(vectorstore_path)
    if disk_bytes <= 0:
        return

    mem = snapshot or _read_meminfo()
    if mem.available_bytes <= 0:
        return

    try:
        multiplier = max(1.0, float(env.get("JLOC_FAISS_LOAD_MULTIPLIER", "1.20")))
    except ValueError:
        multiplier = 1.20
    try:
        reserve_mb = max(512, int(env.get("JLOC_MEMORY_RESERVE_MB", "2048")))
    except ValueError:
        reserve_mb = 2048

    reserve_bytes = reserve_mb * 1024 * 1024
    estimated_load_bytes = int(disk_bytes * multiplier)
    required_available = estimated_load_bytes + reserve_bytes
    if required_available <= mem.available_bytes:
        return

    gib = 1024**3
    raise MemoryError(
        "Unsafe FAISS load refused before model initialization: "
        f"path={Path(vectorstore_path)} disk={disk_bytes / gib:.2f}GiB, "
        f"estimated_load={estimated_load_bytes / gib:.2f}GiB, "
        f"available_ram={mem.available_bytes / gib:.2f}GiB, "
        f"protected_headroom={reserve_bytes / gib:.2f}GiB. "
        "Use a per-repository path such as "
        "data/vectorstore_shards/{repo_key}.faiss. "
        "JLOC_DISABLE_MEMORY_GUARD=1 is an explicit unsafe override."
    )
