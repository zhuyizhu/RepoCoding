from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from agent.agents.session_memory import CognitiveAnchor, MemoryEntry


EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass
class _MemoryRecord:
    vector: np.ndarray
    entry: MemoryEntry


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VectorMemoryBank:
    """Lightweight local vector memory store for CognitiveAnchor retrieval."""

    def __init__(
        self,
        storage_path: str | Path,
        embed_fn: EmbeddingFn,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.embed_fn = embed_fn
        self._records: List[_MemoryRecord] = []
        self._load()

    def add_memory(self, memory_entry: MemoryEntry) -> None:
        """Embed the issue text, then persist vector + MemoryEntry."""
        vec = np.asarray(list(self.embed_fn(memory_entry.issue_text)), dtype=np.float32)
        record = _MemoryRecord(vector=vec, entry=memory_entry)
        self._records.append(record)
        self._persist_append(record)

    def retrieve_top_k(self, current_issue_text: str, k: int = 2) -> List[CognitiveAnchor]:
        """Return top-k CognitiveAnchor by cosine similarity to current issue."""
        if k <= 0 or not self._records:
            return []
        query_vec = np.asarray(list(self.embed_fn(current_issue_text)), dtype=np.float32)
        scored: List[Tuple[float, CognitiveAnchor]] = []
        for rec in self._records:
            score = _cosine_sim(query_vec, rec.vector)
            scored.append((score, rec.entry.anchor))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [anchor for _, anchor in scored[:k]]

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    vec = np.asarray(obj.get("vector", []), dtype=np.float32)
                    entry = MemoryEntry.model_validate(obj.get("entry", {}))
                    self._records.append(_MemoryRecord(vector=vec, entry=entry))
        except Exception:
            self._records = []

    def _persist_append(self, record: _MemoryRecord) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vector": record.vector.tolist(),
            "entry": record.entry.model_dump(),
        }
        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
