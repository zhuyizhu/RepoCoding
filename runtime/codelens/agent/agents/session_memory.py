"""
agent/agents/session_memory.py

单次 Issue 运行的会话记忆。
- 记录每轮 RetrievalAgent / LocalizeAgent / ReflectAgent 的输入输出。
- 支持 few-shot：导出给下一轮 LocalizeAgent 的 few-shot prompt 块。
- 支持持久化（JSON），供 debug 和跨 session 复用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CognitiveAnchor(BaseModel):
    """Trace distillation anchor used for memory bank alignment and retrieval."""

    issue_signature: str = Field(
        ...,
        description=(
            "Defect fingerprint (~30 tokens). Only keep exception type and core entities; do NOT store stacktrace."
        ),
    )
    navigational_path: str = Field(
        ...,
        description=(
            "Abstract navigation path (~40 tokens). Describe hop transitions; do NOT store code text."
        ),
    )
    reflection_epiphany: str = Field(
        ...,
        description=(
            "Key reflection epiphany (~50 tokens). State 'what was wrong -> how corrected'; no generic filler."
        ),
    )
    verification_heuristic: str = Field(
        ...,
        description=(
            "Final verification heuristic (~20 tokens). Static syntax traits of correct target line only."
        ),
    )


class MemoryEntry(BaseModel):
    """Serialized memory entry for trace distillation and retrieval."""

    issue_text: str
    anchor: CognitiveAnchor


TRACE_DISTILLATION_SYSTEM_PROMPT = """
You are a trace distillation system. You must output a single JSON object that exactly matches the CognitiveAnchor schema.

Rules (MUST FOLLOW):
- Output MUST be JSON only. No markdown, no commentary, no extra keys.
- Use with_structured_output(CognitiveAnchor) or an equivalent strict JSON schema mechanism.

What to store vs. what NOT to store:
- issue_signature: STORE only the exception type and core entities. DO NOT store the full stacktrace.
- navigational_path: STORE only abstract hop transitions. DO NOT store any code text or node source.
- reflection_epiphany: STORE only the reasoning correction ("what was wrong -> how corrected"). DO NOT store generic filler.
- verification_heuristic: STORE only static syntax traits of the correct target line. DO NOT store dynamic behavior or full code.
""".strip()


# ───────────────────────────────────────────────
# 数据结构：每一轮的快照
# ───────────────────────────────────────────────

@dataclass
class RoundSnapshot:
    """一轮完整流程的快照：retrieval → localize → reflect。"""
    round_idx: int

    # RetrievalAgent
    subgraph_key: str = ""           # 子图的唯一 key（用于「同粒度不再重复生成」判断）
    retrieval_granularity: str = ""  # file | function | line
    graph_context: str = ""          # 传给 LocalizeAgent 的完整上下文
    retrieval_meta: Dict[str, Any] = field(default_factory=dict)  # 节点数、边数等统计

    # LocalizeAgent
    localize_prompt_used: str = ""   # 本轮完整 prompt（含 retry block）
    localize_output: str = ""        # LLM 原始输出
    retry_block: str = ""            # 注入的 retry prompt（第1轮为空）

    # ReflectAgent
    reflect_json: Dict[str, Any] = field(default_factory=dict)  # 结构化 JSON 输出
    scores: Dict[str, Any] = field(default_factory=dict)         # acc@1/3/5 或 SL
    retry_type: str = ""             # 选中的 retry template 类型
    stop_triggered: bool = False     # 是否由在线 Critic 判定提前终止


@dataclass
class SessionMemory:
    """
    整个 Issue 运行期间的会话记忆。
    - 存放所有轮的 RoundSnapshot。
    - 提供 few-shot 导出（最近 K 轮的成功 localize 输出）。
    - 支持持久化到 JSON 文件。
    """
    issue_text: str
    rounds: List[RoundSnapshot] = field(default_factory=list)
    # 每次 RetrievalAgent.retrieve 的审计记录（含完整 graph_context，供 few-shot / 复盘）
    retrieval_steps: List[Dict[str, Any]] = field(default_factory=list)
    # Policy-driven audit trail. Contains no gold labels.
    policy_steps: List[Dict[str, Any]] = field(default_factory=list)
    policy_stop_reason: str = ""

    # ── 追加一轮 ──────────────────────────────────────────────────────────────
    def add_round(self, snapshot: RoundSnapshot) -> None:
        """追加本轮快照，保持 round_idx 连续。"""
        snapshot.round_idx = len(self.rounds)
        self.rounds.append(snapshot)

    def add_retrieval_step(self, step: Dict[str, Any]) -> None:
        """记录一次检索（粒度、key、meta、完整 graph_context）。"""
        self.retrieval_steps.append(dict(step))

    def add_policy_step(self, step: Dict[str, Any]) -> None:
        """Record one belief/action/observation transition for replay and calibration."""
        self.policy_steps.append(dict(step))

    # ── few-shot 导出 ─────────────────────────────────────────────────────────
    def to_few_shot_block(self, top_k: int = 3) -> str:
        """
        导出最近 top_k 轮中得分最高的 localize 输出，
        格式化为注入 LocalizeAgent prompt 的 few-shot 文本块。
        """
        scored: List[tuple[float, RoundSnapshot]] = []
        for snap in self.rounds:
            s = snap.scores
            combined = float(s.get("normalized_score", s.get("critic_total", 0.0)))
            scored.append((combined, snap))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[:top_k]
        if not best:
            return ""

        blocks: List[str] = []
        for rank, (sc, snap) in enumerate(best, 1):
            blocks.append(
                f"[Few-shot #{rank} | 轮次={snap.round_idx} | 联合分={sc:.3f}]\n"
                f"{snap.localize_output[:2000]}\n"
            )
        return (
            "\n====================\nFew-shot 历史记忆（供参考）\n"
            + "\n".join(blocks)
        )

    # ── 「同粒度不再重复生成子图」的历史记录 ───────────────────────────────────
    def seen_subgraph_keys(self) -> set[str]:
        """返回所有已生成过的子图 key（含独立 retrieval 步骤与各轮 snapshot）。"""
        keys: set[str] = set()
        for r in self.rounds:
            if r.subgraph_key:
                keys.add(r.subgraph_key)
        for s in self.retrieval_steps:
            k = s.get("subgraph_key") or ""
            if k:
                keys.add(str(k))
        return keys

    # ── 持久化 ────────────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """序列化为 JSON 文件。"""
        import dataclasses
        out = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path) -> "SessionMemory":
        """从 JSON 文件恢复。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rounds_raw = data.pop("rounds", [])
        data.setdefault("retrieval_steps", [])
        data.setdefault("policy_steps", [])
        data.setdefault("policy_stop_reason", "")
        snap_allowed = {f.name for f in fields(RoundSnapshot)}
        rounds = [
            RoundSnapshot(**{k: v for k, v in r.items() if k in snap_allowed})
            for r in rounds_raw
        ]
        mem_allowed = {f.name for f in fields(SessionMemory)}
        mem = SessionMemory(**{k: v for k, v in data.items() if k in mem_allowed})
        mem.rounds = rounds
        return mem

    # ── 便利属性 ──────────────────────────────────────────────────────────────
    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def latest(self) -> Optional[RoundSnapshot]:
        return self.rounds[-1] if self.rounds else None
