"""
agent/agents/pipeline.py

LocalizationPipeline: policy-driven belief search by default, with the former
fixed Retrieval/Localize/Critic/Reflect workflow retained as an ablation.

工作流（无「Phase1/Phase2 粗/细」对外语义；评估仍兼容同一 ``PHASE2_MARKER`` 分隔符）：
────────────────────────────────────────────────────────────────────────
1) **RetrievalAgent** ``granularity=file``：文件级 hybrid 子图 ``ctx_file``。
2) **LocalizeAgent**（仅 ``ctx_file``）：步骤 A 产出候选文件路径。
3) **文件级门控通过后**：以 LocalizeAgent 的候选文件作为 ``force_file_paths``，检索并剪枝文件子图。
4) **函数级子图**：仅在在线候选文件范围内保留 class/function 定义节点，弱化 ref 噪声。
5) **行级子图**：依据 CriticAgent 的质量反馈和检索状态细化上下文。
6) 合并进入 Localize：**优先 function 子图**，否则剪枝后的 file 子图 + 可选 line。
7) 并集解析 ``valid_locations`` 白名单。
8) **LocalizeAgent ↔ ReflectAgent** 循环（最多 ``max_reflect_rounds``）：
   - 每轮：Localize（同一 ``ctx_merged`` + retry prompt 注入）→ Reflect（acc@1/3/5 或 SL + 5 模板 + JSON）→
     写入 **SessionMemory**（完整 ``graph_context``、``localize_prompt_used``、``reflect_json`` 等）。
   - CriticAgent 判定 ``PASS`` 或轮次用尽则结束。
9) 可选：``memory_save_path`` 下持久化 ``SessionMemory`` JSON。

长期 few-shot（向量库中的 issue 示例）由调用方通过 ``few_shots_long_term`` 传入，写入每轮 Localize 的 prompt。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from agent.agents.critic_agent import CriticAgent
from agent.agents.file_verifier_agent import FileVerifierAgent
from agent.agents.localize_agent import LocalizeAgent
from agent.agents.models import GraphContext, GraphNode, LocalizeResult
from agent.agents.reflect_agent import ReflectAgent
from agent.agents.retrieval_agent import RetrievalAgent
from agent.agents.session_memory import (
    CognitiveAnchor,
    MemoryEntry,
    RoundSnapshot,
    SessionMemory,
    TRACE_DISTILLATION_SYSTEM_PROMPT,
)
from agent.memory.memory_bank import VectorMemoryBank
from agent.policy.controller import (
    BeliefPolicyController,
    FixedScheduleController,
    HeuristicController,
    PolicyConfig,
)
from agent.policy.ablation_spec import ABLATION_SPECS, specification_for
from agent.policy.evidence import CalibratedEvidenceModel
from agent.policy.models import (
    ActionType,
    BeliefState,
    ExplorationCost,
    PolicyBudget,
    SearchAction,
    SearchTrajectory,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)

_MERGE_SEP = "\n\n====================\n"


def _prune_graph_blocks_to_files(raw: str, keep_paths: List[str]) -> str:
    """
    按 ``File:`` 块裁剪子图文本，只保留路径与 ``keep_paths``（归一化后）匹配的块，去掉无关文件节点。
    """
    from agent.utils.localization_metrics import match_file, normalize_code_path

    if not (raw or "").strip() or not keep_paths:
        return raw
    candidate_paths = {normalize_code_path(x) for x in keep_paths if x}
    candidate_paths.discard("")
    if not candidate_paths:
        return raw
    parts = re.split(r"(?=\[dist=)", raw)
    kept: List[str] = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        m = re.search(r"^File:\s*(\S+)\s*$", p, re.MULTILINE)
        if not m:
            continue
        fp = normalize_code_path(m.group(1).strip())
        if fp and match_file(fp, candidate_paths):
            kept.append(p)
    return "\n".join(kept) if kept else raw


def _truncate_for_prompt(text: str, max_chars: int, label: str) -> str:
    """
    将单块子图压到 ``max_chars`` 以内：优先保留前段（语义/节点更密），末尾保留一条尾巴以免丢掉结尾结构。
    ``max_chars<=0`` 表示不截断。
    """
    t = (text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    msg_budget = min(180, max(80, max_chars // 25))
    inner = max_chars - msg_budget
    if inner < 800:
        return t[:max_chars] + f"\n…({label} 硬截断至 {max_chars} 字符)\n"
    head = max(400, int(inner * 0.62))
    tail = inner - head
    if tail < 400:
        head = inner - 400
        tail = 400
    out = t[:head] + f"\n…({label}: 中间省略 {len(t) - head - tail} 字符)…\n" + t[-tail:]
    return out + f"\n…({label} 子图已压缩至约 {max_chars} 字符；完整文本在 SessionMemory.retrieval_steps)\n"


def _merge_valid_locations(
    *maps: Dict[str, Set[int]],
) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for m in maps:
        for f, lines in (m or {}).items():
            out.setdefault(f, set()).update(lines)
    return out


def _java_symbol_aliases(value: str) -> Set[str]:
    text = str(value or "").strip().strip("`*_~")
    if not text:
        return set()
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    aliases = set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", text))
    aliases.discard("java")
    return {alias for alias in aliases if len(alias) >= 2}


def _linked_symbol_file_paths(
    result: LocalizeResult,
    graph_context: Optional[GraphContext],
) -> List[str]:
    """Resolve pathless model symbols through unique Java graph definitions.

    This is a conservative fallback: a symbol is committed to a file only
    when all exact symbol matches in the retrieved context point to one path.
    Ambiguous basenames/methods are deliberately ignored.
    """
    if graph_context is None:
        return []
    requested: Set[str] = set()
    for target in [result.target, *list(result.alternatives or [])]:
        if target.file_path:
            continue
        requested.update(_java_symbol_aliases(target.function_name))
        requested.update(_java_symbol_aliases(target.target))
    if not requested:
        return []

    paths_by_alias: Dict[str, Set[str]] = {}
    for node in graph_context.nodes:
        if not node.file_path or not node.file_path.replace("\\", "/").endswith(".java"):
            continue
        aliases: Set[str] = set()
        aliases.update(_java_symbol_aliases(node.symbol_name))
        aliases.update(_java_symbol_aliases(node.qualified_name))
        aliases.update(_java_symbol_aliases(node.node_id))
        for alias in requested & aliases:
            paths_by_alias.setdefault(alias, set()).add(node.file_path)

    resolved: List[str] = []
    for alias in sorted(requested):
        paths = paths_by_alias.get(alias, set())
        if len(paths) == 1:
            resolved.extend(paths)
    return list(dict.fromkeys(resolved))


def _collect_candidate_paths(
    result: LocalizeResult,
    graph_context: Optional[GraphContext] = None,
) -> List[str]:
    paths: List[str] = []

    def _maybe_add(path: str) -> None:
        if path:
            paths.append(path)

    if result and result.target:
        _maybe_add(result.target.file_path or (result.target.target if result.target.target_type == "file" else ""))
        for alt in result.alternatives:
            _maybe_add(alt.file_path or (alt.target if alt.target_type == "file" else ""))

    if not paths and (result.raw_output or "").strip():
        from agent.utils.localization_metrics import extract_predicted_file_paths

        paths = extract_predicted_file_paths(result.raw_output)

    if not paths:
        paths = _linked_symbol_file_paths(result, graph_context)

    return list(dict.fromkeys([p for p in paths if p]))


def _serialize_candidates(result: Optional[LocalizeResult]) -> List[Dict[str, Any]]:
    if not result:
        return []
    out = [result.target.model_dump()]
    out.extend([alt.model_dump() for alt in result.alternatives])
    return out


def _serialize_candidates_for_output(
    result: Optional[LocalizeResult],
    granularity: str,
) -> List[Dict[str, Any]]:
    """Apply verifier set-size advice only to the published file head.

    Search and belief updates retain the complete reordered candidate set.  The
    verifier's ``selected_count`` instead controls the externally reported file
    set, so contextual alternatives do not become asserted edit locations.
    """
    candidates = _serialize_candidates(result)
    if granularity != "file" or result is None:
        return candidates
    verifier = dict((result.metadata or {}).get("file_verifier") or {})
    if verifier.get("status") != "ok":
        return candidates
    try:
        selected_count = int(verifier.get("selected_count"))
    except (TypeError, ValueError):
        return candidates
    if selected_count <= 0:
        return candidates
    return candidates[: min(selected_count, len(candidates))]


def _render_policy_belief(
    belief: BeliefState,
    graph_context: Optional[GraphContext],
    *,
    k_top: int,
    fallback: str,
) -> str:
    """Render the posterior ranking in the evaluator's stable text contract."""
    ranked = belief.ranked(max(1, k_top))
    if not ranked:
        return fallback
    lines: List[str] = []
    nodes_by_id = {
        node.node_id: node for node in (graph_context.nodes if graph_context is not None else [])
    }
    if belief.granularity == "file":
        for key, probability in ranked:
            lines.append(f"候选文件: {key} 理由: calibrated_belief={probability:.6f}")
        return "\n".join(lines)
    if belief.granularity == "function":
        for key, probability in ranked:
            node = nodes_by_id.get(key)
            if node is None:
                continue
            start = node.line_start if node.line_start is not None else 0
            end = node.line_end if node.line_end is not None else start
            line_range = str(start) if start == end else f"{start}-{end}"
            lines.append(
                f"函数: {node.symbol_name or key} 文件: {node.file_path} 行: {line_range} "
                f"理由: calibrated_belief={probability:.6f}"
            )
        return "\n".join(lines) if lines else fallback
    for key, probability in ranked:
        file_path, separator, line_text = key.rpartition(":")
        if not separator or not line_text.isdigit():
            continue
        lines.append(
            f"文件: {file_path} 行: {line_text} 理由: calibrated_belief={probability:.6f}"
        )
    return "\n".join(lines) if lines else fallback


def _build_stub_graph_context(
    text_view: str,
    *,
    repo_key: str,
    granularity: str,
    retrieval_meta: Optional[Dict[str, Any]] = None,
) -> GraphContext:
    files = RetrievalAgent.extract_mentioned_files(text_view)
    nodes: List[GraphNode] = []
    for fp in files:
        node = GraphNode(
            node_id=fp,
            rel_fname=fp,
            symbol_name=os.path.basename(fp),
            granularity="file",
            category="file",
            kind="def",
        )
        nodes.append(node)
    g = granularity if granularity in ("file", "function", "line") else "file"
    return GraphContext(
        repo_key=repo_key,
        granularity=g,
        nodes=nodes,
        text_view=text_view,
        retrieval_meta=retrieval_meta or {},
    )


def _build_reflect_prompts(
    *,
    action: str,
    score: float,
    granularity: str,
    critique: str,
    retry_prompt: str,
    regen_request: Dict[str, Any],
) -> Dict[str, str]:
    prefix = f"[Reflect] action={action} score={score:.2f} granularity={granularity}"
    pass_prompt = (
        f"{prefix}\n"
        "Current hypothesis passes at this granularity. Move to the next finer granularity.\n"
        "Preserve the strongest causal chain and focus on tighter code scope."
    )
    retry_text = retry_prompt.strip() or (
        f"{prefix}\n"
        "Retry localization within the current graph only; choose a different target node.\n"
        f"Critique:\n{critique[:1400]}"
    )
    regen_hint = ""
    if regen_request:
        parts = []
        if regen_request.get("increase_hops"):
            parts.append(f"increase_hops(+{regen_request.get('hop_delta', 0)})")
        if regen_request.get("edge_types"):
            parts.append(f"edge_types={','.join(regen_request.get('edge_types') or [])}")
        if parts:
            regen_hint = "\nGraph regeneration hints: " + "; ".join(parts)
    regen_prompt = (
        f"{prefix}\n"
        "Regenerate the graph with broader coverage, then re-localize.\n"
        f"Critique:\n{critique[:1400]}"
        f"{regen_hint}"
    )
    return {
        "pass_prompt": pass_prompt,
        "retry_prompt": retry_text,
        "regenerate_prompt": regen_prompt,
    }


def _format_memory_examples_xml(anchors: List[CognitiveAnchor]) -> str:
    if not anchors:
        return ""
    blocks: List[str] = ["<Examples>"]
    for idx, a in enumerate(anchors, 1):
        blocks.append(
            "\n".join(
                [
                    f"  <Example index=\"{idx}\">",
                    f"    <IssueSignature>{a.issue_signature}</IssueSignature>",
                    f"    <NavigationalPath>{a.navigational_path}</NavigationalPath>",
                    f"    <ReflectionEpiphany>{a.reflection_epiphany}</ReflectionEpiphany>",
                    f"    <VerificationHeuristic>{a.verification_heuristic}</VerificationHeuristic>",
                    "  </Example>",
                ]
            )
        )
    blocks.append("</Examples>")
    return "\n".join(blocks)


def _distill_trajectory(
    *,
    issue_text: str,
    trajectory_text: str,
    llm_overrides: Optional[Dict[str, Any]] = None,
) -> CognitiveAnchor:
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from agent.tools.localization_llm import _get_llm

    overrides = llm_overrides or {}
    llm = _get_llm(
        ollama_num_ctx_override=overrides.get("ollama_num_ctx"),
        temperature_override=overrides.get("temperature"),
        seed_override=overrides.get("seed"),
    )

    parser = PydanticOutputParser(pydantic_object=CognitiveAnchor)
    prompt = ChatPromptTemplate.from_template(
        """{system_prompt}

{format_instructions}

Issue:
{issue_text}

Trajectory:
{trajectory_text}
"""
    ).format(
        system_prompt=TRACE_DISTILLATION_SYSTEM_PROMPT,
        format_instructions=parser.get_format_instructions(),
        issue_text=issue_text[:8000],
        trajectory_text=trajectory_text[:16000],
    )

    if hasattr(llm, "with_structured_output"):
        structured = llm.with_structured_output(CognitiveAnchor)
        result = structured.invoke(prompt)
        if isinstance(result, CognitiveAnchor):
            return result
        return CognitiveAnchor.model_validate(result)

    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)
    return parser.parse(content)


def _llm_overrides_for_granularity(
    cfg: "PipelineConfig",
    granularity: str,
) -> Optional[Dict[str, Any]]:
    g = granularity if granularity in ("file", "function", "line") else "file"
    overrides: Dict[str, Any] = {}
    temp_map = cfg.llm_temperature_by_granularity or {}
    seed_map = cfg.llm_seed_by_granularity or {}
    ctx_map = cfg.ollama_num_ctx_by_granularity or {}
    if g in temp_map:
        overrides["temperature"] = temp_map[g]
    if g in seed_map:
        overrides["seed"] = seed_map[g]
    if g in ctx_map:
        overrides["ollama_num_ctx"] = ctx_map[g]
    return overrides or None


def _llm_overrides_for_action(
    cfg: "PipelineConfig",
    action: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not action:
        return None
    a = action.upper()
    overrides: Dict[str, Any] = {}
    temp_map = cfg.llm_temperature_by_action or {}
    seed_map = cfg.llm_seed_by_action or {}
    ctx_map = cfg.ollama_num_ctx_by_action or {}
    if a in temp_map:
        overrides["temperature"] = temp_map[a]
    if a in seed_map:
        overrides["seed"] = seed_map[a]
    if a in ctx_map:
        overrides["ollama_num_ctx"] = ctx_map[a]
    return overrides or None


def _merge_llm_overrides(
    base: Optional[Dict[str, Any]],
    action_overrides: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not base and not action_overrides:
        return None
    merged = dict(base or {})
    for k, v in (action_overrides or {}).items():
        if v is not None:
            merged[k] = v
    return merged


def merge_multigranularity_contexts(
    ctx_repo: str,
    ctx_file: str,
    ctx_line: str,
) -> str:
    """完整三段合并（仅调试或显式需要时使用）。"""
    parts = [
        "【子图来源: RetrievalAgent | 粒度=file | 文件级上下文】\n" + (ctx_repo or "").strip(),
        "【子图来源: RetrievalAgent | 粒度=file | 文件强制注入】\n" + (ctx_file or "").strip(),
        "【子图来源: RetrievalAgent | 粒度=function | 函数级收敛】\n" + (ctx_file or "").strip(),
        "【子图来源: RetrievalAgent | 粒度=line | 行级增强（hybrid 仍保留）】\n" + (ctx_line or "").strip(),
    ]
    return _MERGE_SEP.join(p for p in parts if p.strip())


def build_localize_merged_context(
    ctx_repo: str,
    ctx_file: str,
    ctx_line: str,
    *,
    suspected_files_summary: str,
    repo_excerpt_max_chars: int = 3000,
    merged_budget_chars: int = 0,
    file_context_max_chars: int = 0,
    line_context_max_chars: int = 0,
) -> str:
    """
    供 LocalizeAgent 使用的**预算友好**合并上下文：

    - file：只保留前 ``repo_excerpt_max_chars`` 字符作「文件级检索摘要」，避免与 function/line 全文三重叠加。
    - function / line：在通过门控后才会非空；默认对两块分别做 **head+tail 软截断**（环境变量可调），整体约减半。
    - 若 ``merged_budget_chars>0``，对最终字符串做尾部硬截断（最后手段）。
    """
    repo = _truncate_for_prompt((ctx_repo or "").strip(), repo_excerpt_max_chars, "file")

    f_cap = file_context_max_chars or int(os.environ.get("JLOC_FILE_CONTEXT_MAX_CHARS", "110000") or "110000")
    l_cap = line_context_max_chars or int(os.environ.get("JLOC_LINE_CONTEXT_MAX_CHARS", "110000") or "110000")

    header = (
        "【RetrievalAgent | 文件级摘要（截断）】\n"
        + repo
        + "\n"
        + _MERGE_SEP
        + "【步骤 A 候选文件路径（用于 file/function/line 子图门控）】\n"
        + (suspected_files_summary or "(无)").strip()
        + "\n"
    )
    body_parts: List[str] = []
    if (ctx_file or "").strip():
        body_parts.append(
            "【子图: file | 文件级强制注入（hybrid）】\n"
            + _truncate_for_prompt((ctx_file or "").strip(), f_cap, "file")
        )
    if (ctx_line or "").strip():
        body_parts.append(
            "【子图: line | 行级增强（hybrid，较紧 hop）】\n"
            + _truncate_for_prompt((ctx_line or "").strip(), l_cap, "line")
        )
    body = _MERGE_SEP.join(body_parts) if body_parts else ""
    out = header + (_MERGE_SEP + body if body else "")
    budget = merged_budget_chars or int(os.environ.get("JLOC_MERGED_CONTEXT_MAX_CHARS", "0") or "0")
    if budget > 0 and len(out) > budget:
        out = out[:budget] + f"\n…(合并上下文硬截断至 {budget} 字符)\n"
    return out


@dataclass
class PipelineConfig:
    repograph_dir: str
    repo_key: str
    vectorstore_path: str = ""
    embeddings_model: str = ""
    hops: int = 2
    seed_k: int = 120
    max_context_chars: int = 0
    k_top: int = 8
    max_reflect_rounds: int = 2
    memory_save_path: Optional[str] = None
    rebuild_vectorstore: bool = False
    llm_temperature_by_granularity: Optional[Dict[str, float]] = None
    llm_seed_by_granularity: Optional[Dict[str, int]] = None
    ollama_num_ctx_by_granularity: Optional[Dict[str, int]] = None
    llm_temperature_by_action: Optional[Dict[str, float]] = None
    llm_seed_by_action: Optional[Dict[str, int]] = None
    ollama_num_ctx_by_action: Optional[Dict[str, int]] = None
    memory_bank_path: Optional[str] = None
    memory_bank_embed_fn: Optional[Callable[[str], List[float]]] = None
    memory_bank_top_k: int = 2
    policy_driven: bool = True
    policy_max_steps: int = 8
    policy_max_tokens: int = 0
    policy_max_llm_calls: int = 8
    policy_max_wall_time_seconds: float = 0.0
    policy_max_graph_nodes: int = 0
    policy_stop_probability: float = 0.82
    policy_stable_steps: int = 2
    policy_min_action_value: float = 0.015
    policy_target_granularity: Granularity = "line"
    policy_initial_outside_probability: float = 0.20
    policy_expansion_outside_recovery: float = 0.50
    policy_open_world_recovery_gain: float = 0.08
    policy_cost_weight: Optional[float] = None
    evidence_calibration_slope: float = 0.85
    evidence_calibration_intercept: float = -0.15
    file_verifier_enabled: bool = True
    file_verifier_max_candidates: int = 5
    file_verifier_evidence_chars_per_candidate: int = 3000
    file_verifier_seed: int = 20260909
    ablation_variant: str = ""
    source_repository_root: str = ""
    verification_max_tool_calls: int = 2


@dataclass
class PipelineResult:
    instance_id: str
    best_localization_output: str
    best_candidates: List[Dict[str, Any]]
    best_scores: Dict[str, float]
    best_combined_score: float
    stop_triggered: bool
    total_retrieval_rounds: int
    total_localize_rounds: int
    total_reflect_rounds: int
    graph_context_file: str
    graph_context_function: str
    graph_context_line: str
    merged_graph_context: str
    graph_context_history: List[Dict[str, Any]]
    three_agent_trace: List[str]
    session_memory: SessionMemory
    policy_trajectory: Optional[Dict[str, Any]] = None
    ranked_candidates_by_granularity: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )
    verification_history: List[Dict[str, Any]] = field(default_factory=list)


class LocalizationPipeline:
    """Asynchronous policy-driven pipeline; ``run_sync`` is the sync entry point."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        allowed = {
            "",
            "full",
            "single_agent",
            "basic_evidence",
            "closed_world",
            "no_line_candidates",
            "fixed_policy",
            *ABLATION_SPECS,
        }
        if config.ablation_variant not in allowed:
            raise ValueError("Unknown ablation variant: " + config.ablation_variant)
        self.ablation_spec = specification_for(config.ablation_variant)
        from agent.tools.code_context import CodeContextTool
        tool_type = CodeContextTool if config.ablation_variant else RetrievalAgent
        tool_options = {
            "source_repository_root": config.source_repository_root,
            "allow_unconstrained_search": self.ablation_spec.scope_recovery,
        } if config.ablation_variant else {}
        self.retrieval_agent = tool_type(
            **tool_options,
            repograph_dir=config.repograph_dir,
            repo_key=config.repo_key,
            vectorstore_path=config.vectorstore_path,
            embeddings_model=config.embeddings_model,
            hops=config.hops,
            seed_k=config.seed_k,
            max_context_chars=config.max_context_chars,
            rebuild_vectorstore=config.rebuild_vectorstore,
        )
        self.localize_agent = LocalizeAgent(
            max_retries=config.max_reflect_rounds,
            k_top=config.k_top,
        )
        self.context_tool = self.retrieval_agent  # legacy name retained for old callers
        from agent.agents.verification_agent import VerificationAgent
        self.verification_agent = (
            self.localize_agent if config.ablation_variant == "single_agent"
            else VerificationAgent(k_top=config.k_top)
        )
        self.file_verifier_agent = FileVerifierAgent(
            max_candidates=config.file_verifier_max_candidates,
            evidence_chars_per_candidate=config.file_verifier_evidence_chars_per_candidate,
            seed=config.file_verifier_seed,
        )

    def _get_reflect_agent(self, tag_rows: Optional[List[Dict[str, Any]]]) -> ReflectAgent:
        return ReflectAgent(tag_rows=tag_rows or [])

    async def run(
        self,
        instance_id: str,
        issue_text: str,
        tag_rows: Optional[List[Dict[str, Any]]] = None,
        *,
        few_shots_long_term: str = "",
    ) -> PipelineResult:
        cfg = self.config
        tag_rows = tag_rows or []

        # Few-shot filtering may use source-scope metadata, never benchmark labels.
        if few_shots_long_term and few_shots_long_term.strip():
            from agent.tools.issue_memory_tools import filter_and_rank_few_shots

            few_shots_long_term = filter_and_rank_few_shots(
                few_shots_long_term,
                min_src_main_java_ratio=0.5,
            )

        if cfg.policy_driven:
            return await self._run_policy(
                instance_id=instance_id,
                issue_text=issue_text,
                tag_rows=tag_rows,
                few_shots_long_term=few_shots_long_term,
            )

        memory = SessionMemory(issue_text=issue_text)
        critic_agent = CriticAgent()
        reflect_agent = self._get_reflect_agent(tag_rows)
        trace: List[str] = []

        memory_bank: Optional[VectorMemoryBank] = None
        if cfg.memory_bank_path and cfg.memory_bank_embed_fn:
            memory_bank = VectorMemoryBank(
                storage_path=cfg.memory_bank_path,
                embed_fn=cfg.memory_bank_embed_fn,
            )

        n_retrieval = 0
        n_localize = 0
        n_reflect = 0

        best_output = ""
        best_candidates: List[Dict[str, Any]] = []
        best_scores: Dict[str, float] = {}
        best_combined = -1.0
        stop_triggered = False
        best_result: Optional[LocalizeResult] = None
        best_critic_score = -1.0
        prev_action: Optional[str] = None

        # ── R1: file ───────────────────────────────────────────────────────
        trace.append("RetrievalAgent.retrieve(granularity=file)")
        key_r, ctx_r, meta_r = await self.retrieval_agent.retrieve(
            issue_text=issue_text,
            force_file_paths=None,
            seen_subgraph_keys=memory.seen_subgraph_keys(),
            granularity="file",
        )
        n_retrieval += 1
        memory.add_retrieval_step(
            {
                "subgraph_key": key_r,
                "retrieval_granularity": "file",
                "graph_context": ctx_r,
                "meta": dict(meta_r),
                "issue_excerpt": issue_text[:800],
            }
        )

        # ── L1: 文件发现（文件级子图）────────────────────────────────────────
        trace.append("LocalizeAgent.run(scope=file_graph, valid_locations=off)")
        dynamic_examples = ""
        if memory_bank is not None:
            anchors = memory_bank.retrieve_top_k(issue_text, k=cfg.memory_bank_top_k)
            dynamic_examples = _format_memory_examples_xml(anchors)

        dynamic_few_shots = "\n".join([x for x in [dynamic_examples, few_shots_long_term] if x])

        disc_result = await self.localize_agent.run(
            issue_text=issue_text,
            graph_context=ctx_r,
            few_shots=(dynamic_few_shots or "").strip(),
            retry_block="",
            valid_locations=None,
            retry_count=0,
            llm_overrides=_merge_llm_overrides(
                    _llm_overrides_for_granularity(cfg, "file"),
                _llm_overrides_for_action(cfg, prev_action),
            ),
                granularity="file",
        )
        n_localize += 1
        snap_disc = RoundSnapshot(
            round_idx=0,
            subgraph_key=key_r,
            retrieval_granularity="file",
            graph_context=ctx_r,
            retrieval_meta={
                **dict(meta_r),
                "llm_overrides": _merge_llm_overrides(
                    _llm_overrides_for_granularity(cfg, "file"),
                    _llm_overrides_for_action(cfg, prev_action),
                ),
                "prompt_granularity": "file",
            },
            localize_prompt_used=disc_result.prompt_used,
            localize_output=disc_result.raw_output,
            retry_block="",
        )
        file_graph_context = self.retrieval_agent.get_structured_context(key_r)
        if file_graph_context is None:
            # Compatibility fallback for mocked/legacy RetrievalAgent implementations.
            file_graph_context = _build_stub_graph_context(
                ctx_r,
                repo_key=cfg.repo_key,
                granularity="file",
                retrieval_meta=dict(meta_r),
            )
        file_eval = critic_agent.evaluate(
            issue_text=issue_text,
            graph_context=file_graph_context,
            localize_result=disc_result,
        )
        file_context = {"valid_files": len(RetrievalAgent.extract_valid_locations(ctx_r))}
        file_decision = reflect_agent.decide(
            evaluation=file_eval,
            current_granularity="file",
            retry_count=0,
            context=file_context,
        )
        n_reflect += 1
        file_prompts = _build_reflect_prompts(
            action=file_decision.action.value,
            score=file_eval.total_score,
            granularity="file",
            critique=file_eval.critique,
            retry_prompt=file_decision.retry_prompt,
            regen_request=file_decision.graph_regeneration_request,
        )
        snap_disc.reflect_json = {
            "decision": file_decision.model_dump(),
            "critic": file_eval.model_dump(),
            "prompts": file_prompts,
            "llm_overrides": _merge_llm_overrides(
                _llm_overrides_for_granularity(cfg, "file"),
                _llm_overrides_for_action(cfg, prev_action),
            ),
            "prompt_granularity": "file",
            "action_override": prev_action or "",
        }
        snap_disc.scores = {
            "critic_total": file_eval.total_score,
            "semantic_alignment": file_eval.semantic_alignment,
            "topological_reachability": file_eval.topological_reachability,
            "causal_validity": file_eval.causal_validity,
            "normalized_score": file_eval.normalized_score,
        }
        snap_disc.retry_type = file_decision.action.value
        snap_disc.stop_triggered = False
        trace.append(
            f"ReflectAgent(file) decision={file_decision.action.value} score={file_eval.total_score:.3f} retry=0"
        )
        memory.add_round(snap_disc)

        force_paths = _collect_candidate_paths(disc_result, file_graph_context)[:25]
        if not force_paths:
            from agent.utils.localization_metrics import extract_predicted_file_paths

            force_paths = extract_predicted_file_paths(disc_result.raw_output)[:15]

        # ── 文件级门控：无候选路径 → 不再建 file/line 子图（省 token）────────────────
        gate_file_scope = bool(force_paths)
        l1_line_rec = 0.0
        l1_file_rec = 0.0
        l1_func_rec = 0.0
        file_gate = float(os.environ.get("JLOC_FILE_RECALL_FOR_FUNCTION_RETRIEVAL", "0.82") or "0.82")
        func_gate = float(os.environ.get("JLOC_FUNC_RECALL_FOR_LINE_RETRIEVAL", "0.45") or "0.45")

        narrow_paths: List[str] = []
        paths_for_narrow = list(force_paths)

        need_line_subgraph = bool(force_paths)
        line_node_filter = "defs_only" if (l1_func_rec >= func_gate) else "none"

        ctx_f = ""
        ctx_fn_tight = ""
        ctx_l = ""
        key_f = ""
        key_fn = ""
        key_l = ""
        meta_f: Dict[str, Any] = {}
        meta_fn: Dict[str, Any] = {}
        meta_l: Dict[str, Any] = {}

        if gate_file_scope:
            trace.append("RetrievalAgent.retrieve(granularity=file): 文件级门控通过（有候选路径）")
            key_f, ctx_f, meta_f = await self.retrieval_agent.retrieve(
                issue_text=issue_text,
                force_file_paths=force_paths,
                seen_subgraph_keys=memory.seen_subgraph_keys(),
                granularity="file",
            )
            n_retrieval += 1
            memory.add_retrieval_step(
                {
                    "subgraph_key": key_f,
                    "retrieval_granularity": "file",
                    "force_file_paths": list(force_paths),
                    "graph_context": ctx_f,
                    "meta": dict(meta_f),
                    "issue_excerpt": issue_text[:800],
                }
            )
            if (ctx_f or "").strip() and paths_for_narrow:
                ctx_f = _prune_graph_blocks_to_files(ctx_f, paths_for_narrow)

            if paths_for_narrow:
                trace.append(
                    "RetrievalAgent.retrieve(granularity=function): 强制函数子图（按候选路径窄化）"
                )
                key_fn, ctx_fn_tight, meta_fn = await self.retrieval_agent.retrieve(
                    issue_text=issue_text,
                    force_file_paths=paths_for_narrow,
                    seen_subgraph_keys=memory.seen_subgraph_keys(),
                    granularity="function",
                )
                n_retrieval += 1
                memory.add_retrieval_step(
                    {
                        "subgraph_key": key_fn,
                        "retrieval_granularity": "function",
                        "force_file_paths": list(paths_for_narrow),
                        "graph_context": ctx_fn_tight,
                        "meta": dict(meta_fn),
                        "issue_excerpt": issue_text[:800],
                    }
                )

            if need_line_subgraph:
                trace.append(
                    "RetrievalAgent.retrieve(granularity=line): 行级子图（line 未足或函数分达阈需行聚焦）"
                )
                line_paths = paths_for_narrow if paths_for_narrow else force_paths
                key_l, ctx_l, meta_l = await self.retrieval_agent.retrieve(
                    issue_text=issue_text,
                    force_file_paths=line_paths,
                    seen_subgraph_keys=memory.seen_subgraph_keys(),
                    granularity="line",
                    node_context_filter=line_node_filter,
                )
                n_retrieval += 1
                memory.add_retrieval_step(
                    {
                        "subgraph_key": key_l,
                        "retrieval_granularity": "line",
                        "force_file_paths": list(line_paths),
                        "graph_context": ctx_l,
                        "meta": dict(meta_l),
                        "issue_excerpt": issue_text[:800],
                    }
                )
            else:
                trace.append("RetrievalAgent: 跳过 line 子图（line 已足且未触发函数→行聚焦）")
        else:
            trace.append("文件级门控: 无候选路径，跳过 file/line 子图")

        ctx_file_for_merge = ctx_fn_tight if (ctx_fn_tight or "").strip() else ctx_f
        summary_lines = "\n".join(f"- {p}" for p in force_paths) if force_paths else "(无候选路径)"
        excerpt = int(os.environ.get("JLOC_REPO_EXCERPT_CHARS", "3000") or "3000")
        budget = int(os.environ.get("JLOC_MERGED_CONTEXT_MAX_CHARS", "0") or "0")
        ctx_file = build_localize_merged_context(
            ctx_r,
            ctx_f,
            "",
            suspected_files_summary=summary_lines,
            repo_excerpt_max_chars=max(1500, excerpt),
            merged_budget_chars=budget,
        )
        loc_maps_file: List[Dict[str, Set[int]]] = [RetrievalAgent.extract_valid_locations(ctx_r)]
        if (ctx_f or "").strip():
            loc_maps_file.append(RetrievalAgent.extract_valid_locations(ctx_f))
        valid_file = _merge_valid_locations(*loc_maps_file)

        ctx_function = ""
        valid_function: Dict[str, Set[int]] = {}
        if (ctx_fn_tight or "").strip():
            ctx_function = build_localize_merged_context(
                ctx_r,
                ctx_fn_tight,
                "",
                suspected_files_summary=summary_lines,
                repo_excerpt_max_chars=max(1500, excerpt),
                merged_budget_chars=budget,
            )
            loc_maps_fn: List[Dict[str, Set[int]]] = [RetrievalAgent.extract_valid_locations(ctx_r)]
            loc_maps_fn.append(RetrievalAgent.extract_valid_locations(ctx_fn_tight))
            valid_function = _merge_valid_locations(*loc_maps_fn)
        if not (ctx_function or "").strip() and (ctx_f or "").strip():
            # Fallback: function stage still runs using merged file context.
            ctx_function = ctx_file
            valid_function = dict(valid_file)

        ctx_line = build_localize_merged_context(
            ctx_r,
            ctx_file_for_merge,
            ctx_l,
            suspected_files_summary=summary_lines,
            repo_excerpt_max_chars=max(1500, excerpt),
            merged_budget_chars=budget,
        )
        loc_maps_line: List[Dict[str, Set[int]]] = [RetrievalAgent.extract_valid_locations(ctx_r)]
        if (ctx_file_for_merge or "").strip():
            loc_maps_line.append(RetrievalAgent.extract_valid_locations(ctx_file_for_merge))
        if (ctx_l or "").strip():
            loc_maps_line.append(RetrievalAgent.extract_valid_locations(ctx_l))
        valid_line = _merge_valid_locations(*loc_maps_line)

        graph_context_history: List[Dict[str, Any]] = [
            {"pass": 1, "granularity": "file", "subgraph_key": key_r, "chars": len(ctx_r), "text": ctx_r},
        ]
        _pass = 2
        if gate_file_scope and (ctx_f or "").strip():
            graph_context_history.append(
                {
                    "pass": _pass,
                    "granularity": "file",
                    "subgraph_key": key_f,
                    "chars": len(ctx_f),
                    "text": ctx_f,
                    "force_file_paths": force_paths,
                }
            )
            _pass += 1
        if gate_file_scope and (ctx_fn_tight or "").strip():
            graph_context_history.append(
                {
                    "pass": _pass,
                    "granularity": "function",
                    "subgraph_key": key_fn,
                    "chars": len(ctx_fn_tight),
                    "text": ctx_fn_tight,
                    "force_file_paths": paths_for_narrow,
                }
            )
            _pass += 1
        if gate_file_scope and need_line_subgraph and (ctx_l or "").strip():
            graph_context_history.append(
                {
                    "pass": _pass,
                    "granularity": "line",
                    "subgraph_key": key_l,
                    "chars": len(ctx_l),
                    "text": ctx_l,
                    "force_file_paths": paths_for_narrow if paths_for_narrow else force_paths,
                }
            )

        stage_plan: List[Dict[str, Any]] = [
            {"granularity": "file", "context": ctx_file, "valid": valid_file, "available": bool(ctx_f)},
            {"granularity": "function", "context": ctx_function, "valid": valid_function, "available": bool(ctx_function)},
            {"granularity": "line", "context": ctx_line, "valid": valid_line, "available": bool(ctx_l)},
        ]
        stage_map = {s["granularity"]: s for s in stage_plan}
        global_round_idx = 0

        def _stage_subgraph_keys(granularity: str) -> List[str]:
            if granularity == "file":
                return [key_r, key_f]
            if granularity == "function":
                return [key_r, key_f, key_fn]
            return [key_r, key_f, key_fn, key_l]

        # ── 文件级回溯状态机常量 ──────────────────────────────────────────
        FILE_MAX_RETRIES_BEFORE_BACKTRACK = 2   # 连续 2 次 RETRY 后强制 BACKTRACK
        FILE_MAX_BACKTRACK_RETRIES = 2           # backtrack 后最多再 RETRY 2 次
        FILE_MAX_BACKTRACKS = 2                  # 总共最多 2 次 BACKTRACK

        for stage in stage_plan:
            stage_granularity = stage["granularity"]
            if not stage["available"]:
                trace.append(f"LocalizeAgent stage={stage_granularity} skipped (no subgraph)")
                continue

            prev_retry_block = ""
            prev_guidance = ""

            # ── 文件级：状态机回溯循环（while-True + 显式退出）────────────────
            if stage_granularity == "file":
                _file_phase = "initial"           # "initial" | "post_backtrack"
                _file_retry_streak = 0            # 当前阶段连续 RETRY 次数
                _file_backtrack_count = 0          # 累计 BACKTRACK 次数
                _file_round_in_phase = 0           # 当前阶段内轮次（用于日志/快照）

                while True:
                    _file_round_in_phase += 1
                    trace.append(
                        f"LocalizeAgent↔ReflectAgent stage=file phase={_file_phase} "
                        f"round={_file_round_in_phase} retry_streak={_file_retry_streak} "
                        f"backtrack_count={_file_backtrack_count}"
                    )
                    fs = (dynamic_few_shots or "").strip()
                    if _file_round_in_phase > 1:
                        fs = fs + "\n" + memory.to_few_shot_block(top_k=2)

                    localize_overrides = _merge_llm_overrides(
                        _llm_overrides_for_granularity(cfg, "file"),
                        _llm_overrides_for_action(cfg, prev_action),
                    )
                    localize_result = await self.localize_agent.run(
                        issue_text=issue_text,
                        graph_context=stage["context"],
                        few_shots=fs,
                        retry_block=prev_retry_block,
                        valid_locations=stage["valid"],
                        retry_count=_file_round_in_phase - 1,
                        llm_overrides=localize_overrides,
                        granularity="file",
                    )
                    n_localize += 1
                    global_round_idx += 1

                    snap = RoundSnapshot(
                        round_idx=global_round_idx,
                        subgraph_key=key_l or key_fn or key_f or key_r,
                        retrieval_granularity="merged",
                        graph_context=stage["context"],
                        retrieval_meta={
                            "valid_files": len(stage["valid"]),
                            "merged_chars": len(stage["context"]),
                            "stage": "file",
                            "file_phase": _file_phase,
                            "file_backtrack_count": _file_backtrack_count,
                        },
                        localize_prompt_used=localize_result.prompt_used,
                        localize_output=localize_result.raw_output,
                        retry_block=prev_retry_block,
                    )
                    graph_context = self.retrieval_agent.merge_structured_contexts(
                        _stage_subgraph_keys("file"),
                        text_view=stage["context"],
                        granularity="file",
                        retrieval_meta={
                            "valid_files": len(stage["valid"]),
                            "merged_chars": len(stage["context"]),
                            "stage": "file",
                        },
                    )
                    evaluation = critic_agent.evaluate(
                        issue_text=issue_text,
                        graph_context=graph_context,
                        localize_result=localize_result,
                    )
                    # 文件级使用独立 retry_count（_file_retry_streak），与 ReflectAgent 内部 max 解耦
                    decision = reflect_agent.decide(
                        evaluation=evaluation,
                        current_granularity="file",
                        retry_count=_file_retry_streak,
                        context={"valid_files": len(stage["valid"]), "file_phase": _file_phase},
                    )
                    n_reflect += 1

                    prompts = _build_reflect_prompts(
                        action=decision.action.value,
                        score=evaluation.total_score,
                        granularity="file",
                        critique=evaluation.critique,
                        retry_prompt=decision.retry_prompt,
                        regen_request=decision.graph_regeneration_request,
                    )

                    snap.reflect_json = {
                        "decision": decision.model_dump(),
                        "critic": evaluation.model_dump(),
                        "prompts": prompts,
                        "llm_overrides": localize_overrides or {},
                        "prompt_granularity": "file",
                        "action_override": prev_action or "",
                        "file_phase": _file_phase,
                        "file_backtrack_count": _file_backtrack_count,
                    }
                    snap.scores = {
                        "critic_total": evaluation.total_score,
                        "semantic_alignment": evaluation.semantic_alignment,
                        "topological_reachability": evaluation.topological_reachability,
                        "causal_validity": evaluation.causal_validity,
                        "normalized_score": evaluation.normalized_score,
                    }
                    snap.retry_type = decision.action.value
                    snap.stop_triggered = False
                    memory.add_round(snap)

                    if evaluation.total_score > best_critic_score:
                        best_critic_score = evaluation.total_score
                        best_result = localize_result
                        best_scores = dict(snap.scores)
                        best_combined = evaluation.total_score

                    trace.append(
                        f"ReflectAgent(file) decision={decision.action.value} "
                        f"score={evaluation.total_score:.3f} phase={_file_phase} "
                        f"retry_streak={_file_retry_streak} backtrack={_file_backtrack_count}"
                    )

                    # ── PASS：文件级通过 → 构建 function 子图并进入下一粒度 ──
                    if decision.is_pass:
                        from agent.utils.localization_metrics import (
                            extract_phase1_candidate_files,
                            extract_predicted_file_paths,
                        )

                        predicted_files = extract_phase1_candidate_files(localize_result.raw_output)
                        if not predicted_files:
                            predicted_files = extract_predicted_file_paths(localize_result.raw_output)
                        predicted_files = predicted_files[:25]
                        if predicted_files:
                            paths_for_narrow = list(dict.fromkeys(predicted_files))
                            trace.append(
                                "RetrievalAgent.retrieve(granularity=function): 依据文件级预测结果收敛"
                            )
                            key_fn, ctx_fn_tight, meta_fn = await self.retrieval_agent.retrieve(
                                issue_text=issue_text,
                                force_file_paths=paths_for_narrow,
                                seen_subgraph_keys=memory.seen_subgraph_keys(),
                                granularity="function",
                            )
                            n_retrieval += 1
                            memory.add_retrieval_step(
                                {
                                    "subgraph_key": key_fn,
                                    "retrieval_granularity": "function",
                                    "force_file_paths": list(paths_for_narrow),
                                    "graph_context": ctx_fn_tight,
                                    "meta": dict(meta_fn),
                                    "issue_excerpt": issue_text[:800],
                                    "regenerated": True,
                                    "regen_request": {"source": "file_pass"},
                                }
                            )
                            if (ctx_fn_tight or "").strip():
                                ctx_function = build_localize_merged_context(
                                    ctx_r,
                                    ctx_fn_tight,
                                    "",
                                    suspected_files_summary="\n".join(f"- {p}" for p in paths_for_narrow),
                                    repo_excerpt_max_chars=max(1500, excerpt),
                                    merged_budget_chars=budget,
                                )
                                loc_maps_fn: List[Dict[str, Set[int]]] = [
                                    RetrievalAgent.extract_valid_locations(ctx_r),
                                    RetrievalAgent.extract_valid_locations(ctx_fn_tight),
                                ]
                                valid_function = _merge_valid_locations(*loc_maps_fn)
                                stage_map["function"].update(
                                    {
                                        "context": ctx_function,
                                        "valid": valid_function,
                                        "available": bool(ctx_function),
                                    }
                                )
                        trace.append("File-level PASS: 退出文件级回溯循环，进入后续粒度")
                        break

                    # ── RETRY 处理 ──
                    if decision.is_retry:
                        _file_retry_streak += 1
                        if _file_phase == "initial" and _file_retry_streak >= FILE_MAX_RETRIES_BEFORE_BACKTRACK:
                            # 连续 2 次 RETRY → 强制 BACKTRACK，不进入下一粒度
                            trace.append(
                                f"File-level: {FILE_MAX_RETRIES_BEFORE_BACKTRACK} 次连续 RETRY → 强制 BACKTRACK"
                            )
                            _file_phase = "post_backtrack"
                            _file_retry_streak = 0
                            _file_backtrack_count = 1
                            # ── 强制回溯：重新生成文件级子图 ──
                            regen = decision.graph_regeneration_request or {}
                            hop_delta = int(regen.get("hop_delta", 0) or 0) or 2
                            edge_types = list(regen.get("edge_types") or []) or ["invokes", "implements", "extends", "contains"]
                            old_hops = self.retrieval_agent.hops
                            try:
                                self.retrieval_agent.hops = max(1, old_hops + hop_delta)
                                key_r, ctx_r, meta_r = await self.retrieval_agent.retrieve(
                                    issue_text=issue_text,
                                    force_file_paths=None,
                                    seen_subgraph_keys=memory.seen_subgraph_keys(),
                                    granularity="file",
                                    edge_types=edge_types,
                                )
                                n_retrieval += 1
                                memory.add_retrieval_step(
                                    {
                                        "subgraph_key": key_r,
                                        "retrieval_granularity": "file",
                                        "graph_context": ctx_r,
                                        "meta": dict(meta_r),
                                        "issue_excerpt": issue_text[:800],
                                        "regenerated": True,
                                        "regen_request": {"source": "file_forced_backtrack", "hop_delta": hop_delta},
                                    }
                                )
                            finally:
                                self.retrieval_agent.hops = old_hops
                            # 重建文件级 context
                            ctx_file = build_localize_merged_context(
                                ctx_r, "", "",
                                suspected_files_summary=summary_lines,
                                repo_excerpt_max_chars=max(1500, excerpt),
                                merged_budget_chars=budget,
                            )
                            valid_file = RetrievalAgent.extract_valid_locations(ctx_r)
                            stage["context"] = ctx_file
                            stage["valid"] = valid_file
                            stage_map["file"].update(
                                {"context": ctx_file, "valid": valid_file, "available": bool(ctx_r)}
                            )
                            prev_retry_block = (
                                "[BACKTRACK] 文件级子图已重新生成（hops 扩展）。"
                                "请在新的子图上下文中重新定位，优先关注 src/main/java/ 下的核心业务实现。\n"
                                + evaluation.critique[:1200]
                            )
                            prev_action = "REGENERATE"
                            continue
                        elif _file_phase == "post_backtrack" and _file_retry_streak >= FILE_MAX_BACKTRACK_RETRIES:
                            # backtrack 后又 RETRY 达到上限 → 再次 BACKTRACK（若未超总上限）
                            if _file_backtrack_count < FILE_MAX_BACKTRACKS:
                                trace.append(
                                    f"File-level post-backtrack: {FILE_MAX_BACKTRACK_RETRIES} 次 RETRY → "
                                    f"再次 BACKTRACK (#{_file_backtrack_count + 1})"
                                )
                                _file_backtrack_count += 1
                                _file_retry_streak = 0
                                # 再次回溯
                                regen = decision.graph_regeneration_request or {}
                                hop_delta = int(regen.get("hop_delta", 0) or 0) or 2
                                edge_types = list(regen.get("edge_types") or []) or ["invokes", "implements", "extends", "contains"]
                                old_hops = self.retrieval_agent.hops
                                try:
                                    self.retrieval_agent.hops = max(1, old_hops + hop_delta + _file_backtrack_count)
                                    key_r, ctx_r, meta_r = await self.retrieval_agent.retrieve(
                                        issue_text=issue_text,
                                        force_file_paths=None,
                                        seen_subgraph_keys=memory.seen_subgraph_keys(),
                                        granularity="file",
                                        edge_types=edge_types,
                                    )
                                    n_retrieval += 1
                                    memory.add_retrieval_step(
                                        {
                                            "subgraph_key": key_r,
                                            "retrieval_granularity": "file",
                                            "graph_context": ctx_r,
                                            "meta": dict(meta_r),
                                            "issue_excerpt": issue_text[:800],
                                            "regenerated": True,
                                            "regen_request": {"source": "file_backtrack_again", "backtrack": _file_backtrack_count},
                                        }
                                    )
                                finally:
                                    self.retrieval_agent.hops = old_hops
                                ctx_file = build_localize_merged_context(
                                    ctx_r, "", "",
                                    suspected_files_summary=summary_lines,
                                    repo_excerpt_max_chars=max(1500, excerpt),
                                    merged_budget_chars=budget,
                                )
                                valid_file = RetrievalAgent.extract_valid_locations(ctx_r)
                                stage["context"] = ctx_file
                                stage["valid"] = valid_file
                                stage_map["file"].update(
                                    {"context": ctx_file, "valid": valid_file, "available": bool(ctx_r)}
                                )
                                prev_retry_block = (
                                    f"[BACKTRACK #{_file_backtrack_count}] 文件级子图已再次重新生成。"
                                    "请优先在 src/main/java/ 中定位核心实现文件。\n"
                                    + evaluation.critique[:1200]
                                )
                                prev_action = "REGENERATE"
                                continue
                            else:
                                # 所有 backtrack 预算耗尽 → 文件级失败，不进入后续粒度
                                trace.append(
                                    "File-level: backtrack 总次数已耗尽，文件级定位失败，"
                                    "跳过 function/line 粒度"
                                )
                                stage_map["function"]["available"] = False
                                stage_map["line"]["available"] = False
                                break
                        else:
                            # 普通 RETRY（未达上限）
                            prev_retry_block = prompts["retry_prompt"]
                            prev_guidance = evaluation.critique[:1200]
                            prev_action = decision.action.value
                            continue

                    # ── REGENERATE (BACKTRACK) 处理 ──
                    if decision.is_regenerate:
                        if _file_phase == "initial":
                            # 初始阶段直接被 ReflectAgent 判定为 BACKTRACK
                            _file_phase = "post_backtrack"
                            _file_backtrack_count = 1
                            _file_retry_streak = 0
                        else:
                            # post_backtrack 阶段再次收到 BACKTRACK
                            if _file_backtrack_count < FILE_MAX_BACKTRACKS:
                                _file_backtrack_count += 1
                                _file_retry_streak = 0
                            else:
                                trace.append(
                                    "File-level: backtrack 总次数已耗尽，文件级定位失败，"
                                    "跳过 function/line 粒度"
                                )
                                stage_map["function"]["available"] = False
                                stage_map["line"]["available"] = False
                                break

                        regen = decision.graph_regeneration_request or {}
                        regen_granularity = regen.get("granularity") or "file"
                        hop_delta = int(regen.get("hop_delta", 0) or 0) or 2
                        increase_hops = bool(regen.get("increase_hops", True))
                        edge_types = list(regen.get("edge_types") or []) or ["invokes", "implements", "extends", "contains"]
                        trace.append(
                            "ReflectAgent requested file graph regeneration"
                            + (f" (g={regen_granularity}, hop+={hop_delta})" if increase_hops else "")
                            + (f" edge_types={edge_types}" if edge_types else "")
                            + f" backtrack=#{_file_backtrack_count}"
                        )

                        old_hops = self.retrieval_agent.hops
                        try:
                            if increase_hops and hop_delta:
                                self.retrieval_agent.hops = max(1, old_hops + hop_delta)

                            # 文件级回溯：重新生成 file 级子图
                            key_r, ctx_r, meta_r = await self.retrieval_agent.retrieve(
                                issue_text=issue_text,
                                force_file_paths=None,
                                seen_subgraph_keys=memory.seen_subgraph_keys(),
                                granularity="file",
                                edge_types=edge_types,
                            )
                            n_retrieval += 1
                            memory.add_retrieval_step(
                                {
                                    "subgraph_key": key_r,
                                    "retrieval_granularity": "file",
                                    "graph_context": ctx_r,
                                    "meta": dict(meta_r),
                                    "issue_excerpt": issue_text[:800],
                                    "regenerated": True,
                                    "regen_request": dict(regen),
                                }
                            )
                        finally:
                            self.retrieval_agent.hops = old_hops

                        # 重建文件级 context 并更新 stage_plan
                        ctx_file = build_localize_merged_context(
                            ctx_r, "", "",
                            suspected_files_summary=summary_lines,
                            repo_excerpt_max_chars=max(1500, excerpt),
                            merged_budget_chars=budget,
                        )
                        valid_file = RetrievalAgent.extract_valid_locations(ctx_r)
                        stage["context"] = ctx_file
                        stage["valid"] = valid_file
                        stage_map["file"].update(
                            {"context": ctx_file, "valid": valid_file, "available": bool(ctx_r)}
                        )
                        prev_retry_block = prompts["regenerate_prompt"]
                        prev_guidance = evaluation.critique[:1200]
                        prev_action = decision.action.value
                        continue

                    prev_action = decision.action.value

                # ── 文件级 while 循环出口：若 PASS 则继续遍历 stage_plan 中的 function/line ──
                # （若因 backtrack 耗尽而 break，function/line 已被标记 unavailable，会被跳过）
                continue

            # ── 函数级 & 行级：保持原有 for 循环逻辑（不做回溯状态机）────────
            for retry_idx in range(cfg.max_reflect_rounds):
                trace.append(
                    f"LocalizeAgent↔ReflectAgent stage={stage_granularity} round={retry_idx + 1}"
                )
                fs = (dynamic_few_shots or "").strip()
                if retry_idx > 0:
                    fs = fs + "\n" + memory.to_few_shot_block(top_k=2)

                localize_overrides = _merge_llm_overrides(
                    _llm_overrides_for_granularity(cfg, stage_granularity),
                    _llm_overrides_for_action(cfg, prev_action),
                )
                localize_result = await self.localize_agent.run(
                    issue_text=issue_text,
                    graph_context=stage["context"],
                    few_shots=fs,
                    retry_block=prev_retry_block,
                    valid_locations=stage["valid"],
                    retry_count=retry_idx,
                    llm_overrides=localize_overrides,
                    granularity=stage_granularity,
                )
                n_localize += 1
                global_round_idx += 1

                snap = RoundSnapshot(
                    round_idx=global_round_idx,
                    subgraph_key=key_l or key_fn or key_f or key_r,
                    retrieval_granularity="merged",
                    graph_context=stage["context"],
                    retrieval_meta={
                        "valid_files": len(stage["valid"]),
                        "merged_chars": len(stage["context"]),
                        "stage": stage_granularity,
                    },
                    localize_prompt_used=localize_result.prompt_used,
                    localize_output=localize_result.raw_output,
                    retry_block=prev_retry_block,
                )
                current_granularity = stage_granularity
                graph_context = self.retrieval_agent.merge_structured_contexts(
                    _stage_subgraph_keys(current_granularity),
                    text_view=stage["context"],
                    granularity=current_granularity,
                    retrieval_meta={
                        "valid_files": len(stage["valid"]),
                        "merged_chars": len(stage["context"]),
                        "stage": stage_granularity,
                    },
                )
                evaluation = critic_agent.evaluate(
                    issue_text=issue_text,
                    graph_context=graph_context,
                    localize_result=localize_result,
                )
                decision = reflect_agent.decide(
                    evaluation=evaluation,
                    current_granularity=current_granularity,
                    retry_count=retry_idx,
                    context={"valid_files": len(stage["valid"])},
                )
                n_reflect += 1

                prompts = _build_reflect_prompts(
                    action=decision.action.value,
                    score=evaluation.total_score,
                    granularity=current_granularity,
                    critique=evaluation.critique,
                    retry_prompt=decision.retry_prompt,
                    regen_request=decision.graph_regeneration_request,
                )

                snap.reflect_json = {
                    "decision": decision.model_dump(),
                    "critic": evaluation.model_dump(),
                    "prompts": prompts,
                    "llm_overrides": localize_overrides or {},
                    "prompt_granularity": stage_granularity,
                    "action_override": prev_action or "",
                }
                snap.scores = {
                    "critic_total": evaluation.total_score,
                    "semantic_alignment": evaluation.semantic_alignment,
                    "topological_reachability": evaluation.topological_reachability,
                    "causal_validity": evaluation.causal_validity,
                    "normalized_score": evaluation.normalized_score,
                }
                snap.retry_type = decision.action.value
                snap.stop_triggered = False
                memory.add_round(snap)

                if evaluation.total_score > best_critic_score:
                    best_critic_score = evaluation.total_score
                    best_result = localize_result
                    best_scores = dict(snap.scores)
                    best_combined = evaluation.total_score

                trace.append(
                    f"ReflectAgent decision={decision.action.value} score={evaluation.total_score:.3f} retry={decision.retry_count}"
                )

                if decision.is_pass:
                    if current_granularity == "function":
                        from agent.utils.localization_metrics import (
                            extract_phase1_function_candidates,
                            extract_predicted_file_paths,
                        )

                        func_objs = extract_phase1_function_candidates(localize_result.raw_output)
                        predicted_files = [o.get("file_path", "") for o in func_objs if o.get("file_path")]
                        if not predicted_files:
                            predicted_files = extract_predicted_file_paths(localize_result.raw_output)
                        predicted_files = [p for p in predicted_files if p][:25]
                        if predicted_files:
                            trace.append(
                                "RetrievalAgent.retrieve(granularity=line): 依据函数级预测结果收敛"
                            )
                            key_l, ctx_l, meta_l = await self.retrieval_agent.retrieve(
                                issue_text=issue_text,
                                force_file_paths=list(dict.fromkeys(predicted_files)),
                                seen_subgraph_keys=memory.seen_subgraph_keys(),
                                granularity="line",
                                node_context_filter=line_node_filter,
                            )
                            n_retrieval += 1
                            memory.add_retrieval_step(
                                {
                                    "subgraph_key": key_l,
                                    "retrieval_granularity": "line",
                                    "force_file_paths": list(dict.fromkeys(predicted_files)),
                                    "graph_context": ctx_l,
                                    "meta": dict(meta_l),
                                    "issue_excerpt": issue_text[:800],
                                    "regenerated": True,
                                    "regen_request": {"source": "function_pass"},
                                }
                            )
                            ctx_file_for_merge = ctx_fn_tight if (ctx_fn_tight or "").strip() else ctx_f
                            ctx_line = build_localize_merged_context(
                                ctx_r,
                                ctx_file_for_merge,
                                ctx_l,
                                suspected_files_summary="\n".join(
                                    f"- {p}" for p in dict.fromkeys(predicted_files)
                                ),
                                repo_excerpt_max_chars=max(1500, excerpt),
                                merged_budget_chars=budget,
                            )
                            loc_maps_line = [RetrievalAgent.extract_valid_locations(ctx_r)]
                            if (ctx_file_for_merge or "").strip():
                                loc_maps_line.append(
                                    RetrievalAgent.extract_valid_locations(ctx_file_for_merge)
                                )
                            if (ctx_l or "").strip():
                                loc_maps_line.append(RetrievalAgent.extract_valid_locations(ctx_l))
                            valid_line = _merge_valid_locations(*loc_maps_line)
                            stage_map["line"].update(
                                {
                                    "context": ctx_line,
                                    "valid": valid_line,
                                    "available": bool(ctx_l),
                                }
                            )
                    if current_granularity == "line" and memory_bank is not None:
                        trajectory = json.dumps(
                            {
                                "trace": trace,
                                "retrieval_steps": memory.retrieval_steps,
                                "rounds": [
                                    {
                                        "round_idx": r.round_idx,
                                        "retrieval_granularity": r.retrieval_granularity,
                                        "localize_output": r.localize_output,
                                        "reflect_json": r.reflect_json,
                                        "scores": r.scores,
                                        "retry_block": r.retry_block,
                                    }
                                    for r in memory.rounds
                                ],
                            },
                            ensure_ascii=False,
                        )
                        distill_overrides = _merge_llm_overrides(
                            _llm_overrides_for_granularity(cfg, "line"),
                            _llm_overrides_for_action(cfg, "PASS"),
                        )
                        anchor = _distill_trajectory(
                            issue_text=issue_text,
                            trajectory_text=trajectory,
                            llm_overrides=distill_overrides,
                        )
                        print(anchor.model_dump_json(ensure_ascii=False))
                        memory_bank.add_memory(MemoryEntry(issue_text=issue_text, anchor=anchor))
                    if current_granularity == "line":
                        stop_triggered = True
                    break

                if decision.is_retry:
                    prev_retry_block = prompts["retry_prompt"]
                    prev_guidance = evaluation.critique[:1200]
                    prev_action = decision.action.value
                    continue

                if decision.is_regenerate:
                    regen = decision.graph_regeneration_request or {}
                    regen_granularity = regen.get("granularity") or current_granularity
                    hop_delta = int(regen.get("hop_delta", 0) or 0)
                    increase_hops = bool(regen.get("increase_hops"))
                    edge_types = list(regen.get("edge_types") or [])
                    trace.append(
                        "ReflectAgent requested graph regeneration"
                        + (f" (g={regen_granularity}, hop+={hop_delta})" if increase_hops else "")
                        + (f" edge_types={edge_types}" if edge_types else "")
                    )

                    old_hops = self.retrieval_agent.hops
                    try:
                        if increase_hops and hop_delta:
                            self.retrieval_agent.hops = max(1, old_hops + hop_delta)

                        if regen_granularity == "function":
                            key_fn, ctx_fn_tight, meta_fn = await self.retrieval_agent.retrieve(
                                issue_text=issue_text,
                                force_file_paths=paths_for_narrow,
                                seen_subgraph_keys=memory.seen_subgraph_keys(),
                                granularity="function",
                                edge_types=edge_types,
                            )
                            n_retrieval += 1
                            memory.add_retrieval_step(
                                {
                                    "subgraph_key": key_fn,
                                    "retrieval_granularity": "function",
                                    "force_file_paths": list(paths_for_narrow),
                                    "graph_context": ctx_fn_tight,
                                    "meta": dict(meta_fn),
                                    "issue_excerpt": issue_text[:800],
                                    "regenerated": True,
                                    "regen_request": regen,
                                }
                            )
                        elif regen_granularity == "line":
                            line_paths = paths_for_narrow if paths_for_narrow else force_paths
                            key_l, ctx_l, meta_l = await self.retrieval_agent.retrieve(
                                issue_text=issue_text,
                                force_file_paths=line_paths,
                                seen_subgraph_keys=memory.seen_subgraph_keys(),
                                granularity="line",
                                node_context_filter=line_node_filter,
                                edge_types=edge_types,
                            )
                            n_retrieval += 1
                            memory.add_retrieval_step(
                                {
                                    "subgraph_key": key_l,
                                    "retrieval_granularity": "line",
                                    "force_file_paths": list(line_paths),
                                    "graph_context": ctx_l,
                                    "meta": dict(meta_l),
                                    "issue_excerpt": issue_text[:800],
                                    "regenerated": True,
                                    "regen_request": regen,
                                }
                            )
                    finally:
                        self.retrieval_agent.hops = old_hops

                    ctx_file_for_merge = ctx_fn_tight if (ctx_fn_tight or "").strip() else ctx_f
                    ctx_function = ""
                    valid_function = {}
                    if (ctx_fn_tight or "").strip():
                        ctx_function = build_localize_merged_context(
                            ctx_r,
                            ctx_fn_tight,
                            "",
                            suspected_files_summary=summary_lines,
                            repo_excerpt_max_chars=max(1500, excerpt),
                            merged_budget_chars=budget,
                        )
                        loc_maps_fn = [RetrievalAgent.extract_valid_locations(ctx_r)]
                        loc_maps_fn.append(RetrievalAgent.extract_valid_locations(ctx_fn_tight))
                        valid_function = _merge_valid_locations(*loc_maps_fn)

                    ctx_line = build_localize_merged_context(
                        ctx_r,
                        ctx_file_for_merge,
                        ctx_l,
                        suspected_files_summary=summary_lines,
                        repo_excerpt_max_chars=max(1500, excerpt),
                        merged_budget_chars=budget,
                    )
                    loc_maps_line = [RetrievalAgent.extract_valid_locations(ctx_r)]
                    if (ctx_file_for_merge or "").strip():
                        loc_maps_line.append(RetrievalAgent.extract_valid_locations(ctx_file_for_merge))
                    if (ctx_l or "").strip():
                        loc_maps_line.append(RetrievalAgent.extract_valid_locations(ctx_l))
                    valid_line = _merge_valid_locations(*loc_maps_line)
                    for plan in stage_plan:
                        if plan["granularity"] == "function":
                            plan["context"] = ctx_function
                            plan["valid"] = valid_function
                            plan["available"] = bool(ctx_fn_tight)
                        elif plan["granularity"] == "line":
                            plan["context"] = ctx_line
                            plan["valid"] = valid_line
                            plan["available"] = bool(ctx_l)
                    prev_retry_block = prompts["regenerate_prompt"]
                    prev_guidance = evaluation.critique[:1200]
                    prev_action = decision.action.value
                    continue

                prev_action = decision.action.value

        if best_result is not None:
            best_output = best_result.raw_output
            best_candidates = _serialize_candidates(best_result)
        elif (disc_result.raw_output or "").strip():
            best_output = disc_result.raw_output
            best_candidates = _serialize_candidates(disc_result)

        ctx_merged = ctx_line

        if cfg.memory_save_path:
            save_path = Path(cfg.memory_save_path) / f"{instance_id}_memory.json"
            try:
                memory.save(save_path)
                logger.info("[Pipeline] SessionMemory -> %s", save_path)
            except Exception as exc:
                logger.warning("[Pipeline] memory save failed: %s", exc)

        return PipelineResult(
            instance_id=instance_id,
            best_localization_output=best_output,
            best_candidates=best_candidates,
            best_scores=best_scores,
            best_combined_score=best_combined,
            stop_triggered=stop_triggered,
            total_retrieval_rounds=n_retrieval,
            total_localize_rounds=n_localize,
            total_reflect_rounds=n_reflect,
            graph_context_file=ctx_file_for_merge,
            graph_context_function=ctx_function,
            graph_context_line=ctx_l,
            merged_graph_context=ctx_merged,
            graph_context_history=graph_context_history,
            three_agent_trace=trace,
            session_memory=memory,
        )

    async def _run_policy(
        self,
        *,
        instance_id: str,
        issue_text: str,
        tag_rows: List[Dict[str, Any]],
        few_shots_long_term: str,
    ) -> PipelineResult:
        """Run lazy, budgeted belief-space localization.

        Gold labels are deliberately absent from this signature. Every transition is
        recorded so an observation model can later be fitted on training repositories.
        """
        from agent.utils.llm_usage import snapshot_session_llm_usage

        cfg = self.config
        memory = SessionMemory(issue_text=issue_text)
        from agent.tools.evidence_inspector import EvidenceInspector
        from agent.tools.scope_state import ScopeStateTool
        critic_agent = EvidenceInspector(structured=cfg.ablation_variant != "basic_evidence") if cfg.ablation_variant else CriticAgent()
        state_type = ScopeStateTool if cfg.ablation_variant else CalibratedEvidenceModel
        state_options = {
            "open_world": cfg.ablation_variant != "closed_world" and self.ablation_spec.outside_mass,
            "propagate_parent_prior": self.ablation_spec.propagation,
        } if cfg.ablation_variant else {}
        evidence_model = state_type(
            **state_options,
            slope=cfg.evidence_calibration_slope,
            intercept=cfg.evidence_calibration_intercept,
            initial_outside_probability=cfg.policy_initial_outside_probability,
            expansion_outside_recovery=cfg.policy_expansion_outside_recovery,
        )
        budget = PolicyBudget(
            max_tokens=cfg.policy_max_tokens,
            max_llm_calls=cfg.policy_max_llm_calls,
            max_wall_time_seconds=cfg.policy_max_wall_time_seconds,
            max_graph_nodes=cfg.policy_max_graph_nodes,
        )
        effective_cost_weight = (
            float(cfg.policy_cost_weight)
            if cfg.policy_cost_weight is not None
            else (0.0 if cfg.ablation_variant else 0.10)
        )
        controller_type = (
            FixedScheduleController if cfg.ablation_variant == "fixed_policy" else
            HeuristicController if self.ablation_spec.controller == "heuristic" else
            BeliefPolicyController
        )
        controller = controller_type(
            PolicyConfig(
                top_k=min(5, max(1, cfg.k_top)),
                stop_probability=cfg.policy_stop_probability,
                stable_steps_to_stop=cfg.policy_stable_steps,
                min_action_value=cfg.policy_min_action_value,
                target_granularity=cfg.policy_target_granularity,
                max_steps=cfg.policy_max_steps,
                cost_weight=effective_cost_weight,
                open_world_recovery_gain=cfg.policy_open_world_recovery_gain,
                allow_scope_recovery=self.ablation_spec.scope_recovery,
                allow_relocalize=self.ablation_spec.relocalize,
            )
        )
        trajectory = SearchTrajectory(
            instance_id=instance_id,
            budget=budget,
            target_granularity=cfg.policy_target_granularity,
        )
        trace: List[str] = ["PolicyController.start(granularity=file)"]
        graph_context_history: List[Dict[str, Any]] = []
        context_keys: List[str] = []
        raw_contexts: Dict[str, str] = {"file": "", "function": "", "line": ""}
        current_context: Optional[GraphContext] = None
        belief = BeliefState(granularity="file")
        candidate_paths: List[str] = []
        expanded_scopes: Set[str] = set()
        round_counts: Dict[str, int] = {"file": 0, "function": 0, "line": 0}
        previous_observation_candidates: Dict[str, List[str]] = {}
        # Keep the Actor's first grounded ranking at every resolution.  The
        # terminal line posterior is useful for search control, but it is not a
        # statistically valid replacement for the earlier file/function heads.
        # In particular, marginalising a bounded line candidate universe can
        # discard a high-quality file ranking simply because no statement node
        # was retrieved for that file.  These heads let evaluation and clients
        # consume each resolution without cross-granularity information loss.
        ranked_candidates_by_granularity: Dict[str, List[Dict[str, Any]]] = {}
        # A line head must contain agent-nominated source locations, not every
        # method entry inherited from graph retrieval.
        from agent.tools.line_candidates import LineCandidateTool
        line_candidates = (
            LineCandidateTool(self.context_tool._source)
            if cfg.ablation_variant
            and cfg.ablation_variant != "no_line_candidates"
            else None
        )
        verification_history: List[Dict[str, Any]] = []
        file_verifier_attempted = False
        verified_granularities = set()
        last_result = None
        initial_candidates = []
        initial_outputs = {}
        audit = {
            "variant": cfg.ablation_variant,
            "mechanism_switches": vars(self.ablation_spec),
            "initial_candidates": initial_candidates,
            "initial_outputs": initial_outputs,
            "cost_objective_weight": effective_cost_weight,
            "line_candidate_preservation": line_candidates is not None,
            "policy_selector": "fixed_schedule"
            if cfg.ablation_variant == "fixed_policy"
            else "heuristic" if self.ablation_spec.controller == "heuristic" else "adaptive_voi",
        }
        consecutive_backtracks = 0
        n_retrieval = n_localize = n_policy = 0
        best_result: Optional[LocalizeResult] = None
        best_eval = None
        best_utility = -1.0
        stop_reason = ""
        initial_usage = snapshot_session_llm_usage()
        cumulative = ExplorationCost()
        action = SearchAction(
            action_type=ActionType.RETRIEVE,
            granularity="file",
            reason="initialize_file_belief",
            predicted_cost=controller.predicted_cost(ActionType.REFINE, "file"),
        )

        def _usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> ExplorationCost:
            return ExplorationCost(
                prompt_tokens=max(0, int(after.get("prompt_tokens", 0)) - int(before.get("prompt_tokens", 0))),
                completion_tokens=max(0, int(after.get("completion_tokens", 0)) - int(before.get("completion_tokens", 0))),
                llm_calls=max(0, int(after.get("call_count", 0)) - int(before.get("call_count", 0))),
            )

        while True:
            if action.action_type is ActionType.STOP:
                stop_reason = action.reason or "policy_stop"
                n_policy += 1
                trace.append(f"PolicyController.STOP reason={stop_reason}")
                break
            if (
                (action.action_type in {ActionType.EXPAND, ActionType.BACKTRACK}
                 and not self.ablation_spec.scope_recovery)
                or (action.action_type is ActionType.RELOCALIZE
                    and not self.ablation_spec.relocalize)
            ):
                stop_reason = "ablation_action_disabled"
                trace.append(f"PolicyController.STOP reason={stop_reason} action={action.action_type.value}")
                break
            if not budget.can_afford(cumulative, action.predicted_cost):
                stop_reason = "predicted_budget_exhausted"
                trace.append("PolicyController.STOP reason=predicted_budget_exhausted")
                break

            usage_before = snapshot_session_llm_usage()
            wall_start = time.perf_counter()
            retrieved_nodes = 0
            retrieved_key = ""
            retrieved_meta: Dict[str, Any] = {}
            previous_granularity = belief.granularity

            needs_retrieval = action.action_type in {
                ActionType.RETRIEVE,
                ActionType.REFINE,
                ActionType.EXPAND,
                ActionType.BACKTRACK,
            }
            if needs_retrieval:
                force_paths: Optional[List[str]] = None
                hops_override: Optional[int] = None
                edge_types: Optional[List[str]] = None
                if action.action_type is ActionType.REFINE:
                    force_paths = action.scope_paths or candidate_paths
                elif action.action_type is ActionType.EXPAND:
                    force_paths = action.scope_paths or candidate_paths or None
                    hops_override = max(cfg.hops + 1, action.hops)
                    edge_types = action.edge_types
                elif action.action_type is ActionType.BACKTRACK:
                    force_paths = None if action.granularity == "file" else (action.scope_paths or candidate_paths or None)
                    hops_override = max(cfg.hops + 2, action.hops)
                    edge_types = action.edge_types

                retrieved_key, raw_context, retrieved_meta = await self.context_tool.retrieve(
                    issue_text=issue_text,
                    force_file_paths=force_paths,
                    seen_subgraph_keys=memory.seen_subgraph_keys(),
                    granularity=action.granularity,
                    node_context_filter="none",
                    hops_override=hops_override,
                    edge_types=edge_types,
                )
                n_retrieval += 1
                if (
                    action.granularity != "file"
                    and not raw_context.strip()
                    and force_paths
                    and self.ablation_spec.scope_recovery
                ):
                    trace.append(
                        f"Policy retrieval empty g={action.granularity}; retrying once without forced paths"
                    )
                    retrieved_key, raw_context, retrieved_meta = await self.context_tool.retrieve(
                        issue_text=issue_text,
                        force_file_paths=None,
                        seen_subgraph_keys=memory.seen_subgraph_keys(),
                        granularity=action.granularity,
                        node_context_filter="none",
                        hops_override=hops_override,
                        edge_types=edge_types,
                    )
                    n_retrieval += 1
                if action.granularity != "file" and not raw_context.strip():
                    stop_reason = f"empty_{action.granularity}_retrieval"
                    trace.append(f"PolicyController.STOP reason={stop_reason}")
                    break
                context_keys.append(retrieved_key)
                raw_contexts[action.granularity] = raw_context
                retrieved_nodes = int(retrieved_meta.get("node_count", 0) or 0)
                memory.add_retrieval_step(
                    {
                        "subgraph_key": retrieved_key,
                        "retrieval_granularity": action.granularity,
                        "force_file_paths": list(force_paths or []),
                        "graph_context": raw_context,
                        "meta": dict(retrieved_meta),
                        "policy_action": action.model_dump(mode="json"),
                        "issue_excerpt": issue_text[:800],
                    }
                )
                graph_context_history.append(
                    {
                        "pass": len(graph_context_history) + 1,
                        "granularity": action.granularity,
                        "subgraph_key": retrieved_key,
                        "chars": len(raw_context),
                        "text": raw_context,
                        "force_file_paths": list(force_paths or []),
                    }
                )
                if action.action_type in {ActionType.EXPAND, ActionType.BACKTRACK}:
                    expanded_scopes.add(action.granularity)

                summary = "\n".join(f"- {path}" for path in (force_paths or candidate_paths)) or "(无候选路径)"
                merge_budget = cfg.max_context_chars or int(os.environ.get("JLOC_MERGED_CONTEXT_MAX_CHARS", "0") or "0")
                if action.granularity == "file":
                    prompt_context = raw_contexts["file"]
                elif action.granularity == "function":
                    prompt_context = build_localize_merged_context(
                        raw_contexts["file"],
                        raw_contexts["function"],
                        "",
                        suspected_files_summary=summary,
                        merged_budget_chars=merge_budget,
                    )
                else:
                    prompt_context = build_localize_merged_context(
                        raw_contexts["file"],
                        raw_contexts["function"],
                        raw_contexts["line"],
                        suspected_files_summary=summary,
                        merged_budget_chars=merge_budget,
                    )
                current_context = self.context_tool.merge_structured_contexts(
                    context_keys,
                    text_view=prompt_context,
                    granularity=action.granularity,
                    retrieval_meta={"policy_step": len(trajectory.steps), **dict(retrieved_meta)},
                )
                if action.granularity != previous_granularity or not belief.marginals:
                    belief = evidence_model.lift_to_context(belief, issue_text, current_context)
                else:
                    # Same-granularity EXPAND/BACKTRACK may reveal candidates
                    # outside the original belief universe. Reconcile the
                    # posterior with the merged context before observing the
                    # new Localize/Critic result.
                    belief = evidence_model.expand_to_context(
                        belief,
                        issue_text,
                        current_context,
                    )

            if current_context is None:
                stop_reason = "missing_graph_context"
                trace.append("PolicyController.STOP reason=missing_graph_context")
                break

            if cfg.ablation_variant and not trajectory.steps and not initial_candidates:
                candidate_node_ids = set(current_context.candidate_nodes)
                initial_candidates.extend(
                    sorted(
                        {
                            node.file_path
                            for node in current_context.nodes
                            if node.node_id in candidate_node_ids and node.file_path
                        }
                    )
                )
                audit["initial_context_sha256"] = __import__("hashlib").sha256(current_context.text_view.encode()).hexdigest()
            retry_block = ""
            if action.action_type is ActionType.RELOCALIZE:
                retry_block = "Generate a materially different localization hypothesis from the same evidence; avoid repeating the previous ranking."
            elif action.action_type is ActionType.VERIFY:
                retry_block = "Independently verify the leading candidates against concrete source and graph evidence; demote unsupported candidates."
            elif action.action_type in {ActionType.EXPAND, ActionType.BACKTRACK}:
                retry_block = "The search scope changed. Re-evaluate candidates using newly observed nodes and edges, including alternatives outside the previous top set."
            elif action.action_type is ActionType.REFINE:
                retry_block = "Refine the current parent candidates at the requested finer granularity while preserving explicit file paths."

            round_counts[action.granularity] += 1
            few_shots = (few_shots_long_term or "").strip()
            if memory.rounds:
                few_shots = "\n".join(x for x in [few_shots, memory.to_few_shot_block(top_k=2)] if x)
            if cfg.ablation_variant and action.action_type is ActionType.VERIFY and last_result is not None:
                localize_result = last_result
            else:
                localize_result = await self.localize_agent.run(
                    issue_text=issue_text,
                    graph_context=current_context,
                    few_shots=few_shots,
                    retry_block=retry_block,
                    valid_locations=current_context.valid_locations,
                    retry_count=round_counts[action.granularity] - 1,
                    llm_overrides=_merge_llm_overrides(
                        _llm_overrides_for_granularity(cfg, action.granularity),
                        _llm_overrides_for_action(cfg, action.action_type.value),
                    ),
                    granularity=action.granularity,
                )
                n_localize += 1
                if (
                    not localize_result.target.target
                    and (
                        not cfg.policy_max_llm_calls
                        or cumulative.llm_calls + 2 <= cfg.policy_max_llm_calls
                    )
                ):
                    trace.append(
                        f"LocalizeAgent empty target g={action.granularity}; retrying parse/localization once"
                    )
                    localize_result = await self.localize_agent.run(
                        issue_text=issue_text,
                        graph_context=current_context,
                        few_shots=few_shots,
                        retry_block=(retry_block + " Return one concrete target from the valid locations; do not return an empty target.").strip(),
                        valid_locations=current_context.valid_locations,
                        retry_count=round_counts[action.granularity],
                        llm_overrides=_merge_llm_overrides(
                            _llm_overrides_for_granularity(cfg, action.granularity),
                            _llm_overrides_for_action(cfg, action.action_type.value),
                        ),
                        granularity=action.granularity,
                    )
                    n_localize += 1
            if line_candidates is not None and action.granularity == "line":
                if action.action_type is not ActionType.VERIFY:
                    current_context = await line_candidates.observe(
                        localize_result, current_context, role="localization")
            if (
                cfg.file_verifier_enabled and not cfg.ablation_variant
                and action.granularity == "file"
                and not file_verifier_attempted
            ):
                file_verifier_attempted = True
                usage_after_actor = snapshot_session_llm_usage()
                actor_cost = _usage_delta(usage_before, usage_after_actor)
                verifier_affordable = (
                    not cfg.policy_max_llm_calls
                    or cumulative.llm_calls + actor_cost.llm_calls + 1
                    <= cfg.policy_max_llm_calls
                )
                if verifier_affordable:
                    localize_result, verification = await self.file_verifier_agent.run(
                        issue_text=issue_text,
                        graph_context=current_context,
                        localize_result=localize_result,
                    )
                    verification_history.append(verification)
                    trace.append(
                        "FileVerifierAgent "
                        f"status={verification.get('status', 'unknown')} "
                        f"candidates={len(verification.get('original_ranking') or [])}"
                    )
                else:
                    trace.append("FileVerifierAgent skipped reason=llm_call_budget")
            if cfg.ablation_variant:
                initial_outputs.setdefault(action.granularity, localize_result.raw_output)
                should_verify = action.granularity not in verified_granularities or action.action_type is ActionType.VERIFY
                used_calls = int(snapshot_session_llm_usage().get("call_count", 0)) - int(initial_usage.get("call_count", 0))
                allowance = max(0, cfg.policy_max_llm_calls - used_calls) if cfg.policy_max_llm_calls else 3
                if should_verify and allowance:
                    localize_result, current_context, report = await self.verification_agent.verify(
                        issue_text=issue_text, graph_context=current_context,
                        localize_result=localize_result, context_tool=self.context_tool,
                        inspector=critic_agent, max_llm_calls=min(3, allowance),
                        max_tool_calls=cfg.verification_max_tool_calls,
                        llm_overrides=_merge_llm_overrides(
                            _llm_overrides_for_granularity(cfg, action.granularity),
                            _llm_overrides_for_action(cfg, ActionType.VERIFY.value)))
                    if line_candidates is not None and action.granularity == "line":
                        current_context = await line_candidates.observe(
                            localize_result, current_context, role="verification")
                    report["granularity"] = action.granularity
                    verification_history.append(report)
                    verified_granularities.add(action.granularity)
                    n_localize += int(report.get("llm_calls", 0))
                    n_retrieval += sum(e["request"].get("tool") == "search" for e in report.get("tool_events", []))
                    trace.append(f"VerificationAgent role={self.verification_agent.role_name} g={action.granularity} tools={len(report.get('tool_events', []))}")
                    belief = evidence_model.expand_to_context(belief, issue_text, current_context)
                last_result = localize_result
            evaluation = critic_agent.evaluate(
                issue_text=issue_text,
                graph_context=current_context,
                localize_result=localize_result,
            )
            belief_before = belief.model_copy(deep=True)
            prior_observation_candidates = previous_observation_candidates.get(
                action.granularity, []
            )
            observation = evidence_model.observe(
                action=action,
                context=current_context,
                result=localize_result,
                evaluation=evaluation,
                previous_top_keys=belief.last_top_keys,
                previous_candidate_keys=prior_observation_candidates,
            )
            serialized_actor_candidates = _serialize_candidates_for_output(
                localize_result, action.granularity
            )
            if (
                action.granularity not in ranked_candidates_by_granularity
                and serialized_actor_candidates
                and any(str(item.get("target") or "").strip() for item in serialized_actor_candidates)
            ):
                ranked_candidates_by_granularity[action.granularity] = (
                    serialized_actor_candidates
                )
            repeated_candidate_set = bool(prior_observation_candidates) and (
                set(observation.candidate_keys) == set(prior_observation_candidates)
            )
            previous_observation_candidates[action.granularity] = list(
                observation.candidate_keys
            )
            belief = evidence_model.update(belief, observation)
            if cfg.ablation_variant:
                rendered = _render_policy_belief(belief, current_context, k_top=cfg.k_top,
                                                 fallback=localize_result.raw_output)
                ranked_candidates_by_granularity[action.granularity] = [
                    t.model_dump() for t in LocalizeAgent._parse_targets(rendered, action.granularity)
                ]
            candidate_paths = _collect_candidate_paths(localize_result, current_context)[:25]

            usage_after = snapshot_session_llm_usage()
            observed_cost = _usage_delta(usage_before, usage_after)
            observed_cost.wall_time_seconds = max(0.0, time.perf_counter() - wall_start)
            observed_cost.graph_nodes = retrieved_nodes
            cumulative = cumulative.plus(observed_cost)
            utility = belief.topk_success_mass(min(5, max(1, cfg.k_top)))
            if utility > best_utility:
                best_utility = utility
                best_result = localize_result
                best_eval = evaluation

            step = TrajectoryStep(
                step_idx=len(trajectory.steps),
                action=action,
                observation=observation,
                belief_before=belief_before,
                belief_after=belief.model_copy(deep=True),
                observed_cost=observed_cost,
                cumulative_cost=cumulative,
            )
            trajectory.append(step)
            memory.add_policy_step(step.model_dump(mode="json"))
            snap = RoundSnapshot(
                round_idx=len(memory.rounds),
                subgraph_key=retrieved_key or (context_keys[-1] if context_keys else ""),
                retrieval_granularity=action.granularity,
                graph_context=current_context.text_view,
                retrieval_meta={
                    **dict(retrieved_meta),
                    "policy_action": action.model_dump(mode="json"),
                    "belief_entropy": belief.entropy(),
                    "belief_topk_mass": utility,
                    "belief_outside_probability": belief.outside_probability,
                    "cumulative_cost": cumulative.model_dump(),
                },
                localize_prompt_used=localize_result.prompt_used,
                localize_output=localize_result.raw_output,
                retry_block=retry_block,
                reflect_json={
                    "controller": "belief_voi",
                    "action": action.model_dump(mode="json"),
                    "observation": observation.model_dump(mode="json"),
                    "belief": belief.model_dump(mode="json"),
                    "critic": evaluation.model_dump(),
                },
                scores={
                    "critic_total": evaluation.total_score,
                    "semantic_alignment": evaluation.semantic_alignment,
                    "topological_reachability": evaluation.topological_reachability,
                    "causal_validity": evaluation.causal_validity,
                    "belief_topk_mass": utility,
                    "belief_entropy": belief.entropy(),
                },
                retry_type=action.action_type.value,
            )
            memory.add_round(snap)
            trace.append(
                f"Policy step={step.step_idx} action={action.action_type.value} g={action.granularity} "
                f"critic={evaluation.total_score:.2f} topk_mass={utility:.4f} "
                f"outside={belief.outside_probability:.4f} cost_tokens={observed_cost.total_tokens}"
            )

            actual_budget_exhausted = not budget.can_afford(cumulative, ExplorationCost())
            if actual_budget_exhausted:
                stop_reason = "actual_budget_exhausted"
                trace.append("PolicyController.STOP reason=actual_budget_exhausted")
                break
            if action.action_type is ActionType.BACKTRACK:
                consecutive_backtracks += 1
            else:
                consecutive_backtracks = 0
            next_action = controller.choose_action(
                belief=belief,
                evaluation=evaluation,
                budget=budget,
                spent=cumulative,
                step_count=len(trajectory.steps),
                candidate_paths=candidate_paths,
                expanded_current_scope=belief.granularity in expanded_scopes,
                evidence_novelty=observation.evidence_novelty,
                repeated_candidate_set=repeated_candidate_set,
                consecutive_backtracks=consecutive_backtracks,
            )
            step.decision_after = next_action
            if memory.policy_steps:
                memory.policy_steps[-1] = step.model_dump(mode="json")
            snap.reflect_json["decision_after"] = next_action.model_dump(mode="json")
            considered = list(next_action.metadata.get("considered_actions") or [])
            if considered:
                compact = ",".join(
                    f"{row.get('action_type')}:{float(row.get('value', 0.0)):.4f}"
                    if row.get("affordable")
                    else f"{row.get('action_type')}:budget"
                    for row in considered
                )
                label = "Policy legal actions" if self.ablation_spec.controller == "heuristic" else "Policy values"
                trace.append(f"{label} after step={step.step_idx} [{compact}]")
            action = next_action
            n_policy += 1

        if not stop_reason:
            stop_reason = "completed"
        trajectory.stop_reason = stop_reason
        memory.policy_stop_reason = stop_reason
        if trajectory.steps:
            trajectory.steps[-1].stop_reason = stop_reason
        fallback_output = best_result.raw_output if best_result is not None else ""
        best_output = _render_policy_belief(
            belief,
            current_context,
            k_top=cfg.k_top,
            fallback=fallback_output,
        )
        if line_candidates is not None and line_candidates.candidates:
            best_output = line_candidates.render(limit=cfg.k_top)
            ranked_candidates_by_granularity["line"] = [
                t.model_dump() for t in LocalizeAgent._parse_targets(best_output, "line")
            ]
            audit["line_proposals"] = line_candidates.audit()
        best_candidates = (
            ranked_candidates_by_granularity.get(belief.granularity, [])
            if cfg.ablation_variant else _serialize_candidates(best_result)
        )
        best_scores: Dict[str, float] = {}
        best_combined = -1.0
        if best_eval is not None:
            best_scores = {
                "critic_total": best_eval.total_score,
                "semantic_alignment": best_eval.semantic_alignment,
                "topological_reachability": best_eval.topological_reachability,
                "causal_validity": best_eval.causal_validity,
                "belief_topk_mass": max(0.0, best_utility),
                "belief_outside_probability": belief.outside_probability,
            }
            best_combined = best_eval.total_score

        if cfg.memory_save_path:
            save_path = Path(cfg.memory_save_path) / f"{instance_id}_memory.json"
            try:
                memory.save(save_path)
            except Exception as exc:
                logger.warning("[PolicyPipeline] memory save failed: %s", exc)

        final_usage = snapshot_session_llm_usage()
        trace.append(
            "PolicyController.done "
            f"steps={len(trajectory.steps)} calls={max(0, int(final_usage.get('call_count', 0)) - int(initial_usage.get('call_count', 0)))} "
            f"stop={stop_reason}"
        )
        return PipelineResult(
            instance_id=instance_id,
            best_localization_output=best_output,
            best_candidates=best_candidates,
            best_scores=best_scores,
            best_combined_score=best_combined,
            stop_triggered=True,
            total_retrieval_rounds=n_retrieval,
            total_localize_rounds=n_localize,
            total_reflect_rounds=n_policy,
            graph_context_file=raw_contexts["file"],
            graph_context_function=raw_contexts["function"],
            graph_context_line=raw_contexts["line"],
            merged_graph_context=current_context.text_view if current_context is not None else "",
            graph_context_history=graph_context_history,
            three_agent_trace=trace,
            session_memory=memory,
            policy_trajectory={**trajectory.model_dump(mode="json"), "ablation_audit": audit},
            ranked_candidates_by_granularity=ranked_candidates_by_granularity,
            verification_history=verification_history,
        )

    def run_sync(self, **kwargs: Any) -> PipelineResult:
        return asyncio.run(self.run(**kwargs))
