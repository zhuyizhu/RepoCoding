"""
单次定位批跑的运行时配置快照：把 GraphRAG / JLOC 相关环境变量与 CLI 的**生效值**写入 JSON，便于论文与复现实验对齐。

说明（写入快照文件正文 ``notes``）：
- ``export VAR=...`` 只作用于**随后启动的** Python 进程；本快照在 ``run_dataset_localization.py`` 启动后立刻读取当前进程环境。
- ``GRAPHRAG_MAX_CONTEXT_CHARS`` 等限制的是 **GraphRAG 每次构建子图时拼出的上下文字符串长度上限**（file/function/line 各次调用分别受 cap），不是「把整个仓库一次性读进一个字符串」；图仍由 ``load_repograph`` 按常规加载，子图文本在达到上限时截断。
"""

from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

from agent.utils.llm_http_timeout import resolve_llm_request_timeout_sec


def _env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if v.isdigit():
        return int(v)
    return default


def _env_str(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_runtime_config_snapshot(args: Optional[Namespace] = None, *, run_started_utc: str = "") -> Dict[str, Any]:
    """
    与 ``graph_rag_tools.graphrag_context_impl``、``pipeline.build_localize_merged_context``、
    ``run_dataset_localization._invoke_three_agent_pipeline`` 中使用的默认值对齐。
    """
    cli: Dict[str, Any] = {}
    if args is not None:
        cli = {
            "dataset_jsonl_file": getattr(args, "dataset_jsonl_file", None),
            "dataset_jsonl_dir": getattr(args, "dataset_jsonl_dir", None),
            "limit": int(getattr(args, "limit", 0) or 0),
            "instance_ids": list(getattr(args, "instance_id", []) or []),
            "issue_input_policy": "resolved_issues_only",
            "max_issue_chars": int(getattr(args, "max_issue_chars", 12_000)),
            "include_dataset_hints": bool(getattr(args, "include_dataset_hints", False)),
            "repograph_root": getattr(args, "repograph_root", None),
            "k": int(getattr(args, "k", 20)),
            "hops": int(getattr(args, "hops", 2)),
            "max_context_chars": int(getattr(args, "max_context_chars", 8000)),
            "few_shots_k": int(getattr(args, "few_shots_k", 0)),
            "persist_memory": bool(getattr(args, "persist_memory", False)),
            "vectorstore_path": getattr(args, "vectorstore_path", None),
            "embeddings_model": getattr(args, "embeddings_model", None),
            "rebuild_vectorstore": bool(getattr(args, "rebuild_vectorstore", False)),
            "file_verifier_enabled": not bool(
                getattr(args, "disable_file_verifier", False)
            ),
            "file_verifier_max_candidates": int(
                getattr(args, "file_verifier_max_candidates", 5)
            ),
            "file_verifier_evidence_chars": int(
                getattr(args, "file_verifier_evidence_chars", 3000)
            ),
            "entity_anchors_enabled": not bool(
                getattr(args, "disable_entity_anchors", False)
            ),
            "topn": getattr(args, "topn", None),
            "combined_output": bool(getattr(args, "combined_output", False)),
            "resume": bool(getattr(args, "resume", False)),
            "policy_driven": not bool(getattr(args, "legacy_workflow", False)),
            "policy_max_steps": int(getattr(args, "policy_max_steps", 8)),
            "policy_max_tokens": int(getattr(args, "policy_max_tokens", 0)),
            "policy_max_llm_calls": int(getattr(args, "policy_max_llm_calls", 8)),
            "policy_max_wall_time_seconds": float(getattr(args, "policy_max_wall_time_seconds", 0.0)),
            "policy_max_graph_nodes": int(getattr(args, "policy_max_graph_nodes", 0)),
            "policy_cost_weight": getattr(args, "policy_cost_weight", None),
            "policy_stop_probability": float(getattr(args, "policy_stop_probability", 0.82)),
            "policy_stable_steps": int(getattr(args, "policy_stable_steps", 2)),
            "policy_target_granularity": getattr(args, "policy_target_granularity", "line"),
            "policy_calibration_file": getattr(args, "policy_calibration_file", None),
        }

    # GraphRAG：与 graphrag_context_impl 入口一致（file/function/line 各自受粒度内调整）
    max_ctx = _env_int("GRAPHRAG_MAX_CONTEXT_CHARS", 200_000)

    # Pipeline 合并上下文（Localize 阶段）
    file_cap = _env_int("JLOC_FILE_CONTEXT_MAX_CHARS", 110_000)
    line_cap = _env_int("JLOC_LINE_CONTEXT_MAX_CHARS", 110_000)
    merged_cap = _env_int("JLOC_MERGED_CONTEXT_MAX_CHARS", 0)
    repo_excerpt = _env_int("JLOC_REPO_EXCERPT_CHARS", 3000)

    # run_dataset 里 Reflect 轮次上界
    reflect_rounds_cap = max(1, min(int(os.environ.get("JLOC_MAX_REFLECT_ROUNDS", "2") or 2), 5))
    base_k = int(getattr(args, "k", 8) or 8) if args is not None else 8
    phase2_topk = max(3, min(int(os.environ.get("JLOC_PHASE2_TOPK", str(base_k)) or base_k), 20))

    env_keys = [
        "GRAPHRAG_MAX_CONTEXT_CHARS",
        "GRAPHRAG_HOPS",
        "GRAPHRAG_SEMANTIC_K",
        "GRAPHRAG_GRAPH_NEIGHBOR_CAP",
        "GRAPHRAG_BM25_CANDIDATE_CAP",
        "GRAPHRAG_FUSION_TOP",
        "GRAPHRAG_RRF_K",
        "GRAPHRAG_EGO_CENTER_CAP",
        "GRAPHRAG_ENTITY_ANCHORS",
        "GRAPHRAG_ENTITY_ANCHOR_FILE_CAP",
        "GRAPHRAG_ENTITY_ANCHOR_NODE_CAP",
        "GRAPHRAG_ENTITY_ANCHOR_NODES_PER_FILE",
        "JLOC_FILE_CONTEXT_MAX_CHARS",
        "JLOC_LINE_CONTEXT_MAX_CHARS",
        "JLOC_MERGED_CONTEXT_MAX_CHARS",
        "JLOC_REPO_EXCERPT_CHARS",
        "JLOC_MAX_REFLECT_ROUNDS",
        "JLOC_PHASE2_TOPK",
        "JLOC_REFLECT_EARLY_STOP_COMBINED",
        "JLOC_VALID_LOC_FILES_CAP",
        "LLM_TEMPERATURE",
        "LLM_SEED",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_URL",
        "OLLAMA_MODEL",
        "OLLAMA_URL",
        "OLLAMA_NUM_CTX",
        "OLLAMA_NUM_CTX_FALLBACK",
        "JLOC_LLM_REQUEST_TIMEOUT_SEC",
        "JLOC_LLM_RETRY_MAX",
        "JLOC_LLM_RETRY_SLEEP_SEC",
        "JLOC_LLM_RETRY_BACKOFF_FACTOR",
        "JLOC_LLM_RETRY_MAX_SLEEP_SEC",
        "JLOC_INSTANCE_RETRY_MAX",
        "JLOC_INSTANCE_RETRY_SLEEP_SEC",
        "JLOC_INSTANCE_RETRY_BACKOFF_FACTOR",
        "JLOC_INSTANCE_RETRY_MAX_SLEEP_SEC",
        "JLOC_REPOGRAPH_CACHE_SIZE",
        "CODE_EMBEDDING_DEVICE",
        "CODE_EMBEDDING_BATCH_SIZE",
    ]
    environment_raw = {k: _env_str(k) for k in env_keys}

    return {
        "schema": "jlocagent.runtime_config.v1",
        "run_started_utc": run_started_utc,
        "notes_zh": (
            "1) 在运行本脚本的终端里先 export 变量，再启动 python，子进程会继承这些环境变量。\n"
            "2) effective_* 为代码实际采用的数值（与未设置时的默认一致）。\n"
            "3) GRAPHRAG_MAX_CONTEXT_CHARS：限制**每一次** GraphRAG 子图输出字符串的最大字符数；file/function/line 各会单独建子图，"
            "file/line 等粒度还会在 graphrag_context_impl 内进一步调整 hops/seed_k/cap（见源码）。\n"
            "4) JLOC_FILE/LINE/MERGED/REPO_EXCERPT：作用于 Localize 前合并后的 prompt 内各块截断与总预算。\n"
            "5) JLOC_MAX_REFLECT_ROUNDS：与 Localize↔Reflect 循环上界相关（另受 pipeline 内 early_stop 影响）。\n"
            "6) JLOC_LLM_REQUEST_TIMEOUT_SEC：单次 LLM HTTP 读超时（秒）；未设置时对非 ollama 默认 1200；"
            "设为 0/none/off 可关闭。避免远程 API 挂起时进程无限等待。\n"
            "7) ISSUE_INPUT_POLICY 固定为 resolved_issues_only：在线定位不读取顶层 PR title/body、"
            "fix/test patch 或运行结果；dataset hints 默认关闭，仅允许显式消融。"
        ),
        "cli_args": cli,
        "environment_raw": environment_raw,
        "effective": {
            "GRAPHRAG_MAX_CONTEXT_CHARS": max_ctx,
            "JLOC_FILE_CONTEXT_MAX_CHARS": file_cap,
            "JLOC_LINE_CONTEXT_MAX_CHARS": line_cap,
            "JLOC_MERGED_CONTEXT_MAX_CHARS": merged_cap,
            "JLOC_REPO_EXCERPT_CHARS": repo_excerpt,
            "JLOC_MAX_REFLECT_ROUNDS": reflect_rounds_cap,
            "JLOC_PHASE2_TOPK": phase2_topk,
            "JLOC_REFLECT_EARLY_STOP_COMBINED": _float_env("JLOC_REFLECT_EARLY_STOP_COMBINED", 0.88),
            "POLICY_DRIVEN": not bool(getattr(args, "legacy_workflow", False)) if args is not None else True,
            "POLICY_MAX_STEPS": int(getattr(args, "policy_max_steps", 8)) if args is not None else 8,
            "POLICY_MAX_TOKENS": int(getattr(args, "policy_max_tokens", 0)) if args is not None else 0,
            "POLICY_MAX_LLM_CALLS": int(getattr(args, "policy_max_llm_calls", 8)) if args is not None else 8,
            "POLICY_COST_WEIGHT": getattr(args, "policy_cost_weight", None) if args is not None else None,
            "POLICY_STOP_PROBABILITY": float(getattr(args, "policy_stop_probability", 0.82)) if args is not None else 0.82,
            "POLICY_TARGET_GRANULARITY": getattr(args, "policy_target_granularity", "line") if args is not None else "line",
            "ISSUE_INPUT_POLICY": "resolved_issues_only",
            "MAX_ISSUE_CHARS": int(getattr(args, "max_issue_chars", 12_000)) if args is not None else 12_000,
            "DATASET_HINTS_ENABLED": bool(getattr(args, "include_dataset_hints", False)) if args is not None else False,
            "FILE_VERIFIER_ENABLED": not bool(
                getattr(args, "disable_file_verifier", False)
            ) if args is not None else True,
            "FILE_VERIFIER_MAX_CANDIDATES": int(
                getattr(args, "file_verifier_max_candidates", 5)
            ) if args is not None else 5,
            "FILE_VERIFIER_EVIDENCE_CHARS": int(
                getattr(args, "file_verifier_evidence_chars", 3000)
            ) if args is not None else 3000,
            "ENTITY_ANCHORS_ENABLED": bool(
                _env_int("GRAPHRAG_ENTITY_ANCHORS", 1)
            ),
            "EXCLUDED_INFERENCE_FIELDS": [
                "title",
                "body",
                "hints",
                "fix_patch",
                "test_patch",
                "fix_patch_result",
                "test_patch_result",
                "run_result",
            ],
            "llm_http_read_timeout_sec_effective": resolve_llm_request_timeout_sec(),
        },
    }


def write_runtime_config_snapshot(path: Path, *, args: Optional[Namespace] = None, run_started_utc: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = build_runtime_config_snapshot(args, run_started_utc=run_started_utc)
    path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
