from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


PHASE2_MARKER = "===PHASE2_TOPK_LOCATION==="


def _load_project_env_for_llm() -> None:
    """Load project .env only when the whole-graph path is about to call an LLM."""
    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None
    if load_dotenv is not None:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        load_dotenv()

    if not os.environ.get("LLM_API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["LLM_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("LLM_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
    if not os.environ.get("LLM_MODEL") and os.environ.get("OPENAI_MODEL"):
        os.environ["LLM_MODEL"] = os.environ["OPENAI_MODEL"]


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth=_depth + 1) for v in value]
    if isinstance(value, set):
        return [_jsonable(v, _depth=_depth + 1) for v in sorted(value, key=lambda x: repr(x))]
    return repr(value)


def _iter_graph_nodes(graph: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if hasattr(graph, "nodes"):
        try:
            for node_id, data in graph.nodes(data=True):
                yield str(node_id), dict(data) if isinstance(data, dict) else {"value": _jsonable(data)}
            return
        except TypeError:
            pass
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        for node_id, data in nodes.items():
            yield str(node_id), dict(data) if isinstance(data, dict) else {"value": _jsonable(data)}


def _iter_graph_edges(graph: Any) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    if hasattr(graph, "edges"):
        try:
            for src, dst, data in graph.edges(data=True):
                yield str(src), str(dst), dict(data) if isinstance(data, dict) else {"value": _jsonable(data)}
            return
        except TypeError:
            pass
    edges = getattr(graph, "edges", None)
    if isinstance(edges, dict):
        for key, data in edges.items():
            if isinstance(key, tuple) and len(key) >= 2:
                yield str(key[0]), str(key[1]), dict(data) if isinstance(data, dict) else {"value": _jsonable(data)}


def _line_for_node(node_id: str, data: Dict[str, Any]) -> str:
    payload = {"node_id": node_id, **_jsonable(data)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _line_for_edge(src: str, dst: str, data: Dict[str, Any]) -> str:
    payload = {"src": src, "dst": dst, **_jsonable(data)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def serialize_graph_pkl_for_llm(graph_pkl_path: Union[str, Path], *, max_chars: int = 0) -> Tuple[str, Dict[str, Any]]:
    """Serialize the entire graph.pkl into deterministic text for direct LLM input.

    ``max_chars <= 0`` means no truncation. The default intentionally preserves the
    full graph representation for the whole-graph baseline entry.
    """
    p = Path(graph_pkl_path)
    with p.open("rb") as f:
        graph = pickle.load(f)

    graph_attrs: Dict[str, Any] = {}
    if hasattr(graph, "graph") and isinstance(getattr(graph, "graph"), dict):
        graph_attrs = _jsonable(getattr(graph, "graph"))

    node_lines = [_line_for_node(node_id, data) for node_id, data in _iter_graph_nodes(graph)]
    edge_lines = [_line_for_edge(src, dst, data) for src, dst, data in _iter_graph_edges(graph)]

    lines: List[str] = [
        "# graph.pkl whole-graph serialization",
        f"source_path: {p}",
        f"graph_type: {type(graph).__module__}.{type(graph).__name__}",
        f"node_count: {len(node_lines)}",
        f"edge_count: {len(edge_lines)}",
        "graph_attrs_json:",
        json.dumps(graph_attrs, ensure_ascii=False, sort_keys=True),
        "",
        "## nodes_jsonl",
    ]
    lines.extend(node_lines)
    lines.append("")
    lines.append("## edges_jsonl")
    lines.extend(edge_lines)
    text = "\n".join(lines)

    truncated = False
    original_chars = len(text)
    if max_chars and max_chars > 0 and len(text) > max_chars:
        truncated = True
        text = text[:max_chars] + f"\n…(whole graph serialization truncated to {max_chars} chars by caller)"

    meta = {
        "graph_pkl_path": str(p),
        "graph_type": f"{type(graph).__module__}.{type(graph).__name__}",
        "node_count": len(node_lines),
        "edge_count": len(edge_lines),
        "chars": len(text),
        "original_chars": original_chars,
        "truncated": truncated,
    }
    return text, meta


def run_whole_graph_localize_issue(
    *,
    issue_text: str,
    graph_pkl_path: Union[str, Path],
    k: int = 20,
    max_graph_chars: int = 200000,
    llm_temperature: Optional[float] = None,
    llm_seed: Optional[int] = None,
    ollama_num_ctx: Optional[int] = None,
    return_prompt: bool = False,
) -> Union[str, Tuple[str, str, Dict[str, Any]]]:
    graph_text, graph_meta = serialize_graph_pkl_for_llm(graph_pkl_path, max_chars=max_graph_chars)
    prompt = f"""你是资深 Java Bug 定位专家。当前实验版本不使用 GraphRAG 检索裁剪，而是把生成的整个 graph.pkl 序列化后直接交给你分析。

你必须只依据下方 whole graph 序列化内容和 Issue 描述预测最可能修改位置。路径、符号、行号必须来自 graph.pkl 内容；禁止臆造未出现的路径。

输出结构（两步都必须完成；分隔符用于下游解析，勿改）

【步骤 A — 候选修改文件（文件级）】
- 仅列出你认为最可能需改动的 .java 文件（最多 12 个），每行一条。
- 排序：优先 src/main/java/ 下业务实现；压低 src/test/、benchmark、jmh 等除非 Issue 明确只关它们。
- 格式：
  候选文件: <相对路径.java> 理由: <一句中文>

【步骤 B — Top-K 函数/行候选】
- 在步骤 A 的文件范围内，选出最多 {k} 个最可能修改点。
- 如果 graph.pkl 中有行号，尽量输出行号；没有行号时输出函数/类名。
- 格式优先使用：
  1. 文件: <路径.java> 行: <整数> 原因: <一句中文>
- 没有可靠行号时使用：
  1. 函数: <方法或类名> 文件: <路径.java> 行: <起始行或范围或 unknown> 原因: <一句中文>

输出时先完整输出步骤 A，然后单独一行输出分隔标记（必须完全一致、单独成行）：
{PHASE2_MARKER}
然后输出步骤 B；步骤 B 中不要重复步骤 A 的候选文件列表。

====================
Issue 描述
{issue_text}

====================
Whole graph.pkl 序列化内容
{graph_text}
"""

    _load_project_env_for_llm()

    from agent.tools.localization_llm import invoke_llm_with_ollama_fallback

    # Try invoking LLM; if the input is too large for the API or run-tracking
    # ingestion, retry once with a much smaller truncated graph serialization.
    try:
        resp = invoke_llm_with_ollama_fallback(
            prompt,
            call_site="run_whole_graph_localize_issue",
            temperature=llm_temperature,
            seed=llm_seed,
            ollama_num_ctx=ollama_num_ctx,
        )
    except Exception as e:  # fallback on oversized payloads
        err_str = repr(e)
        # If we detect a payload-too-large style error, retry with a heavy truncation.
        if "413" in err_str or "Content-Length" in err_str or "exceeds maximum allowed size" in err_str:
            try:
                small_cap = min(20000, max(10000, int(max_graph_chars or 20000)))
            except Exception:
                small_cap = 20000
            print(f"LLM or run-ingest error looks like oversized payload; retrying with truncated graph ({small_cap} chars)...")
            graph_text, graph_meta = serialize_graph_pkl_for_llm(graph_pkl_path, max_chars=small_cap)
            prompt = prompt.split("\n\nWhole graph.pkl 序列化内容\n", 1)[0] + f"\n\nWhole graph.pkl 序列化内容\n{graph_text}\n"
            resp = invoke_llm_with_ollama_fallback(
                prompt,
                call_site="run_whole_graph_localize_issue_retry_truncated",
                temperature=llm_temperature,
                seed=llm_seed,
                ollama_num_ctx=ollama_num_ctx,
            )
        else:
            raise
    out = (resp.content or "").strip()
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1].strip()
    if return_prompt:
        return out, prompt, graph_meta
    return out


def graph_pkl_path_for_repograph_dir(repograph_dir: Union[str, Path]) -> Path:
    p = Path(repograph_dir) / "graph.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"whole-graph entry requires graph.pkl: {p}")
    return p