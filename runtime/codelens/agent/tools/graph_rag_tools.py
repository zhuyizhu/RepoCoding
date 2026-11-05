from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Dict, Any, Optional, Set, List, Tuple, Union

from langchain_core.tools import tool

import config
from agent.agents.models import GraphContext, GraphNode
from agent.memory.unified_vectorstore import UnifiedVectorStore, get_shared_unified_vector_store
from agent.repograph.java_path_filters import is_test_or_benchmark_java_path
from agent.repograph.repograph_loader import load_repograph, normalize_node_text
from agent.retrieval.bm25_okapi import BM25Okapi, tokenize_for_bm25
from agent.retrieval.path_rerank import (
    issue_keyword_tokens,
    keyword_hits_in_text,
    path_priority_for_localization,
    rank_issue_entity_nodes,
)


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    v = (os.environ.get(name) or "").strip()
    if v.isdigit():
        return int(v)
    return default


def _normalize_path_for_match(path_str: str) -> str:
    """路径归一化（与 localization_metrics.normalize_code_path 逻辑一致，避免循环导入）。"""
    p = (path_str or "").strip().replace("\\", "/")
    java_paths = re.findall(
        r"(?:[A-Za-z]:)?[A-Za-z0-9_.$@+~/-]*[A-Za-z0-9_$-]+\.java",
        p,
    )
    if java_paths:
        p = max(java_paths, key=len)
    else:
        p = p.strip("`*_~'\"()[]{}<>")
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    for anchor in ("src/main/java/", "src/test/java/", "src/"):
        idx = p.find(anchor)
        if idx >= 0:
            return p[idx:]
    return p.lstrip("/")


def _force_path_match_keys(paths: List[str]) -> Set[str]:
    """与 gold / 预测路径对齐：同时收录 repograph 相对路径与 ``src/main/java`` 归一化形式。"""
    from agent.utils.localization_metrics import normalize_code_path

    keys: Set[str] = set()
    for p in paths or []:
        if not p:
            continue
        keys.add(_normalize_path_for_match(p))
        np = normalize_code_path(p)
        if np:
            keys.add(_normalize_path_for_match(np))
            keys.add(np)
    keys.discard("")
    return keys


def _resolve_force_path_keys(repograph, paths: List[str]) -> Set[str]:
    """Resolve requested Java paths against the graph without basename guessing.

    Exact and segment-suffix matches are accepted. A basename-only request is
    accepted only when that basename is unique in the current RepoGraph.
    """
    available = {
        _normalize_path_for_match(
            str(data.get("rel_fname") or data.get("fname") or "")
        )
        for data in repograph.nodes.values()
    }
    available = {path for path in available if path.endswith(".java")}
    resolved: Set[str] = set()
    for requested in _force_path_match_keys(paths):
        if not requested:
            continue
        exact = {path for path in available if path == requested}
        if exact:
            resolved.update(exact)
            continue
        suffix = {
            path
            for path in available
            if path.endswith("/" + requested) or requested.endswith("/" + path)
        }
        if len(suffix) == 1:
            resolved.update(suffix)
            continue
        if "/" not in requested:
            basename = {path for path in available if path.rsplit("/", 1)[-1] == requested}
            if len(basename) == 1:
                resolved.update(basename)
    return resolved


def _rel_matches_any_force_path(rel: str, keys: Set[str]) -> bool:
    r = _normalize_path_for_match(rel.replace("\\", "/"))
    if not r or not keys:
        return False
    if r in keys:
        return True
    for k in keys:
        if not k:
            continue
        if r.endswith("/" + k) or k.endswith("/" + r):
            return True
    return False


def _filter_ordered_nodes_for_files(
    repograph,
    ordered_nodes: List[Tuple[str, int]],
    force_file_paths: List[str],
    mode: str,
) -> List[Tuple[str, int]]:
    """
    在已定序的 (node_id, dist) 列表上裁剪，只保留与 ``force_file_paths`` 路径相关、且符合粒度的节点。

    ``mode``:
    - ``function_focus``: class + function(def)，用于函数级子图，去掉大量 ref/无关符号。
    - ``defs_only``: 仅 function(def) + class，比 function_focus 略紧（当前与 function_focus 相同语义）。
    """
    mode = (mode or "none").strip().lower()
    if mode in ("none", "") or not force_file_paths:
        return ordered_nodes
    tset = _resolve_force_path_keys(repograph, list(force_file_paths))
    if not tset:
        return ordered_nodes
    out: List[Tuple[str, int]] = []
    for nid, dd in ordered_nodes:
        data = repograph.nodes.get(nid, {})
        rel = str(data.get("rel_fname") or data.get("fname") or "").replace("\\", "/")
        if not _rel_matches_any_force_path(rel, tset):
            continue
        cat = (data.get("category") or "").lower()
        kind = (data.get("kind") or "").lower()
        if mode in ("function_focus", "defs_only"):
            if cat == "class" or (cat == "function" and kind == "def"):
                out.append((nid, dd))
        else:
            out.append((nid, dd))
    return out if out else ordered_nodes


def _nodes_for_file_paths(
    repograph,
    file_paths: List[str],
    allowed: Set[str],
) -> List[str]:
    """
    返回 repograph 中属于指定文件路径的所有节点 ID（经路径归一化后匹配）。
    用于 graphRAG pass-2 的「文件强制注入」：保证定位阶段 1 的怀疑文件节点进入检索上下文。
    """
    target = _resolve_force_path_keys(repograph, list(file_paths))
    if not target:
        return []
    result: List[str] = []
    for nid, data in repograph.nodes.items():
        rel = str(data.get("rel_fname") or data.get("fname") or "").replace("\\", "/")
        if _rel_matches_any_force_path(rel, target) and str(nid) in allowed:
            result.append(str(nid))
    return result


def _non_test_node_ids(repograph) -> Set[str]:
    out: Set[str] = set()
    for nid, data in repograph.nodes.items():
        rel = str(data.get("rel_fname") or data.get("fname") or "").replace("\\", "/")
        if rel and is_test_or_benchmark_java_path(rel):
            continue
        out.add(str(nid))
    return out


def _expand_seeds_with_graph_neighbors(
    repograph,
    seed_nodes: List[str],
    issue_text: str,
    allowed: Set[str],
    max_extra: int,
    neighbors: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
    tokens = issue_keyword_tokens(issue_text)
    scored: List[Tuple[float, str]] = []
    seen = set(seed_nodes)
    adj = neighbors or repograph.neighbors
    for s in seed_nodes:
        if s not in adj:
            continue
        for nb in adj.get(s, set()):
            if nb not in allowed or nb in seen:
                continue
            data = repograph.nodes.get(nb, {})
            rel = str(data.get("rel_fname") or data.get("fname") or "")
            blob = normalize_node_text(nb, data).lower()
            kh = keyword_hits_in_text(blob, tokens)
            pp = path_priority_for_localization(rel)
            score = kh * 2.5 + max(0.0, pp)
            if kh > 0 or pp >= 2.0:
                scored.append((score, nb))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = list(seed_nodes)
    for sc, nb in scored:
        if sc <= 0 or nb in seen:
            continue
        seen.add(nb)
        out.append(nb)
        if len(out) - len(seed_nodes) >= max_extra:
            break
    return out


def _build_filtered_neighbors(repograph, edge_types: Optional[List[str]]) -> Optional[Dict[str, Set[str]]]:
    if not edge_types:
        return None
    if not getattr(repograph, "edges", None):
        return None
    allowed = {str(x).lower() for x in edge_types if x}
    if not allowed:
        return None
    neighbors: Dict[str, Set[str]] = {}
    for edge in repograph.edges:
        et = str(getattr(edge, "edge_type", "")).lower()
        if et not in allowed:
            continue
        src = str(edge.src)
        dst = str(edge.dst)
        neighbors.setdefault(src, set()).add(dst)
        neighbors.setdefault(dst, set()).add(src)
    return neighbors


def _ego_nodes_with_dist(neighbors: Dict[str, Set[str]], center: str, radius: int) -> List[Tuple[str, int]]:
    if radius < 0:
        return []
    if center not in neighbors:
        return []

    dist: Dict[str, int] = {center: 0}
    queue: List[str] = [center]
    while queue:
        n = queue.pop(0)
        d = dist[n]
        if d >= radius:
            continue
        for m in neighbors.get(n, set()):
            if m not in dist:
                dist[m] = d + 1
                queue.append(m)

    items = sorted(dist.items(), key=lambda x: (x[1], x[0]))
    return items


def reciprocal_rank_fusion(rank_lists: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for i, item in enumerate(ranks):
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _get_store(
    repograph_dir: str,
    repo_key: str,
    vectorstore_path: str,
    embeddings_model: str,
    rebuild_vectorstore: bool,
) -> UnifiedVectorStore:
    store = get_shared_unified_vector_store(
        vectorstore_path,
        embeddings_model,
        rebuild=rebuild_vectorstore,
    )
    store.ensure_init(repograph_dir=repograph_dir, repo_key=repo_key)
    return store


def graphrag_context_impl(
    repograph_dir: str,
    repo_key: str,
    issue_text: str,
    hops: int = 2,
    seed_k: int = 120,
    extra_code_k: int = 0,
    max_context_chars: int = 0,
    vectorstore_path: str = config.VECTORSTORE_PATH,
    embeddings_model: str = config.CODE_EMBEDDING_MODEL,
    rebuild_vectorstore: bool = False,
    force_file_paths: Optional[List[str]] = None,
    retrieval_granularity: str = "file",
    node_context_filter: str = "none",
    edge_types: Optional[List[str]] = None,
    return_structured: bool = False,
) -> Union[str, GraphContext]:
    """
    GraphRAG 核心实现（纯 Python 可调用）。

    请优先通过本函数在代码中调用；``@tool`` 包装器 ``graphrag_context`` 仅用于 LangChain 绑定。
    直接 ``graphrag_context.invoke({..., "force_file_paths": [...]})`` 在部分 LangChain 版本下会经
    内部 ``get_context_wrapper`` 转发失败（unexpected keyword ``force_file_paths``），故工作流与
    RetrievalAgent 应调用 ``graphrag_context_impl(**kwargs)``。
    """
    hops = _env_int("GRAPHRAG_HOPS", hops)
    seed_k = _env_int("GRAPHRAG_SEMANTIC_K", seed_k)
    max_context_chars = max_context_chars or _env_int("GRAPHRAG_MAX_CONTEXT_CHARS", 200_000)
    neighbor_cap = _env_int("GRAPHRAG_GRAPH_NEIGHBOR_CAP", 96)
    candidate_cap = _env_int("GRAPHRAG_BM25_CANDIDATE_CAP", 2500)
    fusion_top = _env_int("GRAPHRAG_FUSION_TOP", 320)
    rrf_k = _env_int("GRAPHRAG_RRF_K", 60)
    center_cap = _env_int("GRAPHRAG_EGO_CENTER_CAP", 72)

    # ── 检索粒度：file / function / line（function 为「窄文件内的类+方法 def」子图）────────
    gran = (retrieval_granularity or "file").strip().lower()
    ncf = (node_context_filter or "none").strip().lower()
    if gran == "function":
        gran = "function"
        hops = max(1, min(hops, _env_int("GRAPHRAG_FUNCTION_GRANULARITY_MAX_HOPS", 2)))
        seed_k = max(seed_k, _env_int("GRAPHRAG_FUNCTION_GRANULARITY_SEED_K", min(200, seed_k + 40)))
        neighbor_cap = min(neighbor_cap, _env_int("GRAPHRAG_FUNCTION_GRANULARITY_NEIGHBOR_CAP", 64))
        candidate_cap = min(max(candidate_cap, 1800), 4000)
        fusion_top = min(fusion_top, _env_int("GRAPHRAG_FUNCTION_FUSION_TOP", 220))
        center_cap = min(center_cap, _env_int("GRAPHRAG_FUNCTION_CENTER_CAP", 48))
        ncf = "function_focus"
    if gran == "line":
        # 行级子图：更紧的 hop、更大语义种子与邻域，突出强制文件附近的符号（hybrid 仍保留）
        hops = max(1, min(hops, _env_int("GRAPHRAG_LINE_GRANULARITY_MAX_HOPS", 2)))
        seed_k = max(seed_k, _env_int("GRAPHRAG_LINE_GRANULARITY_SEED_K", min(240, seed_k + 80)))
        neighbor_cap = max(neighbor_cap, _env_int("GRAPHRAG_LINE_GRANULARITY_NEIGHBOR_CAP", 180))
        candidate_cap = min(
            max(candidate_cap, _env_int("GRAPHRAG_LINE_GRANULARITY_CANDIDATE_CAP", 2800)),
            6000,
        )
        center_cap = max(center_cap, _env_int("GRAPHRAG_LINE_GRANULARITY_CENTER_CAP", 96))
    elif gran == "file":
        # 文件级：在强制注入基础上略扩候选池，便于跨文件调用边进入 BM25
        candidate_cap = min(max(candidate_cap, 2600), 6000)

    logger.info(
        "[GraphRAG] params g=%s hops=%d seed_k=%d candidate_cap=%d fusion_top=%d neighbor_cap=%d center_cap=%d rrf_k=%d max_chars=%d ncf=%s force_files=%d",
        gran,
        hops,
        seed_k,
        candidate_cap,
        fusion_top,
        neighbor_cap,
        center_cap,
        rrf_k,
        max_context_chars,
        ncf,
        len(force_file_paths or []),
    )

    repograph = load_repograph(
        repograph_dir,
        prefer="pkl",
        enrich_with_tags=gran in ("function", "line"),
    )
    print(
        f"[GraphRAG] repograph ready nodes={len(repograph.nodes)} source={getattr(repograph, 'source', '')}",
        flush=True,
    )
    allowed = _non_test_node_ids(repograph)
    # Complement dense/BM25 retrieval with exact code entities explicitly
    # present in the pre-fix issue.  This channel is enabled only for the
    # initial file search: finer passes remain scoped to the Actor's files.
    entity_ids: List[str] = []
    if (
        gran == "file"
        and not force_file_paths
        and _env_int("GRAPHRAG_ENTITY_ANCHORS", 1)
    ):
        entity_ids = rank_issue_entity_nodes(
            repograph.nodes,
            issue_text,
            allowed_node_ids=allowed,
            max_files=_env_int("GRAPHRAG_ENTITY_ANCHOR_FILE_CAP", 24),
            max_nodes=_env_int("GRAPHRAG_ENTITY_ANCHOR_NODE_CAP", 64),
            max_nodes_per_file=_env_int("GRAPHRAG_ENTITY_ANCHOR_NODES_PER_FILE", 3),
        )
        logger.info("[GraphRAG] structured issue-entity anchors=%d", len(entity_ids))
    store = _get_store(
        repograph_dir=repograph_dir,
        repo_key=repo_key,
        vectorstore_path=vectorstore_path,
        embeddings_model=embeddings_model,
        rebuild_vectorstore=rebuild_vectorstore,
    )

    filtered_neighbors = _build_filtered_neighbors(repograph, edge_types)
    neighbors = filtered_neighbors or repograph.neighbors

    # --- 1) Semantic (dense embedding) ---
    sem_k = max(seed_k, 40)
    seed_docs = store.search_code_nodes(issue_text=issue_text, repo_key=repo_key, k=sem_k)
    sem_ids: List[str] = []
    for d in seed_docs:
        nid = d.metadata.get("node_id")
        if not nid:
            continue
        s = str(nid)
        if s not in allowed:
            continue
        if s not in sem_ids:
            sem_ids.append(s)

    if not sem_ids:
        sem_ids = [str(x) for x in list(repograph.nodes.keys())[:3] if str(x) in allowed]

    # --- 1b) 文件强制注入 (Pass-2 file boost) ---
    # 把 force_file_paths 对应的所有节点插入到 sem_ids 最前面（最高优先级），
    # 保证定位阶段 1 的怀疑文件内容必然进入候选池和上下文，解决 GraphRAG pass-1 遗漏关键文件的问题。
    boost_nids: List[str] = []
    if force_file_paths:
        boost_nids = _nodes_for_file_paths(repograph, list(force_file_paths), allowed)
        if boost_nids:
            seen_boost = set(sem_ids)
            forced = [n for n in boost_nids if n not in seen_boost]
            sem_ids = forced + sem_ids  # 强制节点排在最前

    sem_ids = _expand_seeds_with_graph_neighbors(
        repograph, sem_ids, issue_text, allowed, max_extra=neighbor_cap, neighbors=neighbors
    )

    # Candidate pool for BM25 (semantic + issue entities + 1-hop neighbors),
    # size-capped. Entity anchors are a recall channel, not a fault oracle.
    candidates: List[str] = []
    seen_c: Set[str] = set()
    pool_seeds = list(dict.fromkeys([*sem_ids, *entity_ids]))
    for nid in pool_seeds:
        if nid not in allowed or nid in seen_c:
            continue
        seen_c.add(nid)
        candidates.append(nid)
        if len(candidates) >= candidate_cap:
            break
    if len(candidates) < candidate_cap:
        for nid in pool_seeds:
            for nb in neighbors.get(nid, set()):
                if nb not in allowed or nb in seen_c:
                    continue
                seen_c.add(nb)
                candidates.append(nb)
                if len(candidates) >= candidate_cap:
                    break
            if len(candidates) >= candidate_cap:
                break

    # --- 2) BM25 on candidate pool ---
    sem_rank = [n for n in sem_ids if n in allowed]
    bm_order: List[str] = []
    if candidates:
        corpus_texts = [normalize_node_text(nid, repograph.nodes.get(nid, {})) for nid in candidates]
        corpus_tokens = [tokenize_for_bm25(t) for t in corpus_texts]
        bm25 = BM25Okapi(corpus_tokens)
        qtok = tokenize_for_bm25(issue_text)
        bm_scores = bm25.scores(qtok)
        bm_order = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: -bm_scores[i])]

    # --- 3) RRF fusion ---
    rank_lists = (
        [sem_rank]
        + ([bm_order] if bm_order else [])
        + ([entity_ids] if entity_ids else [])
    )
    fused = reciprocal_rank_fusion(rank_lists, k=rrf_k)
    fused_ids = [nid for nid, _ in fused[:fusion_top]]

    # --- 4) k-hop ego from top fused nodes (call-graph / structure expansion) ---
    centers = fused_ids[:center_cap] if fused_ids else sem_rank[: min(20, len(sem_rank))]
    # 把强制注入的文件节点也作为 ego 中心，确保它们的 k-hop 邻居被扩展进来
    if boost_nids:
        extra_centers = [n for n in boost_nids if n in allowed and n not in centers]
        centers = list(centers) + extra_centers[: max(0, center_cap - len(centers))]
    dist: Dict[str, int] = {}
    for center in centers:
        if center not in allowed:
            continue
        if filtered_neighbors is not None:
            ego_nodes = _ego_nodes_with_dist(neighbors, center=center, radius=hops)
        else:
            ego_nodes = repograph.ego_nodes_with_dist(center=center, radius=hops)
        for node_id, d in ego_nodes:
            if node_id not in allowed:
                continue
            if node_id not in dist or d < dist[node_id]:
                dist[node_id] = d

    tokens = issue_keyword_tokens(issue_text)
    priority: Dict[str, int] = {nid: i for i, nid in enumerate(fused_ids)}

    def _final_sort_key(it: Tuple[str, int]):
        nid, d = it
        pr = priority.get(nid, 10**9)
        data = repograph.nodes.get(nid, {})
        rel = str(data.get("rel_fname") or data.get("fname") or "")
        blob = normalize_node_text(nid, data).lower()
        kh = keyword_hits_in_text(blob, tokens)
        pp = path_priority_for_localization(rel)
        # Strengthened: path priority is now a primary sort dimension.
        # src/main/java nodes (pp=6.0) sort before distant/unrelated nodes.
        # Priority bucket: src/main/java first, then by fusion rank, then distance.
        bucket = 0 if pp >= 5.0 else (1 if pp >= 2.0 else (2 if pp >= 0.0 else 3))
        return (bucket, pr, d, -pp, -kh, nid)

    ordered_nodes = sorted(dist.items(), key=_final_sort_key)
    if force_file_paths and ncf != "none":
        filtered = _filter_ordered_nodes_for_files(repograph, ordered_nodes, list(force_file_paths), ncf)
        if filtered:
            ordered_nodes = filtered

    context_parts: List[str] = []
    used_nodes: Set[str] = set()
    used_node_order: List[str] = []
    total = 0

    for node_id, d in ordered_nodes:
        if node_id in used_nodes:
            continue
        used_nodes.add(node_id)
        used_node_order.append(node_id)
        node_text = normalize_node_text(node_id, repograph.nodes.get(node_id, {}))
        block = f"[dist={d}]\n{node_text}\n"
        context_parts.append(block)
        total += len(block)
        if total >= max_context_chars:
            break

    # Optional: second semantic pass (extra_code_k legacy) — only if room left
    room = max_context_chars - total
    extra_k = 0 if gran == "function" else (extra_code_k or _env_int("GRAPHRAG_EXTRA_SEMANTIC_K", 40))
    if room > 2000 and extra_k > 0:
        extra_docs = store.search_code_nodes(issue_text=issue_text, repo_key=repo_key, k=extra_k)
        for d in extra_docs:
            node_id = d.metadata.get("node_id")
            if not node_id or str(node_id) not in allowed:
                continue
            s = str(node_id)
            if s in used_nodes:
                continue
            node_text = normalize_node_text(s, repograph.nodes.get(s, {}))
            block = f"[semantic_extra]\n{node_text}\n"
            context_parts.append(block)
            total += len(block)
            used_nodes.add(s)
            used_node_order.append(s)
            if total >= max_context_chars:
                break

    context = "\n".join(context_parts)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]
    if not return_structured:
        return context

    selected_ids = set(used_nodes)
    typed_nodes: List[GraphNode] = []
    for node_id in used_node_order:
        payload = dict(repograph.nodes.get(node_id, {}))
        payload.setdefault("node_id", str(node_id))
        payload.setdefault("repo_key", repo_key)
        typed_nodes.append(GraphNode.model_validate(payload))
    typed_edges = [
        edge
        for edge in repograph.edges
        if edge.src in selected_ids and edge.dst in selected_ids
    ]
    valid_locations: Dict[str, List[int]] = {}
    for node in typed_nodes:
        if not node.file_path or node.line_start is None:
            continue
        valid_locations.setdefault(node.file_path, []).append(int(node.line_start))
    valid_locations = {
        path: sorted(set(lines)) for path, lines in valid_locations.items()
    }
    return GraphContext(
        context_id=f"{repo_key}:{gran}:{hashlib.sha256(context.encode('utf-8')).hexdigest()[:16]}",
        repo_key=repo_key,
        granularity=gran if gran in ("file", "function", "line") else "file",
        seed_entities=list(
            dict.fromkeys(
                [
                    *sem_ids[: min(len(sem_ids), 48)],
                    *entity_ids[: min(len(entity_ids), 16)],
                ]
            )
        ),
        hops=hops,
        edge_policy=[str(x).lower() for x in (edge_types or []) if x],
        nodes=typed_nodes,
        edges=typed_edges,
        entry_nodes=list(dict.fromkeys(centers[: min(len(centers), 32)])),
        candidate_nodes=list(dict.fromkeys(fused_ids[: min(len(fused_ids), 64)])),
        text_view=context,
        valid_locations=valid_locations,
        retrieval_meta={
            "source": getattr(repograph, "source", ""),
            "selected_node_count": len(typed_nodes),
            "selected_edge_count": len(typed_edges),
            "seed_count": len(sem_ids),
            "entity_anchor_count": len(entity_ids),
            "entity_anchor_node_ids": list(entity_ids[:64]),
            "center_count": len(centers),
            "retrieval_granularity": gran,
        },
    )


@tool
def graphrag_context(
    repograph_dir: str,
    repo_key: str,
    issue_text: str,
    hops: int = 2,
    seed_k: int = 120,
    extra_code_k: int = 0,
    max_context_chars: int = 0,
    vectorstore_path: str = config.VECTORSTORE_PATH,
    embeddings_model: str = config.CODE_EMBEDDING_MODEL,
    rebuild_vectorstore: bool = False,
    force_file_paths: Optional[List[str]] = None,
    retrieval_granularity: str = "file",
    edge_types: Optional[List[str]] = None,
) -> str:
    """
    GraphRAG context builder (hybrid) — LangChain tool 入口，逻辑见 ``graphrag_context_impl``。
    """
    return graphrag_context_impl(
        repograph_dir=repograph_dir,
        repo_key=repo_key,
        issue_text=issue_text,
        hops=hops,
        seed_k=seed_k,
        extra_code_k=extra_code_k,
        max_context_chars=max_context_chars,
        vectorstore_path=vectorstore_path,
        embeddings_model=embeddings_model,
        rebuild_vectorstore=rebuild_vectorstore,
        force_file_paths=force_file_paths,
        retrieval_granularity=retrieval_granularity,
        edge_types=edge_types,
    )
