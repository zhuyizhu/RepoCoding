"""
agent/agents/retrieval_agent.py

RetrievalAgent：为 Issue 生成 GraphRAG 子图上下文（hybrid：dense + BM25 + RRF + k-hop）。

与论文流程对齐的三级子图（由 ``retrieval_granularity`` 区分，并参与 ``subgraph_key`` 哈希）：
    - ``file``：文件级子图（对 LocalizeAgent 给出的怀疑路径 ``force_file_paths`` 强制注入节点与 ego 中心）。
    - ``function``：在 ``force_file_paths`` 窄文件内，优先保留 **class + function(def)** 节点，弱化 ref 噪声。
    - ``line``：行级增强子图（仍 hybrid；在 ``graphrag_context_impl`` 内收紧 hop、放大种子与邻域，
        使上下文更贴近可疑文件附近的符号，利于行号对齐）。

「同一粒度不重复生成」：``subgraph_key`` 含粒度标签；若 key 已在 ``seen_subgraph_keys`` 且
进程内 ``_context_cache`` 命中，则直接复用，不再次调用 ``graphrag_context_impl``。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from agent.agents.models import GraphContext, GraphEdge, GraphNode

logger = logging.getLogger(__name__)

RetrievalGranularity = Literal["file", "function", "line"]


class RetrievalAgent:
    """异步 GraphRAG 检索 Agent（独立模块，可 ``asyncio`` 调用）。"""

    def __init__(
        self,
        repograph_dir: str,
        repo_key: str,
        vectorstore_path: str = "",
        embeddings_model: str = "",
        hops: int = 2,
        seed_k: int = 120,
        max_context_chars: int = 0,
        rebuild_vectorstore: bool = False,
    ) -> None:
        import config as _cfg

        self.repograph_dir = repograph_dir
        self.repo_key = repo_key
        self.vectorstore_path = vectorstore_path or _cfg.VECTORSTORE_PATH
        self.embeddings_model = embeddings_model or _cfg.CODE_EMBEDDING_MODEL
        self.hops = hops
        self.seed_k = seed_k
        self.max_context_chars = max_context_chars
        self.rebuild_vectorstore = rebuild_vectorstore
        self._context_cache: Dict[str, str] = {}
        self._structured_context_cache: Dict[str, GraphContext] = {}

    @staticmethod
    def build_subgraph_key(
        issue_text: str,
        granularity: str,
        force_file_paths: Optional[List[str]] = None,
        hops: int = 2,
        seed_k: int = 120,
        node_context_filter: str = "none",
        edge_types: Optional[List[str]] = None,
    ) -> str:
        """子图唯一 key：issue 摘要 + 粒度 + hop/seed + 强制文件集合 + 节点过滤。"""
        issue_hash = hashlib.md5(issue_text.encode("utf-8")).hexdigest()[:12]
        files_sig = ",".join(sorted(force_file_paths or []))
        g = (granularity or "file").strip().lower()
        nf = (node_context_filter or "none").strip().lower()
        et = ",".join(sorted([str(x).lower() for x in (edge_types or []) if x]))
        raw = f"{issue_hash}|g={g}|h{hops}|k{seed_k}|nf={nf}|et={et}|{files_sig}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _build_context_sync(
        self,
        issue_text: str,
        force_file_paths: Optional[List[str]],
        granularity: str,
        node_context_filter: str = "none",
        edge_types: Optional[List[str]] = None,
        hops: Optional[int] = None,
    ) -> Tuple[GraphContext, Dict[str, Any]]:
        from agent.tools.graph_rag_tools import graphrag_context_impl

        structured = graphrag_context_impl(
            repograph_dir=self.repograph_dir,
            repo_key=self.repo_key,
            issue_text=issue_text,
            hops=self.hops if hops is None else max(0, int(hops)),
            seed_k=self.seed_k,
            max_context_chars=self.max_context_chars,
            vectorstore_path=self.vectorstore_path,
            embeddings_model=self.embeddings_model,
            rebuild_vectorstore=self.rebuild_vectorstore,
            force_file_paths=force_file_paths or None,
            retrieval_granularity=granularity,
            node_context_filter=node_context_filter,
            edge_types=edge_types,
            return_structured=True,
        )
        if not isinstance(structured, GraphContext):
            raise TypeError("GraphRAG structured retrieval did not return GraphContext")
        context = structured.text_view
        lines = context.splitlines()
        meta: Dict[str, Any] = {
            "context_chars": len(context),
            "context_lines": len(lines),
            "file_blocks": sum(1 for l in lines if l.startswith("File:")),
            "forced_files": len(force_file_paths or []),
            "retrieval_granularity": granularity,
            "node_count": len(structured.nodes),
            "edge_count": len(structured.edges),
        }
        meta.update(structured.retrieval_meta)
        return structured, meta

    async def retrieve(
        self,
        issue_text: str,
        force_file_paths: Optional[List[str]] = None,
        seen_subgraph_keys: Optional[Set[str]] = None,
        *,
        granularity: RetrievalGranularity = "file",
        node_context_filter: str = "none",
        hops_override: Optional[int] = None,
        edge_types: Optional[List[str]] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        返回 ``(subgraph_key, graph_context, meta)``。

        ``granularity``：``file`` | ``function`` | ``line``，驱动 ``graphrag_context_impl`` 内不同子图策略。
        ``node_context_filter``：``none`` | ``defs_only``，在 ``line`` 等粒度下进一步只保留类与方法 def 节点。
        """
        g = (granularity or "file").strip().lower()
        if g not in ("file", "function", "line"):
            g = "file"

        effective_hops = self.hops if hops_override is None else max(0, int(hops_override))
        key = self.build_subgraph_key(
            issue_text,
            g,
            force_file_paths,
            effective_hops,
            self.seed_k,
            (node_context_filter or "none").strip().lower(),
            edge_types,
        )

        if seen_subgraph_keys and key in seen_subgraph_keys:
            cached = self._context_cache.get(key, "")
            if cached:
                cached_structured = self._structured_context_cache.get(key)
                logger.info(
                    "[RetrievalAgent] 同粒度子图已存在 key=%s g=%s，复用进程内缓存",
                    key,
                    g,
                )
                meta: Dict[str, Any] = {
                    "context_chars": len(cached),
                    "from_cache": True,
                    "forced_files": len(force_file_paths or []),
                    "retrieval_granularity": g,
                    "node_count": len(cached_structured.nodes) if cached_structured else 0,
                    "edge_count": len(cached_structured.edges) if cached_structured else 0,
                }
                return key, cached, meta

        logger.info(
            "[RetrievalAgent] 构建子图 key=%s g=%s force=%s",
            key,
            g,
            force_file_paths or [],
        )
        loop = asyncio.get_running_loop()
        structured, meta = await loop.run_in_executor(
            None,
            self._build_context_sync,
            issue_text,
            force_file_paths,
            g,
            (node_context_filter or "none").strip().lower(),
            edge_types,
            effective_hops,
        )
        context = structured.text_view
        meta["from_cache"] = False
        self._context_cache[key] = context
        self._structured_context_cache[key] = structured
        logger.info(
            "[RetrievalAgent] 子图完成 key=%s g=%s | chars=%d | file_blocks=%d",
            key,
            g,
            meta["context_chars"],
            meta.get("file_blocks", -1),
        )
        return key, context, meta

    def get_structured_context(self, subgraph_key: str) -> Optional[GraphContext]:
        """Return the exact typed subgraph produced for ``subgraph_key``."""
        return self._structured_context_cache.get(subgraph_key)

    def merge_structured_contexts(
        self,
        subgraph_keys: List[str],
        *,
        text_view: str,
        granularity: RetrievalGranularity,
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> GraphContext:
        """Merge typed retrieved subgraphs without inventing nodes or edges from prompt text."""
        contexts = [
            self._structured_context_cache[key]
            for key in subgraph_keys
            if key and key in self._structured_context_cache
        ]
        nodes_by_id: Dict[str, GraphNode] = {}
        edges_by_key: Dict[Tuple[str, str, str], GraphEdge] = {}
        entry_nodes: List[str] = []
        candidate_nodes: List[str] = []
        valid_locations: Dict[str, Set[int]] = {}
        for context in contexts:
            for node in context.nodes:
                nodes_by_id[node.node_id] = node
            for edge in context.edges:
                edges_by_key[(edge.src, edge.dst, edge.edge_type)] = edge
            entry_nodes.extend(context.entry_nodes)
            candidate_nodes.extend(context.candidate_nodes)
            for path, lines in context.valid_locations.items():
                valid_locations.setdefault(path, set()).update(lines)
        meta = dict(retrieval_meta or {})
        meta.update(
            {
                "source_subgraph_keys": [key for key in subgraph_keys if key],
                "node_count": len(nodes_by_id),
                "edge_count": len(edges_by_key),
            }
        )
        return GraphContext(
            repo_key=self.repo_key,
            granularity=granularity,
            hops=self.hops,
            nodes=list(nodes_by_id.values()),
            edges=list(edges_by_key.values()),
            entry_nodes=list(dict.fromkeys(entry_nodes)),
            candidate_nodes=list(dict.fromkeys(candidate_nodes)),
            text_view=text_view,
            valid_locations={path: sorted(lines) for path, lines in valid_locations.items()},
            retrieval_meta=meta,
        )

    @staticmethod
    def extract_valid_locations(graph_context: str) -> Dict[str, Set[int]]:
        from agent.utils.localization_metrics import extract_valid_locations_from_context

        return extract_valid_locations_from_context(graph_context)

    @staticmethod
    def extract_mentioned_files(graph_context: str) -> List[str]:
        files: List[str] = []
        for line in graph_context.splitlines():
            if line.startswith("File:"):
                fp = line[5:].strip()
                if fp:
                    files.append(fp)
        return list(dict.fromkeys(files))
