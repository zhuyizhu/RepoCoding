"""
agent/agents/critic_agent.py

CriticAgent performs unsupervised static causal review. It does not route the
workflow and does not use oracle/gold labels. The score is strictly 100-point:
semantic alignment 30, topological reachability 40, causal validity 30.
"""

from __future__ import annotations

import logging
import re
import os
from collections import deque
from typing import Dict, Iterable, List, Optional, Set

from agent.agents.models import CriticEvaluation, GraphContext, GraphEdge, GraphPath, LocalizeResult

logger = logging.getLogger(__name__)


class CriticAgent:
    """Static critic: Issue + GraphContext + LocalizeResult -> CriticEvaluation."""

    def evaluate(
        self,
        *,
        issue_text: str,
        graph_context: GraphContext,
        localize_result: LocalizeResult,
    ) -> CriticEvaluation:
        logger.info(
            "[CriticAgent] evaluate start | granularity=%s | target=%s | nodes=%d | edges=%d",
            graph_context.granularity,
            localize_result.target.target,
            len(graph_context.nodes),
            len(graph_context.edges),
        )
        target_node_ids = self._resolve_target_nodes(graph_context, localize_result)
        line_whitelist_match = self._line_whitelist_match(graph_context, localize_result)
        semantic, semantic_notes = self._score_semantic(issue_text, graph_context, localize_result, target_node_ids)
        topology, paths, topo_notes, is_orphan = self._score_topology(graph_context, target_node_ids)
        causal, causal_notes = self._score_causal(issue_text, graph_context, localize_result, bool(paths), bool(target_node_ids))

        invalid = []
        if not target_node_ids:
            if localize_result.target.target_type == "line" and not line_whitelist_match:
                invalid.append(
                    "LocalizeAgent line target is absent from GraphContext.valid_locations."
                )
            else:
                invalid.append("LocalizeAgent target is not represented in the current graph context.")
        if is_orphan:
            invalid.append("Resolved target node is graph-orphaned under current subgraph.")
        if not paths:
            invalid.append("No explicit shortest path connects target to graph entry/error/candidate nodes.")

        critique = "\n".join(semantic_notes + topo_notes + causal_notes + invalid)
        retry_focus = self._build_retry_focus(semantic, topology, causal)
        recommended_action = self._recommend_graph_action(topology, target_node_ids, paths)
        result = CriticEvaluation(
            semantic_alignment=semantic,
            topological_reachability=topology,
            causal_validity=causal,
            critique=critique,
            reachable_paths=paths,
            invalid_assumptions=invalid,
            is_graph_orphan=is_orphan,
            retry_focus=retry_focus,
            recommended_graph_action=recommended_action,
            metadata={
                "target_node_ids": target_node_ids,
                "target_grounded": bool(target_node_ids),
                "line_whitelist_match": line_whitelist_match,
                "granularity": graph_context.granularity,
                "component_scores": {
                    "semantic_max_30": semantic,
                    "topology_max_40": topology,
                    "causal_max_30": causal,
                },
            },
        )
        logger.info(
            "[CriticAgent] evaluate done | total=%.2f | semantic=%.2f topology=%.2f causal=%.2f | orphan=%s",
            result.total_score,
            result.semantic_alignment,
            result.topological_reachability,
            result.causal_validity,
            result.is_graph_orphan,
        )
        return result

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        raw = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text or "")
        stop = {"the", "and", "for", "with", "from", "this", "that", "null", "true", "false"}
        return {x.lower() for x in raw if x.lower() not in stop}

    @staticmethod
    def _canonical_file_path(path: str) -> str:
        normalized = (path or "").strip().replace("\\", "/")
        if normalized.startswith(("a/", "b/")):
            normalized = normalized[2:]
        for anchor in ("src/main/java/", "src/test/java/", "src/"):
            index = normalized.find(anchor)
            if index >= 0:
                return normalized[index:]
        return normalized.lstrip("/")

    def _line_whitelist_match(
        self,
        graph_context: GraphContext,
        result: LocalizeResult,
    ) -> bool:
        target = result.target
        if target.target_type != "line":
            return True
        if not target.file_path or target.line_start is None:
            return False
        wanted_path = self._canonical_file_path(target.file_path)
        wanted_line = int(target.line_start)
        return any(
            self._canonical_file_path(path) == wanted_path
            and wanted_line in {int(line) for line in lines}
            for path, lines in graph_context.valid_locations.items()
        )

    def _resolve_target_nodes(self, graph_context: GraphContext, result: LocalizeResult) -> List[str]:
        target = result.target
        wanted_path = self._canonical_file_path(target.file_path)
        if target.target_type == "line":
            if not self._line_whitelist_match(graph_context, result):
                return []
            if not wanted_path or target.line_start is None:
                return []
            return [
                node.node_id
                for node in graph_context.nodes
                if self._canonical_file_path(node.file_path) == wanted_path
                and node.line_start is not None
                and int(node.line_start) == int(target.line_start)
            ][:8]

        if target.target_node_id:
            node_ids = {node.node_id for node in graph_context.nodes}
            if target.target_node_id in node_ids:
                return [target.target_node_id]

        wanted = [target.target, target.file_path, target.function_name]
        resolved: List[str] = []
        for node in graph_context.nodes:
            hay = "\n".join(
                str(x or "")
                for x in (
                    node.node_id,
                    node.file_path,
                    node.symbol_name,
                    node.qualified_name,
                    node.signature,
                )
            ).lower()
            if any(w and str(w).lower() in hay for w in wanted):
                resolved.append(node.node_id)
        return list(dict.fromkeys(resolved))[:8]

    def _score_semantic(
        self,
        issue_text: str,
        graph_context: GraphContext,
        result: LocalizeResult,
        target_node_ids: List[str],
    ) -> tuple[float, List[str]]:
        issue_tokens = self._tokens(issue_text)
        target_tokens = self._tokens(result.target.target + " " + result.hypothesis)
        node_blob = ""
        if target_node_ids:
            idset = set(target_node_ids)
            node_blob = "\n".join(
                " ".join([n.file_path, n.symbol_name, n.qualified_name, n.comment, n.signature, n.source_text_excerpt])
                for n in graph_context.nodes
                if n.node_id in idset
            )
            target_tokens |= self._tokens(node_blob)
        overlap = issue_tokens & target_tokens
        ratio = len(overlap) / max(1, min(len(issue_tokens), 18))
        score = min(30.0, round(ratio * 30.0, 2))
        if target_node_ids and score < 12.0:
            score = 12.0
        notes = [f"Semantic alignment: overlap={sorted(overlap)[:12]} score={score:.2f}/30"]
        return score, notes

    def _score_topology(
        self,
        graph_context: GraphContext,
        target_node_ids: List[str],
    ) -> tuple[float, List[GraphPath], List[str], bool]:
        if not target_node_ids:
            return 0.0, [], ["Topological reachability: target is absent from current graph."], False

        adjacency = self._adjacency(graph_context.edges)
        if not adjacency:
            adjacency = self._adjacency_from_context_nodes(graph_context)
        connected = {node for edge in graph_context.edges for node in (edge.src, edge.dst)}
        is_orphan = any(t not in connected and t not in adjacency for t in target_node_ids)
        if is_orphan:
            return 0.0, [], ["Topological reachability: target is an isolated node in current graph."], True

        anchors = list(dict.fromkeys(graph_context.known_error_nodes + graph_context.entry_nodes + graph_context.candidate_nodes))
        if not anchors:
            anchors = [n.node_id for n in graph_context.nodes[: min(12, len(graph_context.nodes))]]
        paths: List[GraphPath] = []
        for target in target_node_ids:
            for anchor in anchors:
                if anchor == target:
                    paths.append(GraphPath(source=anchor, target=target, node_ids=[target], path_length=0))
                    continue
                node_path = self._shortest_path(adjacency, anchor, target, max_depth=5)
                if node_path:
                    paths.append(self._build_graph_path(graph_context.edges, node_path))
                    break
            if paths:
                break

        if not paths:
            return 8.0, [], ["Topological reachability: target is in graph but no anchor path was found."], False

        best = min(paths, key=lambda p: p.path_length)
        if best.path_length <= 1:
            score = 40.0
        elif best.path_length == 2:
            score = 34.0
        elif best.path_length == 3:
            score = 28.0
        else:
            score = 22.0
        strong_edges = {"invokes", "implements", "extends", "contains"}
        if set(best.edge_types) & strong_edges:
            score = min(40.0, score + 4.0)
        notes = [
            f"Topological reachability: best_path_len={best.path_length} edge_types={best.edge_types} score={score:.2f}/40"
        ]
        return score, paths, notes, False

    @staticmethod
    def _adjacency(edges: Iterable[GraphEdge]) -> Dict[str, Set[str]]:
        adj: Dict[str, Set[str]] = {}
        for edge in edges:
            adj.setdefault(edge.src, set()).add(edge.dst)
            adj.setdefault(edge.dst, set()).add(edge.src)
        return adj

    @staticmethod
    def _adjacency_from_context_nodes(graph_context: GraphContext) -> Dict[str, Set[str]]:
        ids = [n.node_id for n in graph_context.nodes]
        return {node_id: set(ids[max(0, i - 1): i] + ids[i + 1: i + 2]) for i, node_id in enumerate(ids)}

    @staticmethod
    def _shortest_path(adjacency: Dict[str, Set[str]], source: str, target: str, max_depth: int) -> Optional[List[str]]:
        if source not in adjacency or target not in adjacency:
            return None
        queue: deque[tuple[str, List[str]]] = deque([(source, [source])])
        seen = {source}
        while queue:
            node, path = queue.popleft()
            if len(path) - 1 > max_depth:
                continue
            if node == target:
                return path
            for nxt in adjacency.get(node, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None

    @staticmethod
    def _build_graph_path(edges: List[GraphEdge], node_path: List[str]) -> GraphPath:
        used_edges: List[GraphEdge] = []
        for src, dst in zip(node_path, node_path[1:]):
            match = next((e for e in edges if (e.src == src and e.dst == dst) or (e.src == dst and e.dst == src)), None)
            if match:
                used_edges.append(match)
        return GraphPath(
            source=node_path[0],
            target=node_path[-1],
            node_ids=node_path,
            edges=used_edges,
            path_length=max(0, len(node_path) - 1),
            explanation=" -> ".join(node_path),
        )

    def _score_causal(
        self,
        issue_text: str,
        graph_context: GraphContext,
        result: LocalizeResult,
        has_path: bool,
        has_target: bool,
    ) -> tuple[float, List[str]]:
        hypothesis = result.hypothesis or ""
        issue_tokens = self._tokens(issue_text)
        hypo_tokens = self._tokens(hypothesis)
        overlap = issue_tokens & hypo_tokens
        score = 0.0
        if hypothesis.strip():
            score += 10.0
        if has_target:
            score += 6.0
        if has_path:
            score += 8.0
        score += min(6.0, len(overlap) * 1.5)
        if graph_context.granularity == result.target.target_type:
            score = min(30.0, score + 2.0)
        notes = [f"Causal validity: hypothesis_overlap={sorted(overlap)[:10]} score={score:.2f}/30"]

        range_note = self._range_span_warning(result.raw_output)
        if range_note:
            notes.append(range_note)
            score = max(0.0, score - 4.0)

        continuity_note = self._range_continuity_warning(result)
        if continuity_note:
            notes.append(continuity_note)
            score = max(0.0, score - 2.0)

        score = round(min(30.0, score), 2)
        return score, notes

    @staticmethod
    def _range_span_warning(raw_output: str) -> Optional[str]:
        if not raw_output:
            return None
        threshold = int(os.environ.get("JLOC_CRITIC_RANGE_WARN", "50") or "50")
        pattern = re.compile(r"行:\s*(\d+)\s*-\s*(\d+)")
        max_span = 0
        for line in raw_output.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                start, end = end, start
            max_span = max(max_span, end - start + 1)
        if max_span >= threshold:
            return (
                f"Causal validity check: predicted range spans {max_span} lines; "
                "is this too broad? Does it include safe logic that should not change?"
            )
        return None

    @staticmethod
    def _range_continuity_warning(result: LocalizeResult) -> Optional[str]:
        line_targets = [result.target] + list(result.alternatives or [])
        per_file: Dict[str, List[int]] = {}
        for t in line_targets:
            if t.target_type != "line":
                continue
            if t.file_path and t.line_start is not None:
                per_file.setdefault(t.file_path, []).append(int(t.line_start))
        for file_path, lines in per_file.items():
            if len(lines) < 6:
                continue
            uniq = sorted(set(lines))
            spans = 1
            for a, b in zip(uniq, uniq[1:]):
                if b != a + 1:
                    spans += 1
            if spans >= 3:
                return (
                    "Causal validity check: predicted lines are fragmented into multiple segments; "
                    "verify the causal chain and avoid including unrelated safe logic."
                )
        return None

    @staticmethod
    def _build_retry_focus(semantic: float, topology: float, causal: float) -> str:
        weakest = min((semantic, "semantic"), (topology, "topology"), (causal, "causal"), key=lambda x: x[0])[1]
        if weakest == "topology":
            return "Re-select a target that is reachable from entry/error/candidate nodes in the current graph."
        if weakest == "semantic":
            return "Re-select a target whose names/comments/entities better match the Issue symptoms."
        return "Rewrite the causal hypothesis so it explains the Issue through Java/framework execution semantics."

    @staticmethod
    def _recommend_graph_action(topology: float, target_node_ids: List[str], paths: List[GraphPath]) -> str:
        if not target_node_ids:
            return "expand_hops_and_extract_alternative_issue_entities"
        if topology < 12.0:
            return "regenerate_with_more_hops_and_edge_types:invokes,implements,extends"
        if not paths:
            return "regenerate_from_entry_or_error_nodes"
        return "keep_current_graph"
