"""E: auditable candidate checks. Missing evidence stays unavailable."""
from __future__ import annotations

from agent.agents.critic_agent import CriticAgent
from agent.agents.models import CriticEvaluation


class EvidenceInspector(CriticAgent):
    def __init__(self, structured=True):
        self.structured = structured

    def _resolve_target_nodes(self, context, result):
        target = result.target
        path = self._canonical_file_path(target.file_path or target.target)
        if target.target_type == "line" and not self._line_whitelist_match(context, result):
            return []
        matches = []
        for node in context.nodes:
            if self._canonical_file_path(node.file_path) != path:
                continue
            if target.target_type == "function":
                if node.category != "function" or target.function_name not in {node.symbol_name, node.qualified_name}:
                    continue
            if target.target_type == "line":
                if node.line_start != target.line_start:
                    continue
            matches.append(node.node_id)
        return matches[:8]

    def _score_topology(self, context, target_node_ids):
        if not target_node_ids:
            return 0.0, [], ["Target not observed in this context."], True
        if not context.edges:
            return 0.0, [], ["Graph edges unobserved; no synthetic adjacency is permitted."], True
        # Exclude the target itself from anchors: a zero-hop self path is not support.
        independent = context.model_copy(deep=True)
        excluded = set(target_node_ids)
        independent.entry_nodes = [n for n in context.entry_nodes if n not in excluded]
        independent.known_error_nodes = [n for n in context.known_error_nodes if n not in excluded]
        independent.candidate_nodes = [n for n in context.candidate_nodes if n not in excluded]
        if not (independent.entry_nodes or independent.known_error_nodes or independent.candidate_nodes):
            return 0.0, [], ["No independent graph anchor observed."], True
        return super()._score_topology(independent, target_node_ids)

    @staticmethod
    def _adjacency(edges):
        graph = {}
        for edge in edges:
            graph.setdefault(edge.src, set()).add(edge.dst)
            graph.setdefault(edge.dst, set())
            if not edge.directed:
                graph[edge.dst].add(edge.src)
        return graph

    def evaluate(self, *, issue_text, graph_context, localize_result):
        if self.structured:
            result = super().evaluate(issue_text=issue_text, graph_context=graph_context,
                                      localize_result=localize_result)
            result.metadata["available_channels"] = ["semantic", "grounding", "heuristic_support"]
            if result.reachable_paths:
                result.metadata["available_channels"].append("topology")
            result.metadata["evidence_kind"] = "static_support_not_causal_proof"
        else:
            ids = self._resolve_target_nodes(graph_context, localize_result)
            judgment = localize_result.metadata.get("verification_judgment", {})
            quality = {"supported": 0.75, "rejected": 0.25, "unknown": 0.5}.get(judgment.get("verdict"), 0.5)
            result = CriticEvaluation(
                semantic_alignment=30 * quality, causal_validity=30 * quality,
                critique="Basic source grounding and model judgment; structural channels unavailable.",
                metadata={"target_grounded": bool(ids), "target_node_ids": ids,
                          "available_channels": ["grounding", "model_judgment"],
                          "quality_override": quality})
        result.metadata["inspector"] = "structured" if self.structured else "basic"
        result.metadata["verification_judgment"] = localize_result.metadata.get("verification_judgment", {})
        return result

    def inspect(self, issue_text, context, candidates):
        rows = []
        for candidate in [candidates.target, *candidates.alternatives][:5]:
            item = candidates.model_copy(update={"target": candidate})
            evaluated = self.evaluate(issue_text=issue_text, graph_context=context, localize_result=item)
            rows.append({"target": candidate.target, "grounded": evaluated.metadata["target_grounded"],
                         "channels": evaluated.metadata["available_channels"],
                         "paths": [p.model_dump() for p in evaluated.reachable_paths],
                         "notes": evaluated.critique[:500]})
        return rows
