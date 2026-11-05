"""
agent/agents/reflect_agent.py

ReflectAgent is the controller in the upgraded workflow. It does not score.
It routes strictly from CriticEvaluation.total_score:
  - A: score >= 80 -> PASS
  - B: 40 <= score < 80 -> RETRY on the same graph, up to 2 times
  - C: score < 40 -> REGENERATE graph
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.agents.models import CriticEvaluation, Granularity, ReflectionAction, ReflectionDecision

logger = logging.getLogger(__name__)

# Backward-compatible semantic alias. New code should use ReflectionDecision.
ReflectOutput = ReflectionDecision


class ReflectAgent:
    """Controller agent: CriticEvaluation -> ReflectionDecision."""

    def __init__(self, max_localize_retries: int = 2, tag_rows: Optional[list[dict[str, Any]]] = None) -> None:
        self.max_localize_retries = max_localize_retries
        self.tag_rows = tag_rows or []

    def decide(
        self,
        *,
        evaluation: CriticEvaluation,
        current_granularity: Granularity,
        retry_count: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReflectionDecision:
        score = float(evaluation.total_score)
        logger.info(
            "[ReflectAgent] route start | granularity=%s | score=%.2f | retry_count=%d/%d",
            current_granularity,
            score,
            retry_count,
            self.max_localize_retries,
        )

        if score >= 80.0:
            next_granularity = self._next_granularity(current_granularity)
            decision = ReflectionDecision(
                action=ReflectionAction.PASS,
                current_granularity=current_granularity,
                next_granularity=next_granularity,
                retry_count=retry_count,
                reason=f"Threshold A met: score={score:.2f} >= 80. Pass to next granularity.",
                memory_candidate=self._build_memory_candidate(evaluation, context),
            )
            logger.info(
                "[ReflectAgent] route decision=PASS | current=%s | next=%s | score=%.2f",
                current_granularity,
                next_granularity,
                score,
            )
            return decision

        if score >= 40.0 and retry_count < self.max_localize_retries:
            next_retry = retry_count + 1
            prompt = self._build_retry_prompt(evaluation, current_granularity, next_retry)
            decision = ReflectionDecision(
                action=ReflectionAction.RETRY,
                current_granularity=current_granularity,
                next_granularity=current_granularity,
                retry_count=next_retry,
                retry_prompt=prompt,
                reason=(
                    f"Threshold B met: 40 <= score={score:.2f} < 80. "
                    f"Retry LocalizeAgent on the same graph ({next_retry}/{self.max_localize_retries})."
                ),
            )
            logger.info(
                "[ReflectAgent] route decision=RETRY | granularity=%s | next_retry=%d | score=%.2f",
                current_granularity,
                next_retry,
                score,
            )
            return decision

        if score >= 40.0:
            logger.info(
                "[ReflectAgent] retry budget exhausted | forcing REGENERATE | granularity=%s | score=%.2f",
                current_granularity,
                score,
            )

        decision = ReflectionDecision(
            action=ReflectionAction.REGENERATE,
            current_granularity=current_granularity,
            next_granularity=current_granularity,
            retry_count=0,
            backtrack_signal="BACKTRACK",
            graph_regeneration_request=self._build_regeneration_request(evaluation, current_granularity),
            reason=f"Threshold C path: score={score:.2f} or retry budget exhausted. Regenerate current graph.",
        )
        logger.info(
            "[ReflectAgent] route decision=REGENERATE | granularity=%s | score=%.2f | request=%s",
            current_granularity,
            score,
            decision.graph_regeneration_request,
        )
        return decision

    async def reflect(
        self,
        *,
        evaluation: CriticEvaluation,
        current_granularity: Granularity,
        retry_count: int = 0,
        context: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> ReflectionDecision:
        return self.decide(
            evaluation=evaluation,
            current_granularity=current_granularity,
            retry_count=retry_count,
            context=context,
        )

    @staticmethod
    def _next_granularity(current: Granularity) -> Optional[Granularity]:
        if current == "file":
            return "function"
        if current == "function":
            return "line"
        return None

    @staticmethod
    def _build_retry_prompt(evaluation: CriticEvaluation, granularity: Granularity, retry_count: int) -> str:
        return (
            f"Critic score is {evaluation.total_score:.2f}/100 at {granularity} granularity.\n"
            f"Retry #{retry_count}: stay within the current graph; do not request new graph context.\n"
            f"Critic focus: {evaluation.retry_focus}\n"
            f"Critique:\n{evaluation.critique[:1600]}\n"
            "Select a different target node if the current one is weak, and provide a tighter causal hypothesis."
        )

    @staticmethod
    def _build_regeneration_request(evaluation: CriticEvaluation, granularity: Granularity) -> Dict[str, Any]:
        edge_types = ["invokes", "implements", "extends", "contains"]
        if "implements" in evaluation.recommended_graph_action or "extends" in evaluation.recommended_graph_action:
            edge_types = ["implements", "extends", "invokes", "contains"]
        return {
            "granularity": granularity,
            "signal": "BACKTRACK",
            "increase_hops": True,
            "hop_delta": 2,
            "edge_types": edge_types,
            "strategy": evaluation.recommended_graph_action or "expand_hops_and_alternative_edges",
            "critique_excerpt": evaluation.critique[:1200],
        }

    @staticmethod
    def _build_memory_candidate(evaluation: CriticEvaluation, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "critic_score": evaluation.total_score,
            "critic_components": {
                "semantic_alignment": evaluation.semantic_alignment,
                "topological_reachability": evaluation.topological_reachability,
                "causal_validity": evaluation.causal_validity,
            },
            "context": context or {},
        }