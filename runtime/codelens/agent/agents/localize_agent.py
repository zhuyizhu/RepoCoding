"""
agent/agents/localize_agent.py

LocalizeAgent is the Actor in the upgraded workflow. Its only runtime
responsibility is to read the Issue plus graph context and emit a LocalizeResult.
It does not score, route, stop, or decide whether the graph should be rebuilt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agent.agents.models import GraphContext, LocalizeResult, TargetLocation

logger = logging.getLogger(__name__)

# Backward-compatible semantic alias. New code should use TargetLocation / LocalizeResult.
LocalizeCandidate = TargetLocation


class LocalizeAgent:
    """Actor agent: Issue + GraphContext -> LocalizeResult."""

    role_name = "localization"

    def __init__(self, max_retries: int = 2, k_top: int = 8) -> None:
        self.max_retries = max_retries
        self.k_top = k_top
        self.reasoning_history: List[Dict[str, Any]] = []

    @staticmethod
    def _context_text(graph_context: GraphContext | str) -> str:
        if isinstance(graph_context, GraphContext):
            return graph_context.text_view or "\n".join(
                node.to_legacy_payload().get("Info", "") for node in graph_context.nodes
            )
        return str(graph_context or "")

    def _call_llm_sync(
        self,
        issue_text: str,
        graph_context: GraphContext | str,
        few_shots: str,
        retry_block: str,
        valid_locations: Optional[Dict[str, Any]] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
        granularity: str = "file",
    ) -> tuple[str, str]:
        from agent.tools.localization_llm import run_localize_issue

        overrides = dict(llm_overrides or {})

        raw_output, prompt_used = run_localize_issue(
            issue_text=issue_text,
            graph_context=self._context_text(graph_context),
            few_shots=few_shots,
            k=self.k_top,
            refinement_feedback=retry_block.strip() or None,
            valid_locations=valid_locations,
            granularity=granularity,
            llm_temperature=overrides.get("temperature"),
            llm_seed=overrides.get("seed"),
            ollama_num_ctx=overrides.get("ollama_num_ctx"),
            return_prompt=True,
        )
        return raw_output, prompt_used

    @staticmethod
    def _parse_targets(raw_output: str, granularity: str) -> List[TargetLocation]:
        from agent.utils.localization_metrics import (
            extract_predicted_file_paths,
            extract_predicted_line_locations,
            extract_phase1_function_candidates,
        )

        targets: List[TargetLocation] = []
        g = (granularity or "file").strip().lower()
        if g == "line":
            for file_path, line_num in extract_predicted_line_locations(raw_output):
                targets.append(
                    TargetLocation(
                        target=f"{file_path}:{line_num}",
                        target_type="line",
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        confidence=0.0,
                    )
                )
            if targets:
                return targets

        if g == "function":
            for obj in extract_phase1_function_candidates(raw_output):
                targets.append(
                    TargetLocation(
                        target=f"{obj['file_path']}::{obj['function_name']}",
                        target_type="function",
                        file_path=obj["file_path"],
                        function_name=obj["function_name"],
                        line_start=obj.get("line_start"),
                        line_end=obj.get("line_end"),
                        confidence=0.0,
                    )
                )
            if targets:
                return targets

        if g == "file":
            for file_path in extract_predicted_file_paths(raw_output):
                targets.append(
                    TargetLocation(
                        target=file_path,
                        target_type="file",
                        file_path=file_path,
                        confidence=0.0,
                    )
                )
            if targets:
                return targets

        for file_path, line_num in extract_predicted_line_locations(raw_output):
            targets.append(
                TargetLocation(
                    target=f"{file_path}:{line_num}",
                    target_type="line",
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    confidence=0.0,
                )
            )
        if targets:
            return targets

        for file_path in extract_predicted_file_paths(raw_output):
            targets.append(
                TargetLocation(
                    target=file_path,
                    target_type="file",
                    file_path=file_path,
                    confidence=0.0,
                )
            )
        return targets

    @staticmethod
    def _extract_hypothesis(raw_output: str) -> str:
        lines = [line.strip() for line in (raw_output or "").splitlines() if line.strip()]
        if not lines:
            return ""
        hypothesis_markers = ("hypothesis", "假设", "原因", "推演", "reason")
        selected = [line for line in lines if any(m in line.lower() for m in hypothesis_markers)]
        text = "\n".join(selected[:6] or lines[:8])
        return text[:1600]

    async def localize(
        self,
        issue_text: str,
        graph_context: GraphContext | str,
        *,
        few_shots: str = "",
        retry_block: str = "",
        valid_locations: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        context: Optional[Dict[str, Any]] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
        granularity: str = "file",
    ) -> LocalizeResult:
        logger.info(
            "[LocalizeAgent] localize start | retry=%d | graph_type=%s | ctx_chars=%d",
            retry_count,
            type(graph_context).__name__,
            len(self._context_text(graph_context)),
        )
        loop = asyncio.get_running_loop()
        raw_output, prompt_used = await loop.run_in_executor(
            None,
            self._call_llm_sync,
            issue_text,
            graph_context,
            few_shots,
            retry_block,
            valid_locations,
            llm_overrides,
            granularity,
        )

        targets = self._parse_targets(raw_output, granularity)
        primary = targets[0] if targets else TargetLocation(target="", target_type="file")
        result = LocalizeResult(
            target=primary,
            hypothesis=self._extract_hypothesis(raw_output),
            graph_evidence=[],
            alternatives=targets[1:],
            prompt_used=prompt_used,
            raw_output=raw_output,
            retry_count=retry_count,
            metadata={"global_context": context or {}},
        )
        logger.info(
            "[LocalizeAgent] localize done | retry=%d | target=%s | alternatives=%d",
            retry_count,
            result.target.target,
            len(result.alternatives),
        )
        self.reasoning_history.append({"granularity": granularity, "target": result.target.target,
                                       "reasoning": result.raw_output[:1400]})
        return result

    async def verify(self, **kwargs):
        from agent.agents.verification_agent import active_verify
        return await active_verify(self, **kwargs)

    async def run(
        self,
        issue_text: str,
        graph_context: GraphContext | str,
        few_shots: str = "",
        retry_block: str = "",
        valid_locations: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        context: Optional[Dict[str, Any]] = None,
        llm_overrides: Optional[Dict[str, Any]] = None,
        granularity: str = "file",
    ) -> LocalizeResult:
        if valid_locations:
            logger.info(
                "[LocalizeAgent] injecting valid_locations into the prompt; CriticAgent will enforce grounding"
            )
        return await self.localize(
            issue_text=issue_text,
            graph_context=graph_context,
            few_shots=few_shots,
            retry_block=retry_block,
            valid_locations=valid_locations,
            retry_count=retry_count,
            context=context,
            llm_overrides=llm_overrides,
            granularity=granularity,
        )

    def can_retry(self, current_retry: int) -> bool:
        return current_retry < self.max_retries
