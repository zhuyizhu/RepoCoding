"""Gold-blind, closed-set verification for file localization candidates.

The verifier is deliberately narrower than the Actor.  It cannot retrieve or
name a new file: it only reorders the Actor's candidates after comparing their
candidate-specific graph evidence with the linked issue.  Keeping generation
and verification independent reduces confirmation bias without exposing any
benchmark label, patch, pull-request text, or changed-file list.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

from agent.agents.models import GraphContext, LocalizeResult, TargetLocation
from agent.utils.localization_metrics import match_file, normalize_code_path

logger = logging.getLogger(__name__)


def _file_candidates(result: LocalizeResult, limit: int = 0) -> List[TargetLocation]:
    candidates: List[TargetLocation] = []
    seen: set[str] = set()
    for candidate in [result.target, *list(result.alternatives or [])]:
        file_path = normalize_code_path(candidate.file_path or candidate.target)
        if not file_path or file_path in seen:
            continue
        seen.add(file_path)
        copied = candidate.model_copy(deep=True)
        copied.file_path = file_path
        copied.target = file_path
        copied.target_type = "file"
        candidates.append(copied)
        if limit > 0 and len(candidates) >= limit:
            break
    return candidates


def _candidate_evidence(
    graph_context: GraphContext,
    candidate_path: str,
    max_chars: int,
) -> str:
    """Extract only blocks belonging to one candidate from the observed graph."""
    selected: List[str] = []
    used = 0
    cap = max(200, max_chars)
    for block in re.split(r"(?=\[dist=)", graph_context.text_view or ""):
        match = re.search(r"(?m)^File:\s*(\S+)\s*$", block)
        if not match or not match_file(normalize_code_path(match.group(1)), {candidate_path}):
            continue
        snippet = block.strip()
        if not snippet:
            continue
        remaining = cap - used
        if remaining <= 0:
            break
        selected.append(snippet[:remaining])
        used += min(len(snippet), remaining)

    if not selected:
        # Structured fallback for contexts whose text renderer was truncated.
        for node in graph_context.nodes:
            node_path = normalize_code_path(node.rel_fname or "")
            if not node_path or not match_file(node_path, {candidate_path}):
                continue
            payload = json.dumps(node.to_legacy_payload(), ensure_ascii=False)
            remaining = cap - used
            if remaining <= 0:
                break
            selected.append(payload[:remaining])
            used += min(len(payload), remaining)

    return "\n\n".join(selected) or "(No candidate-specific graph node survived context truncation.)"


def build_file_verifier_prompt(
    issue_text: str,
    candidates: Sequence[TargetLocation],
    graph_context: GraphContext,
    *,
    evidence_chars_per_candidate: int,
) -> str:
    evidence_blocks = []
    for index, candidate in enumerate(candidates, 1):
        evidence_blocks.append(
            f"### C{index}: {candidate.file_path}\n"
            + _candidate_evidence(
                graph_context,
                candidate.file_path,
                evidence_chars_per_candidate,
            )
        )
    candidate_ids = [f"C{index}" for index in range(1, len(candidates) + 1)]
    max_selected = min(3, len(candidates))
    return f"""You are a Java fault-localization verifier. A separate Actor generated a CLOSED candidate set. Re-rank it by whether editing that file can causally explain and fix the linked issue.

Strict evidence boundary:
- Use only the linked issue and candidate-specific graph evidence below.
- Never infer or request a pull request, patch, changed-file list, benchmark hint, test result, or gold label.
- A name appearing in the issue is supporting evidence, not proof. Penalize files that merely share a term but do not own the failing behavior.
- Prefer implementation ownership and an executable causal path over generic API prominence.
- Consider counter-evidence for every candidate. Preserve the original order when evidence is genuinely tied.
- Return candidate IDs only. You cannot introduce a file outside the closed set.

Selection semantics:
- selected_count is the smallest prefix whose files are likely to require actual code edits for this issue.
- Do not select a file merely because it supplies context, calls the target, or shares terminology.

Internally compare direct entity grounding, behavioral ownership, graph/signature support, edit plausibility, and counter-evidence. Return JSON only:
{{"ranking": {candidate_ids!r}, "selected_count": <integer 1..{max_selected}>, "confidence": <0..1>, "reason": "one short sentence"}}

Linked issue:
{issue_text[:12000]}

Candidate-specific graph evidence:
{chr(10).join(evidence_blocks)}
"""


def parse_closed_set_ranking(
    text: str,
    candidate_count: int,
) -> Tuple[List[int], int, Dict[str, Any]]:
    """Parse a complete or partial candidate-ID permutation without path trust."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError("verifier output contains no JSON object")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("verifier JSON must be an object")

    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("verifier ranking must be a non-empty list")
    expected = [f"C{index}" for index in range(1, candidate_count + 1)]
    normalized = [str(item).strip().upper() for item in ranking]
    if len(set(normalized)) != len(normalized) or any(item not in expected for item in normalized):
        raise ValueError("verifier ranking violates the closed candidate set")

    # Qwen may emit only the plausible prefix.  Append omissions in Actor order;
    # this remains a deterministic permutation and does not manufacture evidence.
    complete = normalized + [item for item in expected if item not in normalized]
    indices = [int(item[1:]) - 1 for item in complete]
    recommended = len(normalized)
    try:
        selected_count = int(payload.get("selected_count", recommended))
    except (TypeError, ValueError):
        selected_count = recommended
    selected_count = max(1, min(min(3, candidate_count), selected_count))
    metadata = {
        "confidence": payload.get("confidence"),
        "reason": str(payload.get("reason") or "")[:500],
    }
    return indices, selected_count, metadata


def _message_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
        if chunks:
            return "\n".join(chunks)
    return str(content or response or "")


class FileVerifierAgent:
    """Independent one-shot verifier for the first file-level Actor ranking."""

    def __init__(
        self,
        *,
        max_candidates: int = 5,
        evidence_chars_per_candidate: int = 3000,
        seed: int = 20260909,
    ) -> None:
        self.max_candidates = max(2, max_candidates)
        self.evidence_chars_per_candidate = max(200, evidence_chars_per_candidate)
        self.seed = seed

    def _run_sync(
        self,
        issue_text: str,
        graph_context: GraphContext,
        candidates: Sequence[TargetLocation],
    ) -> Tuple[str, str]:
        from agent.tools.localization_llm import invoke_llm_with_ollama_fallback

        prompt = build_file_verifier_prompt(
            issue_text,
            candidates,
            graph_context,
            evidence_chars_per_candidate=self.evidence_chars_per_candidate,
        )
        response = invoke_llm_with_ollama_fallback(
            prompt,
            call_site="file_candidate_verifier",
            temperature=0.0,
            seed=self.seed,
        )
        return _message_text(response), prompt

    async def run(
        self,
        *,
        issue_text: str,
        graph_context: GraphContext,
        localize_result: LocalizeResult,
    ) -> Tuple[LocalizeResult, Dict[str, Any]]:
        all_candidates = _file_candidates(localize_result)
        candidates = all_candidates[: self.max_candidates]
        untouched_tail = all_candidates[self.max_candidates :]
        original = [candidate.file_path for candidate in candidates]
        history: Dict[str, Any] = {
            "agent": "closed_set_file_verifier",
            "evidence_boundary": "resolved_issue_plus_candidate_graph_only",
            "original_ranking": original,
            "reranked": list(original),
            "status": "skipped",
        }
        if len(candidates) < 2 or not (issue_text or "").strip():
            history["reason"] = "fewer_than_two_candidates_or_empty_issue"
            return localize_result, history

        try:
            loop = asyncio.get_running_loop()
            raw_output, prompt = await loop.run_in_executor(
                None,
                self._run_sync,
                issue_text,
                graph_context,
                candidates,
            )
            indices, selected_count, parsed = parse_closed_set_ranking(
                raw_output,
                len(candidates),
            )
            ranked_head = [candidates[index].model_copy(deep=True) for index in indices]
            ranked = [*ranked_head, *[item.model_copy(deep=True) for item in untouched_tail]]
            result = localize_result.model_copy(deep=True)
            result.target = ranked[0]
            result.alternatives = ranked[1:]
            verifier_metadata = {
                "status": "ok",
                "selected_count": selected_count,
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
                "original_ranking": original,
                "reranked": [candidate.file_path for candidate in ranked],
                "selected_candidates": [
                    candidate.file_path for candidate in ranked[:selected_count]
                ],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_output": raw_output[:4000],
                "evidence_boundary": history["evidence_boundary"],
            }
            result.metadata = {**dict(result.metadata or {}), "file_verifier": verifier_metadata}
            history.update(verifier_metadata)
            return result, history
        except Exception as exc:
            logger.warning("[FileVerifierAgent] falling back to Actor order: %s", exc)
            history.update(
                {
                    "status": "fallback",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            return localize_result, history
