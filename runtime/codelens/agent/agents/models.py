"""
Unified data contracts for the multi-agent localization workflow.

These models define the data flow for the upgraded architecture:
RetrievalAgent -> LocalizeAgent -> CriticAgent -> ReflectAgent.
They are intentionally compatible with legacy RepoGraph node dictionaries so the
existing graph loading and prompt formatting code can migrate incrementally.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


Granularity = Literal["file", "function", "line"]
NodeCategory = Literal["file", "class", "function", "field", "statement", "unknown"]
NodeKind = Literal["def", "ref", "unknown"]
EdgeType = Literal[
    "contains",
    "invokes",
    "implements",
    "extends",
    "overrides",
    "references",
    "data_flow",
    "unknown",
]


class ReflectionAction(str, Enum):
    """Routing actions emitted by ReflectAgent."""

    PASS = "PASS"
    RETRY = "RETRY"
    REGENERATE = "REGENERATE"


class GraphNode(BaseModel):
    """
    Typed view of a Java RepoGraph node.

    The field names accept legacy payloads produced by ``construct_graph.py`` and
    ``repograph_loader.py``: ``rel_fname``, ``fname``, ``name``, ``symbol_name``,
    ``line``, ``line_str``, ``kind``, ``category``, ``comment`` and ``annotations``.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    node_id: str = ""
    repo_key: str = ""
    file_path: str = Field(default="", alias="rel_fname")
    absolute_path: str = Field(default="", alias="fname")
    symbol_name: str = ""
    qualified_name: str = ""
    granularity: Granularity = "file"
    category: NodeCategory = "unknown"
    kind: NodeKind = "unknown"
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    line_text: str = ""
    signature: str = ""
    comment: str = ""
    annotations: List[str] = Field(default_factory=list)
    source_text_excerpt: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        out.setdefault("node_id", out.get("name") or out.get("symbol_name") or "")
        out.setdefault("symbol_name", out.get("symbol_name") or out.get("name") or "")
        out.setdefault("file_path", out.get("rel_fname") or out.get("file_path") or "")
        out.setdefault("absolute_path", out.get("fname") or out.get("absolute_path") or "")
        out.setdefault("comment", out.get("comment_str") or out.get("comment") or "")

        line_value = out.get("line_start") or out.get("line") or out.get("line_str")
        start, end, line_text = _parse_line_range(line_value)
        out.setdefault("line_start", start)
        out.setdefault("line_end", end)
        out.setdefault("line_text", line_text)

        category = str(out.get("category") or "unknown").lower()
        if category not in {"file", "class", "function", "field", "statement"}:
            category = "unknown"
        out["category"] = category

        kind = str(out.get("kind") or "unknown").lower()
        if kind not in {"def", "ref"}:
            kind = "unknown"
        out["kind"] = kind

        if not out.get("granularity"):
            if category in {"class", "function"}:
                out["granularity"] = "function"
            elif start is not None:
                out["granularity"] = "line"
            else:
                out["granularity"] = "file"
        return out

    def to_legacy_payload(self) -> Dict[str, Any]:
        """Return a dictionary compatible with existing RepoGraph helpers."""
        payload = self.model_dump(by_alias=True, exclude_none=True)
        payload.setdefault("rel_fname", self.file_path)
        payload.setdefault("fname", self.absolute_path)
        payload.setdefault("name", self.symbol_name or self.node_id)
        payload.setdefault("line", self.line_start)
        payload.setdefault("line_str", self.line_text or _format_line_range(self.line_start, self.line_end))
        payload.setdefault("comment_str", self.comment)
        return payload


class GraphEdge(BaseModel):
    """Typed RepoGraph edge with direction and edge-type evidence preserved."""

    model_config = ConfigDict(extra="allow")

    src: str
    dst: str
    edge_type: EdgeType = "unknown"
    directed: bool = True
    src_line: Optional[int] = None
    dst_line: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("edge_type", mode="before")
    @classmethod
    def _normalize_edge_type(cls, value: Any) -> str:
        raw = str(value or "unknown").lower()
        allowed = {
            "contains",
            "invokes",
            "implements",
            "extends",
            "overrides",
            "references",
            "data_flow",
            "unknown",
        }
        return raw if raw in allowed else "unknown"


class GraphPath(BaseModel):
    """A shortest or explanatory path used by CriticAgent."""

    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    node_ids: List[str] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    path_length: int = 0
    edge_types: List[str] = Field(default_factory=list)
    explanation: str = ""

    @model_validator(mode="after")
    def _fill_path_metadata(self) -> "GraphPath":
        if not self.path_length:
            self.path_length = max(0, len(self.node_ids) - 1)
        if not self.edge_types and self.edges:
            self.edge_types = [edge.edge_type for edge in self.edges]
        return self


class GraphContext(BaseModel):
    """Structured graph context passed from RetrievalAgent to downstream agents."""

    model_config = ConfigDict(extra="allow")

    context_id: str = ""
    repo_key: str = ""
    issue_id: str = ""
    granularity: Granularity = "file"
    seed_entities: List[str] = Field(default_factory=list)
    hops: int = Field(default=2, ge=0)
    edge_policy: List[str] = Field(default_factory=list)
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    entry_nodes: List[str] = Field(default_factory=list)
    known_error_nodes: List[str] = Field(default_factory=list)
    candidate_nodes: List[str] = Field(default_factory=list)
    shortest_paths: List[GraphPath] = Field(default_factory=list)
    text_view: str = ""
    valid_locations: Dict[str, List[int]] = Field(default_factory=dict)
    retrieval_meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_legacy_context(
        cls,
        *,
        text_view: str,
        repo_key: str = "",
        granularity: Granularity = "file",
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> "GraphContext":
        return cls(
            repo_key=repo_key,
            granularity=granularity,
            text_view=text_view,
            retrieval_meta=retrieval_meta or {},
        )


class TargetLocation(BaseModel):
    """Predicted target location at file/function/line granularity."""

    model_config = ConfigDict(extra="allow")

    target: str
    target_type: Literal["file", "function", "line"] = "file"
    target_node_id: str = ""
    file_path: str = ""
    function_name: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_candidate(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        file_path = str(out.get("file_path") or out.get("file") or "")
        function_name = str(out.get("function_name") or out.get("function") or "")
        line_value = out.get("line_start") if out.get("line_start") is not None else out.get("line")
        start, end, _ = _parse_line_range(line_value)
        out.setdefault("file_path", file_path)
        out.setdefault("function_name", function_name)
        out.setdefault("line_start", start)
        out.setdefault("line_end", end)
        if start is not None and start >= 0:
            out.setdefault("target_type", "line")
            out.setdefault("target", f"{file_path}:{start}")
        elif function_name:
            out.setdefault("target_type", "function")
            out.setdefault("target", f"{file_path}::{function_name}" if file_path else function_name)
        else:
            out.setdefault("target_type", "file")
            out.setdefault("target", file_path or str(out.get("target") or ""))
        return out


class LocalizeResult(BaseModel):
    """
    LocalizeAgent output for the new architecture.

    The legacy Top-K candidate list is represented as ``alternatives``; the
    primary contract is the single auditable ``target`` plus ``hypothesis``.
    """

    model_config = ConfigDict(extra="allow")

    target: TargetLocation
    hypothesis: str
    causal_chain: List[str] = Field(default_factory=list)
    issue_evidence: List[str] = Field(default_factory=list)
    graph_evidence: List[str] = Field(default_factory=list)
    alternatives: List[TargetLocation] = Field(default_factory=list)
    prompt_used: str = ""
    raw_output: str = ""
    retry_count: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_legacy_candidates(
        cls,
        *,
        candidates: List[Dict[str, Any]],
        raw_output: str = "",
        prompt_used: str = "",
        hypothesis: str = "",
        retry_count: int = 0,
    ) -> "LocalizeResult":
        targets = [TargetLocation.model_validate(c) for c in candidates]
        primary = targets[0] if targets else TargetLocation(target="", target_type="file")
        return cls(
            target=primary,
            hypothesis=hypothesis or raw_output[:800],
            alternatives=targets[1:],
            prompt_used=prompt_used,
            raw_output=raw_output,
            retry_count=retry_count,
        )


class CriticEvaluation(BaseModel):
    """CriticAgent static causal review with weighted component scores."""

    model_config = ConfigDict(extra="allow")

    semantic_alignment: float = Field(default=0.0, ge=0.0, le=30.0)
    topological_reachability: float = Field(default=0.0, ge=0.0, le=40.0)
    causal_validity: float = Field(default=0.0, ge=0.0, le=30.0)
    critique: str = ""
    reachable_paths: List[GraphPath] = Field(default_factory=list)
    invalid_assumptions: List[str] = Field(default_factory=list)
    is_graph_orphan: bool = False
    retry_focus: str = ""
    recommended_graph_action: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def total_score(self) -> float:
        return round(self.semantic_alignment + self.topological_reachability + self.causal_validity, 4)

    @computed_field
    @property
    def normalized_score(self) -> float:
        return round(self.total_score / 100.0, 4)

    @model_validator(mode="after")
    def _enforce_orphan_rule(self) -> "CriticEvaluation":
        if self.is_graph_orphan:
            self.topological_reachability = 0.0
        return self


class ReflectionDecision(BaseModel):
    """ReflectAgent routing decision. ReflectAgent routes; it does not score."""

    model_config = ConfigDict(extra="allow")

    action: ReflectionAction
    current_granularity: Granularity = "file"
    next_granularity: Optional[Granularity] = None
    retry_count: int = Field(default=0, ge=0)
    retry_prompt: str = ""
    backtrack_signal: str = ""
    graph_regeneration_request: Dict[str, Any] = Field(default_factory=dict)
    memory_candidate: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @property
    def is_pass(self) -> bool:
        return self.action is ReflectionAction.PASS

    @property
    def is_retry(self) -> bool:
        return self.action is ReflectionAction.RETRY

    @property
    def is_regenerate(self) -> bool:
        return self.action is ReflectionAction.REGENERATE


# Backward-compatible aliases for code that imports semantic names from older modules.
LocalizeCandidate = TargetLocation
ReflectOutput = ReflectionDecision


def _parse_line_range(value: Any) -> tuple[Optional[int], Optional[int], str]:
    if value is None or value == "":
        return None, None, ""
    if isinstance(value, int):
        return value, value, str(value)
    if isinstance(value, (list, tuple)) and value:
        ints = [x for x in value if isinstance(x, int)]
        if not ints:
            return None, None, "-".join(str(x) for x in value)
        return ints[0], ints[-1], _format_line_range(ints[0], ints[-1])
    raw = str(value).strip()
    if not raw or raw == "?":
        return None, None, raw
    if "-" in raw:
        left, right = raw.split("-", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            return int(left), int(right), raw
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        iv = int(raw)
        return iv, iv, raw
    return None, None, raw


def _format_line_range(start: Optional[int], end: Optional[int]) -> str:
    if start is None:
        return "?"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"