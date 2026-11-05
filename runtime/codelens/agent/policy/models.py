from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.agents.models import Granularity


class ActionType(str, Enum):
    RETRIEVE = "RETRIEVE"
    EXPAND = "EXPAND"
    REFINE = "REFINE"
    VERIFY = "VERIFY"
    RELOCALIZE = "RELOCALIZE"
    BACKTRACK = "BACKTRACK"
    COMPRESS = "COMPRESS"
    STOP = "STOP"


class ExplorationCost(BaseModel):
    """Observed or predicted resource use for one policy action."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    wall_time_seconds: float = Field(default=0.0, ge=0.0)
    graph_nodes: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def plus(self, other: "ExplorationCost") -> "ExplorationCost":
        return ExplorationCost(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            wall_time_seconds=self.wall_time_seconds + other.wall_time_seconds,
            graph_nodes=self.graph_nodes + other.graph_nodes,
        )

    def scalar(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = {
            "tokens": 1.0 / 8192.0,
            "calls": 0.25,
            "seconds": 1.0 / 120.0,
            "nodes": 1.0 / 2000.0,
        }
        w.update(weights or {})
        return (
            w["tokens"] * self.total_tokens
            + w["calls"] * self.llm_calls
            + w["seconds"] * self.wall_time_seconds
            + w["nodes"] * self.graph_nodes
        )


class PolicyBudget(BaseModel):
    """Hard cumulative limits. A zero limit disables that dimension."""

    max_tokens: int = Field(default=0, ge=0)
    max_llm_calls: int = Field(default=8, ge=0)
    max_wall_time_seconds: float = Field(default=0.0, ge=0.0)
    max_graph_nodes: int = Field(default=0, ge=0)

    def can_afford(self, spent: ExplorationCost, predicted: ExplorationCost) -> bool:
        future = spent.plus(predicted)
        checks = (
            not self.max_tokens or future.total_tokens <= self.max_tokens,
            not self.max_llm_calls or future.llm_calls <= self.max_llm_calls,
            not self.max_wall_time_seconds or future.wall_time_seconds <= self.max_wall_time_seconds,
            not self.max_graph_nodes or future.graph_nodes <= self.max_graph_nodes,
        )
        return all(checks)


class SearchAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_type: ActionType
    granularity: Granularity = "file"
    scope_paths: List[str] = Field(default_factory=list)
    hops: int = Field(default=0, ge=0)
    edge_types: List[str] = Field(default_factory=list)
    reason: str = ""
    expected_information_gain: float = 0.0
    expected_utility_gain: float = 0.0
    value: float = 0.0
    predicted_cost: ExplorationCost = Field(default_factory=ExplorationCost)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyObservation(BaseModel):
    """Action-conditioned evidence exposed to the belief updater."""

    action: SearchAction
    granularity: Granularity
    candidate_keys: List[str] = Field(default_factory=list)
    primary_candidate: str = ""
    semantic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    topology_score: float = Field(default=0.0, ge=0.0, le=1.0)
    causal_score: float = Field(default=0.0, ge=0.0, le=1.0)
    target_grounded: bool = False
    graph_orphan: bool = False
    ranking_changed: bool = True
    evidence_novelty: float = Field(default=1.0, ge=0.0, le=1.0)
    context_node_count: int = Field(default=0, ge=0)
    context_edge_count: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BeliefState(BaseModel):
    """Open-world fault posterior over candidates at one granularity.

    ``marginals`` is a categorical distribution conditional on the fault being
    inside the currently observed candidate universe. ``outside_probability``
    represents the joint probability that the true location has not been
    retrieved yet. This prevents a small or truncated subgraph from producing
    artificially confident Top-K stopping decisions.
    """

    granularity: Granularity = "file"
    marginals: Dict[str, float] = Field(default_factory=dict)
    outside_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_by_candidate: Dict[str, str] = Field(default_factory=dict)
    candidate_types: Dict[str, Literal["file", "function", "line", "unknown"]] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)
    stable_steps: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    last_top_keys: List[str] = Field(default_factory=list)

    @field_validator("marginals")
    @classmethod
    def _bound_marginals(cls, value: Dict[str, float]) -> Dict[str, float]:
        bounded: Dict[str, float] = {}
        for key, raw_probability in value.items():
            if not key:
                continue
            probability = float(raw_probability)
            if not math.isfinite(probability):
                raise ValueError(f"Non-finite belief probability for candidate {key!r}")
            bounded[str(key)] = max(1e-12, probability)
        total = sum(bounded.values())
        if total <= 0.0:
            return {}
        return {key: probability / total for key, probability in bounded.items()}

    def normalize(self) -> None:
        """Normalize in-place after internal model updates that bypass validation."""
        if not self.marginals:
            return
        bounded = {
            str(key): max(1e-12, float(probability))
            for key, probability in self.marginals.items()
            if key and math.isfinite(float(probability))
        }
        total = sum(bounded.values())
        self.marginals = (
            {key: probability / total for key, probability in bounded.items()}
            if total > 0.0
            else {}
        )

    def ranked(self, k: int = 0) -> List[tuple[str, float]]:
        known_mass = 1.0 - self.outside_probability
        rows = sorted(
            ((key, known_mass * probability) for key, probability in self.marginals.items()),
            key=lambda item: (-item[1], item[0]),
        )
        return rows[:k] if k > 0 else rows

    def top_keys(self, k: int) -> List[str]:
        return [key for key, _ in self.ranked(k)]

    def entropy(self) -> float:
        outside = min(1.0, max(0.0, self.outside_probability))
        known_mass = 1.0 - outside
        probabilities = [
            known_mass * probability
            for probability in self.marginals.values()
            if probability > 0.0
        ]
        if outside > 0.0:
            probabilities.append(outside)
        return -sum(p * math.log(max(1e-12, p)) for p in probabilities if p > 0.0)

    def topk_success_mass(self, k: int) -> float:
        total_probability = sum(max(0.0, p) for p in self.marginals.values())
        if total_probability <= 0.0:
            return 0.0
        conditional_mass = min(
            1.0,
            sum(max(0.0, probability) for _, probability in self.ranked(k))
            / max(1e-12, 1.0 - self.outside_probability),
        )
        return (1.0 - self.outside_probability) * conditional_mass


class TrajectoryStep(BaseModel):
    step_idx: int = Field(ge=0)
    action: SearchAction
    observation: Optional[PolicyObservation] = None
    belief_before: BeliefState
    belief_after: BeliefState
    observed_cost: ExplorationCost = Field(default_factory=ExplorationCost)
    cumulative_cost: ExplorationCost = Field(default_factory=ExplorationCost)
    decision_after: Optional[SearchAction] = None
    stop_reason: str = ""


class SearchTrajectory(BaseModel):
    instance_id: str
    budget: PolicyBudget
    target_granularity: Granularity = "line"
    steps: List[TrajectoryStep] = Field(default_factory=list)
    cumulative_cost: ExplorationCost = Field(default_factory=ExplorationCost)
    stop_reason: str = ""

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.cumulative_cost = step.cumulative_cost
