from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.agents.models import CriticEvaluation, Granularity
from agent.policy.models import (
    ActionType,
    BeliefState,
    ExplorationCost,
    PolicyBudget,
    SearchAction,
)

OUTSIDE_CANDIDATE = "__OUT_OF_SCOPE__"


@dataclass
class PolicyConfig:
    top_k: int = 5
    stop_probability: float = 0.82
    stable_steps_to_stop: int = 2
    min_action_value: float = 0.015
    information_gain_weight: float = 0.35
    cost_weight: float = 0.10
    refinement_resolution_gain: float = 0.45
    orphan_recovery_gain: float = 0.25
    open_world_recovery_gain: float = 0.08
    max_consecutive_backtracks: int = 1
    target_granularity: Granularity = "line"
    max_steps: int = 8
    allow_scope_recovery: bool = True
    allow_relocalize: bool = True
    cost_weights: Dict[str, float] = field(default_factory=dict)


class BeliefPolicyController:
    """One-step value-of-information policy under hard resource constraints."""

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    @staticmethod
    def _next_granularity(granularity: Granularity) -> Optional[Granularity]:
        if granularity == "file":
            return "function"
        if granularity == "function":
            return "line"
        return None

    @staticmethod
    def _granularity_rank(granularity: Granularity) -> int:
        return {"file": 0, "function": 1, "line": 2}[granularity]

    def _target_reached(self, granularity: Granularity) -> bool:
        return self._granularity_rank(granularity) >= self._granularity_rank(
            self.config.target_granularity
        )

    @staticmethod
    def predicted_cost(action_type: ActionType, granularity: Granularity) -> ExplorationCost:
        if action_type in {ActionType.REFINE, ActionType.EXPAND, ActionType.BACKTRACK}:
            nodes = {"file": 900, "function": 650, "line": 1100}[granularity]
            return ExplorationCost(prompt_tokens=4200, completion_tokens=900, llm_calls=1, graph_nodes=nodes)
        if action_type in {ActionType.RELOCALIZE, ActionType.VERIFY}:
            return ExplorationCost(prompt_tokens=3600, completion_tokens=800, llm_calls=1)
        return ExplorationCost()

    @staticmethod
    def _categorical_distribution(belief: BeliefState) -> Dict[str, float]:
        total = sum(max(0.0, value) for value in belief.marginals.values())
        outside = min(1.0, max(0.0, belief.outside_probability))
        if total <= 0.0:
            return {OUTSIDE_CANDIDATE: 1.0} if outside > 0.0 else {}
        known_mass = 1.0 - outside
        distribution = {
            key: known_mass * max(0.0, value) / total
            for key, value in belief.marginals.items()
        }
        if outside > 0.0:
            distribution[OUTSIDE_CANDIDATE] = outside
        return distribution

    @staticmethod
    def _entropy(distribution: Dict[str, float]) -> float:
        return -sum(p * math.log(max(1e-12, p)) for p in distribution.values() if p > 0.0)

    def _topk_mass(self, distribution: Dict[str, float]) -> float:
        probabilities = sorted(
            (
                probability
                for key, probability in distribution.items()
                if key != OUTSIDE_CANDIDATE
            ),
            reverse=True,
        )[: self.config.top_k]
        return sum(probabilities)

    def _scope_keys(
        self,
        action_type: ActionType,
        belief: BeliefState,
        candidate_paths: List[str],
    ) -> set[str]:
        ranked_keys = belief.top_keys(max(self.config.top_k, 5))
        if action_type is ActionType.VERIFY:
            return set(ranked_keys[:1])
        if action_type is ActionType.RELOCALIZE:
            return set(ranked_keys[: max(2, min(5, len(ranked_keys)))])
        path_scope = [path for path in candidate_paths[:5] if path]
        matched = {
            key
            for key in belief.marginals
            if any(path == key or path in key or key in path for path in path_scope)
        }
        if action_type in {ActionType.EXPAND, ActionType.BACKTRACK}:
            complement = set(belief.marginals) - matched
            return (complement or set(ranked_keys)) | {OUTSIDE_CANDIDATE}
        return matched or set(ranked_keys)

    @staticmethod
    def _observation_likelihoods(
        action_type: ActionType,
        evaluation: CriticEvaluation,
    ) -> tuple[float, float]:
        """Return P(positive evidence|fault in scope), P(positive|outside)."""
        semantic_gap = 1.0 - evaluation.semantic_alignment / 30.0
        topology_gap = 1.0 - evaluation.topological_reachability / 40.0
        causal_gap = 1.0 - evaluation.causal_validity / 30.0
        if action_type is ActionType.VERIFY:
            tpr, fpr = 0.82 + 0.10 * causal_gap, 0.18
        elif action_type is ActionType.REFINE:
            tpr, fpr = 0.84, 0.20 + 0.08 * topology_gap
        elif action_type in {ActionType.EXPAND, ActionType.BACKTRACK}:
            tpr, fpr = 0.72 + 0.16 * topology_gap, 0.26
        else:
            tpr, fpr = 0.68 + 0.14 * semantic_gap, 0.30
        return min(0.98, max(0.51, tpr)), min(0.49, max(0.02, fpr))

    @staticmethod
    def _posterior_for_binary_observation(
        prior: Dict[str, float],
        scope: set[str],
        *,
        tpr: float,
        fpr: float,
        positive: bool,
    ) -> tuple[Dict[str, float], float]:
        weighted: Dict[str, float] = {}
        evidence_probability = 0.0
        for key, probability in prior.items():
            likelihood = tpr if key in scope else fpr
            if not positive:
                likelihood = 1.0 - likelihood
            weighted[key] = probability * likelihood
            evidence_probability += weighted[key]
        if evidence_probability <= 1e-12:
            return dict(prior), 0.0
        return (
            {key: value / evidence_probability for key, value in weighted.items()},
            evidence_probability,
        )

    def _value(
        self,
        *,
        action_type: ActionType,
        belief: BeliefState,
        evaluation: CriticEvaluation,
        predicted_cost: ExplorationCost,
        candidate_paths: List[str],
        evidence_novelty: float = 1.0,
    ) -> tuple[float, float, float]:
        prior = self._categorical_distribution(belief)
        if not prior:
            return -predicted_cost.scalar(self.config.cost_weights), 0.0, 0.0
        scope = self._scope_keys(action_type, belief, candidate_paths)
        tpr, fpr = self._observation_likelihoods(action_type, evaluation)
        posterior_positive, probability_positive = self._posterior_for_binary_observation(
            prior, scope, tpr=tpr, fpr=fpr, positive=True
        )
        posterior_negative, probability_negative = self._posterior_for_binary_observation(
            prior, scope, tpr=tpr, fpr=fpr, positive=False
        )
        expected_entropy = (
            probability_positive * self._entropy(posterior_positive)
            + probability_negative * self._entropy(posterior_negative)
        )
        information_gain = max(0.0, self._entropy(prior) - expected_entropy)
        expected_topk_mass = (
            probability_positive * self._topk_mass(posterior_positive)
            + probability_negative * self._topk_mass(posterior_negative)
        )
        utility_gain = max(0.0, expected_topk_mass - self._topk_mass(prior))
        if action_type is ActionType.REFINE:
            # Refinement changes the target resolution, which cannot be expressed
            # by a posterior over the current coarser candidate universe.
            scope_mass = sum(prior.get(key, 0.0) for key in scope)
            max_entropy = math.log(max(1, len(prior)))
            concentration = (
                1.0 if max_entropy <= 1e-12 else max(0.0, 1.0 - self._entropy(prior) / max_entropy)
            )
            utility_gain += (
                self.config.refinement_resolution_gain
                * scope_mass
                * (0.20 + 0.80 * concentration)
            )
        if action_type is ActionType.EXPAND and evaluation.is_graph_orphan:
            # Missing reachability in a bounded/truncated subgraph is an
            # information-recovery opportunity, not negative fault evidence.
            utility_gain += self.config.orphan_recovery_gain
        if action_type in {ActionType.EXPAND, ActionType.BACKTRACK}:
            # Explicitly reward actions that can recover probability mass not
            # represented by the current bounded candidate universe.
            utility_gain += (
                self.config.open_world_recovery_gain
                * belief.outside_probability
                * min(1.0, max(0.0, evidence_novelty))
            )
        value = (
            utility_gain
            + self.config.information_gain_weight * information_gain
            - self.config.cost_weight * predicted_cost.scalar(self.config.cost_weights)
        )
        return value, utility_gain, information_gain

    def choose_action(
        self,
        *,
        belief: BeliefState,
        evaluation: CriticEvaluation,
        budget: PolicyBudget,
        spent: ExplorationCost,
        step_count: int,
        candidate_paths: List[str],
        expanded_current_scope: bool,
        evidence_novelty: float = 1.0,
        repeated_candidate_set: bool = False,
        consecutive_backtracks: int = 0,
    ) -> SearchAction:
        if step_count >= self.config.max_steps:
            return SearchAction(action_type=ActionType.STOP, granularity=belief.granularity, reason="max_steps")
        target_reached = self._target_reached(belief.granularity)
        if target_reached and belief.stable_steps >= self.config.stable_steps_to_stop and belief.topk_success_mass(self.config.top_k) >= self.config.stop_probability:
            return SearchAction(action_type=ActionType.STOP, granularity=belief.granularity, reason="stable_high_posterior")

        candidates: List[SearchAction] = []
        considered: List[Dict[str, object]] = []
        next_granularity = self._next_granularity(belief.granularity)
        action_specs: List[tuple[ActionType, Granularity]] = []
        if next_granularity and candidate_paths:
            action_specs.append((ActionType.REFINE, next_granularity))
        if self.config.allow_relocalize:
            action_specs.append((ActionType.RELOCALIZE, belief.granularity))
        broaden_type = ActionType.BACKTRACK if expanded_current_scope else ActionType.EXPAND
        backtrack_blocked = broaden_type is ActionType.BACKTRACK and (
            repeated_candidate_set
            or consecutive_backtracks >= self.config.max_consecutive_backtracks
        )
        if self.config.allow_scope_recovery and not backtrack_blocked:
            action_specs.append((broaden_type, belief.granularity))
        action_specs.append((ActionType.VERIFY, belief.granularity))

        for action_type, granularity in action_specs:
            predicted = self.predicted_cost(action_type, granularity)
            if not budget.can_afford(spent, predicted):
                considered.append(
                    {
                        "action_type": action_type.value,
                        "granularity": granularity,
                        "affordable": False,
                        "predicted_cost": predicted.model_dump(),
                    }
                )
                continue
            value, utility_gain, information_gain = self._value(
                action_type=action_type,
                belief=belief,
                evaluation=evaluation,
                predicted_cost=predicted,
                candidate_paths=candidate_paths,
                evidence_novelty=evidence_novelty,
            )
            candidates.append(
                SearchAction(
                    action_type=action_type,
                    granularity=granularity,
                    scope_paths=list(candidate_paths),
                    hops=2 if action_type in {ActionType.EXPAND, ActionType.BACKTRACK} else 0,
                    edge_types=["invokes", "implements", "extends", "contains"]
                    if action_type in {ActionType.EXPAND, ActionType.BACKTRACK}
                    else [],
                    reason="one_step_expected_value",
                    expected_information_gain=information_gain,
                    expected_utility_gain=utility_gain,
                    value=value,
                    predicted_cost=predicted,
                )
            )
            considered.append(
                {
                    "action_type": action_type.value,
                    "granularity": granularity,
                    "affordable": True,
                    "value": value,
                    "expected_utility_gain": utility_gain,
                    "expected_information_gain": information_gain,
                    "predicted_cost": predicted.model_dump(),
                }
            )
        if not candidates:
            return SearchAction(
                action_type=ActionType.STOP,
                granularity=belief.granularity,
                reason="budget_exhausted",
                metadata={"considered_actions": considered},
            )
        grounded = bool(
            evaluation.metadata.get(
                "target_grounded", evaluation.metadata.get("target_node_ids")
            )
        )
        if evaluation.is_graph_orphan and grounded and next_granularity:
            refinement = next(
                (candidate for candidate in candidates if candidate.action_type is ActionType.REFINE),
                None,
            )
            if refinement is not None:
                refinement.reason = "grounded_resolution_progress"
                refinement.metadata = {
                    "considered_actions": considered,
                    "target_granularity": self.config.target_granularity,
                    "backtrack_blocked": backtrack_blocked,
                }
                return refinement
        best = max(candidates, key=lambda action: (action.value, action.action_type.value))
        if best.value <= self.config.min_action_value:
            if not target_reached:
                # Goal completeness is a terminal constraint, not a fixed
                # orchestration loop. Recover missing topology once, then move
                # to the next resolution whenever the hard budget permits it.
                preferred_type = (
                    ActionType.EXPAND
                    if evaluation.is_graph_orphan and not expanded_current_scope
                    else ActionType.REFINE
                )
                progress = next(
                    (candidate for candidate in candidates if candidate.action_type is preferred_type),
                    None,
                )
                if progress is None:
                    progress = next(
                        (candidate for candidate in candidates if candidate.action_type is ActionType.REFINE),
                        None,
                    )
                if progress is not None:
                    progress.reason = "required_granularity_progress"
                    progress.metadata = {
                        "considered_actions": considered,
                        "target_granularity": self.config.target_granularity,
                    }
                    return progress
            return SearchAction(
                action_type=ActionType.STOP,
                granularity=belief.granularity,
                reason="non_positive_value",
                metadata={"considered_actions": considered},
            )
        best.metadata = {
            "considered_actions": considered,
            "target_granularity": self.config.target_granularity,
        }
        return best


class HeuristicController(BeliefPolicyController):
    """Frozen, gold-blind action order with the same hard budget as VOI.

    This controller deliberately never calls ``_value`` or the binary-outcome
    simulator.  Signatures prevent repeated verification/relocalization of an
    unchanged candidate universe.
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__(config)
        self._verified: set[tuple[str, tuple[str, ...]]] = set()
        self._relocalized: set[tuple[str, tuple[str, ...]]] = set()

    def choose_action(
        self, *, belief: BeliefState, evaluation: CriticEvaluation,
        budget: PolicyBudget, spent: ExplorationCost, step_count: int,
        candidate_paths: List[str], expanded_current_scope: bool,
        evidence_novelty: float = 1.0, repeated_candidate_set: bool = False,
        consecutive_backtracks: int = 0,
    ) -> SearchAction:
        del evidence_novelty
        granularity = belief.granularity
        def stop(reason: str, considered: Optional[List[Dict[str, object]]] = None) -> SearchAction:
            return SearchAction(action_type=ActionType.STOP, granularity=granularity,
                                reason=reason, metadata={"considered_actions": considered or []})

        if step_count >= self.config.max_steps:
            return stop("max_steps")
        target_reached = self._target_reached(granularity)
        if (target_reached and belief.marginals
                and belief.stable_steps >= self.config.stable_steps_to_stop
                and belief.topk_success_mass(self.config.top_k) >= self.config.stop_probability):
            return stop("stable_high_posterior")

        next_granularity = self._next_granularity(granularity)
        signature = (granularity, tuple(sorted(belief.marginals)))
        specs: List[tuple[ActionType, Granularity]] = []
        if next_granularity and candidate_paths:
            specs.append((ActionType.REFINE, next_granularity))
        specs.append((ActionType.VERIFY, granularity))
        if self.config.allow_relocalize and signature not in self._relocalized:
            specs.append((ActionType.RELOCALIZE, granularity))
        if self.config.allow_scope_recovery:
            if not expanded_current_scope:
                specs.append((ActionType.EXPAND, granularity))
            elif not repeated_candidate_set and consecutive_backtracks < self.config.max_consecutive_backtracks:
                specs.append((ActionType.BACKTRACK, granularity))

        legal: Dict[ActionType, tuple[Granularity, ExplorationCost]] = {}
        considered: List[Dict[str, object]] = []
        for action_type, action_granularity in specs:
            predicted = self.predicted_cost(action_type, action_granularity)
            affordable = budget.can_afford(spent, predicted)
            considered.append({"action_type": action_type.value,
                               "granularity": action_granularity,
                               "affordable": affordable,
                               "predicted_cost": predicted.model_dump()})
            if affordable:
                legal[action_type] = (action_granularity, predicted)

        if not belief.marginals:
            order = (ActionType.EXPAND, ActionType.BACKTRACK)
            rule = "empty_candidates"
        elif not target_reached:
            order = (ActionType.REFINE, ActionType.VERIFY, ActionType.RELOCALIZE,
                     ActionType.EXPAND, ActionType.BACKTRACK)
            rule = "forward_refinement"
        else:
            judgment = evaluation.metadata.get("verification_judgment") or {}
            verdict = judgment.get("verdict", "unknown")
            if signature not in self._verified:
                order = (ActionType.VERIFY, ActionType.EXPAND, ActionType.BACKTRACK,
                         ActionType.RELOCALIZE)
                rule = "first_verification"
            elif verdict in {"rejected", "unknown"} and signature not in self._relocalized:
                order = (ActionType.RELOCALIZE, ActionType.EXPAND,
                         ActionType.BACKTRACK)
                rule = "rejected_or_unknown"
            else:
                order = (ActionType.EXPAND, ActionType.BACKTRACK,
                         ActionType.RELOCALIZE)
                rule = "remaining_exploration"

        for action_type in order:
            if action_type is ActionType.VERIFY and signature in self._verified:
                continue
            if action_type is ActionType.RELOCALIZE and signature in self._relocalized:
                continue
            if action_type not in legal:
                continue
            if action_type is ActionType.VERIFY:
                self._verified.add(signature)
            elif action_type is ActionType.RELOCALIZE:
                self._relocalized.add(signature)
            action_granularity, predicted = legal[action_type]
            return SearchAction(
                action_type=action_type, granularity=action_granularity,
                scope_paths=list(candidate_paths),
                hops=2 if action_type in {ActionType.EXPAND, ActionType.BACKTRACK} else 0,
                edge_types=["invokes", "implements", "extends", "contains"]
                if action_type in {ActionType.EXPAND, ActionType.BACKTRACK} else [],
                reason="heuristic_" + rule, predicted_cost=predicted,
                metadata={"rule_id": rule, "considered_actions": considered},
            )
        return stop("no_legal_heuristic_action", considered)


class FixedScheduleController(BeliefPolicyController):
    """Gold-blind fixed action schedule used as a matched policy ablation.

    The controller shares the same candidate state, hard budget, retrieval
    interfaces, prompts and output logic as the adaptive controller.  Only the
    action-selection rule changes, so the comparison does not fall back to the
    unrelated legacy workflow.
    """

    _LINE_SCHEDULE = (
        ActionType.VERIFY,
        ActionType.RELOCALIZE,
        ActionType.EXPAND,
        ActionType.RELOCALIZE,
        ActionType.VERIFY,
    )

    def choose_action(
        self,
        *,
        belief: BeliefState,
        evaluation: CriticEvaluation,
        budget: PolicyBudget,
        spent: ExplorationCost,
        step_count: int,
        candidate_paths: List[str],
        expanded_current_scope: bool,
        evidence_novelty: float = 1.0,
        repeated_candidate_set: bool = False,
        consecutive_backtracks: int = 0,
    ) -> SearchAction:
        del evaluation, evidence_novelty, repeated_candidate_set, consecutive_backtracks
        if step_count >= self.config.max_steps:
            return SearchAction(
                action_type=ActionType.STOP,
                granularity=belief.granularity,
                reason="fixed_schedule_max_steps",
            )

        next_granularity = self._next_granularity(belief.granularity)
        if next_granularity is not None and candidate_paths:
            action_type = ActionType.REFINE
            granularity = next_granularity
        elif not self._target_reached(belief.granularity):
            action_type = ActionType.EXPAND
            granularity = belief.granularity
        else:
            line_index = max(0, step_count - 3)
            action_type = self._LINE_SCHEDULE[line_index % len(self._LINE_SCHEDULE)]
            if action_type is ActionType.EXPAND and expanded_current_scope:
                action_type = ActionType.RELOCALIZE
            granularity = belief.granularity

        predicted = self.predicted_cost(action_type, granularity)
        if not budget.can_afford(spent, predicted):
            return SearchAction(
                action_type=ActionType.STOP,
                granularity=belief.granularity,
                reason="fixed_schedule_budget_exhausted",
                metadata={"scheduled_action": action_type.value},
            )
        return SearchAction(
            action_type=action_type,
            granularity=granularity,
            scope_paths=list(candidate_paths),
            hops=2 if action_type in {ActionType.EXPAND, ActionType.BACKTRACK} else 0,
            edge_types=["invokes", "implements", "extends", "contains"]
            if action_type in {ActionType.EXPAND, ActionType.BACKTRACK}
            else [],
            reason="fixed_matched_schedule",
            predicted_cost=predicted,
            metadata={"schedule_step": step_count},
        )
