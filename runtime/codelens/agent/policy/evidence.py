from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional

from agent.agents.models import CriticEvaluation, GraphContext, GraphNode, LocalizeResult
from agent.policy.models import BeliefState, PolicyObservation, SearchAction


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or "")
        if token.lower() not in {"the", "and", "for", "with", "from", "this", "that"}
    }


def _canonical_file_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    if p.startswith(("a/", "b/")):
        p = p[2:]
    for anchor in ("src/main/java/", "src/test/java/", "src/"):
        index = p.find(anchor)
        if index >= 0:
            return p[index:]
    return p.lstrip("/")


def _candidate_key(node: GraphNode, granularity: str) -> str:
    file_path = _canonical_file_path(node.file_path)
    if granularity == "file":
        return file_path
    if granularity == "function":
        if node.category not in {"class", "function"}:
            return ""
        return node.node_id or (
            f"{file_path}::{node.symbol_name}" if file_path else node.symbol_name
        )
    if file_path and node.line_start is not None:
        return f"{file_path}:{node.line_start}"
    return ""


class CalibratedEvidenceModel:
    """Deterministic, auditable belief updater with externally calibratable parameters.

    The defaults are conservative priors, not claims of probability calibration. Paper
    experiments should fit ``slope`` and ``intercept`` on training repositories only.
    """

    def __init__(
        self,
        *,
        slope: float = 0.85,
        intercept: float = -0.15,
        selected_candidate_boost: float = 0.0,
        grounded_primary_min_evidence: float = 0.35,
        grounded_primary_probability_floor: float = 0.08,
        expanded_candidate_mass: float = 0.20,
        initial_outside_probability: float = 0.20,
        expansion_outside_recovery: float = 0.50,
        orphan_outside_boost: float = 0.45,
        grounded_outside_reduction: float = 0.55,
        max_outside_log_odds_step: float = 0.55,
        minimum_probability: float = 1e-4,
        propagate_parent_prior: bool = True,
    ) -> None:
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.selected_candidate_boost = float(selected_candidate_boost)
        self.grounded_primary_min_evidence = float(grounded_primary_min_evidence)
        self.grounded_primary_probability_floor = min(
            0.5, max(0.0, float(grounded_primary_probability_floor))
        )
        self.expanded_candidate_mass = min(0.8, max(0.01, float(expanded_candidate_mass)))
        self.initial_outside_probability = min(
            0.95, max(0.0, float(initial_outside_probability))
        )
        self.expansion_outside_recovery = min(
            1.0, max(0.0, float(expansion_outside_recovery))
        )
        self.orphan_outside_boost = max(0.0, float(orphan_outside_boost))
        self.grounded_outside_reduction = max(0.0, float(grounded_outside_reduction))
        self.max_outside_log_odds_step = max(
            0.0, float(max_outside_log_odds_step)
        )
        self.minimum_probability = max(1e-8, float(minimum_probability))
        self.propagate_parent_prior = bool(propagate_parent_prior)

    def initialize(self, issue_text: str, context: GraphContext) -> BeliefState:
        issue_tokens = _tokens(issue_text)
        raw_scores: Dict[str, float] = {}
        parents: Dict[str, str] = {}
        types: Dict[str, str] = {}
        candidate_set = set(context.candidate_nodes)
        for node in context.nodes:
            key = _candidate_key(node, context.granularity)
            if not key:
                continue
            blob = " ".join(
                [node.file_path, node.symbol_name, node.qualified_name, node.signature, node.comment]
            )
            overlap = len(issue_tokens & _tokens(blob))
            retrieval_bonus = 0.7 if node.node_id in candidate_set else 0.0
            path_bonus = 0.4 if "src/main/java/" in node.file_path.replace("\\", "/") else 0.0
            raw_scores[key] = max(raw_scores.get(key, -10.0), overlap * 0.55 + retrieval_bonus + path_bonus)
            if context.granularity != "file" and node.file_path:
                parents[key] = _canonical_file_path(node.file_path)
            types[key] = context.granularity
        if not raw_scores:
            return BeliefState(granularity=context.granularity, outside_probability=1.0)
        maximum = max(raw_scores.values())
        exp_scores = {key: math.exp(min(20.0, score - maximum)) for key, score in raw_scores.items()}
        normalizer = sum(exp_scores.values()) or 1.0
        marginals = {
            key: max(self.minimum_probability, score / normalizer)
            for key, score in exp_scores.items()
        }
        belief = BeliefState(
            granularity=context.granularity,
            marginals=marginals,
            outside_probability=self.initial_outside_probability,
            parent_by_candidate=parents,
            candidate_types=types,
        )
        belief.last_top_keys = belief.top_keys(5)
        return belief

    def expand_to_context(
        self,
        previous: BeliefState,
        issue_text: str,
        context: GraphContext,
    ) -> BeliefState:
        """Add candidates discovered by same-granularity retrieval to the posterior.

        The previous candidate universe retains most of its probability mass while
        a fixed, auditable exploration mass is distributed among newly observed
        candidates according to the retrieval prior. Without this reconciliation,
        EXPAND/BACKTRACK could retrieve new nodes that the belief updater could
        never rank.
        """
        context_prior = self.initialize(issue_text, context)
        if not previous.marginals:
            return context_prior
        if previous.granularity != context.granularity:
            return self.lift_to_context(previous, issue_text, context)

        new_keys = [key for key in context_prior.marginals if key not in previous.marginals]
        updated = previous.model_copy(deep=True)
        updated.parent_by_candidate.update(context_prior.parent_by_candidate)
        updated.candidate_types.update(context_prior.candidate_types)
        if not new_keys:
            updated.normalize()
            return updated

        old_total = sum(previous.marginals.values()) or 1.0
        new_total = sum(context_prior.marginals[key] for key in new_keys) or 1.0
        new_mass = self.expanded_candidate_mass
        updated.marginals = {
            key: (1.0 - new_mass) * probability / old_total
            for key, probability in previous.marginals.items()
        }
        updated.marginals.update(
            {
                key: new_mass * context_prior.marginals[key] / new_total
                for key in new_keys
            }
        )
        # Newly observed candidates explain part of the previously unmodelled
        # search space. Transfer that mass to the known candidate universe,
        # while keeping a residual open-world probability for further search.
        updated.outside_probability = previous.outside_probability * (
            1.0 - self.expansion_outside_recovery
        )
        updated.normalize()
        updated.revision += 1
        updated.stable_steps = 0
        updated.last_top_keys = updated.top_keys(5)
        return updated

    def observe(
        self,
        *,
        action: SearchAction,
        context: GraphContext,
        result: LocalizeResult,
        evaluation: CriticEvaluation,
        previous_top_keys: Optional[List[str]] = None,
        previous_candidate_keys: Optional[List[str]] = None,
    ) -> PolicyObservation:
        targets = [result.target] + list(result.alternatives)
        candidate_keys: List[str] = []
        for target in targets:
            if not target.target:
                continue
            if target.target_type == "file":
                candidate_keys.append(_canonical_file_path(target.file_path or target.target))
            elif target.target_type == "function":
                path = _canonical_file_path(target.file_path)
                candidate_keys.append(
                    f"{path}::{target.function_name}" if path and target.function_name else target.target
                )
            elif target.file_path and target.line_start is not None:
                candidate_keys.append(f"{_canonical_file_path(target.file_path)}:{target.line_start}")
            else:
                candidate_keys.append(target.target)
        # A malformed/empty localizer response must not erase useful retrieval
        # evidence. At file granularity, seed the policy observation from the
        # retrieved candidates so that the controller can recover gracefully.
        candidate_source = "localizer"
        if not candidate_keys and context.granularity == "file":
            candidate_node_ids = set(context.candidate_nodes)
            fallback_keys = []
            for node in context.nodes:
                if candidate_node_ids and node.node_id not in candidate_node_ids:
                    continue
                key = _candidate_key(node, "file")
                if key:
                    fallback_keys.append(key)
            candidate_keys = list(dict.fromkeys(fallback_keys))[:10]
            candidate_source = "retrieval_fallback"
        primary_candidate = candidate_keys[0] if candidate_keys else ""
        grounded = bool(
            evaluation.metadata.get(
                "target_grounded",
                bool(evaluation.metadata.get("target_node_ids")),
            )
        )
        prior_top = previous_top_keys or []
        prior_candidates = set(previous_candidate_keys or [])
        current_candidates = set(candidate_keys)
        novelty = (
            1.0
            if not previous_candidate_keys
            else float(bool(current_candidates - prior_candidates))
        )
        return PolicyObservation(
            action=action,
            granularity=context.granularity,
            candidate_keys=list(dict.fromkeys(candidate_keys)),
            primary_candidate=primary_candidate,
            semantic_score=evaluation.semantic_alignment / 30.0,
            topology_score=evaluation.topological_reachability / 40.0,
            causal_score=evaluation.causal_validity / 30.0,
            target_grounded=grounded,
            graph_orphan=evaluation.is_graph_orphan,
            ranking_changed=not prior_top or candidate_keys[: len(prior_top)] != prior_top,
            evidence_novelty=novelty,
            context_node_count=len(context.nodes),
            context_edge_count=len(context.edges),
            metadata={
                "critic_total": evaluation.total_score,
                "candidate_source": candidate_source,
            },
        )

    @staticmethod
    def observation_quality(observation: PolicyObservation) -> float:
        if observation.graph_orphan:
            # An orphan in a retrieved/truncated subgraph means topology is
            # unobserved, not that the semantically grounded candidate is false.
            quality = (
                0.35 * observation.semantic_score + 0.25 * observation.causal_score
            ) / 0.60
            if observation.target_grounded:
                quality += 0.08
        else:
            quality = (
                0.35 * observation.semantic_score
                + 0.40 * observation.topology_score
                + 0.25 * observation.causal_score
            )
            if observation.target_grounded:
                quality += 0.12
        if not observation.ranking_changed:
            quality += 0.08
        return min(1.0, max(0.0, quality))

    def _update_outside_probability(
        self,
        belief: BeliefState,
        observation: PolicyObservation,
        quality: float,
    ) -> None:
        """Update open-world mass in log-odds space from retrieval evidence."""
        probability = min(1.0 - 1e-6, max(1e-6, belief.outside_probability))
        log_odds = math.log(probability / (1.0 - probability))
        if observation.target_grounded:
            # Grounding directly shows that the target is represented. Missing
            # edges in a bounded graph must not override that signal.
            delta = -self.grounded_outside_reduction * (0.5 + quality)
        elif observation.graph_orphan and observation.evidence_novelty > 0.0:
            # Only genuinely new orphan evidence can increase open-world mass.
            delta = self.orphan_outside_boost * observation.evidence_novelty
        elif not observation.candidate_keys:
            delta = 0.35
        else:
            delta = 0.15 * (1.0 - quality) * observation.evidence_novelty
        delta = min(
            self.max_outside_log_odds_step,
            max(-self.max_outside_log_odds_step, delta),
        )
        log_odds += delta
        belief.outside_probability = min(
            0.999,
            max(0.001, 1.0 / (1.0 + math.exp(-log_odds))),
        )

    def update(self, belief: BeliefState, observation: PolicyObservation) -> BeliefState:
        if not belief.marginals:
            return belief.model_copy(
                update={
                    "outside_probability": 1.0,
                    "observation_count": belief.observation_count + 1,
                }
            )
        selected = set(observation.candidate_keys)
        quality = self.observation_quality(observation)
        new_weights: Dict[str, float] = {}
        for key, prior in belief.marginals.items():
            exact = key in selected
            # A substring match is especially unsafe for line keys (e.g. :9
            # matching :99). Function node IDs still need compatibility with
            # the legacy ``file::symbol`` representation.
            fuzzy = observation.granularity == "function" and any(
                candidate and (candidate in key or key in candidate)
                for candidate in selected
            )
            primary_match = bool(
                observation.primary_candidate
                and (
                    observation.primary_candidate == key
                    or (
                        observation.granularity == "function"
                        and (
                            observation.primary_candidate in key
                            or key in observation.primary_candidate
                        )
                    )
                )
            )
            if exact or fuzzy:
                # This is the calibrated log-likelihood contribution fitted by
                # scripts/fit_policy_calibration.py on selected candidates.
                evidence = (
                    self.intercept
                    + self.slope * (2.0 * quality - 1.0)
                    + self.selected_candidate_boost
                )
                if observation.target_grounded and primary_match:
                    evidence = max(evidence, self.grounded_primary_min_evidence)
            else:
                evidence = -0.12 - 0.20 * quality * max(0.2, observation.evidence_novelty)
            # Categorical Bayes-style update: the calibrated score is treated
            # as a log evidence factor and the complete candidate universe is
            # normalized afterwards.
            new_weights[key] = max(
                self.minimum_probability,
                prior * math.exp(min(20.0, max(-20.0, evidence))),
            )
        updated = belief.model_copy(deep=True)
        updated.marginals = new_weights
        updated.normalize()
        self._update_outside_probability(updated, observation, quality)
        if observation.target_grounded and observation.primary_candidate:
            self._enforce_probability_floor(
                updated,
                observation.primary_candidate,
                observation.granularity,
            )
        updated.revision += 1
        updated.observation_count += 1
        new_top = updated.top_keys(5)
        updated.stable_steps = belief.stable_steps + 1 if new_top == belief.last_top_keys else 0
        updated.last_top_keys = new_top
        return updated

    def _enforce_probability_floor(
        self,
        belief: BeliefState,
        primary_candidate: str,
        granularity: str,
    ) -> None:
        matching_keys = [
            key
            for key in belief.marginals
            if key == primary_candidate
            or (
                granularity == "function"
                and (primary_candidate in key or key in primary_candidate)
            )
        ]
        if not matching_keys:
            return
        key = matching_keys[0]
        floor = self.grounded_primary_probability_floor
        current = belief.marginals[key]
        if current >= floor or len(belief.marginals) == 1:
            return
        other_total = 1.0 - current
        remaining = 1.0 - floor
        belief.marginals[key] = floor
        for other_key in belief.marginals:
            if other_key != key:
                belief.marginals[other_key] *= remaining / max(1e-12, other_total)

    def lift_to_context(
        self,
        previous: BeliefState,
        issue_text: str,
        context: GraphContext,
    ) -> BeliefState:
        """Initialize a finer level and inherit probability from matching parent files."""
        lifted = self.initialize(issue_text, context)
        if not previous.marginals:
            return lifted
        if not self.propagate_parent_prior:
            lifted.revision = previous.revision + 1
            lifted.outside_probability = previous.outside_probability
            lifted.observation_count = previous.observation_count
            lifted.stable_steps = 0
            lifted.last_top_keys = lifted.top_keys(5)
            return lifted
        previous_file_probabilities: Dict[str, float] = {}
        for key, probability in previous.marginals.items():
            parent_file = previous.parent_by_candidate.get(key)
            if previous.granularity == "file":
                parent_file = key
            if parent_file:
                previous_file_probabilities[parent_file] = (
                    previous_file_probabilities.get(parent_file, 0.0) + probability
                )
        weighted: Dict[str, float] = {}
        for key, parent in lifted.parent_by_candidate.items():
            parent_probability = previous_file_probabilities.get(parent)
            if parent_probability is not None:
                weighted[key] = lifted.marginals[key] * max(
                    self.minimum_probability, parent_probability
                )
            else:
                weighted[key] = lifted.marginals[key] * self.minimum_probability
        if weighted:
            lifted.marginals = weighted
            lifted.normalize()
        lifted.revision = previous.revision + 1
        lifted.outside_probability = previous.outside_probability
        lifted.observation_count = previous.observation_count
        lifted.stable_steps = 0
        lifted.last_top_keys = lifted.top_keys(5)
        return lifted
