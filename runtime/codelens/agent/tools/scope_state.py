"""S: stateful, host-independent evidence aggregation and search-scope diagnostics."""
from __future__ import annotations

import hashlib
import json
import math

from agent.policy.evidence import CalibratedEvidenceModel


class ScopeStateTool(CalibratedEvidenceModel):
    """The outside value is a heuristic diagnostic, not a calibrated probability."""

    def __init__(self, *, open_world=True, **kwargs):
        super().__init__(**kwargs)
        self.open_world = open_world
        self.seen_evidence = set()

    def _bounded(self, state):
        if not self.open_world:
            state.outside_probability = 0.0
        return state

    def initialize(self, issue_text, context):
        return self._bounded(super().initialize(issue_text, context))

    def lift_to_context(self, previous, issue_text, context):
        return self._bounded(super().lift_to_context(previous, issue_text, context))

    def expand_to_context(self, previous, issue_text, context):
        # New nodes establish wider coverage, not that the fault was found.
        state = super().expand_to_context(previous, issue_text, context)
        state.outside_probability = previous.outside_probability
        return self._bounded(state)

    def observe(self, **kwargs):
        observation = super().observe(**kwargs)
        context = kwargs["context"]
        evaluation = kwargs["evaluation"]
        identity = {
            "granularity": context.granularity,
            "nodes": sorted(n.node_id for n in context.nodes),
            "edges": sorted((e.src, e.dst, e.edge_type) for e in context.edges),
            "text": hashlib.sha256(context.text_view.encode()).hexdigest(),
            "primary": observation.primary_candidate,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        observation.evidence_novelty = float(digest not in self.seen_evidence)
        observation.metadata.update({
            "evidence_id": digest,
            "available_channels": evaluation.metadata.get("available_channels", []),
            "verification_judgment": evaluation.metadata.get("verification_judgment", {}),
        })
        if "quality_override" in evaluation.metadata:
            observation.metadata["quality_override"] = evaluation.metadata["quality_override"]
        return observation

    @staticmethod
    def observation_quality(observation):
        if "quality_override" in observation.metadata:
            return float(observation.metadata["quality_override"])
        channels = observation.metadata.get("available_channels", [])
        weighted = [(0.35, observation.semantic_score), (0.25, observation.causal_score)]
        if "topology" in channels:
            weighted.append((0.4, observation.topology_score))
        return min(1.0, max(0.0, sum(w * q for w, q in weighted) / sum(w for w, _ in weighted)))

    def _update_outside_probability(self, belief, observation, quality):
        if not self.open_world:
            belief.outside_probability = 0.0
            return
        judgment = observation.metadata.get("verification_judgment", {})
        verdict = judgment.get("verdict", "unknown")
        # Grounding alone is not coverage evidence. Repeated evidence cannot
        # repeatedly suppress outside mass.
        if not observation.evidence_novelty:
            return
        observed = bool(judgment.get("successful_tool_observations"))
        delta = 0.0
        if verdict == "supported" and observed and observation.target_grounded:
            delta = -0.35 * quality
        elif verdict == "rejected" and observed:
            delta = 0.35
        elif not observation.candidate_keys:
            delta = 0.35
        p = min(0.999, max(0.001, belief.outside_probability))
        belief.outside_probability = min(0.999, max(0.001, 1 / (1 + math.exp(-(math.log(p / (1-p)) + delta)))))

    def update(self, belief, observation):
        identity = observation.metadata.get("evidence_id")
        if identity in self.seen_evidence:
            state = belief.model_copy(deep=True)
            state.stable_steps += 1
            return self._bounded(state)
        state = super().update(belief, observation)
        if identity:
            self.seen_evidence.add(identity)
        return self._bounded(state)

    def describe(self, state):
        return {"candidate_ranking": state.ranked(5),
                "outside_estimate": state.outside_probability if self.open_world else None,
                "scope_mode": "open_world" if self.open_world else "candidate_only"}
