"""Pure fixed-budget transition guard. No host calls or state mutations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from ..schemas import validate_transition


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def policy_config(policy):
    result = dict(policy)
    timeout = result.get("barrier_timeout_ms")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("barrier_timeout_ms must be explicitly configured and positive")
    if not isinstance(result.get("strict_cardinality", True), bool):
        raise ValueError("strict_cardinality must be boolean")
    result.setdefault("strict_cardinality", True)
    result.setdefault("policy_version", "guard-v1")
    if result["strict_cardinality"]:
        for key in ("native_llm_calls_limit", "native_tool_calls_limit"):
            if type(result.get(key)) is not int or result[key] < 0:
                raise ValueError(key + " must be explicitly configured for the strict pilot")
    return result


def active_bindings(hid, snapshot):
    """Recheck only this hypothesis's bindings; never infer support from rank."""
    supports, attacks = [], []
    h = snapshot["hypotheses"][hid]
    # Tracker publishes a local adjacency index; older explicit test snapshots
    # may provide binding IDs on H instead.
    index = snapshot.get("binding_index", snapshot.get("bindings_by_hypothesis", {}))
    ids = index.get(hid, h.get("active_binding_ids", h.get("binding_ids", [])))
    for bid in ids:
        b = snapshot["bindings"][bid]
        e = snapshot["evidence"].get(b["evidence_id"], {})
        if (b.get("validity") == "VALID" and e.get("validity") == "VALID"
                and all(b.get(k) is True for k in
                        ("semantic_check", "granularity_check", "provenance_check"))):
            if b["relation"] == "SUPPORT":
                supports.append(bid)
            elif b["relation"] == "ATTACK":
                attacks.append(bid)
    return sorted(supports), sorted(attacks)


class TransitionGuard:
    def evaluate(self, proposal, snapshot, policy):
        policy = policy_config(policy)
        p, hs = proposal, snapshot["hypotheses"]
        validate_transition(p)
        original = list(p["proposed_candidate_ids"])
        if p["run_id"] != snapshot["run_id"]:
            raise ValueError("RUN_MISMATCH")
        if snapshot.get("schema_version") != "2.0.0":
            return self.fallback(p, policy, "UNRESOLVED_SNAPSHOT_VERSION", snapshot)
        if snapshot["seq"] != p["event_seq_watermark"]:
            return self.fallback(p, policy, "SNAPSHOT_WATERMARK_MISMATCH", snapshot)
        ids = p["prev_candidate_ids"] + original
        if any(x not in hs for x in ids):
            raise ValueError("UNKNOWN_HYPOTHESIS_ID")
        if len(set(original)) != len(original) or len(original) > p["capacity_k"]:
            raise ValueError("INVALID_NATIVE_CARDINALITY")
        desc = p["reachability_descriptor"]
        if not desc.get("irreversibility_basis"):
            return self.fallback(p, policy, "UNRESOLVED_REACHABILITY", snapshot)
        if desc.get("irreversible") is not True:
            return self.fallback(p, policy, "NO_PROVEN_IRREVERSIBLE_LOSS", snapshot)
        if not desc.get("native_next_actions") and p["kind"] != "FINAL_COMMIT":
            return self.fallback(p, policy, "UNRESOLVED_NATIVE_ACTION", snapshot)
        executable = set(desc.get("executable_candidate_ids", []))
        if not executable or not set(original) <= executable:
            return self.fallback(p, policy, "UNRESOLVED_EXECUTABILITY", snapshot)
        if not executable <= set(hs):
            raise ValueError("UNKNOWN_EXECUTABLE_ID")
        # A child keeps its parent reachable only with explicit native-path
        # proof referencing a genuine REFINES chain, not merely its filename.
        parents = {}
        for edge in snapshot.get("refinements", []):
            parents.setdefault(edge["child_id"], set()).add(edge["parent_id"])
        retained = desc.get("retained_paths", {})
        for parent, children in retained.items():
            for child in children:
                if child not in hs or parent not in hs:
                    return self.fallback(p, policy, "INVALID_REFINEMENT_MAPPING", snapshot)
                frontier, visited = list(parents.get(child, ())), set()
                while frontier:
                    current = frontier.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    frontier.extend(parents.get(current, ()))
                if parent not in visited:
                    return self.fallback(p, policy, "INVALID_REFINEMENT_MAPPING", snapshot)
        independent = (set(desc.get("continuable_parent_ids", []))
                       - set(original) - set(retained))
        def reachable_from(selection):
            result = set(selection) | independent
            for parent, children in retained.items():
                if set(children) & set(selection):
                    result.add(parent)
            return result
        reachable = reachable_from(original)
        dropped = [x for x in p["prev_candidate_ids"] if x not in reachable]
        d = self.fallback(p, policy, "NO_PROTECTED_DROP", snapshot)
        d["irreversibly_dropped"] = dropped
        d["irreversibility_basis"] = deepcopy(desc)
        supports, attacks, eligible = {}, {}, set()
        # Inspect local retained ancestors as well: removing a child must not
        # sever a supported parent's sole surviving native refinement route.
        affected = set(p["prev_candidate_ids"] + original) | set(retained)
        for hid in affected:
            s, a = active_bindings(hid, snapshot)
            supports[hid], attacks[hid] = s, a
            h = hs[hid]
            superseded = h.get("valid_supersession_ref")
            superseded = bool(superseded and superseded.get("validity") == "VALID")
            if (h["epistemic_state"] == "SUPPORTED" and s and not a
                    and h.get("provenance_valid") is True and not superseded):
                eligible.add(hid)
        protected = [h for h in dropped if h in eligible and h in executable]
        d["support_binding_ids"], d["attack_binding_ids"] = supports, attacks
        d["epistemic_states"] = {h: hs[h]["epistemic_state"] for h in affected}
        d["shadow_counterfactuals"] = [{"hypothesis_id": h, "state": hs[h]["epistemic_state"],
            "reason": "DEFAULT_ARCHIVE_ONLY"} for h in dropped
            if hs[h]["epistemic_state"] in ("UNKNOWN", "CONFLICTED")]
        scope, order = p.get("native_rank_scope"), p["native_order"]
        rank = {h: i for i, h in enumerate(order)} if scope else {}
        if protected and all(h in rank for h in protected):
            protected.sort(key=lambda h: (rank[h], hs[h]["created_seq"], h))
        else:
            protected.sort(key=lambda h: (hs[h]["created_seq"], h))
        d["protected_candidates"] = protected
        result, operations = list(original), []
        for h in protected:
            if not policy["strict_cardinality"] and len(result) < p["capacity_k"]:
                result.append(h)
                operations.append({"action": "PROTECT", "protected": h})
                continue
            currently_retained = eligible & reachable_from(result)
            candidates = [x for x in result if x not in eligible and x in original
                          and currently_retained <= reachable_from([y for y in result if y != x] + [h])]
            refuted = [x for x in candidates if hs[x]["epistemic_state"] == "REFUTED"
                       and attacks.get(x) and not supports.get(x)]
            invalid = [x for x in candidates if hs[x].get("provenance_valid") is False]
            ranked = [x for x in candidates if x in rank and h in rank]
            pool = refuted or invalid or ranked
            if not pool:
                operations.append({"action": "PASS", "protected": h,
                                   "reason": "CAPACITY_CONFLICT"})
                continue
            victim = max(pool, key=lambda x: (rank.get(x, -1), original.index(x), x))
            result[result.index(victim)] = h
            operations.append({"action": "SUBSTITUTE", "protected": h, "victim": victim})
        changed = result != original
        if changed:
            if not all(x in rank for x in result):
                # A frozen positional fallback keeps native slot order; it
                # never maps epistemic state to a numeric sorting score.
                if policy.get("native_fallback_rule") != "preserve_native_slots":
                    return self.fallback(p, policy, "UNRESOLVED_NATIVE_RANK", snapshot)
            else:
                result.sort(key=lambda x: rank[x])
        if policy["strict_cardinality"] and len(result) != len(original):
            raise AssertionError("STRICT_CARDINALITY_VIOLATION")
        if len(result) > p["capacity_k"] or len(result) != len(set(result)):
            raise AssertionError("CAPACITY_OR_UNIQUENESS_VIOLATION")
        d["operations"] = operations
        d["victim_candidates"] = [x["victim"] for x in operations if "victim" in x]
        d["committed_candidates"] = result  # intended set; actual recorded by Barrier
        d["recommended_candidates"] = list(result)
        if d["victim_candidates"]:
            d["action"] = "SUBSTITUTE"
        elif changed:
            d["action"] = "PROTECT"
        conflicts = [x for x in operations if x.get("reason") == "CAPACITY_CONFLICT"]
        d["decision_reason"] = "GUARDED_IRREVERSIBLE_LOSS" if changed else "NO_LEGAL_INTERVENTION"
        d["unresolved_reason"] = "CAPACITY_CONFLICT" if conflicts else None
        return self._seal(d)

    @staticmethod
    def _seal(d):
        stable = {k: v for k, v in d.items() if k not in ("decision_id", "latency_ms", "wait_ms")}
        d["decision_id"] = "decision-" + digest(stable)[:24]
        return d

    def fallback(self, p, policy, reason, snapshot=None):
        d = {"run_id": p["run_id"], "schema_version": "2.0.0", "policy_version": policy["policy_version"],
             "transition_id": p["transition_id"], "snapshot_seq": snapshot["seq"] if snapshot else p["event_seq_watermark"],
             "prev_candidates": list(p["prev_candidate_ids"]),
             "proposed_candidates": list(p["proposed_candidate_ids"]),
             "committed_candidates": list(p["proposed_candidate_ids"]),
             "recommended_candidates": list(p["proposed_candidate_ids"]),
             "irreversibly_dropped": [], "protected_candidates": [], "victim_candidates": [],
             "action": "PASS", "capacity_k": p["capacity_k"],
             "native_proposal_cardinality": len(p["proposed_candidate_ids"]),
             "strict_cardinality": policy["strict_cardinality"],
             "native_rank_scope": p.get("native_rank_scope"), "epistemic_states": {},
             "support_binding_ids": {}, "attack_binding_ids": {}, "operations": [],
             "shadow_counterfactuals": [], "irreversibility_basis": deepcopy(p["reachability_descriptor"]),
             "decision_reason": reason, "unresolved_reason": reason, "timeout": False,
             "latency_ms": 0.0, "wait_ms": 0.0,
             "budget_before": deepcopy(p["budget_before"]), "budget_after": deepcopy(p["budget_before"])}
        return self._seal(d)
