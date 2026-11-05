"""Synchronous pre-commit barrier, with honest failure and host receipts."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter

from ..schemas import make_event, validate_transition
from .guard import policy_config


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def check_budget(budget, policy):
    for key in ("additional_llm_calls", "additional_tool_calls"):
        value = budget.get(key)
        if type(value) is not int or value < 0 or value > policy.get(key + "_limit", 0):
            raise ValueError("BUDGET_VIOLATION:" + key)
    for key in ("native_llm_calls", "native_tool_calls"):
        if type(budget.get(key)) is not int or budget[key] < 0:
            raise ValueError("INVALID_NATIVE_COUNTER:" + key)


class Barrier:
    def __init__(self, tracker, guard, policy):
        self.tracker, self.guard = tracker, guard
        self.policy = policy_config(policy)
        self.decisions = []
        self.receipts = []
        self.failures = []
        self._completed = {}

    def execute(self, proposal, commit_callback, mode="intervention"):
        if mode not in ("shadow", "intervention"):
            raise ValueError("invalid barrier mode")
        p = deepcopy(proposal)
        p["strict_cardinality"] = self.policy["strict_cardinality"]
        p["guard_policy"] = deepcopy(self.policy)
        p["mode"] = mode
        validate_transition(p)
        check_budget(p["budget_before"], self.policy)
        for counter in ("native_llm_calls", "native_tool_calls"):
            limit = self.policy.get(counter + "_limit")
            if limit is not None and p["budget_before"][counter] > limit:
                raise ValueError("NATIVE_BUDGET_ALREADY_EXCEEDED:" + counter)
        if p["run_id"] != self.tracker.run_id:
            raise ValueError("RUN_MISMATCH")
        tid = p["transition_id"]
        if tid in self._completed:
            old = self._completed[tid]
            if old["proposal"] != p or old["mode"] != mode:
                raise ValueError("TRANSITION_ID_REUSED_WITH_DIFFERENT_CONTENT")
            return deepcopy(old["result"])
        if tid in self.tracker.store.transitions:
            raise ValueError("TRANSITION_ALREADY_PENDING_OR_COMMITTED")
        start = perf_counter()
        snapshot, normalized_pre = None, False
        try:
            snapshot = self.tracker.catch_up(p["run_id"], p["event_seq_watermark"],
                                            self.policy["barrier_timeout_ms"])
            wait_ms = (perf_counter() - start) * 1000
            pre = make_event(self.tracker.run_id, self.tracker.seq + 1, "PRE_COMMIT_TRANSITION", p,
                             native_event_ref="barrier:" + tid, timestamp=utc_now())
            self.tracker.apply(pre)
            normalized_pre = True
            d = self.guard.evaluate(p, snapshot, self.policy)
            elapsed = (perf_counter() - start) * 1000
            if elapsed > self.policy["barrier_timeout_ms"]:
                d = self.guard.fallback(p, self.policy, "BARRIER_TIMEOUT", snapshot)
                d["timeout"] = True
        except TimeoutError:
            wait_ms = (perf_counter() - start) * 1000
            d = self.guard.fallback(p, self.policy, "BARRIER_TIMEOUT", snapshot)
            d["timeout"] = True
        except ValueError as error:
            # A missing/stale tracker watermark is a fail-open workflow case.
            # Invalid native proposals are never silently repaired.
            if snapshot is not None and self.tracker.seq == p["event_seq_watermark"]:
                raise
            wait_ms = (perf_counter() - start) * 1000
            if normalized_pre:
                raise
            d = self.guard.fallback(p, self.policy, "UNRESOLVED_SNAPSHOT:" + str(error), snapshot)
        d["wait_ms"] = wait_ms
        d["latency_ms"] = (perf_counter() - start) * 1000
        d["mode"] = mode
        d["normalized_pre_commit"] = normalized_pre
        d["decision_id"] = self.guard._seal(d)["decision_id"]
        selected = list(p["proposed_candidate_ids"] if mode == "shadow" else d["recommended_candidates"])
        decision_time = utc_now()
        d["decision_timestamp"] = decision_time
        self.decisions.append(d)
        # The store audit extension is append-only and not an epistemic input.
        if not hasattr(self.tracker.store, "guard_decisions"):
            self.tracker.store.guard_decisions = []
        self.tracker.store.guard_decisions.append(d)
        try:
            receipt = deepcopy(commit_callback(list(selected)))
            if not isinstance(receipt, dict):
                raise ValueError("COMMIT_RESULT_MUST_BE_OBJECT")
            actual = receipt.get("actual_candidate_ids")
            if not isinstance(actual, list) or actual != selected:
                raise ValueError("COMMIT_MISMATCH")
            if len(set(actual)) != len(actual) or len(actual) > p["capacity_k"]:
                raise ValueError("COMMIT_CAPACITY_VIOLATION")
            if self.policy["strict_cardinality"] and len(actual) != len(p["proposed_candidate_ids"]):
                raise ValueError("STRICT_CARDINALITY_VIOLATION")
            if not receipt.get("commit_ref"):
                raise ValueError("MISSING_NATIVE_COMMIT_RECEIPT")
            check_budget(receipt["budget_after"], self.policy)
            for counter in ("native_llm_calls", "native_tool_calls"):
                if receipt["budget_after"][counter] < p["budget_before"][counter]:
                    raise ValueError("NATIVE_COUNTER_REGRESSION:" + counter)
                limit = self.policy.get(counter + "_limit")
                if limit is not None and receipt["budget_after"][counter] > limit:
                    raise ValueError("NATIVE_BUDGET_EXCEEDED:" + counter)
            receipt.update(transition_id=tid, decision_id=d["decision_id"],
                           actual_native_order=list(actual), capacity_k=p["capacity_k"],
                           commit_timestamp=utc_now())
            receipt["guard_outcome"] = {"action": d["action"], "timeout": d["timeout"],
                                        "reason": d["decision_reason"], "mode": mode,
                                        "recommended_candidates": list(d["recommended_candidates"])}
            # Continuation roots from the proposal cannot keep an evicted
            # proposed parent alive after substitution. Independent retained
            # branches and proven retained descendants are preserved.
            desc = p["reachability_descriptor"]
            continuation = (set(desc["continuable_parent_ids"]) - set(p["proposed_candidate_ids"])
                            - set(desc.get("retained_paths", {})))
            continuation.update(actual)
            for parent, children in desc.get("retained_paths", {}).items():
                if set(children) & set(actual):
                    continuation.add(parent)
            receipt["actual_continuable_parent_ids"] = sorted(continuation)
            if normalized_pre:
                event = make_event(self.tracker.run_id, self.tracker.seq + 1, "TRANSITION_COMMITTED", receipt,
                                   native_event_ref=receipt["commit_ref"], timestamp=receipt["commit_timestamp"])
                self.tracker.apply(event)
            else:
                # Do not fabricate missing sequence events to make a lagging
                # tracker appear current. Actual native fallback is journalled.
                unresolved = {"transition_id": tid, "reason": "UNRESOLVED_TRACKER_COMMIT",
                              "proposal": p, "actual_receipt": deepcopy(receipt)}
                self.failures.append(unresolved)
                self.tracker.store.errors.append(deepcopy(unresolved))
            d["committed_candidates"] = list(actual)
            d["budget_after"] = deepcopy(receipt["budget_after"])
            d["commit_confirmed"] = True
            d["commit_ref"] = receipt["commit_ref"]
            d["commit_timestamp"] = receipt["commit_timestamp"]
            self.receipts.append(deepcopy(receipt))
        except Exception as error:
            failure = {"transition_id": tid, "reason": type(error).__name__ + ":" + str(error),
                       "selected_ids": selected, "decision_id": d["decision_id"], "timestamp": utc_now()}
            self.failures.append(failure)
            self.tracker.store.errors.append(failure)
            d["commit_confirmed"] = False
            d["commit_error"] = failure["reason"]
            # A callback may already have performed a native side effect.
            # Never retry it blindly or claim a completed protection.
            raise
        result = {"decision": deepcopy(d), "commit": deepcopy(receipt)}
        self._completed[tid] = {"proposal": p, "mode": mode, "result": deepcopy(result)}
        return result
