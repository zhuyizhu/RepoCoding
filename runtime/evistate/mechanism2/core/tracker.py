"""Ordered local state updates with immutable evidence audit records."""
from copy import deepcopy
import threading
import time
from ..schemas import (SCHEMA_VERSION, SchemaError, GRANULARITIES, canonical_anchor,
                       content_hash, hypothesis_id, require, validate_event, validate_observation)
from .store import StateStore
from .evidence import Binder, EvidenceExtractor, checked_ruleset


class StateTracker:
    def __init__(self, run_id, ruleset=None):
        require(isinstance(run_id, str) and bool(run_id), 'run_id required')
        self.run_id = run_id
        self.ruleset = checked_ruleset(ruleset)
        self.seq = 0
        self.store = StateStore()
        self.extractor = EvidenceExtractor(self.ruleset)
        self.binder = Binder()
        self._condition = threading.Condition(threading.RLock())

    def _refs(self, ids):
        for hid in ids:
            require(hid in self.store.hypotheses, 'unknown hypothesis reference: ' + str(hid))

    def _preflight(self, event):
        validate_event(event)
        require(event['run_id'] == self.run_id, 'event run mismatch')
        require(event['seq'] == self.seq + 1, 'event sequence gap or out of order')
        p, kind = event['payload'], event['type']
        for key in ('candidate_ids', 'actual_candidate_ids', 'native_order', 'actual_native_order', 'prev_candidate_ids', 'proposed_candidate_ids', 'actual_continuable_parent_ids'):
            if key in p:
                self._refs(p[key])
        if kind == 'PROPOSE' and p.get('parent_id'):
            raise SchemaError('parent mapping must use REFINE')
        if kind == 'REFINE':
            self._refs([p['parent_hypothesis_id']])
            parent = self.store.hypotheses[p['parent_hypothesis_id']]
            anchor = canonical_anchor(p['child_anchor'])
            old = parent['anchor']
            require(old['repo_id'] == anchor['repo_id'] and old['code_version'] == anchor['code_version'], 'refinement repo/version mismatch')
            require(GRANULARITIES.index(old['granularity']) < GRANULARITIES.index(anchor['granularity']), 'refinement must become finer')
            require(old['granularity'] == 'REPO' or old['path'] == anchor['path'], 'refinement path mismatch')
            if old['granularity'] == 'CLASS':
                require(anchor.get('class_fqn') == old['class_fqn'], 'refinement class mismatch')
            if old['granularity'] == 'METHOD':
                require(anchor.get('method_signature') == old['method_signature'], 'refinement method mismatch')
            require(bool(p['mapping_rule_id']), 'mapping rule required')
            require(self._provenance_valid(p['provenance_ref'], anchor), 'refinement requires verifiable mapping provenance')
        if kind in ('OBSERVE', 'RETRIEVE'):
            for field in ('producer', 'action', 'origin_id', 'code_version'):
                require(isinstance(p[field], str) and bool(p[field]), 'invalid observation ' + field)
            # Validate extraction before mutating any audit structures.
            observation = self._observation(event)
            validate_observation(observation)
            if kind != 'RETRIEVE' or p['accessed_candidate_ids']:
                self.extractor.extract(observation)
        if kind == 'EVIDENCE_INVALIDATED':
            require(p['evidence_id'] in self.store.evidence, 'unknown evidence reference')
            if p.get('replacement_evidence_id'):
                require(p['replacement_evidence_id'] in self.store.evidence, 'unknown replacement evidence')
        if kind == 'PRE_COMMIT_TRANSITION':
            require(p['transition_id'] not in self.store.transitions, 'duplicate transition proposal')
            require(p['event_seq_watermark'] == self.seq, 'proposal watermark must equal current state')
            self._refs(p['reachability_descriptor']['continuable_parent_ids'])
            self._refs(p['reachability_descriptor'].get('executable_candidate_ids', []))
            for parent, children in p['reachability_descriptor'].get('retained_paths', {}).items():
                self._refs([parent] + children)
        if kind == 'TRANSITION_COMMITTED':
            require(p['transition_id'] in self.store.transitions, 'commit has no proposal')
            transition = self.store.transitions[p['transition_id']]
            require('commit' not in transition, 'transition already committed')
            proposal = transition['proposal']
            require(type(p['capacity_k']) is int and p['capacity_k'] == proposal['capacity_k'], 'commit capacity mismatch')
            require(len(p['actual_candidate_ids']) <= p['capacity_k'], 'commit exceeds capacity')
            if proposal.get('strict_cardinality', True):
                require(len(p['actual_candidate_ids']) == len(proposal['proposed_candidate_ids']), 'strict pilot must preserve native proposal cardinality')
            require(set(p['actual_candidate_ids']) == set(p['actual_native_order']), 'actual order must cover actual candidates exactly')
            require(set(p['actual_candidate_ids']) <= set(proposal['prev_candidate_ids']) | set(proposal['proposed_candidate_ids']), 'commit introduces outside candidate')
            require(isinstance(p['decision_id'], str) and bool(p['decision_id']) and isinstance(p['budget_after'], dict), 'invalid commit metadata')
        if kind == 'RANK':
            require(set(p['candidate_ids']) <= set(p['native_order']), 'rank order must cover candidates')
            for record in p['raw_rank_records']:
                require(isinstance(record, dict), 'rank records must be objects')

    def _provenance_valid(self, provenance, anchor):
        raw = provenance['raw_ref_or_inline']
        # References lacking their content cannot be verified by this passive core.
        if isinstance(raw, dict) and set(raw) <= {'ref', 'uri', 'sha256'}:
            return False
        return provenance['content_sha256'] == content_hash(raw) and provenance['code_version'] == anchor['code_version']

    def _observation(self, event):
        p = event['payload']
        return dict(obs_id='o_' + event['event_id'], run_id=self.run_id, event_seq=event['seq'], producer=p['producer'], action=p['action'], raw_payload_or_ref=deepcopy(p['raw_result_ref_or_inline']), content_sha256=content_hash(p['raw_result_ref_or_inline']), origin_id=p['origin_id'], code_version=p['code_version'], timestamp=event['timestamp'], native_event_ref=event['native_event_ref'], mapping_rule_version=p.get('mapping_rule_version', event['adapter_version']))

    def _change(self, hid, event, delta, before, reason, evidence_ids=(), binding_ids=()):
        hypothesis = self.store.hypotheses[hid]
        after = {k: hypothesis[k] for k in ('epistemic_state', 'lifecycle', 'visibility')}
        if before != after:
            record = dict(run_id=self.run_id, seq=event['seq'], event_id=event['event_id'], hypothesis_id=hid, before=before, after=after, reason=reason, evidence_ids=list(evidence_ids), binding_ids=list(binding_ids), native_event_ref=event['native_event_ref'], event_type=event['type'], trigger_event_id=event['event_id'], parent_hypothesis_id=hypothesis['parent_hypothesis_id'], provenance=deepcopy(hypothesis['source_provenance']))
            self.store.state_changes.append(record)
        if hid not in delta['changed_hypothesis_ids']:
            delta['changed_hypothesis_ids'].append(hid)
        hypothesis['last_updated_seq'] = event['seq']

    def _state(self, hid):
        return {key: self.store.hypotheses[hid][key] for key in ('epistemic_state', 'lifecycle', 'visibility')}

    def _recompute(self, hid, event, delta, reason):
        h = self.store.hypotheses[hid]
        before = self._state(hid)
        families = {'SUPPORT': set(), 'ATTACK': set()}
        active_ids = []
        for bid in sorted(self.store.hypothesis_bindings.get(hid, ())):
            binding = self.store.bindings[bid]
            evidence = self.store.evidence[binding['evidence_id']]
            if self.store.evidence_validity[evidence['evidence_id']] != 'VALID' or binding['validity'] != 'VALID':
                continue
            if binding['relation'] in families and all(binding[k] for k in ('semantic_check', 'granularity_check', 'provenance_check')):
                families[binding['relation']].add(evidence['origin_family'])
                active_ids.append(bid)
        support, attack = bool(families['SUPPORT']), bool(families['ATTACK'])
        h['epistemic_state'] = {(False, False): 'UNKNOWN', (True, False): 'SUPPORTED', (False, True): 'REFUTED', (True, True): 'CONFLICTED'}[(support, attack)]
        h['support_origin_families'] = sorted(families['SUPPORT'])
        h['attack_origin_families'] = sorted(families['ATTACK'])
        h['active_binding_ids'] = active_ids
        self._change(hid, event, delta, before, reason, [self.store.bindings[b]['evidence_id'] for b in active_ids], active_ids)

    def _bind(self, eid, hid, event, delta):
        evidence = deepcopy(self.store.evidence[eid])
        evidence['validity'] = self.store.evidence_validity[eid]
        binding = self.binder.bind(evidence, self.store.hypotheses[hid], self.ruleset, event['seq'])
        bid = binding['binding_id']
        if bid in self.store.bindings:
            prior = self.store.bindings[bid]
            if all(prior[k] == binding[k] for k in ('relation', 'semantic_check', 'granularity_check', 'provenance_check', 'validity')):
                return
            # Keep the previous rejected attempt while recording resolution.
            bid = binding['binding_id'] = bid + '_s' + str(event['seq'])
        self.store.bindings[bid] = binding
        self.store.evidence_bindings.setdefault(eid, set()).add(bid)
        self.store.hypothesis_bindings.setdefault(hid, set()).add(bid)
        delta['added_binding_ids'].append(bid)
        self._recompute(hid, event, delta, binding['reason'])

    def _register(self, p, event, delta, parent=None):
        anchor = canonical_anchor(p['anchor'])
        hid = hypothesis_id(self.run_id, p['issue_id'], p['claim_type'], anchor)
        source = deepcopy(p['provenance_ref'])
        h = self.store.hypotheses.get(hid)
        before = self._state(hid) if h is not None else None
        if h is None:
            h = dict(hypothesis_id=hid, run_id=self.run_id, issue_id=p['issue_id'], claim_type=p['claim_type'], anchor=anchor, granularity=anchor['granularity'], code_version=anchor['code_version'], parent_hypothesis_id=parent, refinement_depth=(self.store.hypotheses[parent]['refinement_depth'] + 1) if parent else 0, native_candidate_ids=[], source_event_ids=[], source_provenance=[], created_seq=event['seq'], last_updated_seq=event['seq'], native_rank_history=[], epistemic_state='UNKNOWN', lifecycle='ACTIVE', visibility='VISIBLE', last_transition_reason=None, valid_supersession_ref=None, provenance_valid=self._provenance_valid(source, anchor), support_origin_families=[], attack_origin_families=[], active_binding_ids=[])
            self.store.hypotheses[hid] = h
            self.store.anchor_hypotheses.setdefault(anchor['canonical_id'], set()).add(hid)
            self.store.state_changes.append(dict(run_id=self.run_id, seq=event['seq'], event_id=event['event_id'],
                hypothesis_id=hid, before=None, after=self._state(hid), reason='REGISTERED_NATIVE_ENTITY',
                evidence_ids=[], binding_ids=[], native_event_ref=event['native_event_ref'], event_type=event['type'],
                trigger_event_id=event['event_id'], parent_hypothesis_id=parent, provenance=[deepcopy(source)]))
        else:
            h.update(lifecycle='ACTIVE', visibility='VISIBLE', last_transition_reason=None)
        if p['native_candidate_id'] not in h['native_candidate_ids']:
            h['native_candidate_ids'].append(p['native_candidate_id'])
        h['source_event_ids'].append(event['event_id'])
        h['source_provenance'].append(source)
        h['provenance_valid'] = h['provenance_valid'] or self._provenance_valid(source, anchor)
        h['last_updated_seq'] = event['seq']
        if before is not None:
            self._change(hid, event, delta, before, 'explicit native re-proposal')
        if hid not in delta['changed_hypothesis_ids']:
            delta['changed_hypothesis_ids'].append(hid)
        for eid in sorted(self.store.anchor_evidence.get(anchor['canonical_id'], ())):
            self._bind(eid, hid, event, delta)
        return hid

    def _workflow(self, hid, lifecycle, visibility, reason, event, delta):
        h = self.store.hypotheses[hid]
        before = self._state(hid)
        h.update(lifecycle=lifecycle, visibility=visibility, last_transition_reason=reason)
        self._change(hid, event, delta, before, reason)
        delta['workflow_changes'].append(dict(hypothesis_id=hid, lifecycle=lifecycle, visibility=visibility, reason=reason))

    def apply(self, event):
        with self._condition:
            event = deepcopy(event)
            fingerprint = content_hash(event)
            try:
                if event.get('event_id') in self.store.event_hashes:
                    require(self.store.event_hashes[event['event_id']] == fingerprint, 'event ID reused with different content')
                    return deepcopy(self.store.event_deltas[event['event_id']])
                self._preflight(event)
            except (SchemaError, KeyError, TypeError, ValueError) as exc:
                self.store.errors.append(dict(event_id=event.get('event_id'), seq=event.get('seq'), error=str(exc)))
                raise
            delta = dict(event_id=event['event_id'], seq=event['seq'], changed_hypothesis_ids=[], added_observation_ids=[], added_evidence_ids=[], added_binding_ids=[], invalidated_binding_ids=[], workflow_changes=[])
            p, kind = event['payload'], event['type']
            if kind == 'PROPOSE':
                self._register(p, event, delta)
            elif kind == 'REFINE':
                parent = self.store.hypotheses[p['parent_hypothesis_id']]
                child = self._register(dict(issue_id=parent['issue_id'], native_candidate_id=p['child_native_candidate_id'], claim_type=p['child_claim_type'], anchor=p['child_anchor'], provenance_ref=p['provenance_ref']), event, delta, parent['hypothesis_id'])
                edge = dict(parent_id=parent['hypothesis_id'], child_id=child, mapping_rule_id=p['mapping_rule_id'], code_version=parent['code_version'], source_event_id=event['event_id'])
                if child not in self.store.children.get(parent['hypothesis_id'], set()):
                    self.store.refinements.append(edge)
                    self.store.children.setdefault(parent['hypothesis_id'], set()).add(child)
            elif kind in ('OBSERVE', 'RETRIEVE'):
                obs = self._observation(event)
                self.store.observations[obs['obs_id']] = obs
                delta['added_observation_ids'].append(obs['obs_id'])
                extracted = self.extractor.extract(obs) if kind != 'RETRIEVE' or p['accessed_candidate_ids'] else []
                for evidence in extracted:
                    eid = evidence['evidence_id']
                    self.store.evidence[eid] = evidence
                    self.store.evidence_validity[eid] = evidence['validity']
                    self.store.anchor_evidence.setdefault(evidence['anchor']['canonical_id'], set()).add(eid)
                    self.store.origin_evidence.setdefault(evidence['origin_family'], set()).add(eid)
                    delta['added_evidence_ids'].append(eid)
                    for hid in sorted(self.store.anchor_hypotheses.get(evidence['anchor']['canonical_id'], ())):
                        self._bind(eid, hid, event, delta)
            elif kind == 'EVIDENCE_INVALIDATED':
                eid = p['evidence_id']
                self.store.evidence_validity[eid] = 'INVALID'
                impacted = set()
                for bid in sorted(self.store.evidence_bindings.get(eid, ())):
                    delta['invalidated_binding_ids'].append(bid)
                    impacted.add(self.store.bindings[bid]['hypothesis_id'])
                for hid in sorted(impacted):
                    self._recompute(hid, event, delta, p['reason'])
            elif kind == 'RANK':
                for hid in p['candidate_ids']:
                    h = self.store.hypotheses[hid]
                    position = p['native_order'].index(hid) + 1
                    records = [r for r in p['raw_rank_records'] if r.get('hypothesis_id', r.get('candidate_id')) == hid or r.get('native_candidate_id') in h['native_candidate_ids']]
                    if not records:
                        records = [dict(position=position)]
                    for record in records:
                        merged = dict(event_id=event['event_id'], seq=event['seq'], scope=p['native_rank_scope'], stage=p.get('stage', p['native_rank_scope']), native_candidate_id=h['native_candidate_ids'][0], position=position)
                        merged.update(deepcopy(record))
                        h['native_rank_history'].append(merged)
                    h['last_updated_seq'] = event['seq']
                    delta['changed_hypothesis_ids'].append(hid)
            elif kind == 'PRUNE':
                for hid in p['candidate_ids']:
                    self._workflow(hid, 'SUSPENDED' if p['reversible'] else 'TERMINATED', 'HIDDEN', p['reason'], event, delta)
            elif kind in ('HANDOFF', 'FINALIZE'):
                for hid in p['actual_candidate_ids']:
                    self._workflow(hid, 'SELECTED' if kind == 'FINALIZE' else 'ACTIVE', 'VISIBLE', 'FINAL_COMMIT' if kind == 'FINALIZE' else p['reason'], event, delta)
            elif kind == 'PRE_COMMIT_TRANSITION':
                self.store.transitions[p['transition_id']] = {'proposal': deepcopy(p), 'proposal_event_id': event['event_id']}
            elif kind == 'TRANSITION_COMMITTED':
                transition = self.store.transitions[p['transition_id']]
                transition['commit'] = deepcopy(p)
                proposal = transition['proposal']
                actual = set(p['actual_candidate_ids'])
                descriptor = proposal['reachability_descriptor']
                if 'actual_continuable_parent_ids' in p:
                    continuable = set(p['actual_continuable_parent_ids'])
                else:
                    # Proposal reachability cannot blindly carry over after
                    # substitution. Retain only proof tied to actual children.
                    continuable = set(descriptor['continuable_parent_ids']) if actual == set(proposal['proposed_candidate_ids']) else set()
                    parents = {}
                    for edge in self.store.refinements:
                        parents.setdefault(edge['child_id'], set()).add(edge['parent_id'])
                    for parent, children in descriptor.get('retained_paths', {}).items():
                        for child in set(children) & actual:
                            frontier, ancestors = list(parents.get(child, ())), set()
                            while frontier:
                                ancestor = frontier.pop()
                                if ancestor not in ancestors:
                                    ancestors.add(ancestor)
                                    frontier.extend(parents.get(ancestor, ()))
                            if parent in ancestors:
                                continuable.add(parent)
                irreversible = descriptor.get('irreversible') is True
                final = proposal['kind'] in ('FINALIZE', 'FINAL_COMMIT')
                for hid in proposal['prev_candidate_ids']:
                    if hid not in actual:
                        self._workflow(hid, 'SUSPENDED' if hid in continuable or not irreversible else 'TERMINATED', 'HIDDEN', 'FINAL_COMMIT' if final else 'STAGE_HANDOFF', event, delta)
                for hid in p['actual_candidate_ids']:
                    self._workflow(hid, 'SELECTED' if final else 'ACTIVE', 'VISIBLE', 'FINAL_COMMIT' if final else 'STAGE_HANDOFF', event, delta)
            self.seq = event['seq']
            self.store.events.append(event)
            self.store.event_hashes[event['event_id']] = fingerprint
            self.store.event_deltas[event['event_id']] = deepcopy(delta)
            self._condition.notify_all()
            return deepcopy(delta)

    def snapshot(self):
        with self._condition:
            evidence = deepcopy(self.store.evidence)
            for eid, claim in evidence.items():
                claim['validity'] = self.store.evidence_validity[eid]
            bindings = deepcopy(self.store.bindings)
            for binding in bindings.values():
                if self.store.evidence_validity[binding['evidence_id']] != 'VALID' and binding['validity'] == 'VALID':
                    binding['validity'] = 'INVALID'
            result = dict(run_id=self.run_id, seq=self.seq, schema_version=SCHEMA_VERSION, ruleset_version=self.ruleset['version'], hypotheses=deepcopy(self.store.hypotheses), observations=deepcopy(self.store.observations), evidence=evidence, bindings=bindings, refinements=deepcopy(self.store.refinements), transitions=deepcopy(self.store.transitions), evidence_index={k: sorted(v) for k, v in self.store.anchor_evidence.items()}, binding_index={k: sorted(v) for k, v in self.store.hypothesis_bindings.items()}, reachability_index={k: sorted(v) for k, v in self.store.children.items()})
            result['bindings_by_hypothesis'] = deepcopy(result['binding_index'])
            result['source_hash'] = content_hash(result)
            return result

    def catch_up(self, run_id, watermark, timeout_ms):
        require(run_id == self.run_id, 'barrier run mismatch')
        require(type(watermark) is int and watermark >= 0, 'invalid barrier watermark')
        require(isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool) and timeout_ms > 0, 'barrier timeout must be configured positive milliseconds')
        deadline = time.monotonic() + timeout_ms / 1000
        with self._condition:
            while self.seq < watermark:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('state did not reach watermark within configured timeout')
                self._condition.wait(remaining)
            require(self.seq == watermark, 'requested watermark is stale')
            return self.snapshot()
