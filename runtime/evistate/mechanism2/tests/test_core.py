import unittest
from copy import deepcopy
from mechanism2.schemas import SchemaError, canonical_anchor, content_hash, hypothesis_id, make_event, validate_binding
from mechanism2.core.tracker import StateTracker
from mechanism2.core.evidence import Binder


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tracker = StateTracker('run-test')
        self.anchor = dict(repo_id='repo', code_version='commit-a', granularity='FILE', path='pkg/code.py')

    def event(self, kind, payload):
        return make_event(self.tracker.run_id, self.tracker.seq + 1, kind, payload)

    def provenance(self, anchor=None):
        anchor = anchor or self.anchor
        raw = {'native_candidate': anchor['path']}
        return dict(native_event_ref='native:1', origin_id='origin:1', raw_ref_or_inline=raw, content_sha256=content_hash(raw), code_version=anchor['code_version'], mapping_rule_version='v1')

    def propose(self, claim='EXISTS', anchor=None, native_id='candidate-a'):
        anchor = anchor or self.anchor
        self.tracker.apply(self.event('PROPOSE', dict(issue_id='issue', native_candidate_id=native_id, claim_type=claim, anchor=anchor, source_kind='native_proposal', provenance_ref=self.provenance(anchor))))
        return hypothesis_id(self.tracker.run_id, 'issue', claim, anchor)

    def observe(self, positive=True, origin='tool:1', anchor=None, **raw_changes):
        anchor = anchor or self.anchor
        raw = dict(carrier='entity_lookup', anchor=anchor, status='found' if positive else 'not_found', exhaustive=True, scope_complete=True, code_version=anchor['code_version'])
        raw.update(raw_changes)
        return self.tracker.apply(self.event('OBSERVE', dict(producer='native', action='lookup', raw_result_ref_or_inline=raw, origin_id=origin, code_version=anchor['code_version'])))

    def state(self, hid):
        return self.tracker.snapshot()['hypotheses'][hid]['epistemic_state']

    def test_four_states_and_local_invalidation(self):
        hid = self.propose()
        self.assertEqual('UNKNOWN', self.state(hid))
        positive = self.observe()['added_evidence_ids'][0]
        self.assertEqual('SUPPORTED', self.state(hid))
        negative = self.observe(False, 'tool:2')['added_evidence_ids'][0]
        self.assertEqual('CONFLICTED', self.state(hid))
        self.tracker.apply(self.event('EVIDENCE_INVALIDATED', dict(evidence_id=positive, reason='hash invalid', source_event_ref='audit:1')))
        self.assertEqual('REFUTED', self.state(hid))
        self.assertEqual('VALID', self.tracker.store.evidence[positive]['validity'])
        self.assertEqual('INVALID', self.tracker.snapshot()['evidence'][positive]['validity'])
        self.tracker.apply(self.event('EVIDENCE_INVALIDATED', dict(evidence_id=negative, reason='version invalid', source_event_ref='audit:2')))
        self.assertEqual('UNKNOWN', self.state(hid))

    def test_evidence_before_registration_and_origin_dedup(self):
        first = self.observe(origin='same-source')['added_evidence_ids'][0]
        hid = self.propose()
        self.observe(origin='same-source')
        self.assertEqual(['same-source'], self.tracker.snapshot()['hypotheses'][hid]['support_origin_families'])
        self.tracker.apply(self.event('EVIDENCE_INVALIDATED', dict(evidence_id=first, reason='one observation invalid', source_event_ref='audit:1')))
        self.assertEqual('SUPPORTED', self.state(hid))

    def test_fault_relevant_rejection_and_no_draft_enabling(self):
        hid = self.propose('FAULT_RELEVANT')
        self.observe()
        self.assertEqual('UNKNOWN', self.state(hid))
        binding = next(iter(self.tracker.store.bindings.values()))
        self.assertEqual('IRRELEVANT', binding['relation'])
        self.assertEqual('DIRECT', binding['directness'])
        self.assertEqual('HARD', binding['strength'])
        with self.assertRaises(SchemaError):
            StateTracker('run', {'version': 'bad', 'enabled_rules': ['R_RETRIEVAL_RELEVANCE_V1']})

    def test_typed_evidence_cannot_be_replaced_by_free_form_text(self):
        hid = self.propose()
        self.observe()
        claim = deepcopy(next(iter(self.tracker.store.evidence.values())))
        for field, value in [('predicate', 'FAULT_RELEVANT'), ('polarity', 'SOMEWHAT'),
                             ('proposition_text', 'Definitely unrelated to the issue'),
                             ('arguments', {'free_form': 'support'})]:
            corrupted = dict(claim, **{field: value})
            with self.assertRaises(SchemaError):
                Binder().bind(corrupted, self.tracker.store.hypotheses[hid])
        binding = deepcopy(next(iter(self.tracker.store.bindings.values())))
        binding['directness'] = 'HARD'
        with self.assertRaises(SchemaError):
            validate_binding(binding)

    def test_nonexhaustive_negative_and_free_text_are_not_evidence(self):
        self.propose()
        result = self.observe(False, exhaustive=False)
        self.assertEqual([], result['added_evidence_ids'])
        self.tracker.apply(self.event('OBSERVE', dict(producer='native', action='read', raw_result_ref_or_inline='pkg/code.py definitely does not exist', origin_id='text', code_version='commit-a')))
        self.assertEqual(2, len(self.tracker.store.observations))
        self.assertEqual(0, len(self.tracker.store.evidence))

    def test_unaccessed_retrieval_remains_observation_only(self):
        raw = dict(carrier='entity_lookup', anchor=self.anchor, status='found',
                   code_version=self.anchor['code_version'])
        result = self.tracker.apply(self.event('RETRIEVE', dict(producer='native',
            action='batch search', raw_result_ref_or_inline=raw, origin_id='batch:1',
            code_version=self.anchor['code_version'], accessed_candidate_ids=[])))
        self.assertEqual(1, len(result['added_observation_ids']))
        self.assertEqual([], result['added_evidence_ids'])
        self.assertEqual({}, self.tracker.store.hypotheses)
        self.assertEqual({}, self.tracker.store.bindings)

    def test_finer_locator_cannot_change_coarse_entity_identity(self):
        method = dict(self.anchor, granularity='METHOD', method_signature='Thing.run(value)')
        self.assertEqual(canonical_anchor(method), canonical_anchor(deepcopy(method)))
        for anchor in (dict(method, line_start=1, line_end=3),
                       dict(method, line_start=99, line_end=101),
                       dict(self.anchor, method_signature='Thing.run(value)'),
                       dict(self.anchor, granularity='CLASS', class_fqn='Thing', line_start=1, line_end=3)):
            with self.assertRaises(SchemaError):
                canonical_anchor(anchor)

    def test_immutable_inputs_snapshot_and_idempotency(self):
        event = self.event('OBSERVE', dict(producer='native', action='read', raw_result_ref_or_inline={'text': 'original'}, origin_id='text', code_version='commit-a'))
        delta = self.tracker.apply(event)
        self.assertEqual(delta, self.tracker.apply(event))
        event['payload']['raw_result_ref_or_inline']['text'] = 'modified'
        with self.assertRaises(SchemaError):
            self.tracker.apply(event)
        snapshot = self.tracker.snapshot()
        snapshot['observations'][delta['added_observation_ids'][0]]['raw_payload_or_ref']['text'] = 'snapshot modified'
        self.assertEqual('original', self.tracker.snapshot()['observations'][delta['added_observation_ids'][0]]['raw_payload_or_ref']['text'])
        self.assertEqual(1, self.tracker.seq)

    def test_schema_sequence_and_unknown_reference(self):
        with self.assertRaises(SchemaError):
            make_event('run-test', 1, 'UNKNOWN', {})
        with self.assertRaises(SchemaError):
            make_event('run-test', 1, 'PROPOSE', {})
        e = make_event('run-test', 2, 'PRUNE', dict(candidate_ids=['missing'], reason='PRUNED', stage='file', reversible=True))
        with self.assertRaises(SchemaError):
            self.tracker.apply(e)
        e['seq'] = 1
        with self.assertRaises(SchemaError):
            self.tracker.apply(e)
        self.assertEqual([], self.tracker.store.events)
        self.assertEqual(2, len(self.tracker.store.errors))

    def test_refinement_does_not_inherit_support(self):
        parent = self.propose()
        self.observe()
        child_anchor = dict(self.anchor, granularity='METHOD', method_signature='Thing.run(value)')
        delta = self.tracker.apply(self.event('REFINE', dict(parent_hypothesis_id=parent, child_native_candidate_id='method', child_claim_type='EXISTS', child_anchor=child_anchor, mapping_rule_id='ast:membership', provenance_ref=self.provenance(child_anchor))))
        child = hypothesis_id('run-test', 'issue', 'EXISTS', child_anchor)
        self.assertEqual('SUPPORTED', self.state(parent))
        self.assertEqual('UNKNOWN', self.state(child))
        self.assertEqual(parent, self.tracker.snapshot()['refinements'][0]['parent_id'])
        evidence = next(iter(self.tracker.store.evidence.values()))
        binding = Binder().bind(evidence, self.tracker.store.hypotheses[child])
        self.assertEqual('IRRELEVANT', binding['relation'])
        self.assertFalse(binding['granularity_check'])

    def test_workflow_state_independence(self):
        hid = self.propose()
        self.observe()
        self.tracker.apply(self.event('RANK', dict(candidate_ids=[hid], native_rank_scope='files', native_order=[hid], raw_rank_records=[dict(hypothesis_id=hid, position=1, raw_score=0.15)])))
        self.tracker.apply(self.event('PRUNE', dict(candidate_ids=[hid], reason='RANK_OUT', stage='files', reversible=False)))
        self.assertEqual('SUPPORTED', self.state(hid))
        self.assertEqual('TERMINATED', self.tracker.snapshot()['hypotheses'][hid]['lifecycle'])
        self.assertEqual(0.15, self.tracker.snapshot()['hypotheses'][hid]['native_rank_history'][0]['raw_score'])

    def test_configurable_barrier_and_stale_snapshot_rejection(self):
        self.propose()
        self.assertEqual(1, self.tracker.catch_up('run-test', 1, 17)['seq'])
        with self.assertRaises(TimeoutError):
            self.tracker.catch_up('run-test', 2, 1)
        with self.assertRaises(SchemaError):
            self.tracker.catch_up('run-test', 0, 17)
        with self.assertRaises(SchemaError):
            self.tracker.catch_up('run-test', 1, 0)

    def transition(self, before, proposed, **changes):
        p = dict(transition_id='t-' + str(self.tracker.seq), run_id=self.tracker.run_id,
                 kind='FINAL_COMMIT', prev_candidate_ids=before,
                 proposed_candidate_ids=proposed, capacity_k=2, native_order=before,
                 event_seq_watermark=self.tracker.seq, budget_before={},
                 reachability_descriptor=dict(continuable_parent_ids=[],
                     native_next_actions=[], refinement_mapping_refs=[],
                     irreversibility_basis='final choice', irreversible=True))
        p.update(changes)
        self.tracker.apply(self.event('PRE_COMMIT_TRANSITION', p))
        return p

    def commit(self, p, actual, **changes):
        payload = dict(transition_id=p['transition_id'], decision_id='d-1',
                       actual_candidate_ids=actual, actual_native_order=actual,
                       capacity_k=p['capacity_k'], commit_ref='native:commit',
                       commit_timestamp='2026-10-04T01:00:00Z', budget_after={})
        payload.update(changes)
        return self.tracker.apply(self.event('TRANSITION_COMMITTED', payload))

    def test_commit_strict_cardinality_and_actual_reachability(self):
        first = self.propose()
        second = self.propose(anchor=dict(self.anchor, path='pkg/other.py'))
        p = self.transition([first, second], [first], reachability_descriptor=dict(
            continuable_parent_ids=[first], native_next_actions=[],
            refinement_mapping_refs=[], irreversibility_basis='final selection', irreversible=True))
        self.assertEqual('ACTIVE', self.tracker.store.hypotheses[first]['lifecycle'])
        with self.assertRaises(SchemaError):
            self.commit(p, [first, second])
        self.commit(p, [second], actual_continuable_parent_ids=[])
        self.assertEqual('TERMINATED', self.tracker.store.hypotheses[first]['lifecycle'])
        self.assertEqual('SELECTED', self.tracker.store.hypotheses[second]['lifecycle'])
        self.propose()
        self.assertEqual('ACTIVE', self.tracker.store.hypotheses[first]['lifecycle'])
        self.assertEqual('VISIBLE', self.tracker.store.hypotheses[first]['visibility'])

    def test_reversible_commit_and_nonpilot_capacity(self):
        first = self.propose()
        second = self.propose(anchor=dict(self.anchor, path='pkg/other.py'))
        p = self.transition([first, second], [first], strict_cardinality=False)
        self.commit(p, [first, second])
        self.assertEqual(2, len(self.tracker.store.transitions[p['transition_id']]['commit']['actual_candidate_ids']))
        p = self.transition([first, second], [first], reachability_descriptor=dict(
            continuable_parent_ids=[], native_next_actions=[], refinement_mapping_refs=[],
            irreversibility_basis='reversible hide', irreversible=False))
        self.commit(p, [first])
        self.assertEqual('SUSPENDED', self.tracker.store.hypotheses[second]['lifecycle'])

    def test_primitive_validation(self):
        for field, bad_value in [('issue_id', 42), ('native_candidate_id', []), ('source_kind', False)]:
            p = dict(issue_id='issue', native_candidate_id='native', claim_type='EXISTS',
                     anchor=self.anchor, source_kind='candidate', provenance_ref=self.provenance())
            p[field] = bad_value
            with self.assertRaises(SchemaError):
                self.event('PROPOSE', p)
        with self.assertRaises(SchemaError):
            self.event('RANK', dict(candidate_ids=[], native_rank_scope=12, native_order=[], raw_rank_records=[]))
        event = self.event('OBSERVE', dict(producer='native', action='read', raw_result_ref_or_inline={}, origin_id='text', code_version='commit-a'))
        self.assertFalse(event['timestamp'].startswith('1970-'))

if __name__ == '__main__':
    unittest.main()
