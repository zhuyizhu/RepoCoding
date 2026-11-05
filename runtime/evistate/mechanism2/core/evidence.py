"""Deterministic evidence extraction and binding.

The default ruleset remains existence-only.  The explicitly opt-in
``heuristic-pilot-v1`` ruleset is a gold-free engineering pilot and is kept
separate from the future E2-frozen paper ruleset.
"""
from copy import deepcopy
from ..schemas import (FAULT_RELEVANT_HEURISTIC_BASES, canonical_anchor, content_hash, require, serialize_predicate,
                       validate_binding, validate_evidence, validate_observation)

REVIEWED_RULES = frozenset({'R_EXISTENCE_POS_V1', 'R_EXISTENCE_NEG_V1'})
HEURISTIC_PILOT_RULES = frozenset({'R_HEURISTIC_FAULT_PROPOSAL_EXACT_FILE_V1', 'R_SOURCE_GROUNDED_REVIEW_EXACT_V1', 'R_SOURCE_GROUNDED_REVIEW_EXACT_LINE_V1'})
DEFAULT_RULESET = {'version': 'evidence-rules-v1-existence-only', 'enabled_rules': sorted(REVIEWED_RULES)}
HEURISTIC_PILOT_RULESET = {
    'version': 'evidence-rules-heuristic-pilot-v1',
    'enabled_rules': sorted(HEURISTIC_PILOT_RULES),
    'paper_ready': False,
    'gold_access': False,
}
BUDGETED_REVIEW_RULESET_VERSION = 'evidence-rules-budgeted-review-v1'
SOURCE_REVIEW_RULESET_VERSION = 'evidence-rules-source-grounded-review-pilot-v1'

def checked_ruleset(ruleset=None):
    result = deepcopy(DEFAULT_RULESET if ruleset is None else ruleset)
    require(isinstance(result, dict) and isinstance(result.get('version'), str), 'ruleset requires version')
    require(isinstance(result.get('enabled_rules'), list), 'ruleset requires enabled_rules list')
    if result['version'] in (HEURISTIC_PILOT_RULESET['version'], BUDGETED_REVIEW_RULESET_VERSION, SOURCE_REVIEW_RULESET_VERSION):
        require(set(result['enabled_rules']) <= HEURISTIC_PILOT_RULES, 'unknown heuristic pilot rule')
        result['paper_ready'] = False
        result['gold_access'] = False
    else:
        require(set(result['enabled_rules']) <= REVIEWED_RULES, 'unreviewed / FAULT_RELEVANT rules cannot be enabled')
    return result

def proposition_text(predicate, polarity, subject, arguments):
    return serialize_predicate(predicate, polarity, subject, arguments)

class EvidenceExtractor:
    def __init__(self, ruleset=None):
        self.ruleset = checked_ruleset(ruleset)

    def extract(self, observation):
        validate_observation(observation)
        raw = observation['raw_payload_or_ref']
        if (self.ruleset['version'] in (HEURISTIC_PILOT_RULESET['version'], BUDGETED_REVIEW_RULESET_VERSION, SOURCE_REVIEW_RULESET_VERSION)
                and isinstance(raw, dict)
                and raw.get('carrier') in ('heuristic_fault_signal', 'source_grounded_fault_signal')):
            return self._extract_heuristic_fault(observation, raw)
        if not isinstance(raw, dict) or raw.get('carrier') not in ('entity_lookup', 'existence'):
            return []
        if raw.get('status') not in ('found', 'not_found') or not isinstance(raw.get('anchor'), dict):
            return []
        anchor = canonical_anchor(raw['anchor'])
        positive = raw['status'] == 'found'
        # Zero hits alone never prove nonexistence.
        if not positive and not (raw.get('exhaustive') is True and raw.get('scope_complete') is True):
            return []
        valid = raw.get('code_version') == observation['code_version'] == anchor['code_version']
        valid = valid and observation['content_sha256'] == content_hash(raw) and raw.get('ambiguous', False) is False
        polarity = 'POSITIVE' if positive else 'NEGATIVE'
        arguments = {} if positive else {'exhaustive': True, 'scope_complete': True}
        claim = dict(evidence_id='e_' + content_hash([observation['obs_id'], polarity, anchor])[:24], run_id=observation['run_id'], created_seq=observation['event_seq'], obs_ids=[observation['obs_id']], predicate='EXISTS', polarity=polarity, subject=deepcopy(anchor), arguments=arguments, proposition_text=proposition_text('EXISTS', polarity, anchor, arguments), claim_type='EXISTENCE', anchor=anchor, granularity=anchor['granularity'], validity='VALID' if valid else 'UNCERTAIN', tier='PRIMARY', origin_family=observation['origin_id'], extractor_rule_id='X_EXISTENCE_POS_V1' if positive else 'X_EXISTENCE_NEG_V1', extractor_version='1.0.0', code_version=anchor['code_version'])
        return [validate_evidence(claim)]

    def _extract_heuristic_fault(self, observation, raw):
        if raw.get('predicate') != 'FAULT_RELEVANT' or raw.get('polarity') != 'POSITIVE':
            return []
        anchor = raw.get('anchor')
        basis = raw.get('basis')
        if not isinstance(anchor, dict) or basis not in ('NATIVE_PROPOSAL_EXACT_FILE', 'SOURCE_GROUNDED_REVIEW', 'SOURCE_GROUNDED_REVIEW_LINE'):
            return []
        anchor = canonical_anchor(anchor)
        expected_granularity = 'LINE' if basis == 'SOURCE_GROUNDED_REVIEW_LINE' else 'FILE'
        if anchor['granularity'] != expected_granularity or raw.get('status') != 'found':
            return []
        valid = (raw.get('code_version') == observation['code_version'] == anchor['code_version']
                 and observation['content_sha256'] == content_hash(raw)
                 and raw.get('ambiguous', False) is False
                 and (('R_SOURCE_GROUNDED_REVIEW_EXACT_LINE_V1' in self.ruleset['enabled_rules']
                       and raw.get('carrier') == 'source_grounded_fault_signal'
                       and basis == 'SOURCE_GROUNDED_REVIEW_LINE'
                       and raw.get('source_quote_verified') is True
                       and raw.get('failure_quote_verified') is True)
                      or ('R_SOURCE_GROUNDED_REVIEW_EXACT_V1' in self.ruleset['enabled_rules']
                       and raw.get('carrier') == 'source_grounded_fault_signal'
                       and basis == 'SOURCE_GROUNDED_REVIEW'
                       and raw.get('source_quote_verified') is True
                       and raw.get('failure_quote_verified') is True)
                      or ('R_HEURISTIC_FAULT_PROPOSAL_EXACT_FILE_V1' in self.ruleset['enabled_rules']
                          and raw.get('carrier') == 'heuristic_fault_signal'
                          and basis == 'NATIVE_PROPOSAL_EXACT_FILE')))
        arguments = {'basis': basis}
        claim = dict(
            evidence_id='e_' + content_hash([observation['obs_id'], 'FAULT_RELEVANT', anchor, basis])[:24],
            run_id=observation['run_id'], created_seq=observation['event_seq'],
            obs_ids=[observation['obs_id']], predicate='FAULT_RELEVANT', polarity='POSITIVE',
            subject=deepcopy(anchor), arguments=arguments,
            proposition_text=proposition_text('FAULT_RELEVANT', 'POSITIVE', anchor, arguments),
            claim_type='FAULT_RELEVANCE', anchor=anchor, granularity=anchor['granularity'],
            validity='VALID' if valid else 'UNCERTAIN', tier='INTERPRETIVE',
            origin_family=observation['origin_id'],
            extractor_rule_id=('X_SOURCE_GROUNDED_REVIEW_EXACT_LINE_V1' if basis == 'SOURCE_GROUNDED_REVIEW_LINE'
                               else 'X_SOURCE_GROUNDED_REVIEW_EXACT_V1' if basis == 'SOURCE_GROUNDED_REVIEW'
                               else 'X_HEURISTIC_FAULT_PROPOSAL_EXACT_FILE_V1'),
            extractor_version='1.0.0', code_version=anchor['code_version'])
        return [validate_evidence(claim)]

class Binder:
    def bind(self, evidence, hypothesis, ruleset=None, created_seq=None):
        validate_evidence(evidence)
        ruleset = checked_ruleset(ruleset)
        exact = evidence['subject']['canonical_id'] == hypothesis['anchor']['canonical_id']
        semantic = ((evidence['predicate'] == 'EXISTS' and hypothesis['claim_type'] == 'EXISTS')
                    or (evidence['predicate'] == 'FAULT_RELEVANT'
                        and hypothesis['claim_type'] == 'FAULT_RELEVANT'))
        provenance = evidence['validity'] == 'VALID' and hypothesis['provenance_valid'] and evidence['code_version'] == hypothesis['code_version']
        if evidence['predicate'] == 'FAULT_RELEVANT':
            rule = ('R_SOURCE_GROUNDED_REVIEW_EXACT_LINE_V1' if evidence['arguments']['basis'] == 'SOURCE_GROUNDED_REVIEW_LINE'
                    else 'R_SOURCE_GROUNDED_REVIEW_EXACT_V1' if evidence['arguments']['basis'] == 'SOURCE_GROUNDED_REVIEW'
                    else 'R_HEURISTIC_FAULT_PROPOSAL_EXACT_FILE_V1')
        else:
            rule = 'R_EXISTENCE_POS_V1' if evidence['polarity'] == 'POSITIVE' else 'R_EXISTENCE_NEG_V1'
        enabled = rule in ruleset['enabled_rules']
        accepted = exact and semantic and provenance and enabled
        relation = ('SUPPORT' if evidence['polarity'] == 'POSITIVE' else 'ATTACK') if accepted else ('IRRELEVANT' if not semantic or not exact else 'UNRESOLVED')
        if accepted and evidence['predicate'] == 'FAULT_RELEVANT':
            reason = ('source-quote and failure-quote checked interpretive review, exact LINE'
                      if rule == 'R_SOURCE_GROUNDED_REVIEW_EXACT_LINE_V1' else
                      'source-quote and failure-quote checked interpretive review, exact FILE'
                      if rule == 'R_SOURCE_GROUNDED_REVIEW_EXACT_V1' else
                      'heuristic pilot exact native proposal FILE predicate')
        else:
            reason = 'reviewed exact existence predicate' if accepted else ('claim type mismatch; draft FAULT_RELEVANT rules disabled' if not semantic else 'anchor mismatch' if not exact else 'rule disabled' if not enabled else 'provenance or evidence validity failed')
        binding = dict(binding_id='b_' + content_hash([evidence['evidence_id'], hypothesis['hypothesis_id'], ruleset['version']])[:24], run_id=hypothesis['run_id'], evidence_id=evidence['evidence_id'], hypothesis_id=hypothesis['hypothesis_id'], relation=relation, semantic_check=semantic, granularity_check=exact, provenance_check=bool(provenance), rule_id=rule, rule_version=ruleset['version'], validity='VALID' if accepted else 'UNCERTAIN', directness='DIRECT', strength='SOFT' if evidence['predicate'] == 'FAULT_RELEVANT' else 'HARD', reason=reason, created_seq=created_seq if created_seq is not None else evidence['created_seq'])
        return validate_binding(binding)
