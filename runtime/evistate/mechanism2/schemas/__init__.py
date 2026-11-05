"""Strict, JSON-native contracts for Mechanism 2.0."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import posixpath

SCHEMA_VERSION = '2.0.0'
GRANULARITIES = ('REPO', 'FILE', 'CLASS', 'METHOD', 'LINE')
EVENT_TYPES = {'PROPOSE', 'RETRIEVE', 'OBSERVE', 'REFINE', 'RANK', 'PRUNE', 'HANDOFF', 'FINALIZE', 'EVIDENCE_INVALIDATED', 'PRE_COMMIT_TRANSITION', 'TRANSITION_COMMITTED'}
REASONS = {'RANK_OUT', 'PRUNED', 'CONTEXT_LOSS', 'STAGE_HANDOFF', 'SUPERSEDED', 'NATIVE_REJECT', 'QUEUE_EVICT', 'BRANCH_CLOSE', 'MEMORY_EVICT', 'FINAL_COMMIT', 'OTHER'}
BindingDirectness = frozenset({'DIRECT', 'INDIRECT'})
BindingStrength = frozenset({'HARD', 'SOFT'})
FAULT_RELEVANT_HEURISTIC_BASES = frozenset({
    'NATIVE_PROPOSAL_EXACT_FILE',
    'SOURCE_GROUNDED_REVIEW',
    'SOURCE_GROUNDED_REVIEW_LINE',
})

class SchemaError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise SchemaError(message)

def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def content_hash(value):
    return hashlib.sha256(stable_json(value).encode('utf-8')).hexdigest()

def required(obj, fields, context):
    require(isinstance(obj, dict), context + ' must be an object')
    for key in fields:
        require(key in obj, context + ' missing ' + key)

def strings(value, context, unique=False):
    require(isinstance(value, list) and all(isinstance(v, str) and v for v in value), context + ' must be a string list')
    if unique:
        require(len(value) == len(set(value)), context + ' has duplicate IDs')

def nonempty_strings(obj, fields, context):
    for key in fields:
        require(isinstance(obj.get(key), str) and bool(obj[key].strip()), context + ' invalid ' + key)

def canonical_anchor(anchor):
    required(anchor, ('repo_id', 'code_version', 'granularity'), 'anchor')
    require(all(isinstance(anchor[k], str) and anchor[k] for k in ('repo_id', 'code_version')), 'anchor identity must be nonempty strings')
    granularity = anchor['granularity']
    require(granularity in GRANULARITIES, 'unknown granularity')
    result = {k: anchor[k] for k in ('repo_id', 'code_version', 'granularity')}
    path = anchor.get('path', '')
    require(isinstance(path, str), 'anchor path must be string')
    path = path.replace('\\', '/')
    require(not path.startswith('/') and ':' not in path and '..' not in path.split('/'), 'anchor path must be repository relative')
    path = posixpath.normpath(path) if path else ''
    if path == '.':
        path = ''
    require(granularity == 'REPO' or bool(path), 'non-repo anchor requires path')
    require(granularity != 'REPO' or not path, 'repo anchor cannot carry a file path')
    result['path'] = path
    allowed_locators = {
        'REPO': set(),
        'FILE': set(),
        'CLASS': {'class_fqn'},
        'METHOD': {'class_fqn', 'method_signature'},
        'LINE': {'class_fqn', 'method_signature', 'line_start', 'line_end'},
    }[granularity]
    for key in ('class_fqn', 'method_signature', 'line_start', 'line_end'):
        if key in anchor:
            require(key in allowed_locators, granularity + ' anchor cannot carry finer locator ' + key)
            result[key] = anchor[key]
    for key in ('class_fqn', 'method_signature'):
        if key in result:
            require(isinstance(result[key], str) and bool(result[key]), 'invalid anchor ' + key)
    if granularity == 'CLASS':
        require(isinstance(result.get('class_fqn'), str) and bool(result['class_fqn']), 'class requires class_fqn')
    if granularity == 'METHOD':
        sig = result.get('method_signature')
        require(isinstance(sig, str) and '(' in sig and sig.endswith(')'), 'method requires unambiguous signature')
    if granularity == 'LINE':
        require(type(result.get('line_start')) is int and type(result.get('line_end')) is int and 0 < result['line_start'] <= result['line_end'], 'line requires positive ordered interval')
    result['canonical_id'] = content_hash(result)
    if 'canonical_id' in anchor:
        require(anchor['canonical_id'] == result['canonical_id'], 'anchor canonical_id mismatch')
    return result

def hypothesis_id(run_id, issue_id, claim_type, anchor):
    return 'h_' + content_hash([run_id, issue_id, claim_type, canonical_anchor(anchor)])[:24]

def make_event(run_id, seq, event_type, payload, native_event_ref='native:inline', timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    event = dict(schema_version=SCHEMA_VERSION, run_id=run_id, seq=seq, type=event_type, payload=deepcopy(payload), native_event_ref=native_event_ref, timestamp=timestamp, adapter_version='2.0.0', raw_payload_sha256=content_hash(payload))
    event['event_id'] = 'evt_' + content_hash(event)[:24]
    validate_event(event)
    return event

def validate_provenance(value):
    required(value, ('native_event_ref', 'origin_id', 'raw_ref_or_inline', 'content_sha256', 'code_version', 'mapping_rule_version'), 'provenance')
    for key in ('native_event_ref', 'origin_id', 'content_sha256', 'code_version', 'mapping_rule_version'):
        require(isinstance(value[key], str) and bool(value[key]), 'invalid provenance ' + key)
    require(len(value['content_sha256']) == 64 and all(c in '0123456789abcdef' for c in value['content_sha256']), 'invalid provenance SHA256')

def validate_observation(value):
    required(value, ('obs_id', 'run_id', 'event_seq', 'producer', 'action', 'raw_payload_or_ref', 'content_sha256', 'origin_id', 'code_version', 'timestamp', 'native_event_ref', 'mapping_rule_version'), 'observation')
    nonempty_strings(value, ('obs_id', 'run_id', 'producer', 'action', 'origin_id', 'code_version', 'native_event_ref', 'mapping_rule_version'), 'observation')
    require(type(value['event_seq']) is int and value['event_seq'] > 0, 'invalid observation sequence')
    require(not any(key in value for key in ('relation', 'support', 'attack', 'SUPPORT', 'ATTACK')), 'observation must not encode binding relations')
    require(value['content_sha256'] == content_hash(value['raw_payload_or_ref']), 'observation hash mismatch')
    return value

def serialize_predicate(predicate, polarity, subject, arguments):
    require(predicate in ('EXISTS', 'FAULT_RELEVANT'), 'unsupported predicate')
    require(polarity in ('POSITIVE', 'NEGATIVE'), 'unsupported predicate/polarity')
    require(isinstance(arguments, dict), 'predicate arguments must be object')
    if predicate == 'EXISTS':
        require(arguments == ({} if polarity == 'POSITIVE' else {'exhaustive': True, 'scope_complete': True}), 'invalid existence predicate arguments')
        if polarity == 'NEGATIVE':
            require(all(type(v) is bool for v in arguments.values()), 'negative completeness fields must be boolean')
    else:
        require(polarity == 'POSITIVE', 'FAULT_RELEVANT only supports positive pilot claims')
        require(arguments.keys() == {'basis'} and arguments.get('basis') in FAULT_RELEVANT_HEURISTIC_BASES,
                'invalid FAULT_RELEVANT heuristic arguments')
    anchor = canonical_anchor(subject)
    label = anchor['path'] or anchor['repo_id']
    if anchor['granularity'] == 'METHOD':
        label += ':' + anchor['method_signature']
    elif anchor['granularity'] == 'CLASS':
        label += ':' + anchor['class_fqn']
    elif anchor['granularity'] == 'LINE':
        label += ':L%d-%d' % (anchor['line_start'], anchor['line_end'])
    if predicate == 'EXISTS':
        return ('Exists' if polarity == 'POSITIVE' else 'NotExists') + '(' + label + '@' + anchor['code_version'] + ')'
    return 'FaultRelevant(' + label + '@' + anchor['code_version'] + ';basis=' + arguments['basis'] + ')'

def validate_evidence(value):
    required(value, ('evidence_id', 'run_id', 'created_seq', 'obs_ids', 'predicate', 'polarity', 'subject', 'arguments', 'proposition_text', 'claim_type', 'anchor', 'granularity', 'validity', 'tier', 'origin_family', 'extractor_rule_id', 'extractor_version', 'code_version'), 'evidence')
    nonempty_strings(value, ('evidence_id', 'run_id', 'origin_family', 'extractor_rule_id', 'extractor_version', 'code_version'), 'evidence')
    strings(value['obs_ids'], 'evidence obs_ids', unique=True)
    require(bool(value['obs_ids']), 'evidence requires observation')
    require(type(value['created_seq']) is int and value['created_seq'] > 0, 'invalid evidence created_seq')
    require(value['claim_type'] in ('EXISTENCE', 'FAULT_RELEVANCE'), 'unsupported evidence claim_type')
    require((value['predicate'], value['claim_type']) in (('EXISTS', 'EXISTENCE'), ('FAULT_RELEVANT', 'FAULT_RELEVANCE')),
            'evidence predicate/claim_type mismatch')
    require(value['validity'] in ('VALID', 'INVALID', 'UNCERTAIN'), 'invalid evidence validity')
    require(value['tier'] in ('PRIMARY', 'DERIVED', 'INTERPRETIVE'), 'invalid evidence tier')
    subject, anchor = canonical_anchor(value['subject']), canonical_anchor(value['anchor'])
    require(subject == anchor and value['granularity'] == anchor['granularity'] and value['code_version'] == anchor['code_version'], 'evidence subject/anchor/version mismatch')
    require(value['proposition_text'] == serialize_predicate(value['predicate'], value['polarity'], subject, value['arguments']), 'evidence display does not match typed proposition')
    return value

def validate_binding(value):
    required(value, ('binding_id', 'run_id', 'evidence_id', 'hypothesis_id', 'relation', 'semantic_check', 'granularity_check', 'provenance_check', 'rule_id', 'rule_version', 'validity', 'directness', 'strength', 'reason', 'created_seq'), 'binding')
    nonempty_strings(value, ('binding_id', 'run_id', 'evidence_id', 'hypothesis_id', 'rule_id', 'rule_version', 'reason'), 'binding')
    require(value['relation'] in ('SUPPORT', 'ATTACK', 'IRRELEVANT', 'UNRESOLVED'), 'invalid binding relation')
    require(value['validity'] in ('VALID', 'INVALID', 'UNCERTAIN'), 'invalid binding validity')
    require(value['directness'] in BindingDirectness and value['strength'] in BindingStrength, 'invalid binding directness/strength')
    require(all(type(value[k]) is bool for k in ('semantic_check', 'granularity_check', 'provenance_check')), 'binding checks must be booleans')
    require(type(value['created_seq']) is int and value['created_seq'] > 0, 'invalid binding created_seq')
    return value

def validate_transition(p):
    required(p, ('transition_id', 'run_id', 'kind', 'prev_candidate_ids', 'proposed_candidate_ids', 'capacity_k', 'native_order', 'reachability_descriptor', 'event_seq_watermark', 'budget_before'), 'transition')
    nonempty_strings(p, ('transition_id', 'run_id', 'kind'), 'transition')
    for key in ('from_stage', 'to_stage', 'native_rank_scope'):
        if key in p and p[key] is not None:
            nonempty_strings(p, (key,), 'transition')
    if 'strict_cardinality' in p:
        require(type(p['strict_cardinality']) is bool, 'strict_cardinality must be boolean')
    for field in ('prev_candidate_ids', 'proposed_candidate_ids', 'native_order'):
        strings(p[field], field, unique=True)
    require(type(p['capacity_k']) is int and p['capacity_k'] >= 0, 'capacity_k must be nonnegative integer')
    require(len(p['proposed_candidate_ids']) <= p['capacity_k'], 'proposal exceeds capacity')
    require(type(p['event_seq_watermark']) is int and p['event_seq_watermark'] >= 0, 'invalid watermark')
    required(p['reachability_descriptor'], ('continuable_parent_ids', 'native_next_actions', 'refinement_mapping_refs', 'irreversibility_basis'), 'reachability_descriptor')
    strings(p['reachability_descriptor']['continuable_parent_ids'], 'continuable_parent_ids', unique=True)
    require(isinstance(p['reachability_descriptor']['native_next_actions'], list) and isinstance(p['reachability_descriptor']['refinement_mapping_refs'], list), 'reachability actions/refs must be lists')
    descriptor = p['reachability_descriptor']
    require(isinstance(descriptor['irreversibility_basis'], (str, dict, list)), 'irreversibility_basis must be text or structured proof')
    if 'irreversible' in descriptor:
        require(type(descriptor['irreversible']) is bool, 'irreversible must be boolean')
    if 'executable_candidate_ids' in descriptor:
        strings(descriptor['executable_candidate_ids'], 'executable_candidate_ids', unique=True)
    if 'retained_paths' in descriptor:
        require(isinstance(descriptor['retained_paths'], dict), 'retained_paths must be object')
        for parent, children in descriptor['retained_paths'].items():
            require(isinstance(parent, str) and bool(parent), 'retained path parent must be ID')
            strings(children, 'retained path children', unique=True)
    require(isinstance(p['budget_before'], dict), 'budget_before must be object')

def validate_event(event):
    required(event, ('schema_version', 'run_id', 'event_id', 'seq', 'timestamp', 'native_event_ref', 'type', 'payload', 'adapter_version', 'raw_payload_sha256'), 'event')
    require(isinstance(event['schema_version'], str) and event['schema_version'].split('.')[0] == '2', 'unsupported schema major')
    require(type(event['seq']) is int and event['seq'] > 0, 'seq must be positive integer')
    for key in ('run_id', 'event_id', 'native_event_ref', 'adapter_version'):
        require(isinstance(event[key], str) and bool(event[key]), 'invalid ' + key)
    try:
        timestamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
        require(timestamp.tzinfo is not None and timestamp.utcoffset().total_seconds() == 0, 'timestamp must be UTC')
    except (ValueError, TypeError, AttributeError) as exc:
        raise SchemaError('timestamp must be UTC RFC3339') from exc
    require(event['type'] in EVENT_TYPES, 'unknown event type')
    p = event['payload']
    require(content_hash(p) == event['raw_payload_sha256'], 'event payload hash mismatch')
    fields = {
        'PROPOSE': ('issue_id', 'native_candidate_id', 'claim_type', 'anchor', 'source_kind', 'provenance_ref'),
        'REFINE': ('parent_hypothesis_id', 'child_native_candidate_id', 'child_claim_type', 'child_anchor', 'mapping_rule_id', 'provenance_ref'),
        'OBSERVE': ('producer', 'action', 'raw_result_ref_or_inline', 'origin_id', 'code_version'),
        'RETRIEVE': ('producer', 'action', 'raw_result_ref_or_inline', 'origin_id', 'code_version', 'accessed_candidate_ids'),
        'RANK': ('candidate_ids', 'native_rank_scope', 'native_order', 'raw_rank_records'),
        'PRUNE': ('candidate_ids', 'reason', 'stage', 'reversible'),
        'HANDOFF': ('from_stage', 'to_stage', 'actual_candidate_ids', 'reason'),
        'FINALIZE': ('actual_candidate_ids', 'native_order', 'native_rank_scope', 'output_ref'),
        'EVIDENCE_INVALIDATED': ('evidence_id', 'reason', 'source_event_ref'),
        'TRANSITION_COMMITTED': ('transition_id', 'decision_id', 'actual_candidate_ids', 'actual_native_order', 'capacity_k', 'commit_ref', 'commit_timestamp', 'budget_after'),
    }
    required(p, fields.get(event['type'], ()), event['type'])
    if event['type'] in ('PROPOSE', 'REFINE'):
        canonical_anchor(p['anchor'] if event['type'] == 'PROPOSE' else p['child_anchor'])
        validate_provenance(p['provenance_ref'])
        claim = p['claim_type'] if event['type'] == 'PROPOSE' else p['child_claim_type']
        require(claim in ('EXISTS', 'FAULT_RELEVANT'), 'unsupported hypothesis claim_type')
        if event['type'] == 'PROPOSE':
            nonempty_strings(p, ('issue_id', 'native_candidate_id', 'source_kind'), 'PROPOSE')
        else:
            nonempty_strings(p, ('parent_hypothesis_id', 'child_native_candidate_id', 'mapping_rule_id'), 'REFINE')
    for field in ('candidate_ids', 'actual_candidate_ids', 'accessed_candidate_ids', 'native_order', 'actual_native_order'):
        if field in p:
            strings(p[field], field, unique=True)
    if 'reason' in p and event['type'] in ('PRUNE', 'HANDOFF'):
        require(p['reason'] in REASONS, 'invalid workflow reason')
    if event['type'] == 'PRUNE':
        require(type(p['reversible']) is bool, 'reversible must be boolean')
        nonempty_strings(p, ('stage',), 'PRUNE')
    if event['type'] == 'HANDOFF':
        nonempty_strings(p, ('from_stage', 'to_stage'), 'HANDOFF')
    if event['type'] == 'EVIDENCE_INVALIDATED':
        nonempty_strings(p, ('evidence_id', 'reason', 'source_event_ref'), 'EVIDENCE_INVALIDATED')
    if event['type'] == 'TRANSITION_COMMITTED':
        nonempty_strings(p, ('transition_id', 'decision_id', 'commit_ref', 'commit_timestamp'), 'TRANSITION_COMMITTED')
        require(type(p['capacity_k']) is int and p['capacity_k'] >= 0, 'invalid commit capacity_k')
        require(isinstance(p['budget_after'], dict), 'budget_after must be object')
        if 'actual_continuable_parent_ids' in p:
            strings(p['actual_continuable_parent_ids'], 'actual_continuable_parent_ids', unique=True)
    if event['type'] == 'PRE_COMMIT_TRANSITION':
        validate_transition(p)
        require(p['run_id'] == event['run_id'] and p['event_seq_watermark'] < event['seq'], 'invalid transition run/watermark')
    if event['type'] == 'RANK':
        nonempty_strings(p, ('native_rank_scope',), 'RANK')
        require(isinstance(p['raw_rank_records'], list), 'raw_rank_records must be list')
        for record in p['raw_rank_records']:
            require(isinstance(record, dict), 'rank records must be objects')
            if 'position' in record and record['position'] is not None:
                require(type(record['position']) is int and record['position'] > 0, 'rank position must be positive integer')
            if 'scope' in record:
                nonempty_strings(record, ('scope',), 'rank record')
    stable_json(event)
    return event
