"""New CodeLens FILE handoff adapter over the unmodified EviState core.

Default existence rules only: rank/reason never establishes fault relevance.
Fine locations stay in native payloads; no invented method/line refinement.
"""
from __future__ import annotations
import importlib
import json
import subprocess
import sys
from pathlib import Path
from app.engine.codelens import IntegrationUnavailable


def handoff(*, candidates, task_id, repo_key, repository, evistate_home, artifact_root, source="codelens"):
    if source not in ('codelens','manual_file_input'):
        raise ValueError('Unknown provenance source.')
    home = Path(evistate_home).resolve()
    if not (home / 'mechanism2/core/tracker.py').is_file():
        raise IntegrationUnavailable('EviState core is missing under EVISTATE_HOME/mechanism2.')
    sys.path.insert(0, str(home))
    try:
        StateTracker = importlib.import_module('mechanism2.core.tracker').StateTracker
        Barrier = importlib.import_module('mechanism2.core.barrier').Barrier
        TransitionGuard = importlib.import_module('mechanism2.core.guard').TransitionGuard
        schemas = importlib.import_module('mechanism2.schemas')
    finally:
        sys.path.remove(str(home))
    repo = Path(repository).resolve()
    version = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    if subprocess.run(['git','-C',str(repo),'status','--porcelain'],check=True,capture_output=True,text=True,timeout=15).stdout.strip():
        raise IntegrationUnavailable('EviState provenance requires a clean source checkout.')
    tracker = StateTracker('repocoding-' + task_id)
    def emit(kind,payload):
        event = schemas.make_event(tracker.run_id,tracker.seq+1,kind,payload,native_event_ref='repocoding:' + source + ':' + str(tracker.seq+1))
        tracker.apply(event)
    by_path = {}
    for candidate in candidates:
        path = str(candidate.get('file_path') or '').replace('\\','/')
        if not path or path in by_path:
            continue
        anchor = schemas.canonical_anchor(dict(repo_id=repo_key,code_version=version,granularity='FILE',path=path))
        location = (repo/path).resolve()
        if repo not in location.parents:
            raise IntegrationUnavailable('Candidate resolves outside the source checkout.')
        provenance = dict(native_event_ref=source+':candidates',origin_id=source+':'+task_id+':'+path,
            raw_ref_or_inline=candidate,content_sha256=schemas.content_hash(candidate),code_version=version,mapping_rule_version='repocoding-'+source+'-file-v1')
        ids = {}
        for claim in ('FAULT_RELEVANT','EXISTS'):
            emit('PROPOSE',dict(issue_id=task_id,native_candidate_id=claim+':'+path,claim_type=claim,anchor=anchor,
                source_kind=source+'_file_projection',provenance_ref=provenance))
            ids[claim] = schemas.hypothesis_id(tracker.run_id,task_id,claim,anchor)
        # Exact regular-file lookup in a clean checkout, not a search hit count.
        raw = dict(carrier='entity_lookup',anchor=anchor,status='found' if location.is_file() else 'not_found',
            exhaustive=True,scope_complete=True,code_version=version,ambiguous=False)
        emit('OBSERVE',dict(producer='repocoding-filesystem',action='exact_file_lookup',raw_result_ref_or_inline=raw,
            origin_id='filesystem:'+version+':'+path,code_version=version,mapping_rule_version='exact-file-v1'))
        by_path[path] = ids
    if not by_path:
        raise IntegrationUnavailable('CodeLens returned no candidates with repository-relative file paths.')
    ordered = [v['FAULT_RELEVANT'] for v in by_path.values()]
    emit('RANK',dict(candidate_ids=ordered,native_rank_scope=source+'-file-projection',native_order=ordered,raw_rank_records=[]))
    budget = dict(native_llm_calls=0,native_tool_calls=len(by_path),additional_llm_calls=0,additional_tool_calls=0)
    # Counters cover this adapter only; CodeLens policy usage is stored separately.
    policy = dict(barrier_timeout_ms=5000,strict_cardinality=True,policy_version='repocoding-existence-v1',
        native_llm_calls_limit=0,native_tool_calls_limit=len(by_path))
    proposal = dict(transition_id='repair-handoff',run_id=tracker.run_id,kind='STAGE_HANDOFF',from_stage='localization',to_stage='repair',
        prev_candidate_ids=ordered,proposed_candidate_ids=ordered[:3],capacity_k=min(3,len(ordered)),native_order=ordered,
        native_rank_scope=source+'-file-projection',event_seq_watermark=tracker.seq,budget_before=budget,
        reachability_descriptor=dict(continuable_parent_ids=[],native_next_actions=['repair_selected_files'],refinement_mapping_refs=[],
            irreversible=True,irreversibility_basis='This repair attempt only receives the selected files; excluded files have no re-entry in this attempt.',executable_candidate_ids=ordered))
    output = Path(artifact_root)/task_id
    output.mkdir(parents=True,exist_ok=True)
    def commit(selected):
        paths = [path for path,ids in by_path.items() if ids['FAULT_RELEVANT'] in selected]
        receipt = dict(actual_candidate_ids=selected,commit_ref=str(output/'repair-input.json'),budget_after=budget)
        (output/'repair-input.json').write_text(json.dumps(dict(paths=paths,code_version=version,receipt=receipt),ensure_ascii=False,indent=2),encoding='utf-8')
        return receipt
    result = Barrier(tracker,TransitionGuard(),policy).execute(proposal,commit,mode='intervention')
    snapshot = tracker.snapshot()
    (output/'tracker-snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    paths = json.loads((output/'repair-input.json').read_text(encoding='utf-8'))['paths']
    return dict(events=tracker.store.events,snapshot=snapshot,decision=result['decision'],paths=paths,code_version=version,
        mapping_version='repocoding-'+source+'-file-v1',ruleset='existence-only; fault relevance remains UNKNOWN')
