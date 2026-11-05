from app.models import CandidateHypothesis, Evidence, SupportAttackBinding
from app.events import record_event


def persist_snapshot(db, task, result):
    snapshot = result['snapshot']
    common = dict(source='EviState.StateTracker.snapshot',source_hash=snapshot['source_hash'],seq=snapshot['seq'])
    task.workflow_state_q = dict(**common,hypotheses={hid:{k:h[k] for k in ('lifecycle','visibility')} for hid,h in snapshot['hypotheses'].items()})
    task.cognitive_state_z = dict(**common,hypotheses={hid:{k:h[k] for k in ('epistemic_state','support_origin_families','attack_origin_families')} for hid,h in snapshot['hypotheses'].items()})
    candidates, evidence = {}, {}
    for rank,(hid,h) in enumerate(snapshot['hypotheses'].items(),1):
        row = CandidateHypothesis(task_id=task.id,rank=rank,target=h['anchor']['path'],target_type='file',file_path=h['anchor']['path'],
            hypothesis=h['claim_type']+'('+h['anchor']['path']+')',source_payload=dict(source='evistate',**h))
        db.add(row)
        db.flush()
        candidates[hid] = row.id
    for eid,e in snapshot['evidence'].items():
        row = Evidence(task_id=task.id,source='evistate',kind=e['predicate'],content=e['proposition_text'],source_payload=e)
        db.add(row)
        db.flush()
        evidence[eid] = row.id
    for b in snapshot['bindings'].values():
        db.add(SupportAttackBinding(candidate_id=candidates[b['hypothesis_id']],evidence_id=evidence[b['evidence_id']],relation=b['relation'],source_payload=b))
    for event in result.get('events',[]):
        record_event(db,task.id,'evistate_event','evistate',event)
    record_event(db,task.id,'evistate_handoff','evistate',dict(paths=result['paths'],decision=result['decision'],ruleset=result['ruleset']))
