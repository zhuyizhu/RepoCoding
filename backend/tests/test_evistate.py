"""Integration tests execute the real copied EviState core on temporary inputs.
These fixtures are not benchmark localization results or model inference.
"""
import subprocess
from pathlib import Path
from app.engine.evistate import handoff

HOME = Path(__file__).resolve().parents[2]/'runtime/evistate'


def test_real_tracker_preserves_q_z_and_uses_typed_bindings(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,text=True)
    git('init')
    (repo/'A.java').write_text('class A {}',encoding='utf-8')
    git('add','.')
    git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')
    candidates = [dict(file_path=p,confidence=1,reason='rank must not establish fault relevance') for p in ['A.java','B.java','C.java','D.java']]
    result = handoff(candidates=candidates,task_id='test-run',repo_key='fixture',repository=repo,evistate_home=HOME,artifact_root=tmp_path/'artifacts')
    snapshot = result['snapshot']
    hs = list(snapshot['hypotheses'].values())
    assert all(h['epistemic_state']=='UNKNOWN' for h in hs if h['claim_type']=='FAULT_RELEVANT')
    exists = {h['anchor']['path']:h for h in hs if h['claim_type']=='EXISTS'}
    assert exists['A.java']['epistemic_state']=='SUPPORTED'
    assert exists['B.java']['epistemic_state']=='REFUTED'
    excluded = next(h for h in hs if h['anchor']['path']=='D.java' and h['claim_type']=='FAULT_RELEVANT')
    assert excluded['lifecycle']=='TERMINATED'
    assert excluded['epistemic_state']=='UNKNOWN'
    assert {'SUPPORT','ATTACK','IRRELEVANT'} <= {b['relation'] for b in snapshot['bindings'].values()}
    assert result['paths']==['A.java','B.java','C.java']
    assert (tmp_path/'artifacts/test-run/repair-input.json').is_file()
    assert git('status','--porcelain').stdout==''

    from sqlalchemy import create_engine,select
    from sqlalchemy.orm import Session
    from app.database import Base
    from app.models import Task,SupportAttackBinding,TaskEvent
    from app.engine.persist_evistate import persist_snapshot
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        task=Task(issue_title='fixture',issue_text='fixture',repo_key='fixture',repository_path='repo',repograph_path='graph')
        db.add(task);db.flush()
        persist_snapshot(db,task,result);db.commit()
        assert task.workflow_state_q['source_hash']==snapshot['source_hash']
        assert 'epistemic_state' not in next(iter(task.workflow_state_q['hypotheses'].values()))
        assert 'lifecycle' not in next(iter(task.cognitive_state_z['hypotheses'].values()))
        assert len(db.scalars(select(SupportAttackBinding)).all())==len(snapshot['bindings'])
        assert len(db.scalars(select(TaskEvent)).all())==len(result['events'])+1
    engine.dispose()
