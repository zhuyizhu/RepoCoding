import pytest
from app.engine.repair import FileTools, run_repair
from app.engine.codelens import IntegrationUnavailable
from app.config import Settings


def test_file_tools_reject_scope_escape_and_ambiguous_edits(tmp_path):
    (tmp_path/'A.java').write_text('class A { int x = 1; }',encoding='utf-8')
    tools=FileTools(tmp_path,['A.java'])
    with pytest.raises(ValueError):tools.read_file('../secret')
    with pytest.raises(ValueError):tools.replace_text('A.java','missing','new')
    with pytest.raises(ValueError):tools.read_file('A.java',0,1)
    tools.replace_text('A.java','int x = 1','int x = 2')
    assert 'int x = 2' in tools.read_file('A.java')
    (tmp_path/'ATest.java').write_text('class ATest {}')
    with pytest.raises(ValueError,match='test files'):
        FileTools(tmp_path,['ATest.java']).replace_text('ATest.java','ATest','AChangedTest')


def test_disabled_repair_never_calls_model_or_git():
    with pytest.raises(IntegrationUnavailable,match='disabled'):
        run_repair(None,None,[],Settings(repair_enabled=False),'unused')
