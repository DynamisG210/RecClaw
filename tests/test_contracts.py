import copy
import importlib.util
from pathlib import Path
import pytest
import recrsi
from recrsi.cli import readonly_inspection


def selector():
    p=Path(recrsi.__file__).parent/'native/task/research/measured_execution/measured_selector.py'
    spec=importlib.util.spec_from_file_location('selector_test',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_selection_preserves_committed_bytes_and_rejects_mutation():
    m=selector()
    pair={'proposals':[{'program':{'source':'original A\n'}},{'program':{'source':'original B\n'}}]}
    commit=m.commit_pair(pair,{'run':'fixture','call':'1'})
    pair['proposals'][0]['program']['source']='changed outside'
    row=commit['programs'][0]
    decision={'proposals':[{'candidate_id':row['candidate_id']}]}
    assert m.choose_from_commit(commit,decision)==row['proposal']
    with pytest.raises(ValueError,match='uncommitted'):
        m.choose_from_commit(commit,{'proposals':[{'candidate_id':'invented'}]})
    row['proposal']['program']['source']='tampered'
    with pytest.raises(ValueError,match='changed'):m.choose_from_commit(commit,decision)


@pytest.mark.parametrize('event,args',[
    ('open',('output','w',0)),('socket.connect',(None,None)),('subprocess.Popen',(None,)),
    ('os.remove',('x',)),('os.mkdir',('x',))])
def test_inspection_denies_effects(event,args):
    with pytest.raises(PermissionError):readonly_inspection(event,args)


def test_inspection_allows_read():
    readonly_inspection('open',('source','r',0))

def test_parent_contrast_keeps_actual_parent_root_and_frontier_distinct(tmp_path,monkeypatch):
    import hashlib,json
    from types import SimpleNamespace
    native=Path(recrsi.__file__).parent/'native/task/research'
    monkeypatch.syspath_prepend(str(native))
    monkeypatch.syspath_prepend(str(native/'mechanism_contrast'))
    import prospective_revision
    root='class FreshCandidateModel:\n    def calculate_loss(self):\n        return 1\n'
    parent='class FreshCandidateModel:\n    def calculate_loss(self):\n        return 2\n'
    child='class FreshCandidateModel:\n    def calculate_loss(self):\n        return 3\n'
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    ledger={'results':[dict(valid=True,root=True,source_sha256=sha(root),primary_metric=.3),
                       dict(valid=True,root=False,source_sha256=sha(parent),primary_metric=.2)]}
    (tmp_path/'EVALUATION_LEDGER.json').write_text(json.dumps(ledger))
    adapter=SimpleNamespace(config={'output':str(tmp_path)},_registry_root_id='root',_parent_source=lambda p:{'root':root,'own-parent':parent}[p])
    commit={'programs':[dict(candidate_id='child',source_sha256=sha(child),proposal={'parent_ref':'own-parent','program':{'source':child}})]}
    context={'objective':{'space':{'branch_allocation':{'mode':'REFINE'}}}}
    record=prospective_revision.program_contrasts(adapter,commit,context)['child']
    assert record['status']=='EXACT_PARENT_SOURCE_LINKED'
    assert record['construction_parent']['own_DEV_NDCG5']==.2
    assert record['current_global_frontier']['own_DEV_NDCG5']==.3
    assert record['original_root']['own_DEV_NDCG5']==.3
    context['objective']['space']['branch_allocation']['mode']='DISCOVER'
    assert prospective_revision.program_contrasts(adapter,commit,context)['child']['status']=='PARENT_OUTSIDE_DISCOVER_BOUNDARY'

def test_negative_feedback_is_projected_with_separate_comparisons(monkeypatch):
    import hashlib,json
    native=Path(recrsi.__file__).parent/'native/task/research'
    monkeypatch.syspath_prepend(str(native))
    import source_bound_context as feedback
    sources={k:f'class FreshCandidateModel:\n    def calculate_loss(self):\n        return {v}\n' for k,v in [('root',1),('parent',2),('child',3)]}
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    values={'root':.3,'parent':.2,'child':.25}
    rows=[dict(valid=True,root=k=='root',genuine=k!='root',source_sha256=sha(sources[k]),primary_metric=values[k],available_index=i) for i,k in enumerate(values)]
    executions={k:dict(implementation={'files':[dict(path='candidate.py',content=sources[k])]},entrypoint_source_sha256=sha(sources[k]),development_metrics={'primary_metric':{'name':'ndcg@5','value':values[k]}},worker_configuration={'dataset':'synthetic'},seed=1) for k in ('parent','child')}
    decisions={'parent':{'parent_id':'root'},'child':{'parent_id':'parent'}}
    cards=feedback.build_cards(executions,{'results':rows},decisions,'root',sources['root'])
    assert cards['parent']['construction_parent']['child_minus_parent']==pytest.approx(-.1)
    assert cards['child']['construction_parent']['child_minus_parent']==pytest.approx(.05)
    assert cards['child']['prior_global_frontier']['child_minus_frontier']==pytest.approx(-.05)
    context={'objective':{'space':{'branch_allocation':{'mode':'REFINE'}}},'context':{'digest':'old'}}
    projected,receipt=feedback.project_prompt(feedback.MARKER+json.dumps(context),cards)
    decoded=json.JSONDecoder().raw_decode(projected[projected.index('{',projected.index(feedback.MARKER)):])[0]
    assert decoded['objective']['space']['OWN_COMPUTATION_CONTRASTS']==cards
    assert receipt['own_measured_candidates']==2
    rows[1]['available_index']=3
    with pytest.raises(ValueError,match='Future evidence'):
        feedback.build_cards(executions,{'results':rows},decisions,'root',sources['root'])
