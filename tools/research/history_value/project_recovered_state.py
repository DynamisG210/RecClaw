"""Offline native state projection of a completed recovery; executes no model or FIT."""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def project(home, original, target):
    def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
    assert not target.exists()
    recovery = read(home/'engineering_recovery/RETAIN_R13_EXECUTION.json')
    assert recovery['stage'] == 'NATIVE_RECOVERY_RETURNED'
    assert recovery['result']['status'] == 'SUCCESS'
    assert recovery['result']['test_accessed'] is False
    target.mkdir()
    (target/'runtime').symlink_to(home/'runtime', target_is_directory=True)
    for name in ('ALLOCATION.json', 'INITIAL_INPUT_CONTRACT_V2.json', 'runner.py', 'initial_input_contract.py'):
        shutil.copy2(home/name, target/name)
    for name in ('BUDGET.json', 'EVALUATOR_CONFIG.json', 'PROVIDER.json', 'LAUNCH.json'):
        p=target/'retain'/name; p.parent.mkdir(exist_ok=True)
        p.write_text((home/'retain'/name).read_text().replace(str(home),str(target)))
    cfg=read(target/'retain/EVALUATOR_CONFIG.json')
    cfg['e2_budget_sha256']=hashlib.sha256((target/'retain/BUDGET.json').read_bytes()).hexdigest()
    (target/'retain/EVALUATOR_CONFIG.json').write_text(json.dumps(cfg))
    sys.path[:0]=[str(target),str(target/'runtime'),str(original),str(Path(cfg['task_root'])/'controller_adapter'),*read(original/'ORIGINAL_RUNTIME_PATHS.json')]
    import auto_search_space
    import e2_ro_search_space_adapter
    import morec_ro_execution
    from recclaw_core.research_line import runtime
    checkpoint=pickle.loads((home/'retain/search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes())
    paid=pickle.loads((home/'retain/search/campaign/ROUND_13_PREPARED_CHECKPOINT.pkl').read_bytes())['prepared']
    (target/'retain_INITIAL_STATE.pkl').write_bytes(pickle.dumps(checkpoint.state_before))
    source=Path(recovery['repair']['repaired'])
    files={name:(source/name).read_text() for name in ('main_code.py','models.py','diagnosis_tools.py','self_evolverec_interface.py','morec_initialization.py')}
    import sqlite3
    databases=list((home/'retain/search/campaign/provider_calls/implementations').rglob('broker.sqlite3'))
    assert len(databases)==1
    with sqlite3.connect(str(databases[0])) as database:
        original_response=json.loads(database.execute('select response_json from calls').fetchone()[0])['proposals'][0]
    import morec_main_evaluator as evaluator
    import runner
    count={'implementations_rebound':0,'qualification_receipts_rebound':0,'results_rebound':0}
    native_lane=runtime._run_innovation_lane
    def lane(candidates, **kw):
        def implementation(request):
            count['implementations_rebound']+=1
            assert count['implementations_rebound']==1
            return {**original_response,'files':[{'path':k,'content':v} for k,v in files.items()]}
        kw['inputs']=replace(kw['inputs'],implementer=implementation)
        result=native_lane(candidates,**kw)
        assert result.selected_outcome.spec.digest == paid.innovation.selected_outcome.spec.digest
        return result
    def qualification(config, command, **kwargs):
        candidate=Path(command[command.index('--candidate-root')+1])
        assert {k:(candidate/k).read_text() for k in files}==files
        dest=Path(command[command.index('--output-dir')+1]);dest.mkdir(parents=True,exist_ok=True)
        (dest/'RO_SHARED_QUALIFICATION_SMOKE_RESULT.json').write_text(json.dumps(recovery['qualification']))
        (dest/'REUSED_QUALIFICATION_PROVENANCE.json').write_text(json.dumps({'source':str(home/'engineering_recovery/RETAIN_R13_EXECUTION.json'),'physical_execution_repeated':False}))
        count['qualification_receipts_rebound']+=1
        return subprocess.CompletedProcess(command,0,'Reused completed identical-source qualification','')
    def result(config, proposed):
        assert proposed==files
        count['results_rebound']+=1
        assert count['results_rebound']==1
        return recovery['result']
    def deny(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
            raise RuntimeError('Offline projection forbids provider/GPU execution: '+event)
    sys.addaudithook(deny)
    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime,'produce_research_specs',lambda *a,**k:paid.producer_outcomes))
        stack.enter_context(patch.object(runtime,'_run_innovation_lane',lane))
        stack.enter_context(patch.object(evaluator,'run_qualification',qualification))
        stack.enter_context(patch.object(evaluator,'run_files',result))
        runner.compose(target,original,'retain',False,{'cpus':sorted(os.sched_getaffinity(0))[:4],'gpu_uuid':'OFFLINE_NO_PHYSICAL_WORK'})
    state=pickle.loads((target/'retain/search/campaign/CAMPAIGN_STATE.pkl').read_bytes())
    record=pickle.loads((target/'retain/search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes())
    assert state.next_round_index==14 and record.result.has_metric_bearing_attempt
    assert count=={'implementations_rebound':1,'qualification_receipts_rebound':1,'results_rebound':1}
    receipt={'stage':'NATIVE_RECOVERY_STATE_PROJECTED_OFFLINE','counts':count,'new_api_calls':0,'new_fits':0,'source_result':str(home/'engineering_recovery/RETAIN_R13_EXECUTION.json'),'raw_failed_round_preserved':True,'state_digest':state.digest,'next_round':state.next_round_index,'record_status':record.status,'state_path':str(target/'retain/search/campaign/CAMPAIGN_STATE.pkl')}
    (target/'PROJECTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('home',type=Path);p.add_argument('original',type=Path);p.add_argument('target',type=Path);a=p.parse_args();project(a.home,a.original,a.target)
