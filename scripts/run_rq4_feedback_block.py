"""Execute an allocated feedback-only block through the original native broker."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--runtime-paths',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--execute',action='store_true')
a=p.parse_args()
for path in reversed(json.loads(a.runtime_paths.read_text())):
    sys.path.insert(0,path)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1

def read(path):return json.loads(path.read_text())
def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)
allocation=read(a.root/'ALLOCATION.json')
report=read(a.root/'payloads/PAYLOAD_REPORT.json')
assert allocation['max_feedback_calls']==14 and allocation['new_fit_cap']==0
assert allocation['api_usd_cap']==4.75 and len(report['states'])==7
lock=(a.root/'OWNER.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ledger_path=a.root/'FINITE_API_LEDGER.json'
ledger=read(ledger_path) if ledger_path.exists() else {'identity':allocation['identity'],'calls':[]}
class Budget:
    def __init__(self,cell,bound):self.cell,self.bound=cell,bound
    def request_token_bound(self,payload,ceiling):return ceiling
    def reserve_call(self,model,request_digest,ceiling):
        assert len(ledger['calls']) < allocation['max_feedback_calls']
        assert sum(r['charged_or_held_usd'] for r in ledger['calls'])+self.bound <= allocation['api_usd_cap']
        row={'cell':self.cell,'model':model,'request_digest':request_digest,
             'status':'IN_FLIGHT_OR_UNKNOWN','reserved_usd':self.bound,
             'charged_or_held_usd':self.bound,'started_ns':time.time_ns()}
        ledger['calls'].append(row);write(ledger_path,ledger)
        return len(ledger['calls'])-1
    def settle_call(self,index,usage):
        row=ledger['calls'][index]
        cost=(usage['input_tokens']*2+usage['output_tokens']*12)/1e6
        row.update(status='USAGE_SETTLED',usage=usage,charged_or_held_usd=cost,finished_ns=time.time_ns())
        write(ledger_path,ledger)

plan=[]
for state in report['states']:
    for arm in ('Summary','Mechanism'):
        cell=f"state_{state['source_state']}/{arm}"
        q=read(a.root/'payloads'/f"state_{state['source_state']}"/f'{arm}.json')
        assert hashlib.sha256(q['prompt'].encode()).hexdigest()==state['arms'][arm]['prompt_sha256']
        assert q['common_facts_sha256']==state['common_facts_sha256']
        home=a.root/'outputs'/cell
        home.mkdir(parents=True,exist_ok=True)
        schema=home/'SCHEMA.json'
        if schema.exists():assert read(schema)==q['response_schema']
        else:write(schema,q['response_schema'])
        bound=state['arms'][arm]['estimated_reference_usd_at_output_cap']+4096*2/1e6
        broker=LabApiCanaryBrokerV1(home/'broker',schema_path=schema,config_path=a.config,
            model=q['model'],max_total_tokens_per_call=148000,reasoning_effort=q['reasoning_effort'],
            wire_api='responses',request_budget=Budget(cell,bound))
        identity=allocation['identity']+':'+cell+':'+state['common_facts_sha256']
        plan.append({'cell':cell,'logical_call_id':identity,'request_bound_usd':bound})
        try:
            if not a.execute:continue
            if any(r['cell']==cell for r in ledger['calls']):
                # Completed or ambiguous requests are never automatically replayed.
                continue
            try:
                receipt=broker.call_with_session(logical_call_id=identity,
                    proposal_generation_session_id=identity,prompt=q['prompt'],
                    expected_proposal_count=1,max_output_tokens=q['max_output_tokens'])
                response=receipt.response
                if isinstance(response,str):response=json.loads(response)
                write(home/'FEEDBACK.json',{'cell':cell,'common_facts_sha256':q['common_facts_sha256'],
                    'response':response,'logical_call_id':identity})
                print(json.dumps({'cell':cell,'status':'COMPLETE'}),flush=True)
            except Exception as error:
                write(home/'FAILURE.json',{'cell':cell,'error_type':type(error).__name__,
                    'message':str(error),'no_automatic_retry':True})
                print(json.dumps({'cell':cell,'status':'FAILED','type':type(error).__name__}),flush=True)
        finally:broker.close()
write(a.root/'REQUEST_BINDINGS.json',{'identity':allocation['identity'],'plan':plan,
    'allocation_sha256':hashlib.sha256((a.root/'ALLOCATION.json').read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'new_fit':0,'gpu_seconds':0})
print(json.dumps({'stage':'execution' if a.execute else 'offline_preflight','cells':len(plan),
    'calls':len(ledger['calls']),'charged_or_held_usd':sum(r['charged_or_held_usd'] for r in ledger['calls'])}),flush=True)
