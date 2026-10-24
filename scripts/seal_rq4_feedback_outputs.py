"""Seal completed feedback bytes against native broker receipts; price unsent role payloads."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lane',type=Path,required=True);a=p.parse_args();root=a.lane;froot=root/'feedback_repeat'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
ledger=read(froot/'FINITE_API_LEDGER.json');rows=[]
assert len(ledger['calls'])==14
for call in ledger['calls']:
    assert call['status']=='USAGE_SETTLED'
    home=froot/'outputs'/call['cell'];feedback=read(home/'FEEDBACK.json')
    with sqlite3.connect((home/'broker/broker.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        receipt=db.execute('select status,response_json,request_digest from calls where logical_call_id=?',(feedback['logical_call_id'],)).fetchone()
    assert receipt and receipt[0]=='SUCCESS'
    assert json.loads(receipt[1])==feedback['response']
    assert receipt[2]==call['request_digest']
    rows.append(dict(cell=call['cell'],logical_call_id=feedback['logical_call_id'],reference_usd=call['charged_or_held_usd'],feedback_sha256=hashlib.sha256((home/'FEEDBACK.json').read_bytes()).hexdigest()))
files={str(p.relative_to(froot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((froot/'outputs').rglob('*')) if p.is_file()}
seal=dict(status='COMPLETE_NATIVE_RECEIPTS_MATCHED',calls=rows,reference_usd=sum(r['reference_usd'] for r in rows),unknown_pending=0,new_candidate_fits=0,files_sha256=files)
write(froot/'SEALED_OUTPUTS.json',seal)
write(froot/'LIVE_STATUS.json',dict(status=seal['status'],completed_calls=14,reference_usd=seal['reference_usd'],unknown_pending=0,new_candidate_fits=0,seal='SEALED_OUTPUTS.json'))
priced=[]
for arm in ['Raw','Summary','Mechanism']:
    for role,r in read(root/'state2_native_check'/arm/'UNSENT_REQUESTS.json').items():
        ip,op=(.2,1.2) if 'luna' in r['model'] else (2,12)
        priced.append(dict(arm=arm,role=role,model=r['model'],guarded_input=r['guarded_input'],output_cap=r['output'],max_reference_usd=(r['guarded_input']*ip+r['output']*op)/1e6,wire_sha256=r['wire_sha256']))
write(root/'STATE2_UNSENT_ROLE_COST.json',dict(new_api_calls=0,rows=priced,total_reference_upper=sum(r['max_reference_usd'] for r in priced),implementation_excluded=True,route='Historical endpoint2 in offline capture; endpoint1 binding required before paid execution'))
print(json.dumps(dict(sealed_calls=len(rows),sealed_files=len(files),reference_usd=seal['reference_usd'],native_unsent_roles=len(priced),role_reference_upper=sum(r['max_reference_usd'] for r in priced))))
