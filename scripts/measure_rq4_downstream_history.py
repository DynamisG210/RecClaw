"""Read original native downstream broker receipts to size the next complete block."""
import argparse
import json
from pathlib import Path
import sqlite3

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
states=[(2,'rq4_action_space_two_state_validation_v1_20261009/state_02'),
 (3,'rq4_action_space_two_state_validation_v1_20261009/state_03'),
 (18,'rq4_feedback_action_space_v1_20261008'),(6,'rq4_additional_validation_20261014/state_06'),
 (10,'rq4_additional_validation_20261014/state_10'),(8,'rq4_differential_feedback_v1_20261014/state_08'),
 (12,'rq4_differential_feedback_v1_20261014/state_12')]
rows=[]
for state,sub in states:
 for arm in ('Raw','Summary','Mechanism'):
  home=a.native_root/sub/arm/'search/campaign/provider_calls'
  for stage in ('proposals','implementations'):
   for db in sorted((home/stage).rglob('broker.sqlite3')):
    release=json.loads((db.parent/'LAB_API_BROKER_RELEASE_V1.json').read_text())
    model=release.get('model')
    with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as connection:
     connection.row_factory=sqlite3.Row
     records=connection.execute('SELECT logical_call_id,request_digest,status,input_tokens,output_tokens,error_type FROM calls').fetchall()
    for row in records:
     record=dict(row)
     rates={'gpt-5.6-luna':(0.2,1.2),'gpt-5.6-terra':(2,12)}.get(model)
     measured=(record['input_tokens']*rates[0]+record['output_tokens']*rates[1])/1e6 if rates else None
     rows.append(dict(state=state,arm=arm,stage=stage,model=model,db=str(db),
       reference_usd_from_recorded_tokens=measured,usage_unknown=(record['status']!='SUCCESS' and record['input_tokens']==record['output_tokens']==0),**record))
result={'scope':'Historical downstream physical broker calls only; excludes feedback, GPU and inherited reference charges. Unknown failures remain unknown.',
 'calls':rows,'physical_records':len(rows),'successful_records':sum(r['status']=='SUCCESS' for r in rows),
 'unknown_usage_records':sum(r['usage_unknown'] for r in rows),
 'known_reference_usd':sum(r['reference_usd_from_recorded_tokens'] or 0 for r in rows),
 'by_cell':[{'state':s,'arm':arm,'records':len([r for r in rows if r['state']==s and r['arm']==arm]),
 'known_reference_usd':sum(r['reference_usd_from_recorded_tokens'] or 0 for r in rows if r['state']==s and r['arm']==arm)} for s,_ in states for arm in ('Raw','Summary','Mechanism')],
 'not_new_allocation':True,'new_api_calls':0,'new_fits':0}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('physical_records','successful_records','unknown_usage_records','known_reference_usd')}))
