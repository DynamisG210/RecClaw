"""Direct-human endpoint2-only routing, installed only in new process entries."""
from pathlib import Path
import json,time,os
def install(provider_path):
 import openevolve_provider,se_shared_api_ledger
 p=Path(provider_path);settings=json.loads(p.read_bytes());target=settings['api_config']
 assert settings['endpoint'] == 'endpoint2'
 output=Path(settings['resource_root'])
 old_settings=openevolve_provider._settings_and_ledger
 old_load=se_shared_api_ledger._load_route_credentials
 def record(kind,requested):
  with (output/'ENDPOINT2_ONLY_RUNTIME.jsonl').open('a') as f:
   f.write(json.dumps(dict(time_ns=time.time_ns(),pid=os.getpid(),kind=kind,requested_config=str(requested),selected_config=target,endpoint=2,authority='DIRECT_HUMAN_ALL_ENDPOINT2_ONLY_20261013'))+'\n')
 def routed_settings(path,*args,**kwargs):
  value,ledger=old_settings(path,*args,**kwargs)
  value=dict(value);value['api_config']=target
  value['model_api_configs']={k:target for k in value.get('model_api_configs',{})}
  if 'embedding_api_config' in value:value['embedding_api_config']=target
  record('PROVIDER_SETTINGS',path)
  return value,ledger
 def load(path,*args,**kwargs):
  record('ACTUAL_CREDENTIAL_ROUTE_LOAD',path)
  return old_load(target,*args,**kwargs)
 openevolve_provider._settings_and_ledger=routed_settings
 se_shared_api_ledger._load_route_credentials=load
