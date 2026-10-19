"""Single fresh Genesis/Critic, same MAIN ledger as native search, no redraw."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request
import graph_autospace as method

def freeze(path,value):
    encoded=json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=encoded.encode():raise ValueError('Frozen graph input/output changed: '+str(path))
    else:path.write_bytes(encoded.encode())

def prepare(inputs):
    launch,config=inputs['launch'],inputs['evaluator']
    if config['phase']=='BOUNDED_GRAPH_DEVELOPMENT':
        if (launch['autospace_seed'],launch['search_seed'],config['train_seed'])!=(94001,43001,51001):
            raise ValueError('Graph development seeds differ from prospective binding')
    elif launch['autospace_seed']!=91025+launch['repeat']:
        raise ValueError('Formal AutoSpace seed differs from MAIN repeat')
    contract=method.build_contract(config,launch,inputs['facts'])
    request=method.genesis_request(contract,inputs['parent'],inputs['corpus'])
    root=Path(launch['output'])/'autospace'
    return root,contract,request

def stage(name,request,root,settings,ledger,*,transport=None):
    from auto_search_space import extract_responses_api_scientific_response
    from ro_main_provider import MainROTransport
    from se_shared_api_ledger import _load_route_credentials
    payload=dict(model='gpt-5.6-terra',reasoning={'effort':'medium'},store=False,
        max_output_tokens=ledger.max_output_tokens,text={'format':{'type':'json_object'}},input=json.dumps(request,sort_keys=True,ensure_ascii=False))
    freeze(root/(name+'_REQUEST.json'),request);freeze(root/(name+'_TRANSPORT.json'),payload)
    raw=root/(name+'_RAW_RESPONSE.json')
    if not raw.exists():
        started=root/(name+'_STARTED.json')
        if started.exists():raise RuntimeError('Original generation interruption needs reconciliation; no new draw')
        routefile=settings.get('model_api_configs',{}).get('gpt-5.6-terra',settings['api_config'])
        route=_load_route_credentials(routefile)
        freeze(started,dict(request_digest=method.digest(payload),status='DISPATCH_INTENT'))
        http=Request(route['base_url'].rstrip('/')+'/responses',data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'Authorization':'Bearer '+route['api_key'],'Content-Type':'application/json'})
        with (transport or MainROTransport(settings,ledger)).urlopen(http,timeout=900) as response:
            data=response.read()
        with raw.open('xb') as stream:stream.write(data)
    body=json.loads(raw.read_bytes())
    if body.get('model') not in ('gpt-5.6-terra','openai/gpt-5.6-terra'):
        raise ValueError('Retain original output: unexpected generation model')
    result=extract_responses_api_scientific_response(body)
    freeze(root/(name+'_RESPONSE.json'),result);method.validate(result,name.lower())
    return result

def generate(inputs,*,stage_call=stage):
    from openevolve_provider import _settings_and_ledger
    root,contract,genesis=prepare(inputs)
    freeze(root/'GRAPH_AUTOSPACE_CONTRACT.json',contract)
    freeze(root/'INPUT_IDENTITY.json',dict(input_hashes=inputs['hashes'],method=method.METHOD))
    settings,ledger=_settings_and_ledger(inputs['evaluator']['provider_config'])
    draft=stage_call('GENESIS',genesis,root,settings,ledger)
    critic=stage_call('CRITIC',method.critic_request(genesis,draft),root,settings,ledger)
    assembled,quality=method.assemble(draft,critic,contract,inputs['parent'])
    freeze(root/'ASSEMBLED_SPACE.json',assembled);freeze(root/'QUALITY_REPORT.json',quality)
    return method.normalized_space(assembled,root)
