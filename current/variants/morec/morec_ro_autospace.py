"""Fresh original six-axis generation, charged once to the same MAIN ledger."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request
from morec_formal_identity import validate_formal_launch

from morec_axis_contract import (
    AUTHORING_GUIDANCE, AUTHORING_VERSION, bind_generator_schema,
    validate_generator_bindings,
)


def freeze(path, value):
    data=json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=data: raise ValueError(f'Original input changed: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(data)


def prepare(launch, config):
    validate_formal_launch(launch, config)
    import auto_search_space as native
    contract_path=Path(launch['autospace_contract'])
    contract=json.loads(contract_path.read_text(encoding='utf-8'))
    native.validate_trusted_contract(contract)
    if contract['task_id']!='MOREC_AMAZON_CDS' or contract['bootstrap_seed']!=launch['autospace_seed']:
        raise ValueError('Fresh MoRec task/seed differs from launch')
    if contract['repeat_policy']['search_seeds'] != [launch['search_seed']]:
        raise ValueError('Fresh MoRec search seed differs from launch')
    parent=json.loads(Path(config['parent_source']).read_text(encoding='utf-8'))
    inputs=[]
    for item in contract['genesis_input_policy']['expected_inputs']:
        role,name=item['role'],item['path']
        if role=='baseline_source': content=parent[name]
        elif role=='task_contract' and name==contract_path.name: content=contract_path.read_bytes().decode('utf-8')
        else: raise ValueError('Unexpected AutoSpace input; do not import old search material')
        inputs.append(dict(role=role,path=name,content=content,sha256=hashlib.sha256(content.encode()).hexdigest()))
    if {x['path'] for x in inputs if x['role']=='baseline_source'} != set(parent):
        raise ValueError('Generator must see the complete frozen five-file parent')
    policy=dict(contract['generation_policy']);policy.pop('authorized_generation_calls',None)
    request=dict(schema='recclaw.auto-space.genesis-request.v1',task_id=contract['task_id'],
        bootstrap_seed=launch['autospace_seed'],generation_policy=policy,
        instructions=native.genesis_provider_instructions(contract),inputs=inputs,
        access_audit=dict(allowlist_mode='TRUSTED_CONTRACT_EXACT_SET',input_file_count=len(inputs),
            literature_accessed=False,test_artifacts_accessed=False,prior_candidates_accessed=False,
            prior_results_accessed=False,historical_search_space_semantics_accessed=False,
            tools_enabled=False,retrieval_enabled=False))
    request['request_digest']=native.canonical_sha256(request)
    native.validate_embedded_genesis_request(request)
    root=Path(launch['output'])/'autospace'
    freeze(root/'GENESIS_REQUEST.json',request)
    freeze(root/'INPUT_IDENTITY.json',dict(training_seed=config['train_seed'],search_seed=launch['search_seed'],
        autospace_seed=launch['autospace_seed'],request_digest=request['request_digest'],
        budget_scope='FRESH_GENERATION_AND_SEARCH_SAME_MAIN_LEDGER'))
    return root,contract,request


def generate(launch,config):
    root=Path(launch['output'])/'autospace'
    # A new authoring version never consumes, edits or redraws an old paid run.
    if not (root/'AUTHORING_PROFILE.json').exists() and (
        (root/'package').exists() or any(
            (root/f'{role}_{suffix}.json').exists()
            for role in ('GENERATOR','CRITIC')
            for suffix in ('REQUEST','TRANSPORT','STARTED','RAW','RESPONSE')
        )
    ):
        raise ValueError('Existing AutoSpace belongs to its original authoring version; no replay or redraw')
    import auto_search_space as native
    import auto_space_v161
    import morec_axis_contract
    from auto_space_v161 import assemble, QUALITY_REPORT_FILE
    from openevolve_provider import _settings_and_ledger
    from ro_main_provider import MainROTransport
    from se_shared_api_ledger import _load_route_credentials
    root,contract,request=prepare(launch,config)
    schema_name=native.scientific_response_schema_name(contract,call_role='GENERATOR')
    generator_schema=bind_generator_schema(native.resolved_schema(schema_name),contract)
    freeze(root/'AUTHORING_PROFILE.json',dict(
        version=AUTHORING_VERSION,
        bound_schema_sha256=native.canonical_sha256(generator_schema),
        guidance_sha256=hashlib.sha256(AUTHORING_GUIDANCE.encode()).hexdigest(),
        source_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__,morec_axis_contract.__file__,native.__file__,auto_space_v161.__file__)},
        original_contract_sha256=native.canonical_sha256(contract),
        retry_policy='NO_ADDITIONAL_GENERATION_OR_CRITIC_CALLS',
    ))
    freeze(root/'GENERATOR_BOUND_SCHEMA.json',generator_schema)
    settings,ledger=_settings_and_ledger(config['provider_config'])
    transport=MainROTransport(settings,ledger)
    route_path=settings.get('model_api_configs',{}).get('gpt-5.6-terra',settings['api_config'])

    def stage(role,content):
        schema_name=native.scientific_response_schema_name(contract,call_role=role)
        schema=generator_schema if role=='GENERATOR' else native.resolved_schema(schema_name)
        payload=dict(model='gpt-5.6-terra',reasoning={'effort':'medium'},store=False,
            max_output_tokens=ledger.max_output_tokens,text={'format':{'type':'json_object'}},
            input=json.dumps(content,sort_keys=True,ensure_ascii=False)+
            ('\n\n'+AUTHORING_GUIDANCE if role=='GENERATOR' else '')+
            '\n\nRequired response JSON Schema (validated locally):\n'+
            json.dumps(schema,ensure_ascii=False,separators=(',',':')))
        freeze(root/f'{role}_REQUEST.json',content);freeze(root/f'{role}_TRANSPORT.json',payload)
        raw=root/f'{role}_RAW.json'
        if not raw.exists():
            intent=root/f'{role}_STARTED.json'
            if intent.exists(): raise RuntimeError('Resume the original interrupted request, do not redraw')
            route=_load_route_credentials(route_path)
            freeze(intent,dict(payload_sha256=native.canonical_sha256(payload)))
            http=Request(route['base_url'].rstrip('/')+'/responses',data=json.dumps(payload,ensure_ascii=False).encode(),
                headers={'Authorization':'Bearer '+route['api_key'],'Content-Type':'application/json'})
            with transport.urlopen(http,timeout=900) as response:
                with raw.open('xb') as output: output.write(response.read())
        envelope=json.loads(raw.read_text(encoding='utf-8'))
        scientific=native.extract_responses_api_scientific_response(envelope)
        freeze(root/f'{role}_RESPONSE.json',scientific)
        native.validate_schema(scientific,schema_name)
        return scientific

    draft=stage('GENERATOR',request)
    try:
        validate_generator_bindings(draft,generator_schema)
    except ValueError as error:
        freeze(root/'GENERATOR_CONTRACT_CHECK.json',dict(status='REJECTED',reason=str(error),
            version=AUTHORING_VERSION,critic_called=False,output_rewritten=False))
        raise
    freeze(root/'GENERATOR_CONTRACT_CHECK.json',dict(status='PASS',version=AUTHORING_VERSION,
        scope='AXIS_TARGETS_AND_DECLARED_ROUTE_ANCHORS_ONLY_NOT_EFFECTIVENESS',output_rewritten=False))
    critic=stage('CRITIC',dict(genesis_request=request,draft=draft,
        instruction=native.V161_CRITIC_INSTRUCTIONS+'\n\n'+AUTHORING_GUIDANCE))
    assembled,quality=assemble(draft,critic,contract,baseline_sources={i['path']:i['content'] for i in request['inputs'] if i['role']=='baseline_source'})
    package=root/'package';package.mkdir(exist_ok=True)
    freeze(package/'AUTO_SPACE_GENESIS_REQUEST.json',request)
    contract_file=package/native.TRUSTED_CONTRACT_FILE
    raw_contract=native.canonical_bytes(contract)
    if contract_file.exists():
        if contract_file.read_bytes()!=raw_contract: raise ValueError('Original trusted contract changed')
    else: contract_file.write_bytes(raw_contract)
    manifest=dict(record_type='search_space_manifest',schema='recclaw.auto-space.manifest.v1',
        source='AUTO_GENERATED',generator_version='recclaw.auto-space-genesis.v4',
        search_space_id=assembled['search_space_id'],search_space_version='v1.6.1',
        family_id=assembled['family']['family_id'],task_id=request['task_id'],bootstrap_seed=request['bootstrap_seed'],
        genesis_request_digest=request['request_digest'],trusted_contract_sha256=hashlib.sha256(raw_contract).hexdigest(),
        conceptual_only=True,executable_registry_initial_count=0)
    for name,value in zip(native.CORE_FILES,[manifest,assembled['family'],assembled['primitive_registry'],assembled['grammar'],assembled['capability_policy']]):
        freeze(package/name,value)
    freeze(package/QUALITY_REPORT_FILE,quality)
    space=native.load_search_space_package(package,expected_source=native.SearchSpaceSource.AUTO_GENERATED,require_frozen=False)
    freeze(root/'MAIN_SPACE_IDENTITY.json',dict(search_space_digest=space.digest,
        core_file_sha256={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in native.CORE_FILES},
        search_seed=launch['search_seed'],training_seed=config['train_seed'],fresh_generation=True))
    return space
