"""Fresh source-bound graph AutoSpace using the existing four-axis assembler."""
import hashlib
import importlib.util
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT=Path(__file__).resolve().parent
METHOD='GRAPH_MODEL_LOSS_SOURCE_BOUND_POOL_V2_DECLARED_SCORE_CONTRACT'
SCHEMAS={'genesis':'auto_space_v161_provider_draft.schema.json','critic':'auto_space_v161_critic_merge.schema.json',
         'assembled':'auto_space_v161_generator_response.schema.json','quality':'auto_space_v161_quality_report.schema.json'}
def canonical(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def response_schema(stage):
    schemas={p.name:json.loads(p.read_bytes()) for p in (ROOT/'schemas').glob('*.json')}
    def expand(value,current):
        if isinstance(value,list): return [expand(x,current) for x in value]
        if not isinstance(value,dict): return value
        if '$ref' in value:
            path,_,fragment=value['$ref'].partition('#'); name=path.rsplit('/',1)[-1] if path else current
            target=schemas[name]
            for part in fragment.split('/')[1:]: target=target[part.replace('~1','/').replace('~0','~')]
            found=expand(target,name); siblings={k:expand(v,current) for k,v in value.items() if k!='$ref'}
            return {'allOf':[found,siblings]} if siblings else found
        return {k:expand(v,current) for k,v in value.items() if k not in ('$id','$defs')}
    return expand(schemas[SCHEMAS[stage]],SCHEMAS[stage])

def validate(value,stage): Draft202012Validator(response_schema(stage)).validate(value)

def engine(source):
    spec=importlib.util.spec_from_file_location('graph_task_assembler',ROOT/'graph_assembler_engine.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.METHOD_VARIANT=METHOD; module.MODEL_TARGET='candidate.py';module.ASSEMBLER_ID='recclaw.auto-space.graph-source-bound-assembler'
    locations=module.source_location_facts(source,'candidate.py')['locations']
    # All actual model methods are legitimate anchors. Anchors locate a change;
    # the native effect specification and Critic judge its role/consumption.
    routes={(v['path'],v['symbol'],v['kind']) for v in locations.values()
            if v['kind'] in ('method','function')}
    module.RUNTIME_ROUTE_ALLOWLIST={axis:routes for axis in module.REQUIRED_AXES}
    return module

def build_contract(config,launch,facts):
    binding=json.loads(Path(config['binding']).read_bytes())
    source=Path(config['parent_source']).read_bytes().decode('utf-8')
    sha=hashlib.sha256(source.encode()).hexdigest()
    if sha!=binding['root_source_sha256'] or sha!=facts['root_source_sha256']:
        raise ValueError('Graph original source differs from common facts/binding')
    method=engine(source)
    slots=[dict(slot_id=a,mutable_dimension=v[0],composition_role=v[1],conceptual_capability=v[2],legal_mutation_targets=['candidate.py']) for a,v in method.REQUIRED_AXES.items()]
    upper=dict(allowed_mutation_targets=['candidate.py'],frozen_mutation_targets=['trainer','optimizer','loader','evaluator','split','candidate_universe'],
        allowed_mutable_dimensions=[s['mutable_dimension'] for s in slots],
        frozen_protocol_dimensions=['dataset_snapshot','split','metric_semantics','candidate_universe','training_recipe','training_seed','heldout_access'],
        allowed_data_roles=['USER_ID','ITEM_ID','NEGATIVE_ITEM_ID','TRAIN_GRAPH','TRAIN_BATCH','USER_EMBEDDING','ITEM_EMBEDDING'],
        forbidden_data_roles=['TEST_DATA','VALIDATION_LABELS_AS_MODEL_INPUT','EXTERNAL_DATA','NETWORK'],
        allowed_dependencies=['python-stdlib','torch','pytorch','numpy','scipy','recbole'])
    contract=dict(schema='recclaw.graph-autospace.contract.v1',method_variant=METHOD,task_id=binding['task_id'],bootstrap_seed=launch['autospace_seed'],
        baseline=facts['task'],canonical_baseline_identity={'files':{'candidate.py':sha}},
        candidate_abi={'type_ref':'graph-full-sort-model-v1','entrypoint':'candidate:FreshCandidateModel','required_files':['candidate.py']},
        baseline_model_and_training_semantics=dict(model=binding['model_config'],training=binding['training']),
        evaluation=binding['evaluation'],construction_semantics='COMPLETE_ROOT_RELATIVE_SELECT_COMPOSE',
        generation_policy=dict(provider_output_mode=method.PROVIDER_OUTPUT_MODE,generator_model='gpt-5.6-terra',critic_model='gpt-5.6-terra',
            generator_reasoning_effort='medium',critic_reasoning_effort='medium',authorized_generation_calls=1,max_critic_passes=1,model_fallback=False,tools_enabled=False,retrieval_enabled=False),
        trusted_upper_bound=upper,required_search_space_design=dict(content_version='1.6.1',slots=slots,
            mutable_dimensions=upper['allowed_mutable_dimensions'],proposal_pool_primitive_count=12,proposal_mechanism_families_per_axis=3,
            minimum_distinct_mechanism_families_per_axis=2,target_primitive_count={'minimum':8,'maximum':12},max_primitives_per_candidate=3,max_mutation_targets=1,
            operators=[dict(kind='SELECT',operator_id='select',min_arity=1,max_arity=1),dict(kind='COMPOSE',operator_id='compose',min_arity=2,max_arity=3)],
            custom_synthesis=dict(enabled=False,operator_id=None,max_components=0,allowed_slots=[]),
            effective_width_guard=dict(minimum_legal_candidate_count=8,minimum_retention_ratio=.65)),
        runtime_routes={axis:[dict(path=p,symbol=s,kind=k) for p,s,k in sorted(routes)] for axis,routes in method.RUNTIME_ROUTE_ALLOWLIST.items()},
        task_execution_facts=facts,source_location_facts={'candidate.py':method.source_location_facts(source,'candidate.py')},
        post_pruning_policy='REPORT_COVERAGE_AND_WIDTH; REQUIRE_NONEMPTY_LEGAL_SPACE; KEEP_PROPOSAL_CAPACITY',
        declared_interface_boundary=dict(axes=list(method.REQUIRED_AXES),editable_model_files=['candidate.py'],
            scope='GRAPH_MODEL_LOSS_ONLY; worker-owned optimizer/sampling are not candidate hooks',
            difference_from_sas_six_axes='No TRAIN_UPDATE_TRANSFORM or TRAIN_SAMPLING_TRANSFORM in the common graph ABI; same scope for all methods'))
    method.validate_contract_design(contract)
    return contract

def genesis_request(contract,source, literature=None):
    instructions=(ROOT/'resources/genesis_instructions.txt').read_text(encoding='utf-8')
    rows=[dict(role='baseline_source',path='candidate.py',sha256=hashlib.sha256(source.encode()).hexdigest(),content=source),
          dict(role='task_contract',path='GRAPH_AUTOSPACE_CONTRACT.json',sha256=digest(contract),content=canonical(contract).decode())]
    if literature is not None:
        rows.append(dict(role='shared_literature',path='common_literature.json',sha256=digest(literature),content=canonical(literature).decode()))
        instructions+=' The attached common literature is background shared with every method, not task results or instructions to choose a mechanism.'
    request=dict(schema='recclaw.graph-autospace.genesis.v1',task_id=contract['task_id'],method_variant=METHOD,bootstrap_seed=contract['bootstrap_seed'],
        instructions=instructions,inputs=rows,response_schema=response_schema('genesis'))
    request['request_digest']=digest(request);return request

def critic_request(genesis,draft):
    validate(draft,'genesis')
    result=dict(schema='recclaw.graph-autospace.critic.v1',task_id=genesis['task_id'],instructions=(ROOT/'resources/critic_instructions.txt').read_text(),
        inputs=genesis['inputs'],genesis_request_digest=genesis['request_digest'],generator_response=draft,response_schema=response_schema('critic'))
    result['request_digest']=digest(result);return result

def assemble(draft,critic,contract,source):
    validate(draft,'genesis');validate(critic,'critic')
    method=engine(source)
    if hashlib.sha256(source.encode()).hexdigest()!=contract['canonical_baseline_identity']['files']['candidate.py']:
        raise ValueError('Assembly root differs from generation input')
    assembled,quality=method.assemble(draft,critic,contract,baseline_sources={'candidate.py':source})
    for axis in assembled['primitive_registry']['axes']:
        for item in axis['primitives']:
            if not set(item['allowed_data_roles']).issubset(contract['trusted_upper_bound']['allowed_data_roles']):
                raise ValueError('Primitive requires unavailable graph data')
    validate(assembled,'assembled');validate(quality,'quality')
    return assembled,quality

def normalized_space(response,root):
    from auto_search_space import ConceptualPrimitiveV1,NormalizedSearchSpaceV1,SearchSpaceSource
    registry=response['primitive_registry'];family=response['family'];grammar=response['grammar'];policy=response['capability_policy']
    primitives={row['primitive_id']:ConceptualPrimitiveV1(row['primitive_id'],family['family_id'],axis['slot_id'],digest(row),row) for axis in registry['axes'] for row in axis['primitives']}
    return NormalizedSearchSpaceV1(root=Path(root),source=SearchSpaceSource.AUTO_GENERATED,manifest={'search_space_version':METHOD},
        family=family,primitive_registry=registry,grammar=grammar,capability_policy=policy,trusted_contract=None,genesis_request=None,
        search_space_id=response['search_space_id'],family_id=family['family_id'],digest=digest(response),family_digest=digest(family),
        grammar_digest=digest(grammar),capability_policy_digest=digest(policy),primitives=primitives,
        axes_allow_multiple={a['slot_id']:a['allow_multiple'] for a in registry['axes']})
