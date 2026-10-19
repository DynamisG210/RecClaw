"""Private whole-program proposal binding; native selection/materialization/qualification."""
import ast
import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from open_mechanism import OpenMechanismAdapter, MARKER, VERSION, FIELDS
from research_actions import ResearchActionAdapter, source_of, MARKER as ACTION_MARKER
from graph_autospace import digest

HANDOFF_DESCRIPTION = 'Selected Research program; deterministic identity metadata only; no model call.'
PARENT_REF_VERSION = 'graph-own-run-parent-ref-v1'
SCORE_TARGET_VERSION = 'graph-score-targets-v1'

CONSTRUCTION = ('Submit complete candidate.py as the sole executable definition. '
    'Choose parent_ref from the supplied discrete own-run registry. The controller binds source identity. '
    'You may add or delete helpers and methods and change multiple computations. '
    'No second model rewrites the selected program. Scientific explanations describe and predict '
    'the submitted code, not another algorithm. Worker, data, score and qualification ABI remain fixed.')

def sha(source):
    return hashlib.sha256(source.encode('utf8')).hexdigest()

def program_schema():
    text = {'type':'string','minLength':1}
    props = dict(source=text,
        score_effect={'type':'object','additionalProperties':False,'properties':{
            'parameterization':{'enum':['UNCHANGED','FIXED','LEARNED']},
            'phase':{'enum':['TRAIN_ONLY','INFERENCE_ONLY','TRAIN_AND_INFERENCE']}},
            'required':['parameterization','phase']},
        score_parameter_targets={'type':'array','items':{
            'type':'object','additionalProperties':False,
            'properties':{'kind':{'enum':['PARAMETER','MODULE']},'path':text},
            'required':['kind','path']},
            'description':'PARAMETER is an exact leaf path; MODULE explicitly declares every recursive parameter below that exact module/container. No wildcard or inferred correction; every resolved leaf must satisfy the declared connection.'},
        recommendation_loss_index={'type':['integer','null'],'minimum':0})
    return dict(type='object',additionalProperties=False,properties=props,required=list(props))

def program_projection(context):
    result=copy.deepcopy(context)
    task=result['knowledge_base']['search_space']
    registry=task['parent_ref_registry']
    task.pop('conceptual_search_space',None)
    task.pop('source_region_contract',None)
    task['construction']=CONSTRUCTION
    task['candidate_mechanism_contract'].update(operator='program',definition_authority='program.source')
    task['parent_ref_registry'] = registry
    task['research_actions']['parents'] = [parent for parent in task.get('research_actions',{}).get('parents',[])
        if parent['candidate_id'] in registry['parents']]
    for parent in task.get('research_actions',{}).get('parents',[]):
        parent.pop('source_regions',None)
    if 'root_relative_comparison' in task:
        old=task['root_relative_comparison']['construction_semantics']
        task['root_relative_comparison']['construction_semantics']=CONSTRUCTION+' '+old[old.index('Native comparator_delta'):]
    return result

class ProgramAdapter(ResearchActionAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Native preparation temporarily switches self.parent/parent_id.
        # The sealed run registry always belongs to the original root.
        self._registry_root_id = self.parent_id
        self._registry_root_sha256 = sha(self.parent)

    def _available_parent_refs(self):
        parents = {self._registry_root_id: dict(source_sha256=self._registry_root_sha256, evidence_digest=None)}
        for candidate_id, row in sorted(self._parents().items()):
            if candidate_id in parents:
                raise ValueError('Executed candidate cannot shadow original root')
            parents[candidate_id] = dict(source_sha256=sha(source_of(row)), evidence_digest=digest(row))
        return parents

    def _registry_path(self, context_ref):
        return Path(self.config['output'])/'parent_ref_contexts'/(digest(context_ref)+'.json')

    def _registry(self, context_ref, context_digest, protocol_digest, *, create=False, allowed_parent_refs=None):
        path = self._registry_path(context_ref)
        identity = dict(version=PARENT_REF_VERSION, context_ref=context_ref,
            context_digest=context_digest, protocol_digest=protocol_digest,
            original_root_id=self._registry_root_id, original_root_sha256=self._registry_root_sha256,
            run_identity_digest=digest(self.config['run_identity']),
            binding_sha256=hashlib.sha256(Path(self.config['binding']).read_bytes()).hexdigest())
        available = self._available_parent_refs()
        if not path.exists():
            if not create:
                raise ValueError('No sealed parent_ref registry for this provider context')
            parents=available if allowed_parent_refs is None else {key:available[key] for key in allowed_parent_refs}
            value = dict(identity, parents=parents)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf8', newline='\n') as stream:
                json.dump(value, stream, sort_keys=True, ensure_ascii=False)
        else:
            value = json.loads(path.read_bytes())
        if any(value.get(key) != expected for key, expected in identity.items()):
            raise ValueError('Parent registry context/run/binding identity drift')
        if allowed_parent_refs is not None and set(value['parents']) != set(allowed_parent_refs):
            raise ValueError('Sealed parent whitelist differs from construction route')
        if any(available.get(key) != entry for key, entry in value['parents'].items()):
            raise ValueError('Sealed parent source/evidence is no longer the same own-run parent')
        return value

    def proposal_schema(self):
        return self._proposal_schema_for(self._available_parent_refs())

    def _proposal_schema_for(self, parents):
        value=super().proposal_schema()
        item=value['properties']['proposals']['items'];props=item['properties']
        for key in (*FIELDS,'open_mechanism'):
            props.pop(key);item['required'].remove(key)
        props['program']=program_schema();item['required'].append('program')
        props.pop('closest_parent');item['required'].remove('closest_parent')
        props['parent_ref']={'type':'string','enum':list(parents)}
        item['required'].append('parent_ref')
        action=props['research_action']['anyOf'][1]
        action['properties'].pop('parent_source_sha256')
        action['required'].remove('parent_source_sha256')
        props['implementation_requirements']={'type':'array','items':{'type':'string'},'maxItems':0}
        return value

    def build_provider_research_producer(self,**kwargs):
        wrapped=super().build_provider_research_producer(**kwargs)
        producer=wrapped._producer
        producer.proposal_template=(Path(__file__).parent/'program_producer.txt').read_text(encoding='utf8')
        original=producer._call
        def call(role,view,**options):
            view=copy.deepcopy(view)
            task=view['knowledge_base']['search_space']
            route=task.get('branch_allocation')
            allowed=None if route is None else [route['parent_id']]
            registry=self._registry(view['context_ref'],view['context_digest'],view['protocol_digest'],create=True,allowed_parent_refs=allowed)
            task['parent_ref_registry']=registry
            # Native schemas are immutable per role/path. A changed round enum
            # needs its own schema/cache namespace; replay reuses this exact one.
            call_root=producer.call_root/SCORE_TARGET_VERSION/'parent-ref-contexts'/digest(registry)
            with patch.object(producer,'proposal_schema',self._proposal_schema_for(registry['parents'])),patch.object(producer,'call_root',call_root):
                return original(role,program_projection(view),**options)
        producer._call=call
        from discovery_contract import OLD_DEFINITION_NOTICE
        provider=producer.provider_call
        def provider_call(**options):
            prompt=options['prompt']

            replacement=('\n\nThe complete program.source is the sole executable definition. '
                'Use explanation fields for the hypothesis, parent-relative mechanism difference and '
                'falsifiable predictions. implementation_requirements is empty. Do not supply '
                'a second algorithm in prose. A matched control is a separate candidate opportunity.')
            if prompt.count(OLD_DEFINITION_NOTICE)==1 and replacement not in prompt:
                prompt=prompt.replace(OLD_DEFINITION_NOTICE,replacement,1)
            elif OLD_DEFINITION_NOTICE in prompt or prompt.count(replacement)!=1:
                raise ValueError('Program definition authority changed')
            return provider(**{**options,'prompt':prompt})
        producer.provider_call=provider_call
        return wrapped

    def _parent_source(self,parent_id):
        return self.parent if parent_id==self.parent_id else source_of(self._parent(parent_id))

    def resolve_confirmation(self,kind,primary_binding,context):
        if context.get('phase')!='PROJECT_PROVIDER_PROPOSAL':
            return super().resolve_confirmation(kind,primary_binding,context)
        from jsonschema import Draft202012Validator,ValidationError
        from recclaw_core.research_line.producers import _project_result
        from recclaw_core.research_line.interfaces import ProducerOutcome,ResearchTaskQueueV2
        from recclaw_core.research_line.search_space_adapter import ConfirmationResolutionV1 as Resolution,ConfirmationResolutionKindV1 as Kind
        draft=copy.deepcopy(dict(primary_binding))
        try:
            rc=context['research_context']
            registry=self._registry(rc.context_ref,rc.digest,rc.protocol_digest)
            wire=copy.deepcopy(draft);ex=wire['execution_contract']
            if 'config' in ex:ex['config_json']=json.dumps(ex.pop('config'),sort_keys=True)
            Draft202012Validator(self._proposal_schema_for(registry['parents'])).validate({'schema':'recclaw.research-line.fresh-open-spec-proposal-response.v1','proposals':[wire]})
            parent_id=draft.pop('parent_ref');draft['closest_parent']=parent_id
            definition=draft.pop('program');action=draft.pop('research_action')
            definition['parent_source_sha256']=registry['parents'][parent_id]['source_sha256']
            definition['parent_ref_binding']=dict(version=PARENT_REF_VERSION,parent_ref=parent_id,
                registry_digest=digest(registry),context_ref=rc.context_ref,context_digest=rc.digest,
                protocol_digest=rc.protocol_digest)
            if action is not None:
                action['parent_source_sha256']=definition['parent_source_sha256']
            if sha(self._parent_source(parent_id))!=definition['parent_source_sha256']:
                raise ValueError('Program parent source differs')
            if parent_id==self.parent_id:
                self._validate_original_root_action(action)
            else:
                if action is None:raise ValueError('Executed-parent program requires explicit action')
                parent=self._parent(parent_id)
                if action['parent_source_sha256']!=definition['parent_source_sha256']:raise ValueError('Action parent source differs')
                if action['operation']!='DIRECT_IMPROVEMENT' and len(action['competing_explanations'])<2:raise ValueError('Control requires competing explanations')
                if action['task_ref'] is not None:
                    memory=context['research_context'].scientific_memory
                    global_memory=memory.get('global_memory',memory.get('global',memory))
                    task=ResearchTaskQueueV2.from_dict(global_memory.get('task_queue')).get(action['task_ref'])
                    if task is None or task.candidate_id!=parent_id or task.operation.value!=action['operation'] or task.status.value not in ('PENDING','ACTIVE'):raise ValueError('Action must reference an open own-parent task')
                action.update(parent_candidate_id=parent_id,parent_binding_sha256=parent['worker_configuration']['binding_sha256'],parent_evidence_digest=digest(parent),run_identity_digest=digest(self.config['run_identity']),protocol_digest=context['research_context'].protocol_digest)
                draft['implementation_requirements'].append(ACTION_MARKER+json.dumps(action,sort_keys=True,separators=(',',':')))
            definition['research_source_sha256']=sha(definition['source'])
            draft['implementation_requirements'].append(MARKER+json.dumps(definition,sort_keys=True,separators=(',',':'),ensure_ascii=False))
            execution=draft['execution_contract']
            if execution.get('config',{})!={} or execution.get('config_json','{}')!='{}':raise ValueError('External model configuration stays fixed')
            draft['execution_contract']={**{k:execution[k] for k in ('capability_family','model','base_model_config')},'config':{}}
            if draft.get('idea_mode')=='DIAGNOSIS_DRIVEN' and draft.get('observed_failure_mode') in (None,'NOT_OBSERVED'):draft.update(idea_mode='FRONTIER_HYPOTHESIS',observed_failure_mode=None)
            spec,facts,source,program,companion=_project_result(context['producer_role'],draft,bindings=context['producer_bindings'],frozen_profile_ref=context['frozen_profile_ref'])
            spec=replace(spec,execution_contract={**spec.execution_contract,'config':{}})
            bound=self._bind_definition(spec,definition);rc=context['research_context']
            outcome=ProducerOutcome(producer_role=context['producer_role'],context_ref=rc.context_ref,context_digest=rc.digest,spec=spec,resolution_facts=facts,source_proposal=source,source_mechanism_program=program,implementation_companion=companion,provenance=dict(status='PRODUCED',producer_role=context['producer_role'],context_ref=rc.context_ref,context_digest=rc.digest,spec_id=spec.spec_id,spec_digest=spec.digest,candidate_mechanism_binding=bound))
            return Resolution(Kind.EXACT_BINDING,binding={'producer_outcome':outcome})
        except (ValueError,KeyError,TypeError,ValidationError) as error:
            return self._unsupported('PROGRAM_PROJECTION:'+str(error))

    def _bind_definition(self,spec,definition):
        parent_ref=definition['parent_ref_binding']
        registry=self._registry(spec.context_ref,spec.context_digest,spec.protocol_digest)
        if (parent_ref != dict(version=PARENT_REF_VERSION,parent_ref=spec.closest_parent,
                registry_digest=digest(registry),context_ref=spec.context_ref,
                context_digest=spec.context_digest,protocol_digest=spec.protocol_digest)
                or registry['parents'][spec.closest_parent]['source_sha256']!=definition['parent_source_sha256']):
            raise ValueError('Program parent_ref recovery identity drift')
        if sha(definition['source'])!=definition['research_source_sha256'] or sha(self._parent_source(spec.closest_parent))!=definition['parent_source_sha256']:
            raise ValueError('Research program/source parent identity drift')
        with self._using_parent(spec.closest_parent):
            bound=OpenMechanismAdapter._bind_definition(self,spec,definition)
        actions=[v[len(ACTION_MARKER):] for v in spec.implementation_requirements if v.startswith(ACTION_MARKER)]
        if actions:
            action=json.loads(actions[0]);row=self._parent(spec.closest_parent)
            if len(actions)!=1 or digest(row)!=action['parent_evidence_digest'] or digest(self.config['run_identity'])!=action['run_identity_digest'] or spec.protocol_digest!=action['protocol_digest']:raise ValueError('Own-parent evidence or run drift')
            bound['research_action']=action
        return bound

    def _prepare_native_innovation(self,outcome):
        with self._using_parent(outcome.spec.closest_parent):
            return OpenMechanismAdapter._prepare_native_innovation(self,outcome)

    def _open_binding(self,program):
        bound=program['candidate_mechanism_binding']
        ref=bound['definition']['parent_ref_binding']
        registry=self._registry(ref['context_ref'],ref['context_digest'],ref['protocol_digest'])
        if (ref['version']!=PARENT_REF_VERSION or ref['parent_ref']!=bound['parent_id']
                or ref['registry_digest']!=digest(registry)
                or registry['parents'][bound['parent_id']]['source_sha256']!=bound['definition']['parent_source_sha256']):
            raise ValueError('Materialized parent_ref identity drift')
        if sha(self._parent_source(bound['parent_id']))!=bound['definition']['parent_source_sha256']:raise ValueError('Parent source drift')
        with self._using_parent(bound['parent_id']):
            return OpenMechanismAdapter._open_binding(self,program)

    def adapt_implementation_request(self,request,*,mechanism_program):
        bound=self._open_binding(mechanism_program)
        return dict(request,candidate_mechanism_recipe=bound,graph_score_contract_request=self._mechanism_score_request(mechanism_program))

    @staticmethod
    def source_response(request):
        bound=request['candidate_mechanism_recipe'];definition=bound['definition'];source=definition['source']
        # Parse only to reject a pre-existing reserved identity declaration; never rewrite AST/code.
        try:tree=ast.parse(source)
        except SyntaxError:tree=None  # Original syntax is passed unchanged to native failure classification.
        if tree is not None and any(isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(t,ast.Name) and t.id=='GRAPH_SCORE_CONTRACT' for t in (n.targets if isinstance(n,ast.Assign) else [n.target])) for n in tree.body):
            raise ValueError('Research source must omit reserved GRAPH_SCORE_CONTRACT metadata')
        required=request['graph_score_contract_request'];entries=[]
        for expected in required['entries']:
            entries.append(dict(primitive_id=expected['primitive_id'],parameter_targets=definition['score_parameter_targets'],recommendation_loss_index=definition['recommendation_loss_index']))
        declaration=dict(selected_spec_digest=required['selected_spec_digest'],selected_primitive_ids=[],entries=entries)
        emitted=source+'\nGRAPH_SCORE_CONTRACT = '+repr(declaration)+'\n'
        return dict(entrypoint='candidate:FreshCandidateModel',implementation_summary='Research-authored source; deterministic identity metadata only',files=[dict(path='candidate.py',content=emitted)])

    def build_provider_implementer(self,**kwargs):
        # This is a local native interface adapter, not an HTTP/model gateway.
        adapter=self
        class SourceHandoff:
            def implementation_template_for_search_space(self,search_space_id):
                return HANDOFF_DESCRIPTION
            def __call__(self,request):
                return adapter.source_response(request)
        return SourceHandoff()

    def implementation_policy(self,policy):
        return replace(super().implementation_policy(policy),prompt_digest=digest(HANDOFF_DESCRIPTION))

    def _qualify_open(self,package,research_spec,candidate_root,mechanism_program):
        request=self.adapt_implementation_request({},mechanism_program=mechanism_program)
        expected=self.source_response(request)['files'][0]['content'].encode('utf8')
        actual=(Path(candidate_root)/'candidate.py').read_bytes()
        if actual!=expected:raise ValueError('Research source computation changed during materialization')
        return OpenMechanismAdapter._qualify_open(self,package,research_spec,candidate_root,mechanism_program)

@contextmanager
def program_scope():
    import research_actions
    from recclaw_core.research_line import runtime
    # Native runtime otherwise rewrites imports before materialization; keep exact authored bytes.
    with patch.object(research_actions,'ResearchActionAdapter',ProgramAdapter),patch.object(runtime,'normalize_candidate_runtime_imports',lambda files:files),patch.object(runtime,'MAX_REPAIR_TURNS',0):
        yield
