"""Graph-only evidence-bound local research actions. No new scheduler or scoring rule."""
import ast,copy,hashlib,json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from open_mechanism import OpenMechanismAdapter
from graph_autospace import digest
from graph_score_connection import bind_source_contract
from recclaw_core.research_line.interfaces import ResearchTaskRecordV2,ResearchTaskQueueV2
from recclaw_core.research_line.search_space_adapter import ConfirmationResolutionV1 as Resolution,ConfirmationResolutionKindV1 as Kind
VERSION='graph-evidence-bound-research-action-v1'
MARKER='GRAPH_RESEARCH_ACTION_JSON:'

def payload(outcome):
    if outcome is None or outcome.spec is None:return None
    values=[s[len(MARKER):] for s in outcome.spec.implementation_requirements if s.startswith(MARKER)]
    if len(values)>1:raise ValueError('Ambiguous research action')
    return json.loads(values[0]) if values else None

def source_of(execution):
    files=execution.get('implementation',{}).get('files',[])
    sources=[x['content'] for x in files if x['path']=='candidate.py']
    if len(sources)!=1 or hashlib.sha256(sources[0].encode()).hexdigest()!=execution.get('entrypoint_source_sha256'):
        raise ValueError('Executed parent source is absent or changed')
    return sources[0]

def regions(source):
    """Exact AST regions, not guessed natural-language patch locations."""
    result={};module=[];imports=[]
    for node in ast.parse(source).body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):imports.append(node);continue
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRAPH_SCORE_CONTRACT' for t in node.targets):continue
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.ClassDef):
            methods=[v for v in node.body if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))]
            for method in methods:result[node.name+'.'+method.name]=ast.dump(method,include_attributes=False)
            shell=copy.deepcopy(node);shell.body=[v for v in shell.body if not isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))]
            result[node.name+'.<class>']=ast.dump(shell,include_attributes=False)
        else:module.append(node)
    result['<imports>']=ast.dump(ast.Module(body=imports,type_ignores=[]),include_attributes=False)
    result['<module>']=ast.dump(ast.Module(body=module,type_ignores=[]),include_attributes=False)
    return result

REGION_RESPONSE_MODE='GRAPH_PARENT_REGION_REPLACEMENTS_V1'

def region_request(request):
    result=dict(request)
    targets=[x['target'] for x in result['candidate_mechanism_recipe']['definition']['operations']]
    result['service_policy']={**result.get('service_policy',{}),'response_mode':REGION_RESPONSE_MODE,'region_targets':targets}
    return result

def assemble_regions(parent_source,files,targets,score_entries,score_request):
    """Build the child from the exact parent's AST, replacing only named regions."""
    names=[f['path'] for f in files]
    if len(names)!=len(set(names)) or set(names)!=set(targets):
        raise ValueError('Expected exactly each declared region')
    if len(targets)!=len(set(targets)) or any(t not in regions(parent_source) for t in targets):
        raise ValueError('Replacement targets differ from actual parent regions')
    nodes={f['path']:ast.parse(f['content']).body for f in files}
    declaration={k:score_request[k] for k in ('selected_spec_digest','selected_primitive_ids')}
    declaration['entries']=score_entries
    score=ast.parse('GRAPH_SCORE_CONTRACT = '+repr(declaration)).body
    tree=ast.parse(parent_source)
    def is_score(n):
        return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRAPH_SCORE_CONTRACT' for t in n.targets)
    def replace_group(predicate,replacements):
        positions=[i for i,n in enumerate(tree.body) if predicate(n)]
        first=positions[0] if positions else len(tree.body)
        tree.body=[n for n in tree.body if not predicate(n)]
        tree.body[first:first]=replacements
    # Shell replacement retains exact parent methods; subsequent method replacements
    # apply to that retained list. No implementation-supplied whole class is accepted.
    for target in sorted(targets,key=lambda t:0 if t.endswith('.<class>') else 1):
        replacement=nodes[target]
        if target=='<imports>':
            if any(not isinstance(n,(ast.Import,ast.ImportFrom)) for n in replacement):raise ValueError('Import region contains other computation')
            replace_group(lambda n:isinstance(n,(ast.Import,ast.ImportFrom)),replacement)
        elif target=='<module>':
            def module_node(n):return not isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not is_score(n)
            if any(not module_node(n) for n in replacement):raise ValueError('Module region contains another named region')
            replace_group(module_node,replacement)
        elif target.endswith('.<class>'):
            name=target.split('.')[0]
            if len(replacement)!=1 or not isinstance(replacement[0],ast.ClassDef) or replacement[0].name!=name or any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) for n in replacement[0].body):raise ValueError('Class shell must not replace methods')
            old=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            shell=replacement[0];shell.body.extend(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)))
            tree.body[tree.body.index(old)]=shell
        else:
            parts=target.split('.')
            if len(replacement)!=1 or not isinstance(replacement[0],(ast.FunctionDef,ast.AsyncFunctionDef)) or replacement[0].name!=parts[-1]:raise ValueError('Region must contain exactly its named function definition')
            body=tree.body if len(parts)==1 else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==parts[0])
            old=next(n for n in body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==parts[-1])
            body[body.index(old)]=replacement[0]
    replace_group(is_score,score)
    child=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    before,after=regions(parent_source),regions(child)
    changed={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    if not changed or not changed<=set(targets):raise ValueError('Local child changed undeclared parent regions or changed no computation')
    return child


class ResearchActionAdapter(OpenMechanismAdapter):
    def _validate_original_root_action(self, action):
        # The original root is an authorized reference, not a prior run's child.
        # Consume equivalent metadata only; never rewrite the paid response/source.
        if action is None:
            return
        if action['operation'] != 'DIRECT_IMPROVEMENT':
            raise ValueError('Original-root action must be DIRECT_IMPROVEMENT')
        if action['parent_source_sha256'] != hashlib.sha256(self.parent.encode('utf8')).hexdigest():
            raise ValueError('Original-root action source differs')
        if action['task_ref'] is not None:
            raise ValueError('Original-root action cannot reference an executed-parent task')

    def _parents(self):
        result={}
        root=Path(self.config['output'])/'research_execution_evidence'
        if not root.exists():return result
        binding_sha=hashlib.sha256(Path(self.config['binding']).read_bytes()).hexdigest()
        for file in root.glob('*.json'):
            row=json.loads(file.read_bytes())
            if row.get('seed')!=self.config['train_seed'] or row.get('worker_configuration',{}).get('binding_sha256')!=binding_sha:continue
            source_of(row)
            result[row['candidate_id']]=row
        return result

    def _parent(self,candidate_id):
        try:return self._parents()[candidate_id]
        except KeyError:raise ValueError('Parent must be an executed candidate of this run and binding') from None

    @contextmanager
    def _using_parent(self,candidate_id):
        if candidate_id==self.parent_id:
            yield;return
        row=self._parent(candidate_id)
        previous=self.parent,self.parent_id
        self.parent,self.parent_id=source_of(row),candidate_id
        try:yield
        finally:self.parent,self.parent_id=previous

    def adapt_initial_context(self,context):
        result=super().adapt_initial_context(context)
        return replace(result,knowledge_base={**dict(result.knowledge_base),'graph_research_action_output':self.config['output']})

    def provider_context(self,context):
        result=super().provider_context(context)
        existing=result['knowledge_base']['search_space']
        task=self.task_context()
        for key in ('incumbent_execution_evidence','root_relative_comparison'):
            if key in existing:task[key]=existing[key]
        task['latest_local_research_result']=context.get('knowledge_base',{}).get('latest_local_research_result')
        result['knowledge_base']['search_space']=task
        return result

    def task_context(self):
        value=super().task_context()
        value['source_region_contract']={'Class.method':'Only that existing method, including __init__ when initialization changes.', 'Class.<class>':'Class header, bases, decorators and class-level statements only; excludes all methods.', '<imports>':'Module imports only.', '<module>':'Other module statements only, excluding named functions/classes and GRAPH_SCORE_CONTRACT.', 'dependency_rule':'Declare every required changed region as its own operation. A forward change requiring a new initialized buffer must also declare __init__; a scoring-route change must name its methods, not Class.<class>.'}
        value['construction']='Choose original-root construction or explicitly inherit one supplied executed parent. Local operations replace only declared source regions; all other parent computations are retained.'
        value['research_actions']={'operations':['DIRECT_IMPROVEMENT','MATCHED_CONTROL','MECHANISM_OFF'],'optional':True,'candidate_cost':'One existing candidate opportunity; measured parent is not retrained','parents':[{ 'candidate_id':k,'source_sha256':v['entrypoint_source_sha256'],'binding_sha256':v['worker_configuration']['binding_sha256'],'source_regions':list(regions(source_of(v))),'observation':{n:v.get(n) for n in ('development_metrics','measured_cost','status','seed')}} for k,v in self._parents().items()]}
        return value

    def proposal_schema(self):
        schema=super().proposal_schema();item=schema['properties']['proposals']['items'];props=item['properties']
        props['closest_parent']={'type':'string','minLength':1}  # Schema freezes at composition; actual own-run membership is checked at projection.
        text={'type':'string','minLength':1}
        action={'type':'object','additionalProperties':False,'properties':{
            'operation':{'enum':['DIRECT_IMPROVEMENT','MATCHED_CONTROL','MECHANISM_OFF']},
            'parent_source_sha256':text,'task_ref':{'type':['string','null']},
            'competing_explanations':{'type':'array','items':text},
            'predicted_distinction':text}}
        action['required']=list(action['properties'])
        props['research_action']={'anyOf':[{'type':'null'},action]};item['required'].append('research_action')
        return schema

    def resolve_confirmation(self,kind,primary_binding,context):
        phase=context.get('phase')
        if phase=='DECLARE_FOLLOWUP' and kind in ('MATCHED_CONTROL','MECHANISM_OFF'):
            return Resolution(Kind.NEEDS_PROPOSAL,reason='One evidence-bound child may be proposed; no automatic parent replay')
        if phase!='PROJECT_PROVIDER_PROPOSAL':return super().resolve_confirmation(kind,primary_binding,context)
        from jsonschema import Draft202012Validator,ValidationError
        draft=copy.deepcopy(dict(primary_binding));wire=copy.deepcopy(draft)
        ex=wire.get('execution_contract',{})
        if 'config' in ex:ex['config_json']=json.dumps(ex.pop('config'),sort_keys=True)
        try:
            Draft202012Validator(self.proposal_schema()).validate({'schema':'recclaw.research-line.fresh-open-spec-proposal-response.v1','proposals':[wire]})
            action=draft.pop('research_action');parent_id=draft.get('closest_parent')
            if parent_id==self.parent_id:
                self._validate_original_root_action(action)
            elif action is None:
                raise ValueError('Executed-parent inheritance requires its explicit action/source binding')
            else:
                if draft.get('conceptual_operator_id')!='mechanism':raise ValueError('Local action requires one explicit mechanism operation table')
                parent=self._parent(parent_id)
                if action['parent_source_sha256']!=parent['entrypoint_source_sha256']:raise ValueError('Action parent source differs')
                if action['operation']!='DIRECT_IMPROVEMENT' and len(action['competing_explanations'])<2:raise ValueError('Discriminating action needs competing explanations')
                if action.get('task_ref') is not None:
                    memory=context['research_context'].scientific_memory
                    global_memory=memory.get('global_memory',memory.get('global',memory))
                    pending=ResearchTaskQueueV2.from_dict(global_memory.get('task_queue')).get(action['task_ref'])
                    if pending is None or pending.candidate_id!=parent_id or pending.operation.value!=action['operation'] or pending.status.value not in ('PENDING','ACTIVE'):
                        raise ValueError('Action must reference an open native task for that parent and operation')
                definition=draft['open_mechanism'];available=regions(source_of(parent))
                targets=[x['target'] for x in definition['operations']]
                if not targets or len(set(targets))!=len(targets) or any(t not in available for t in targets):raise ValueError('Local operations must name distinct actual parent source regions')
                action.update(parent_candidate_id=parent_id,parent_binding_sha256=parent['worker_configuration']['binding_sha256'],parent_evidence_digest=digest(parent),run_identity_digest=digest(self.config['run_identity']),protocol_digest=context['research_context'].protocol_digest)
                draft['implementation_requirements'].append(MARKER+json.dumps(action,sort_keys=True,separators=(',',':')))
            with self._using_parent(parent_id),patch.object(self,'proposal_schema',super().proposal_schema):
                return super().resolve_confirmation(kind,draft,context)
        except (KeyError,ValueError,TypeError,ValidationError) as error:return self._unsupported('RESEARCH_ACTION:'+str(error))

    def _bind_definition(self,spec,definition):
        # The complete child identity consists of exact executed parent + delta.
        values=[v[len(MARKER):] for v in spec.implementation_requirements if v.startswith(MARKER)]
        if not values:return super()._bind_definition(spec,definition)
        if len(values)!=1:raise ValueError('Ambiguous action spec')
        action=json.loads(values[0]);row=self._parent(spec.closest_parent)
        if (action['parent_candidate_id']!=spec.closest_parent or action['parent_source_sha256']!=row['entrypoint_source_sha256'] or action['parent_evidence_digest']!=digest(row) or action['run_identity_digest']!=digest(self.config['run_identity']) or action['protocol_digest']!=spec.protocol_digest):raise ValueError('Research action parent/protocol binding drift')
        with self._using_parent(spec.closest_parent):bound=super()._bind_definition(spec,definition)
        bound['research_action']=action
        return bound

    def _prepare_native_innovation(self,outcome):
        action=payload(outcome)
        if not action:return super()._prepare_native_innovation(outcome)
        with self._using_parent(action['parent_candidate_id']):return super()._prepare_native_innovation(outcome)

    def _open_binding(self,program):
        bound=program['candidate_mechanism_binding'];action=bound.get('research_action')
        if not action:return super()._open_binding(program)
        row=self._parent(action['parent_candidate_id'])
        if row['entrypoint_source_sha256']!=action['parent_source_sha256'] or digest(row)!=action['parent_evidence_digest']:raise ValueError('Executed source changed since prepared round')
        with self._using_parent(action['parent_candidate_id']):return super()._open_binding(program)

    def adapt_implementation_request(self,request,*,mechanism_program):
        bound=mechanism_program.get('candidate_mechanism_binding',{});action=bound.get('research_action')
        if not action:return super().adapt_implementation_request(request,mechanism_program=mechanism_program)
        with self._using_parent(action['parent_candidate_id']):result=super().adapt_implementation_request(request,mechanism_program=mechanism_program)
        result['research_action']=action
        return region_request(result)

    def implementation_context(self,request):
        # This second adapter boundary used to overwrite the selected parent with root.
        action=request.get('candidate_mechanism_recipe',{}).get('research_action')
        if not action:return super().implementation_context(request)
        with self._using_parent(action['parent_candidate_id']):result=super().implementation_context(request)
        result['research_action']=action
        return region_request(result)

    def canonicalize_implementation_response(self,response,request):
        action=request.get('research_action')
        if not action:return super().canonicalize_implementation_response(response,request)
        parent=self._parent(action['parent_candidate_id'])
        if parent['entrypoint_source_sha256']!=action['parent_source_sha256']:
            raise ValueError('Executed parent source differs at implementation assembly')
        if request.get('service_policy',{}).get('response_mode')!=REGION_RESPONSE_MODE:
            raise ValueError('Local action requires explicit region replacements')
        targets=[x['target'] for x in request['candidate_mechanism_recipe']['definition']['operations']]
        if request['service_policy']['region_targets']!=targets:raise ValueError('Implementation targets drifted')
        if response['region_status']=='PROPOSAL_INCOMPLETE':
            raise ValueError('PROPOSAL_INCOMPLETE: '+response['implementation_summary'])
        child=assemble_regions(source_of(parent),response['files'],targets,response['graph_score_entries'],request['graph_score_contract_request'])
        bind_source_contract(child.encode(),request['graph_score_contract_request'])
        result=dict(response,files=[{'path':'candidate.py','content':child}])
        result.pop('region_status')
        result.pop('graph_score_entries')
        return result


def finish_action(state,result,round_index):
    action=payload(result.selected_outcome)
    if not action or action['operation']=='DIRECT_IMPROVEMENT':return state
    from recclaw_core.research_line import runtime
    spec=result.selected_outcome.spec
    child_id='cand-graph-'+spec.digest[:24]
    current=runtime._latest_completed_execution(state.context,allowed_files=('candidate.py',))
    # A paused external stage is not a zero-valued or completed experiment.
    if current is None or current.get('candidate_id')!=child_id:return state
    memory=copy.deepcopy(dict(state.context.scientific_memory));key='global_memory' if 'global_memory' in memory else 'global' if 'global' in memory else None
    global_memory=memory[key] if key else memory
    queue=ResearchTaskQueueV2.from_dict(global_memory.get('task_queue'))
    parent_path=Path(state.context.knowledge_base['graph_research_action_output'])/'research_execution_evidence'/(hashlib.sha256(action['parent_candidate_id'].encode()).hexdigest()+'.json')
    parent=json.loads(parent_path.read_bytes())
    source_of(current);source_of(parent)
    if digest(parent)!=action['parent_evidence_digest']:raise ValueError('Task parent reference drift')
    paired=current.get('worker_configuration',{}).get('binding_sha256')==action['parent_binding_sha256'] and current.get('seed')==parent.get('seed')
    def score(row):
        value=row.get('development_metrics',{}).get('primary_metric',{})
        return value.get('value') if row.get('metric_bearing') and value.get('name')=='ndcg@5' else None
    a,b=score(parent),score(current)
    observation={'child_source_sha256':current['entrypoint_source_sha256'],'parent_source_sha256':action['parent_source_sha256'],'same_training_binding_and_seed':paired,'parent_value':a,'child_value':b,'child_minus_parent':b-a if paired and a is not None and b is not None else None,'child_status':current.get('status'),'child_measured_cost':current.get('measured_cost'),'parent_evidence_reused':True,'causal_identification_claimed':False}
    task_id='graph-research-action:'+spec.digest
    program={'spec':spec.to_dict(),'research_action':action}
    if queue.get(task_id) is None:
        task=ResearchTaskRecordV2(task_id=task_id,operation=action['operation'],candidate_id=child_id,candidate_semantic_digest=digest(program),mechanism_program_digest=digest(program),parent_candidate_id=action['parent_candidate_id'],comparator_identity=action['parent_source_sha256'],protocol_digest=spec.protocol_digest,required_seed_or_control=str(current['seed']),priority=.5,created_round=round_index,mechanism_program=program,producer_role=result.selected_outcome.producer_role,metadata={'research_action':action,'observation':observation,'verification_seed':str(current['seed'])})
        queue=queue.enqueue(task).activate(task_id)
        queue=queue.satisfy(task_id,evidence=(current['entrypoint_source_sha256'],),reason='LOCAL_RESULT_OBSERVED') if paired and a is not None and b is not None else queue.close(task_id,reason='LOCAL_RESULT_NO_COMPARABLE_METRIC')
    ref=action.get('task_ref')
    if ref is not None:
        pending=queue.get(ref)
        if pending is None or pending.candidate_id!=action['parent_candidate_id']:raise ValueError('Referenced task does not belong to measured parent')
        if pending.status.value in ('PENDING','ACTIVE'):
            queue=queue.satisfy(ref,evidence=(current['entrypoint_source_sha256'],),reason='LOCAL_CHILD_RESULT_OBSERVED') if paired and a is not None and b is not None else queue.close(ref,reason='LOCAL_CHILD_NO_COMPARABLE_METRIC')
    global_memory['task_queue']=queue.to_dict()
    knowledge=copy.deepcopy(dict(state.context.knowledge_base));knowledge['latest_local_research_result']={'task_id':task_id,'action':action,'observation':observation}
    return replace(state,context=replace(state.context,scientific_memory=memory,knowledge_base=knowledge))

@contextmanager
def research_actions(adapter_module,policy_module):
    base_policy=policy_module.FrozenSearchPolicy
    class ActionPolicy(base_policy):
        def post_round(self,state,result,round_index,opportunity_ref):
            state=super().post_round(state,result,round_index,opportunity_ref)
            return finish_action(state,result,round_index)
    # Native acquisition/role coverage/ranking/budget stay untouched.
    with patch.object(adapter_module,'GraphSearchAdapter',ResearchActionAdapter),patch.object(policy_module,'FrozenSearchPolicy',ActionPolicy):yield
