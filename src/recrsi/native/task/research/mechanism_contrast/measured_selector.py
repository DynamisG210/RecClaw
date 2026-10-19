"""Commit both original programs, then independently choose unchanged bytes.

The selector receives the actual projected own-run context and no generator choice.
Its native broker, role token ceiling and common physical ledger remain authoritative.
"""
import copy,hashlib,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

VERSION='graph-parent-mechanism-contrast-v3'
MARKER='Use this complete role-scoped Research Context JSON as the authoritative input for this call:'
INSTRUCTION='''Choose one committed program for the next original training opportunity.
Both programs are untrained; their position, size and complexity carry no preference.
Read actual source and own TRAIN/DEV evidence. Mechanism prose is a hypothesis.\nUse exact_parent_program_contrasts to identify the actual changed methods, shared TRAIN/scoring paths, parent source and separate root/frontier references. These are static facts, not execution counts or proof of benefit. Explain when changes are coupled, scale-only, or do not distinguish the proposed mechanism from its parent. Prefer an informative useful improvement, not cosmetic novelty or an unchanged parent retry.
Do not extrapolate a coupled negative result to all nonlinear/linear/regularized models.
For your choice state the actual computation change, a source-grounded evidence claim,
the competing explanation and a distinguishable TRAIN prediction. Unknown observations
stay unknown. Do not rewrite either program, merge programs, invent fitness or request
extra training. Optimize expected final recommendation quality within the remaining
shared budget, not selection flips, novelty, larger gradients or aesthetic simplicity.
Return exactly one decision object in proposals, using the supplied JSON schema.
'''

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

def save_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    blob=(json.dumps(value,sort_keys=True,ensure_ascii=True,indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=blob:raise ValueError('Committed selection input changed')
    else:
        with path.open('xb') as f:f.write(blob)

def commit_pair(response,identity):
    pair=copy.deepcopy(response['proposals'])
    if len(pair)!=2:raise ValueError('Exactly two original programs are required')
    ids=['candidate-'+digest({'identity':identity,'position':i})[:16] for i in range(2)]
    order=[0,1] if int(digest({'identity':identity,'permutation_seed':91029})[:8],16)%2==0 else [1,0]
    return dict(schema=VERSION,identity=identity,
        programs=[dict(candidate_id=ids[i],proposal=pair[i],proposal_digest=digest(pair[i]),
            source_sha256=hashlib.sha256(pair[i]['program']['source'].encode()).hexdigest()) for i in order],
        original_index_by_id={ids[i]:i for i in range(2)},
        generator_response_digest=digest(response),generator_choice_not_exposed=True)

def schema(ids):
    text={'type':'string','minLength':1}
    item={'type':'object','additionalProperties':False,'properties':{
        'candidate_id':{'enum':ids},'source_grounded_comparison':text,
        'measured_evidence_ids':{'type':'array','items':text},
        'actual_computation_change':text,'competing_explanation':text,
        'distinguishable_train_prediction':text,'reason_to_spend_next_fit':text}}
    item['required']=list(item['properties'])
    return {'type':'object','additionalProperties':False,'properties':{
        'proposals':{'type':'array','minItems':1,'maxItems':1,'items':item}},'required':['proposals']}

def choose_from_commit(commit,decision):
    selected=decision['proposals'][0]['candidate_id']
    chosen=next((r for r in commit['programs'] if r['candidate_id']==selected),None)
    if chosen is None:raise ValueError('Selector invented an uncommitted program')
    if digest(chosen['proposal'])!=chosen['proposal_digest']:raise ValueError('Committed program changed')
    return copy.deepcopy(chosen['proposal'])

def load_original(path):
    spec=importlib.util.spec_from_file_location('_graph_original_counterproposal',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def adapter_class(base,original):
    Parent=original.adapter_class(base)
    consume_original=original.consume_choice
    class MeasuredChoice(Parent):
        def build_provider_research_producer(self,**kwargs):
            wrapped=super().build_provider_research_producer(**kwargs)
            producer=wrapped._producer;old_call=producer._call;old_provider=producer.provider_call
            generation={}
            def provider_call(**options):
                options=dict(options,output_token_ceiling=min(options['output_token_ceiling'],17000))
                generation['options']=dict(options)
                result=old_provider(**options);generation['result']=result
                return result
            def call(role,view,**options):
                from recclaw_core.research_line import provider as native
                from ro_main_provider import native_ro_provider
                from run_native_json import include_schema
                generation.clear()
                def consume(response,*,expected_role):
                    trace=copy.deepcopy(producer.last_call_trace)
                    identity={'run':self.config['run_identity'],'logical_call_id':trace['logical_call_id'],
                        'session_id':producer.session_id,'generation_request_digest':trace['receipt']['request_digest']}
                    commit=commit_pair(response,identity)
                    root=Path(self.config['output'])/'independent_selection'/digest(identity)
                    save_once(root/'PAIR_COMMIT.json',commit)
                    if (root/'DECISION.json').exists():
                        record=json.loads((root/'DECISION.json').read_bytes())
                        if record['commit_sha256']!=digest(commit):raise ValueError('Cached selection has different committed programs')
                        budget=json.loads((root/'ROLE_BUDGET.json').read_bytes())
                        saved_trace=record['selector_trace']
                        # Reconstruct logical role accounting from the immutable physical receipt.
                        # This is not a new provider call or a synthetic successful native receipt.
                        producer._charge_output_tokens(budget['context_identity'],SimpleNamespace(attempts=saved_trace['attempts']))
                        producer.call_traces=(*producer.call_traces,copy.deepcopy(saved_trace))
                        return choose_from_commit(commit,record['independent_decision']),record
                    # Parse the exact final generation wire, after all existing projections.
                    payload=trace['sealed_request']['request_payload'];prompt=payload['input']
                    start=prompt.index('{',prompt.index(MARKER)+len(MARKER))
                    context,_=json.JSONDecoder().raw_decode(prompt[start:])
                    from prospective_revision import program_contrasts
                    contrasts=program_contrasts(self,commit,context)
                    save_once(root/'PROSPECTIVE_SOURCE_CONTRASTS.json',contrasts)
                    visible={'own_context':context,'committed_programs':commit['programs'],'exact_parent_program_contrasts':contrasts}
                    selector_prompt=INSTRUCTION+'\n'+json.dumps(visible,ensure_ascii=True,sort_keys=True)
                    ids=[r['candidate_id'] for r in commit['programs']]
                    save_once(root/'SCHEMA.json',schema(ids))
                    # Original per-context total includes generator, declaration completion and selector.
                    # Native producer uses context_digest as context_identity (verified in its caller).
                    key=view['context_ref']
                    matching=[k for k in producer._proposal_output_tokens_by_context if k.startswith(view['context_ref']+':context-')]
                    if len(matching)!=1:raise ValueError('Cannot identify original role token budget')
                    key=matching[0]
                    ceiling=min(3000,producer._budgeted_output_ceiling(key))
                    budget_path=root/'ROLE_BUDGET.json'
                    if budget_path.exists():
                        budget=json.loads(budget_path.read_bytes())
                        if budget['context_identity']!=key:raise ValueError('Selector role budget identity changed')
                        ceiling=min(ceiling,budget['selector_ceiling'])
                    else:
                        save_once(budget_path,dict(context_identity=key,
                            total=producer.proposal_output_token_ceiling_total_per_context,
                            prior_outputs=producer._proposal_output_tokens_by_context[key],selector_ceiling=ceiling))
                    request=dict(generation['options'],call_root=root/'native',schema_path=root/'SCHEMA.json',
                        logical_call_id=trace['logical_call_id']+':independent-source-selection-v2',
                        prompt=selector_prompt,output_token_ceiling=ceiling,expected_proposal_count=1)
                    request.pop('resume_request_digest',None)
                    # The selector has its own committed schema, not the generator's transport release.
                    request['expected_transport_release_digest']=None
                    # Persist the complete fixed request before any dispatch; no successful redraw.
                    serial={k:str(v) if isinstance(v,Path) else v for k,v in request.items()}
                    request_path=root/'REQUEST.json'
                    if request_path.exists():
                        saved=json.loads(request_path.read_bytes())
                        if saved['prompt']!=serial['prompt'] or saved['logical_call_id']!=serial['logical_call_id']:
                            raise ValueError('Selector replay changed committed evidence')
                        request.update(saved)
                        for k in ('call_root','schema_path','credential_config_path'):
                            if request.get(k):request[k]=Path(request[k])
                        if request['output_token_ceiling']>ceiling:raise ValueError('Saved selector exceeds role remainder')
                    else:save_once(request_path,serial)
                    # Native short-storage map is deterministic; native SQLite is authoritative on resume.
                    relative=Path(request['call_root']).relative_to(Path(self.config['output']))
                    mapped=Path(self.config['output'])/'broker_calls'/hashlib.sha256(str(relative).encode()).hexdigest()
                    started,resume=native._stored_provider_request_replay(mapped,
                        logical_call_id=request['logical_call_id'],session_id=producer.session_id)
                    if started and resume is None:raise ValueError('Selector dispatch requires evidence reconciliation')
                    if resume:request['resume_request_digest']=resume
                    with native_ro_provider(self.config['provider_config']) as paid:
                        result=paid(**include_schema(request))
                    native._finalize_provider_result(producer,budget_key=key,kind='independent_source_selector',
                        logical_call_id=request['logical_call_id'],prompt=request['prompt'],result=result,
                        transport_ceiling={'total_tokens':request['token_ceiling'],'output_tokens':request['output_token_ceiling']})
                    decision=native._require_success(result,kind='independent source selector')
                    chosen=choose_from_commit(commit,decision)
                    if chosen['producer_role']!=expected_role:raise ValueError('Selector changed role')
                    index=commit['original_index_by_id'][decision['proposals'][0]['candidate_id']]
                    # Preserve the generator receipt and append the distinct selector trace to its record.
                    selector_trace=copy.deepcopy(producer.last_call_trace)
                    producer.last_call_trace=trace
                    effective=dict(selected='PROPOSAL' if index==0 else 'COUNTERPROPOSAL',
                        comparison=decision['proposals'][0]['source_grounded_comparison'],evidence_updates=[])
                    _,record=consume_original(dict(response,research_decision=effective),expected_role=expected_role)
                    record.update(version=VERSION,generator_decision=response['research_decision'],
                        independent_decision=decision,selector_trace=selector_trace,commit_sha256=digest(commit))
                    save_once(root/'DECISION.json',record)
                    return chosen,record
                with patch.object(original,'consume_choice',consume):return old_call(role,view,**options)
            producer.provider_call=provider_call;producer._call=call
            return wrapped
    return MeasuredChoice
