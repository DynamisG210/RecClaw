"""Actual own-run TRAIN evidence consumed by generation and independent choice."""
import ast,copy,json
from pathlib import Path
from train_signal import summarize

POLICY = '''MEASURED COMPUTATION LEARNING:
Read the same-source TRAIN observations before using an earlier mechanism name as
evidence. Distinguish no measured activation, an unchanged/scale-only learning signal,
an actual update with harmful DEV outcome, and an unmeasured claim. Sampled absence
is not global inactivity; larger gradient/update is not better recommendation.
State which measured limitation the next complete computation addresses and a
TRAIN observation that would distinguish its mechanism from the competing explanation.
An implementation can change propagation, representation, readout and loss together
when the hypothesis needs it. Neither small edits nor large rewrites receive preference.
A coupled negative observation does not eliminate an entire mechanism family.
Previously tested edits need a changed condition and a distinguishable prediction.
Only this run's own TRAIN/DEV is available. Never infer hidden TEST or control results.
'''

def install(context_module):
    build0=context_module.build_cards;project0=context_module.project_prompt
    def build(executions,ledger,decisions,root_id,root_source):
        rows={r['source_sha256']:r for r in ledger['results']}
        measured={cid:e for cid,e in executions.items() if rows[e['entrypoint_source_sha256']].get('genuine') and rows[e['entrypoint_source_sha256']].get('valid')}
        cards=build0(measured,ledger,decisions,root_id,root_source)
        for cid,e in executions.items():
            if cid in measured:continue
            row=rows[e['entrypoint_source_sha256']]
            if row.get('valid') and row.get('root'):continue
            cards[cid]=dict(candidate_id=cid,source_sha256=e['entrypoint_source_sha256'],
                available_index=row['available_index'],value=None,metric=None,
                evidence_authority='PHYSICAL_FAILURE_NO_DEV_METRIC',status=e.get('status'),
                failure_stage=row.get('evidence',{}).get('failure_stage'),
                failure_detail=row.get('evidence',{}).get('error'),
                interpretation='No successful training metric; do not infer mechanism harm or inactive gradients.')
        for cid,card in cards.items():
            row=rows[card['source_sha256']]
            stage=row['evidence'].get('physical_stage',{})
            if not row.get('valid') and stage.get('kind')=='training':
                log=Path(stage['output']).parent/'worker.log'
                if log.exists() and 'FileNotFoundError' in log.read_text() and 'training_runtime/graph_worker_measured.py' in log.read_text():
                    card['failure_class']='CONFIRMED_TRUSTED_WORKER_STAGING_ERROR_BEFORE_TRAIN'
                    card['failure_log']=str(log)
                    card['optimizer_steps_completed']=0
            card['actual_training']=summarize(Path(stage['output'])/'TRAIN_SIGNAL.jsonl',card['source_sha256']) if stage.get('kind')=='training' else {'status':'NOT_MEASURED'}
            source=root_source if cid==root_id else context_module.source_of(executions[cid]);tree=ast.parse(source)
            card['literal_operator_calls']=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call) and any(s in ast.unparse(n.func).lower() for s in ('linear','relu','tanh','dropout','normalize','softmax','sigmoid','softplus'))})
            card['operator_authority']='Literal syntax, not runtime activation; compare sampled TRAIN and exact source.'
        return cards
    def project(prompt,cards,task_facts=None):
        marker=context_module.MARKER;start=prompt.index('{',prompt.index(marker)+len(marker))
        context,_=json.JSONDecoder().raw_decode(prompt[start:]);mode=context['objective']['space']['branch_allocation']['mode']
        cards=copy.deepcopy(cards)
        if mode=='DISCOVER':
            # Preserve the original numeric-only historical boundary. No old code/operator names.
            for card in cards.values():
                card.pop('literal_operator_calls',None);card.pop('operator_authority',None)
                signal=card.get('actual_training',{})
                card['actual_training']={k:v for k,v in signal.items() if k in ('status','epochs','sha256','limits')}
        return project0(POLICY+'\n'+prompt,cards,task_facts)
    context_module.build_cards=build;context_module.project_prompt=project
