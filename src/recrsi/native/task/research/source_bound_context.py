"""Build decision context from ledger contrasts and candidate source facts.
Apply this projection when constructing a new research decision."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from source_effect_facts import compare_sources

VERSION = 'graph-source-bound-contrasts-v1'
MARKER = 'Use this complete role-scoped Research Context JSON as the authoritative input for this call:'
INSTRUCTION = '''OWN_COMPUTATION_CONTRASTS contains own-run facts with explicit authority.
Use EXECUTED_TASK_FACTS and worker_task as the executed task; legacy display configuration is not task identity.
Distinguish child-minus-construction-parent, child-minus-prior-global-frontier and
child-minus-root. Falling below the frontier does not imply the parent intervention
was harmful. A positive parent delta does not establish component-wise causality.
Read the parent/candidate source differences before interpreting archived mechanism
prose. The prose and your new explanation remain hypotheses. A method reachable
from both calculate_loss and scoring cannot be assumed inference-only. Static self
call edges are not runtime coverage; branches, aliases and inherited paths may remain
unresolved. Explain coupled changes rather than calling a changed bundle a one-factor
control. Consult relevant earlier negative outcomes and distinguish a new combination
from an exact computation replay. Neither novelty nor added complexity earns credit.
Compare both full proposed computations using these facts, a competing explanation
and a falsifier. Keep the original response schema, parent registry, two-alternative
selection, task protocol and finite budget. No mechanism is banned or selected for you.
'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def source_of(row):
    files = row['implementation']['files']
    source = next(f['content'] for f in files if f['path'] == 'candidate.py')
    if hashlib.sha256(source.encode()).hexdigest() != row['entrypoint_source_sha256']:
        raise ValueError('Own executed source bytes differ from evidence identity')
    return source


def compact_difference(parent, child):
    facts = compare_sources(parent, child)
    c = facts['candidate']
    return dict(evidence_class=c['evidence_class'],
                parent_source_sha256=facts['parent']['source_sha256'],
                candidate_source_sha256=c['source_sha256'],
                changed_methods=facts['changed_methods'],
                changed_shared_train_inference_methods=facts['changed_shared_train_inference_methods'],
                attribute_changes=facts['attribute_changes'],
                candidate_direct_self_method_edges=c['direct_self_method_edges'],
                candidate_syntactically_reachable=c['syntactically_reachable'],
                candidate_normalize_call_sites={k:v for k,v in c['normalize_call_sites'].items() if v},
                unresolved_self_calls={k:v for k,v in c['unresolved_self_calls'].items() if v},
                limits=c['limits'])


def build_cards(executions, ledger, decisions, root_id, root_source):
    """Only current run's already measured TRAIN/DEV artifacts are inputs."""
    rows = sorted((r for r in ledger['results'] if r.get('valid')),
                  key=lambda r:r['available_index'])
    roots = [r for r in rows if r.get('root')]
    root = roots[0] if roots else None
    source_rows = {r['source_sha256']:r for r in rows if r.get('genuine')}
    cards = {}
    for cid, execution in executions.items():
        source = source_of(execution)
        physical = source_rows[execution['entrypoint_source_sha256']]
        decision = decisions[cid]
        parent_id = decision['parent_id']
        parent_execution = executions.get(parent_id)
        if parent_id == root_id:
            parent_source, parent_row = root_source, root
        elif parent_execution is not None:
            parent_source = source_of(parent_execution)
            parent_row = source_rows[parent_execution['entrypoint_source_sha256']]
        else:
            raise ValueError('Construction parent lacks original own-run evidence')
        if parent_row is None:
            raise ValueError('Construction parent lacks measured outcome')
        if parent_row['available_index'] >= physical['available_index']:
            raise ValueError('Future evidence cannot define the measured parent')
        predecessors = [r for r in rows if r['available_index'] < physical['available_index']]
        frontier = max(predecessors, key=lambda r:r['primary_metric'])
        metric = physical['primary_metric']
        measured = execution['development_metrics']['primary_metric']
        if measured['name'] != 'ndcg@5' or measured['value'] != metric:
            raise ValueError('Native DEV observation differs from physical ledger')
        worker = execution['worker_configuration']
        difference = compact_difference(parent_source, source)
        cards[cid] = dict(candidate_id=cid, available_index=physical['available_index'],
            source_sha256=execution['entrypoint_source_sha256'],
            worker_task={k:worker[k] for k in ('dataset','binding_sha256') if k in worker},
            train_seed=execution['seed'], evidence_authority='PHYSICAL_OWN_DEV_AND_EXACT_SOURCE',
            metric='ndcg@5', value=metric,
            construction_parent=dict(candidate_id=parent_id, source_sha256=parent_row['source_sha256'],
                value=parent_row['primary_metric'], child_minus_parent=metric-parent_row['primary_metric']),
            prior_global_frontier=dict(source_sha256=frontier['source_sha256'],
                value=frontier['primary_metric'], child_minus_frontier=metric-frontier['primary_metric']),
            original_root=dict(source_sha256=root['source_sha256'], value=root['primary_metric'],
                child_minus_root=metric-root['primary_metric']) if root else None,
            source_difference=difference,
            archived_mechanism=dict(authority='MODEL_WRITTEN_HYPOTHESIS_NOT_EXECUTION_FACT',
                text=execution.get('proposed_mechanism')),
            interpretation='Whole-candidate single-search-seed observation; no isolated causal effect established.')
    return dict(sorted(cards.items(), key=lambda kv:kv[1]['available_index']))


def project_prompt(prompt, cards, task_facts=None):
    """Preserve native context except one explicit evidence field and its digest."""
    mark = prompt.index(MARKER)
    start = prompt.index('{', mark+len(MARKER))
    context, length = json.JSONDecoder().raw_decode(prompt[start:])
    original = copy.deepcopy(context)
    mode = context['objective']['space']['branch_allocation']['mode']
    if mode == 'DISCOVER':
        # Preserve the existing independent-discovery information boundary.
        cards = {cid:{k:v for k,v in card.items() if k not in ('source_difference','archived_mechanism')}
                 for cid,card in cards.items()}
    context['objective']['space']['OWN_COMPUTATION_CONTRASTS'] = cards
    if task_facts is not None:
        context['objective']['space']['EXECUTED_TASK_FACTS'] = task_facts
    context['context']['projection_version'] = VERSION
    context['context'].pop('digest', None)
    context['context']['digest'] = digest(context)
    result = (prompt[:mark] + INSTRUCTION + '\n' + prompt[mark:start]
              + json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
              + prompt[start+length:])
    receipt = dict(version=VERSION, original_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   projected_prompt_sha256=hashlib.sha256(result.encode()).hexdigest(),
                   context_digest=context['context']['digest'], cards_digest=digest(cards),
                   own_measured_candidates=len(cards), prompt_chars=len(result), mode=mode,
                   old_context_digest=original['context'].get('digest'),
                   model_choice_and_effect_not_established=True)
    return result, receipt


def attach(adapter, evidence_dir):
    """Fresh prospective adapter; inject after final DISCOVER/REFINE projection."""
    original_build = adapter.build_provider_research_producer
    evidence_dir = Path(evidence_dir)
    root_source = adapter._parent_source(adapter._registry_root_id)
    if hashlib.sha256(root_source.encode()).hexdigest() != adapter._registry_root_sha256:
        raise ValueError('Frozen original root source differs')
    def build(**kwargs):
        wrapped = original_build(**kwargs)
        producer = wrapped._producer
        producer.call_root = producer.call_root / VERSION
        native_provider_call = producer.provider_call
        def provider_call(**options):
            import independent_discovery
            import ro_main_provider
            output = Path(adapter.config['output'])
            binding_path = Path(adapter.config['binding'])
            binding_bytes = binding_path.read_bytes()
            binding = json.loads(binding_bytes)
            task_facts = {k:binding[k] for k in ('dataset','task_id','train_seed','common_protocol_sha256')}
            task_facts['binding_sha256'] = hashlib.sha256(binding_bytes).hexdigest()
            task_facts['authority'] = 'FROZEN_EXECUTION_BINDING_NOT_LEGACY_DISPLAY'
            ledger = json.loads((output/'EVALUATION_LEDGER.json').read_bytes())
            decisions = {r['candidate_id']:r for p in (output/'branch_decisions').glob('*.json')
                         for r in [json.loads(p.read_bytes())]}
            cards = build_cards(adapter._parents(), ledger, decisions,
                                adapter._registry_root_id, root_source)
            original_projection = independent_discovery.project_prompt
            original_send = ro_main_provider.MainROTransport.send
            projected = []
            def after_discovery(prompt, authority_notice):
                final, receipt = project_prompt(original_projection(prompt, authority_notice), cards, task_facts)
                evidence_dir.mkdir(parents=True, exist_ok=True)
                path = evidence_dir/(receipt['projected_prompt_sha256']+'.json')
                payload = dict(receipt=receipt, logical_call_id=options['logical_call_id'])
                if path.exists():
                    if json.loads(path.read_bytes()) != payload:
                        raise ValueError('Prospective request evidence changed under its identity')
                else:
                    with path.open('x', encoding='utf-8') as stream:
                        json.dump(payload, stream, ensure_ascii=False, indent=2)
                projected.append(receipt)
                return final
            async def before_reserve(transport, request, timeout):
                raw = request.data
                payload = json.loads(raw)
                prompt = payload['input']
                start = prompt.index('{', prompt.index(MARKER)+len(MARKER))
                context, _ = json.JSONDecoder().raw_decode(prompt[start:])
                if len(projected)!=1 or context['context']['projection_version']!=VERSION:
                    raise ValueError('Final request lost source-bound projection')
                actual_cards=context['objective']['space']['OWN_COMPUTATION_CONTRASTS']
                if context['objective']['space']['EXECUTED_TASK_FACTS'] != task_facts:
                    raise ValueError('Final request task identity differs from frozen binding')
                if digest(actual_cards)!=projected[0]['cards_digest']:
                    raise ValueError('Final request facts differ from projected evidence')
                wire_sha=hashlib.sha256(raw).hexdigest()
                wire=dict(status='ACTUAL_BEFORE_RESERVE_NOT_EFFECT',wire_sha256=wire_sha,
                          context_digest=context['context']['digest'],cards_digest=digest(actual_cards),
                          model=payload['model'],max_output_tokens=payload['max_output_tokens'],
                          projection_receipt=projected[0])
                path=evidence_dir/('WIRE_'+wire_sha+'.json')
                if not path.exists():
                    with path.open('x',encoding='utf-8') as stream:json.dump(wire,stream,indent=2)
                return await original_send(transport,request,timeout)
            with patch.object(independent_discovery,'project_prompt',after_discovery), \
                 patch.object(ro_main_provider.MainROTransport,'send',before_reserve):
                result=native_provider_call(**options)
            if len(projected)!=1:
                raise ValueError('Expected the original final discovery projection exactly once')
            return result
        producer.provider_call = provider_call
        return wrapped
    adapter.build_provider_research_producer = build
    return adapter
