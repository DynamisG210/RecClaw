"""Construct task and parent-computation contrasts before candidate selection."""
import hashlib,importlib.util,json,sys
from pathlib import Path
N=Path(__file__).resolve().parent
POLICY='''PARENT-CONTROLLED MECHANISM RESEARCH v3:
The dataset and candidate universe in the task fields are the executed binding, not a legacy full-sort display. full_sort_predict is an interface name and does not mean a full-catalog evaluation.
Use the actual allocated construction parent for both complete alternatives. Each alternative should test one explicit mechanism against that already measured parent, preserving its other computational conditions. If several changes are necessary to realize a mechanism, enumerate their coupling and restrict the claim to that bundle. Do not re-propose an unchanged measured parent merely to obtain another draw.
The counterproposal should distinguish a competing explanation using the same parent and matched conditions, not merely offer a large rewrite against a simpler attractive baseline. State the source-level change, mechanism-off behavior and a discriminating TRAIN observation in the existing fields. No architecture or complexity is preferred or banned.
When reading outcomes, use the named child-minus-construction-parent, child-minus-prior-frontier and child-minus-root separately. An active parameter or nonlinear function does not establish a useful nontrivial mechanism: distinguish effective scale, normalization, depth weighting and changed geometry from method names. Read actual source and measured TRAIN, and leave unmeasured activation unknown. Coupled negative outcomes do not reject an entire mechanism class.
Only this new run's own TRAIN/DEV and original task root are available. No prior winner, other system, hidden TEST, or development outcome is an input. Keep the original schema, allocated parent, finite budget, and independent source selector.
'''

def program_contrasts(adapter,commit,context):
    import source_bound_context
    mode=context['objective']['space']['branch_allocation']['mode']
    ledger=json.loads((Path(adapter.config['output'])/'EVALUATION_LEDGER.json').read_bytes())
    valid=[r for r in ledger['results'] if r.get('valid')]
    root=next((r for r in valid if r.get('root')),None)
    frontier=max(valid,key=lambda r:r['primary_metric']) if valid else None
    result={}
    for row in commit['programs']:
        p=row['proposal'];parent=p['parent_ref'];source=p['program']['source']
        record=dict(candidate_id=row['candidate_id'],source_sha256=row['source_sha256'],declared_parent=parent,authority='UNTRAINED_EXACT_SOURCE_STATIC_CONTRAST_NOT_EFFECT')
        if mode=='DISCOVER' and parent!=adapter._registry_root_id:
            record['status']='PARENT_OUTSIDE_DISCOVER_BOUNDARY';result[row['candidate_id']]=record;continue
        try:parent_source=adapter._parent_source(parent)
        except (KeyError,ValueError):
            record['status']='UNRESOLVED_PARENT_RETAIN_NATIVE_QUALIFICATION';result[row['candidate_id']]=record;continue
        h=hashlib.sha256(parent_source.encode()).hexdigest();pr=next((r for r in reversed(valid) if r['source_sha256']==h),None)
        record['construction_parent']=dict(candidate_id=parent,source_sha256=h,own_DEV_NDCG5=pr['primary_metric'] if pr else None)
        record['original_root']=dict(source_sha256=root['source_sha256'],own_DEV_NDCG5=root['primary_metric']) if root else None
        record['current_global_frontier']=dict(source_sha256=frontier['source_sha256'],own_DEV_NDCG5=frontier['primary_metric']) if frontier else None
        try:
            record['source_difference']=source_bound_context.compact_difference(parent_source,source)
            record['status']='EXACT_PARENT_SOURCE_LINKED'
        except (SyntaxError,ValueError) as error:
            record.update(status='STATIC_EXTRACTION_UNAVAILABLE_RETAIN_NATIVE_QUALIFICATION',reason=type(error).__name__+': '+str(error))
        record['limits']='Static calls and changed methods do not establish execution counts, activation, numerical equivalence or quality. Proposed programs are untrained. A literal full_sort_predict method is legal under fixed100 scoring.'
        result[row['candidate_id']]=record
    return result

def install_context(module):
    import measured_context_complete
    measured_context_complete.install(module)
    old=module.project_prompt
    def project(prompt,cards,task_facts=None):return old(POLICY+'\n'+prompt,cards,task_facts)
    module.project_prompt=project

def install():
    import graph_search_adapter,measured_context
    path=N/'graph_search_adapter.prospective.py'
    spec=importlib.util.spec_from_file_location('_prospective_truth_adapter',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for name in ['__init__','task_context','provider_context']:setattr(graph_search_adapter.GraphSearchAdapter,name,getattr(m.GraphSearchAdapter,name))
    measured_context.install=install_context
    old=sys.modules['_revision_comparators']
    old.SEMANTICS='Use the actual allocated construction parent, which may be original_root or an own measured candidate. original_root is a fixed comparison, never an instruction to replace the allocated parent. '+old.SEMANTICS[old.SEMANTICS.index('Compare declared intent'):]
    old.INPUT_VERSION='graph-executed-task-parent-contrasts-v3'
