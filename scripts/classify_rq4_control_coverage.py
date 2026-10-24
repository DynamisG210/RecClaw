"""Offline control coverage using only frozen prefix evidence, never successor scores."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
parents={'cand-e2-7d7c8adaa7d9ce93df7eff1e':1,'cand-e2-dbccd6479033e80a9b0557ba':2}
rows=[];inputs={};registry={}
for state in (2,3):
 prefix=a.root/f'state{state}_native_check/PREFIX_RESULT.json'
 pool=a.root/f'state{state}_decisions/DECISION_SOURCE_CHAIN.json'
 for path in (prefix,pool):inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 results=json.loads(prefix.read_text())['evaluation']['results']
 assert {r['available_index'] for r in results}==set(range(state+1))
 refs={}
 for r in results:
  evidence=r['evidence'];idx=r['available_index'];key=f'state{state}:prefix:{idx}'
  refs[idx]=key
  registry[key]={'available_index':idx,'source_sha256':r['source_sha256'],'ndcg_at_5':r['primary_metric'],'native_artifacts':evidence.get('artifacts'),'train_seed':evidence.get('train_seed'),'eval_seed':evidence.get('eval_seed'),'test_accessed':evidence.get('test_accessed'),'prefix_file':str(prefix)}
 for arm in json.loads(pool.read_text())['rows']:
  for proposal in arm['complete_pool']:
   spec=proposal.get('spec')
   row={'state_cutoff':state,'arm':arm['arm'],'role':proposal['role'],'proposal_digest':proposal['digest'],'pool_sha256':arm['pool_sha256'],'failure_code':proposal.get('failure_code'),'primitives':proposal.get('primitives')}
   if not spec:
    row.update(classification='REJECTED_NO_PROPOSAL',available_controls=[],unresolved=['No valid proposal; do not impute a scientific comparison.'])
   else:
    parent=spec['closest_parent'];idx=parents[parent]
    row.update(classification='CONTROL_OBSERVED_CANDIDATE_CONTRAST_UNRESOLVED',closest_parent=parent,available_controls=[refs[idx]],expected_evidence_verbatim=spec.get('expected_evidence'),predictions_verbatim=spec.get('discriminative_predictions'),unresolved=['Proposed candidate quality versus the measured parent is not available at this cutoff.','New candidate source/qualification and requested mechanism diagnostics are not established by the parent score.'])
    if idx!=1:row['additional_available_reference']=refs[1];row['comparison_note']='Selected memory+gate parent differs from gate-only. A future convolution replacement versus memory is not an isolated convolution-on/off intervention.'
    else:row['comparison_note']='Reuse measured gate-only parent for candidate-level comparison; this does not replace matched branch interventions or requested gradients/stratified diagnostics.'
    if state==3:
     row['already_answered_history_contrast']={'control':refs[1],'treated':refs[3],'gate_minus_adapter':registry[refs[1]]['ndcg_at_5']-registry[refs[3]]['ndcg_at_5'],'scope':'Historical gate-only versus adapter+gate candidate comparison; not the effect of this new proposal.'}
   rows.append(row)
assert len(rows)==24
assert sum(r['classification']=='REJECTED_NO_PROPOSAL' for r in rows)==1
assert sum(bool(r['available_controls']) for r in rows)==23
report={'method':'Deterministic offline classification of original complete pools and original frozen PREFIX_RESULT only. No successor outcomes loaded. Parent ID mapping is explicit and reviewable. Diagnostic absence means not established by these inputs, not proof that no other artifact exists.','input_hashes':inputs,'evidence_registry':registry,'rows':rows,'counts':{'proposal_slots':24,'valid_proposals':23,'rejected':1,'measured_parent_control_available':23,'new_candidate_contrast_unresolved_at_cutoff':23,'memory_parent_with_gate_reference':2},'decision':'A minimal existing-control lookup can make already measured comparator IDs accessible before generation and avoid requesting redundant control FITs. It does not justify forbidding new candidates or a paid 2x2: all 23 valid candidate contrasts remain unresolved at their cutoffs. No runtime inputs changed.','new_api_calls':0,'new_fits':0}
out=a.root/'CONTROL_COVERAGE_24_PROPOSALS.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report['counts']))
