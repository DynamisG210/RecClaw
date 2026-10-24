"""Connect all state18 selected native proposals to sealed source and measured results."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();A=a.root
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
s=read(A/'state18_sealed/SETTLEMENT.json');pool=read(A/'state18_decisions/DECISION_SOURCE_CHAIN.json');verified=0
for item in read(A/'state18_sealed/MANIFEST.json'):
 if item.get('local_relative'):
  f=A/'state18_sealed'/item['local_relative'];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'];verified+=1
rows=[]
for r in s['rows']:
 arm=r['arm'];trace=read(A/f'state18_decisions/{arm}/ROUND_19_TRACE.json')['summary'];ar=next(x for x in pool['rows'] if x['arm']==arm);selected=next(x for x in ar['complete_pool'] if x['digest']==trace['selected_outcome_digest'])
 models=(A/f'state18_sealed/{arm}/source/models.py').read_text();main=(A/f'state18_sealed/{arm}/source/main_code.py').read_text()
 row={'arm':arm,'selected_producer':selected['role'],'selected_digest':selected['digest'],'selected_primitives':selected['primitives'],'hypothesis':selected['spec']['hypothesis'],'expected_evidence':selected['spec'].get('expected_evidence'),'source_hashes':r['source_sha256'],'ndcg5':r['ndcg5'],'retained_frontier':r['retained_frontier'],'frontier_gain':0.0,'delta_from_existing_frontier':r['ndcg5']-r['retained_frontier'],'source_observation':{'adapter_present':'adapter_up' in models,'causal_conv_present':'local_conv' in models,'cosine_scoring':'torch.sqrt((q*q)' in models,'query_gate_scoring':('score_gate' in models or 'query_gate' in models),'recent_suffix_training':'sequence_width-20' in main}}
 assert row['source_observation']['adapter_present'] and row['source_observation']['causal_conv_present'];rows.append(row)
assert rows[0]['selected_primitives']==['sample_recent'] and rows[0]['source_observation']['recent_suffix_training'] and rows[0]['source_observation']['cosine_scoring']
assert all(r['source_observation']['query_gate_scoring'] for r in rows[1:])
result={'state':18,'rows':rows,'manifest_local_files_verified':verified,'valid_candidate_fits':3,'test_fits':0,'api_calls':15,'api_usd':s['api_usd'],'new_gpu_seconds':s['new_gpu_seconds'],'mechanism_minus_summary':s['mechanism_minus_summary'],'mechanism_minus_raw':rows[2]['ndcg5']-rows[0]['ndcg5'],'interpretation':'Raw selected recent-position training while retaining cosine scoring. Summary and Mechanism selected query-gated scoring retaining adapter and convolution. All three underperformed the inherited frontier. Mechanism slightly exceeds Summary but both pursue the same broad scoring intervention; this is not evidence that mechanism feedback improved discovery. Fine implementation differences remain, and diagnostic requests are not measured mediation evidence.','next_evidence_gap':'A subsequent controlled decision must resolve why shared query-gate hypotheses fail on this parent; current single successor observations do not establish a causal explanation or sustained research benefit.'}
(A/'STATE18_DECISION_QUALITY_ANALYSIS.json').write_text(json.dumps(result,indent=2));print(json.dumps({'verified':verified,'selections':[(r['arm'],r['selected_producer'],r['selected_primitives']) for r in rows],'mechanism_minus_summary':result['mechanism_minus_summary'],'api_usd':result['api_usd'],'gpu_seconds':result['new_gpu_seconds']}))
