"""Describe independent-search quality and resource variation in adopted RQ1 cohorts."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def summary(values):
    avg=statistics.mean(values)
    sd=statistics.stdev(values) if len(values)>1 else None
    return dict(n=len(values),mean=avg,sample_sd=sd,min=min(values),max=max(values),
                coefficient_of_variation=sd/abs(avg) if avg and sd is not None else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--searches',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.searches.read_text(encoding='utf-8-sig'))
    protocol=json.loads(a.protocol.read_text(encoding='utf-8-sig'))
    adopted=protocol['cohort']['bert_electronics_source_repeats']
    rows=[r for r in data['searches'] if r['group']!='BERT_Electronics' or r['repeat'] in adopted]
    identities=[(r['group'],r['method'],r['repeat']) for r in rows]
    assert len(identities)==len(set(identities)), 'Duplicate source search'
    assert all(abs((r['gpu_job_seconds']+r['gpu_root_reference_seconds'])/3600-r['gpu_hours'])<1e-8 for r in rows), 'GPU units/scope mismatch'
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['group'],r['method']].append(r)
    cells=[]
    for (group,method),rs in sorted(grouped.items()):
        cell=dict(group=group,method=method,repeats=[r['repeat'] for r in rs],
                  quality=summary([r['ndcg5'] for r in rs]),
                  reference_api_usd=summary([r['api_usd'] for r in rs]),
                  metered_job_gpu_seconds=summary([r['gpu_job_seconds'] for r in rs]),
                  allocated_root_gpu_seconds=summary([r['gpu_root_reference_seconds'] for r in rs]),
                  reference_search_gpu_hours=summary([r['gpu_hours'] for r in rs]),
                  reported_final_evaluations=[r['n_final_fits'] for r in rs],
                  unknown_api_events=sum(r.get('unknown_excluded',0) or 0 for r in rs),
                  excluded_remote_events=sum(r.get('remote_excluded',0) or 0 for r in rs),
                  unknown_gpu_jobs=[{'repeat':r['repeat'],'jobs':r.get('unknown_gpu_job_ids',[])} for r in rs if r.get('unknown_gpu_job_ids')])
        cells.append(cell)
    comparisons=[]
    for group in sorted({r['group'] for r in rows}):
        cs=[c for c in cells if c['group']==group];ro=next(c for c in cs if c['method']=='ro')
        best=max((c for c in cs if c['method']!='ro'),key=lambda c:c['quality']['mean'])
        comparisons.append(dict(group=group,best_comparator=best['method'],
                                ro_mean=ro['quality']['mean'],ro_sd=ro['quality']['sample_sd'],
                                comparator_mean=best['quality']['mean'],comparator_sd=best['quality']['sample_sd'],
                                mean_difference=ro['quality']['mean']-best['quality']['mean'],
                                independent_ro_searches_above_comparator_mean=sum(r['ndcg5']>best['quality']['mean'] for r in grouped[group,'ro']),
                                ro_reference_api_ratio=ro['reference_api_usd']['mean']/best['reference_api_usd']['mean'],
                                ro_reference_gpu_ratio=ro['reference_search_gpu_hours']['mean']/best['reference_search_gpu_hours']['mean']))
    result=dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.searches,a.protocol]},
                retained_searches=len(rows),cells=cells,comparisons=comparisons,
                source_cost_policy=data['cost_policy'],source_gpu_policy=data['gpu_policy'],
                interpretation='Descriptive search-level dispersion. No paired-seed or causal inference. Reference costs exclude the listed unknown/remote events; root allocation is not repeated physical training.',
                new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'searches':len(rows),'cells':len(cells),'comparisons':comparisons},indent=2))

if __name__=='__main__':
    main()
