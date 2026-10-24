"""Compare recorded NGCF original/replacement cohorts without adopting a winner."""
import argparse
import datetime
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); hashes={}
    def read(name):
        raw=(a.evidence/name).read_bytes(); hashes[name]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    current=read('CURRENT_RESULTS.json')
    cds=read('CDS_SIX_SYSTEM_SELECTED_REPORTING_20261014.json')
    office=read('OFFICE_THREE_SEED_COMPARISON_20261014.json')
    ml=read('ML_COMPARISON.json'); el=read('EL_COMPARISON.json')
    cells=[]
    for dataset,new in [('cds',cds),('office',office)]:
        rows=[r for r in current['test_results'] if r['slot'].startswith(dataset+'-ro-')]
        assert len(rows)==9
        metrics={}
        for metric in ['ndcg@5','hit@5']:
            old=statistics.mean(r['metrics'][metric] for r in rows)
            remaining=[r['metrics'][metric] for r in rows if r['slot'].split('-')[2]!='01']
            replacement=(sum(remaining)+new['mean'][metric]*3)/9
            metrics[metric]={'original':old,'uniform_replacement':replacement,'change':replacement-old}
        cells.append({'dataset':dataset,'metrics':metrics,'new_ro1_refits':3})
    cells.append({'dataset':'ml1m','metrics':{m:{'original':ml['old_systems']['ro'][m],'uniform_replacement':ml['replacement_RO'][m],'change':ml['replacement_RO'][m]-ml['old_systems']['ro'][m]} for m in ['ndcg@5','hit@5']},'new_ro1_refits':3})
    cells.append({'dataset':'electronics','uniform_replacement':None,'reason':'New source transfer has one seed and batch1024; old transfer batch128. Independent EL searches are a different cohort. No matched three-search replacement calculable.','single_seed_descriptive':el})
    timeline=[]
    for name in ['SEARCH_HUMAN.json','SEARCH_PLAN.json','DIRECT_HUMAN.json','CDS_COMPARISON.json','OFFICE_DIRECT_HUMAN_20261014.json','OFFICE_USER_SEED_CONFIRMED_20261014.json','CDS_SIX_SYSTEM_SELECTED_REPORTING_20261014.json','OFFICE_THREE_SEED_COMPARISON_20261014.json','ML_COMPARISON.json']:
        d=read(name); when=d.get('utc')
        if not when and d.get('created_ns'):when=datetime.datetime.fromtimestamp(d['created_ns']/1e9,datetime.timezone.utc).isoformat()
        timeline.append({'file':name,'recorded_utc':when,'selection':d.get('selection'),'request':d.get('request',d.get('direct_human_correction')),'status':d.get('status'),'interpretation':d.get('interpretation',d.get('caveat'))})
    recovery={}
    for task in ['cds','office']:
        d=read(task+'_VERIFICATION_OUTCOME.json')
        recovery[task]={k:d[k] for k in ['checkpoint_verification','limitation','original_training_exit_code','new_training_epochs','new_validation_evaluations','test_accessed']}
    result={'cells':cells,'timeline':timeline,'recovery':recovery,'input_hashes':hashes,'conclusion':'No examined pre-TEST rule explains selective CDS replacement and Office retention based on recovery class. Both use EXACT_STATE_GRAPH_AND_CPU_REPLAY with the same original-CUDA limitation. This is absence of evidence in inspected records, not proof of deliberate TEST selection. Uniform replacement is sensitivity analysis, not adopted cohort or three fresh independent searches. Electronics is noncomparable and not pooled.','new_api':0,'new_fit':0,'new_test':0}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(cells[:3],indent=2))


if __name__=='__main__':main()
