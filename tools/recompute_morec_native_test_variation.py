"""Recompute MoRec search variation from retained native TEST result files."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--ml1m-v3',type=Path,help='Explicit adopted ML1M RO execution-grounded-v3 report')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    snapshot=json.loads(a.snapshot.read_text(encoding='utf-8-sig'))
    replacement_hash=None
    if a.ml1m_v3:
        replacement=json.loads(a.ml1m_v3.read_text(encoding='utf-8-sig'))
        replacement_hash=hashlib.sha256(a.ml1m_v3.read_bytes()).hexdigest()
        snapshot['rows']=[r for r in snapshot['rows'] if not (r['dataset']=='ML1M' and r['system']=='RO')]+[r for r in replacement['rows'] if r['system']=='RO']
        snapshot['aggregate']=[r for r in snapshot['aggregate'] if not (r['dataset']=='ML1M' and r['system']=='RO')]+[dict(r,dataset='ML1M') for r in replacement['aggregate'] if r['system']=='RO']
    rows=[]
    for row in snapshot['rows']:
        b=(a.native/(row['result_sha256']+'.json')).read_bytes()
        native=json.loads(b)
        checks={'test_hash':hashlib.sha256(b).hexdigest()==row['result_sha256'],
                'metrics':native['metrics']==row['metrics'],
                'source':native['source_sha256']==row['source_sha256'],
                'checkpoint':native['checkpoint_sha256']==row['checkpoint_sha256']}
        rows.append({'dataset':row['dataset'],'method':row['system'],'repeat':row['repeat'],
                     'checks':checks,'metrics':native['metrics'],'native_path':row['result'],
                     'source_sha256':native['source_sha256'],'checkpoint_sha256':native['checkpoint_sha256'],
                     'training':native.get('training'),'cached':row['cached'],
                     'root_fallback':row['ROOTfallback']})
    stats=[]
    for group in snapshot['aggregate']:
        method=group['system']
        selected=[r for r in rows if r['dataset']==group['dataset'] and r['method']==method]
        for metric in ['ndcg@5','hr@5']:
            values=[r['metrics'][metric] for r in selected]
            avg,sd=statistics.mean(values),statistics.stdev(values)
            expected=group['metrics'][metric]
            stats.append({'dataset':group['dataset'],'method':method,'metric':metric,
                          'searches':len(values),'values':values,'mean':avg,'sample_sd':sd,
                          'matches_summary':abs(avg-expected['mean'])<1e-12 and abs(sd-expected['sample_sd'])<1e-12})
    result={'replacement_sha256':replacement_hash,'input_sha256':hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),'rows':rows,'statistics':stats,
            'new_api':0,'new_fit':0,'new_test':0,
            'scope':'Native TEST results match adopted report; source/checkpoint identities here are recorded hashes, not independent byte verification.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'native_results':len(rows),'checks_passed':sum(sum(r['checks'].values()) for r in rows),
                      'statistics_match':sum(r['matches_summary'] for r in stats),'statistics_total':len(stats)}))

if __name__=='__main__':
    main()
