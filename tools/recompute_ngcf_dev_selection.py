"""Recompute frozen NGCF DEV winner selection, preserving historical-pin differences."""
import argparse,json,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for key in ['freezes','ledgers','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();freezes=json.loads(a.freezes.read_text())['rows'];ledgers={r['path']:r for r in json.loads(a.ledgers.read_text())};rows=[]
    for f in freezes:
        d=f['data'];path=next(k for k in d['original_pins'] if k.endswith('/EVALUATION_LEDGER.json'));l=ledgers[path]
        eligible=[r for r in l['data']['results'] if r.get('valid') and r.get('admitted')]
        checks=[r['evidence'].get('primary_metric')=='NDCG@5' and r['evidence'].get('test_accessed') is False and r['primary_metric']==r['evidence']['metrics']['ndcg@5'] for r in eligible]
        assert eligible and all(checks),(f['method'],f['repeat'])
        winner=max(eligible,key=lambda r:r['primary_metric'])
        source=d.get('source_sha256',d.get('winner_source_sha256'))
        rows.append(dict(method=f['method'],repeat=f['repeat'],eligible=len(eligible),all_eligible_dev_only=all(checks),selected_source=winner['source_sha256'],source_matches=winner['source_sha256']==source,value=winner['primary_metric'],value_matches=winner['primary_metric']==d['selection_value'],freeze_bytes_match=f['byte_match'],ledger_bytes_match=l['actual']==l['expected']))
    result=dict(rows=rows,inputs={k:hashlib.sha256(getattr(a,k).read_bytes()).hexdigest() for k in ['freezes','ledgers']},rule='Native select_winner: valid and admitted; NDCG@5 and test_accessed false; stable first maximum. Live matching winner is not historical byte equality.',new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
