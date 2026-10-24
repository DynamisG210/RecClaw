"""Describe original NGCF DEV discovery trajectories without treating DEV as TEST."""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ledgers', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for record in json.loads(a.ledgers.read_text()):
        d = record['data']
        identity = d['contract']['identity']
        eligible = [r for r in d['results'] if r.get('valid') and r.get('admitted')]
        roots = [r for r in eligible if r.get('root')]
        assert len(roots) == 1
        root = roots[0]['primary_metric']
        candidates = [r for r in eligible if not r.get('root')]
        assert len({r['source_sha256'] for r in candidates}) == len(candidates)
        frontier = root
        curve = []
        for r in sorted(candidates, key=lambda r: r['available_index']):
            assert r['evidence']['test_accessed'] is False
            value = r['primary_metric']
            improved = value > frontier
            frontier = max(frontier, value)
            curve.append(dict(source=r['source_sha256'], available_index=r['available_index'],
                              ndcg5=value, gain_over_root=value-root,
                              frontier=frontier, frontier_improvement=improved))
        rows.append(dict(identity=identity, ledger=record['path'],
                         historical_pin_matches=record['expected']==record['actual'],
                         root_ndcg5=root, eligible_candidates=len(candidates),
                         above_root=sum(r['gain_over_root']>0 for r in curve),
                         frontier_updates=sum(r['frontier_improvement'] for r in curve),
                         final_gain=frontier-root, curve=curve))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['identity']['method']].append(r)
    summary = {m: dict(searches=len(rs), mean_final_gain=statistics.mean(r['final_gain'] for r in rs),
                       sd_final_gain=statistics.stdev(r['final_gain'] for r in rs),
                       above_root=sum(r['above_root'] for r in rs),
                       eligible_candidates=sum(r['eligible_candidates'] for r in rs),
                       frontier_updates=sum(r['frontier_updates'] for r in rs))
               for m, rs in grouped.items()}
    result = dict(rows=rows, summary=summary,
                  input_sha256=hashlib.sha256(a.ledgers.read_bytes()).hexdigest(),
                  scope='Original 18 ML1M searches; DEV only. Eligible unique candidate records, not all paid attempts. SE1/SE3 current ledgers differ from historical pins. No causal or TEST superiority inference. Replacement RO1 excluded.',
                  new_api=0, new_fit=0, new_test=0)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
