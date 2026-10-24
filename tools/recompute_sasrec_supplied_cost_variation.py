"""Recompute supplied SASRec cost variability without inferring TEST or token rows."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.costs.open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['group'] == 'SASRec_CDs']
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    cells = []
    for ref in reference['sasrec_cells']:
        rs = [r for r in rows if r['method'] == ref['method']]
        assert len(rs) == 3 and {r['repeat'] for r in rs} == {'R1', 'R2', 'R3'}
        metrics = {}
        for key in ['api_usd', 'gpu_hours']:
            values = [float(r[key]) for r in rs]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            assert math.isclose(mean, ref[key], rel_tol=1e-12), (ref['method'], key)
            metrics[key] = dict(mean=mean, sample_sd=sd, cv=sd/mean,
                                minimum=min(values), maximum=max(values), values=values,
                                reference_mean_matches=True)
        cells.append(dict(method=ref['method'], repeats=[r['repeat'] for r in rs], metrics=metrics))
    result = dict(source_sha256=hashlib.sha256(args.costs.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  supplied_rows=rows, cells=cells,
                  scope='Supplied SASRec/CDs cost records only. GPU entries are fit time, not uniformly scoped total search cost. Native ledger, UNKNOWN charges, final-fit/TEST scope and search-to-TEST identities remain unresolved. Missing tokens are not zero. No quality-cost pairing inferred from method means.',
                  new_api=0, new_fit=0, new_test=0)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
