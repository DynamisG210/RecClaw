"""Prepare a source-bound BERT RO08 ablation; never launches training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--objective-source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ledger_path = a.raw / 'EVALUATION_LEDGER.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    rows = {r['available_index']: r for r in ledger['results']}
    combo = a.raw / 'executed/R07/recclaw_ext/models/bert4rec.py'
    assert sha(combo) == rows[7]['source_sha256']
    assert sha(a.objective_source) == rows[3]['source_sha256']
    source = combo.read_text(encoding='utf-8')
    old = 'slot_loss = -(0.9 * target_log_probs + 0.1 * mean_catalog_log_probs)'
    assert source.count(old) == 1
    score_only = source.replace(old, 'slot_loss = -target_log_probs')
    # Keep the whole scoring architecture and constructor unchanged, including
    # RNG consumption. Only the CE target distribution is intervened upon.
    ast.parse(score_only)
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / 'score_only/recclaw_ext/models/bert4rec.py'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(score_only, encoding='utf-8', newline='\n')
    for parent in (dest.parent, dest.parent.parent):
        (parent / '__init__.py').write_text('', encoding='utf-8')
    original_ast, new_ast = ast.parse(source), ast.parse(score_only)
    def constructor(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
    assert constructor(original_ast) == constructor(new_ast)
    observations = []
    for idx, r in rows.items():
        evidence = r['evidence']
        original = evidence.get('original_worker_result', evidence)
        observations.append(dict(index=idx, metric=r.get('primary_metric'), valid=r.get('valid'),
                                 source_sha256=r['source_sha256'], identity=original.get('identity'),
                                 wall_seconds=original.get('wall_time_secs')))
    base, objective, combination = [rows[i]['primary_metric'] for i in (0, 3, 7)]
    result = dict(schema='recrsi.algorithm-factorial-preparation.v1', ledger_sha256=sha(ledger_path),
        selection='Human-designated paper RO3 = raw ro08 round7; post-search selected case, not independent replication',
        partition='DEVELOPMENT_VALIDATION', observations=observations,
        cells={'root':base, 'objective_only':objective, 'score_only':None, 'combination':combination},
        differences={'objective_minus_root':objective-base, 'combination_minus_objective':combination-objective,
                     'combination_minus_root':combination-base, 'factorial_interaction':None},
        source_only_sha256=sha(dest), combination_sha256=sha(combo), objective_sha256=sha(a.objective_source),
        implementation='Only smoothed CE changed to hard CE; combination constructor and RNG consumption preserved',
        limits=['Missing score-only FIT; no interaction estimate is available',
                'Root result is legacy ABI-equivalence reuse, not a new matched training run',
                'Adding query_projection before apply(_init_weights) changes shared-parameter initialization under equal seed',
                'A complete historical four-cell comparison alone cannot establish seed-robust causal synergy'],
        new_fit=0, new_test=0, api_usd=0, training_launched=False)
    (a.out/'PREPARATION.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'cells':result['cells'], 'score_only_sha256':sha(dest), 'new_fit':0}))


if __name__ == '__main__':
    main()
