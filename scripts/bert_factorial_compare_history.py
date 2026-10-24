"""Reproduce the historical R10 versus redundant confirmation comparison."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def compare(inputs):
    data = {name: path.read_bytes() for name, path in inputs.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    old, new = (json.loads(data[k]) for k in ('historical_result', 'confirmation_result'))
    for label, result in (('historical', old), ('confirmation', new)):
        if result['identity']['source_sha256'] != hashes[label+'_source']:
            raise ValueError(label+' result/source mismatch')
    identity_keys = ('worker_sha256', 'recipe_sha256', 'manifest_sha256', 'seed')
    if any(old['identity'][k] != new['identity'][k] for k in identity_keys):
        raise ValueError('Different execution identities')
    if old['candidate_universe_sha256'] != new['candidate_universe_sha256']:
        raise ValueError('Different candidate universes')
    trajectories = []
    for name in ('historical_epochs', 'confirmation_epochs'):
        rows = [json.loads(line) for line in data[name].splitlines() if line.strip()]
        indices = [row['epoch'] for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError('Epochs must be contiguous and unique')
        trajectories.append(rows)
    a, b = trajectories
    if len(a) != len(b) or len(a) != old['epochs_completed'] or len(b) != new['epochs_completed']:
        raise ValueError('Incomplete trajectory')
    aligned = [dict(epoch=x['epoch'], loss_difference=y['loss_sum']-x['loss_sum'],
                    metrics_equal=x['metrics']==y['metrics'], best_epoch_equal=x['best_epoch']==y['best_epoch'])
               for x, y in zip(a, b)]
    metrics = ('ndcg_at_5', 'ndcg_at_10', 'recall_at_5', 'recall_at_10', 'best_epoch', 'epochs_completed', 'status')
    sources = [data[k].decode('utf-8-sig') for k in ('historical_source', 'confirmation_source')]
    def methods(source):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
    ma, mb = map(methods, sources)
    diff = ''.join(difflib.unified_diff(sources[0].splitlines(True), sources[1].splitlines(True),
                                      fromfile='historical_R10', tofile='redundant_confirmation'))
    return dict(input_files={k:str(v) for k,v in inputs.items()}, input_sha256=hashes,
                historical_epochs=len(a), confirmation_epochs=len(b), aligned_epochs=aligned,
                all_losses_equal=all(row['loss_difference']==0 for row in aligned),
                all_metrics_identical=all(row['metrics_equal'] for row in aligned),
                all_best_epochs_equal=all(row['best_epoch_equal'] for row in aligned),
                final_result_equal={k:old[k]==new[k] for k in metrics},
                ast_equal_methods=[k for k in ma if ma[k]==mb.get(k)],
                ast_different_methods=[k for k in ma if ma[k]!=mb.get(k)],
                interpretation='Same-seed redundant confirmation; no additional independent sample',
                source_comparison='AST differences and full diff are evidence for manual semantic review, not a general equivalence proof'), diff


def main():
    parser=argparse.ArgumentParser()
    for name in ('historical-source','confirmation-source','historical-result','confirmation-result','historical-epochs','confirmation-epochs'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diff-out', type=Path, required=True)
    args=parser.parse_args()
    inputs={k:v for k,v in vars(args).items() if k not in ('out','diff_out')}
    report,diff=compare(inputs)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    args.diff_out.write_text(diff,encoding='utf-8')


if __name__ == '__main__':
    main()
