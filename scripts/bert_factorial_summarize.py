"""Recompute four-cell DEV contrasts without refitting or selecting outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(preparation, score_result=None):
    cells = dict(preparation['cells'])
    if score_result is not None:
        expected = next(row['identity'] for row in preparation['observations'] if row['index'] == 7)
        identity = score_result['identity']
        for key in ('manifest_sha256', 'recipe_sha256', 'worker_sha256', 'seed'):
            if identity[key] != expected[key]:
                raise ValueError('Incompatible score-only result: ' + key)
        if identity['source_sha256'] != next(row['source_sha256'] for row in preparation['observations'] if row['index'] == 10):
            raise ValueError('Score-only source mismatch')
        if score_result['status'] != 'ok':
            raise ValueError('Unsuccessful result must be diagnosed, not used as a metric')
        cells['score_only'] = score_result['ndcg_at_5']
    root, objective, score, combo = (cells[k] for k in ('root', 'objective_only', 'score_only', 'combination'))
    effects = dict(objective_without_projection=objective-root,
                   projection_with_smoothing=combo-objective,
                   combination_vs_root=combo-root)
    if score is not None:
        effects.update(projection_without_smoothing=score-root,
                       objective_with_projection=combo-score,
                       descriptive_interaction=(combo-score)-(objective-root),
                       mean_projection_contrast=((score-root)+(combo-objective))/2,
                       mean_objective_contrast=((objective-root)+(combo-score))/2)
    return dict(partition='DEVELOPMENT_VALIDATION', cells=cells, contrasts=effects,
                complete=score is not None,
                comparison_type='HISTORICAL_FOUR_CELL_DESCRIPTIVE',
                relative_combination_gain=(combo/root-1),
                hypotheses=dict(composition_exceeds_root=combo>root,
                    composition_exceeds_both_single_changes=None if score is None else combo>max(score,objective),
                    positive_descriptive_interaction=None if score is None else effects['descriptive_interaction']>0),
                history=preparation['observations'],
                provenance=[x for x in preparation['limits'] if not x.startswith('Missing score-only')],
                score_only_origin='HISTORICAL_RO08_ROUND10',
                score_only_result=score_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--score-result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths=[args.preparation]+([args.score_result] if args.score_result else [])
    report=summarize(json.loads(args.preparation.read_text(encoding='utf-8-sig')),
                     json.loads(args.score_result.read_text(encoding='utf-8-sig')) if args.score_result else None)
    report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
