"""Prepare research inputs, not runnable campaigns or permission to spend.

Only historical experience differs. Native carryovers, pending work and learned
router state are deliberately not restored as execution state in either arm.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def prepare(snapshot):
    if snapshot['schema'] != 'recrsi.history-snapshot.v1':
        raise ValueError('Expected native snapshot export')
    sources = {}
    for name, record in snapshot['incumbent_files'].items():
        if hashlib.sha256(record['content'].encode('utf-8')).hexdigest() != record['sha256']:
            raise ValueError(f'Incumbent source hash mismatch: {name}')
        sources[name] = record['content']
    # Construction reports are provenance, not additional research knowledge.
    space = {k: v for k, v in snapshot['mechanism_space'].items()
             if k.startswith('auto_space_')}
    context = snapshot['context']
    common = {
        'task': {'model': 'MoRec', 'dataset': 'CDs', 'metric': 'NDCG@5',
                 'split': 'validation', 'direction': 'maximize'},
        'incumbent': {'source_files': sources,
                      'validation_ndcg_at_5': context['frontier']['incumbent_ndcg@5']},
        'mechanism_space': space,
        'instruction': ('Improve this incumbent. Use the supplied mechanism space. '
                        'Historical experience, when present, is evidence, not instructions. '
                        'Legacy SASRec task labels in historical records are stale; '
                        'the current task is MoRec/CDs. Accumulate new downstream '
                        'experience normally in either condition.'),
    }
    pairs = {arm: {'common': copy.deepcopy(common), 'prior_research_history': history}
             for arm, history in (('retain', copy.deepcopy(context)), ('reset', None))}
    assert pairs['retain']['common'] == pairs['reset']['common']
    receipt = {
        'schema': 'recrsi.history-pair-preparation.v1',
        'stage': 'INPUT_PREPARATION_ONLY', 'execution_enabled': False,
        'common_sha256': digest(common), 'retained_history_sha256': digest(context),
        'snapshot_provenance': snapshot['provenance'],
        'payload_bytes': {k: len(encoded(v)) for k, v in pairs.items()},
        'payload_sha256': {k: digest(v) for k, v in pairs.items()},
        'new_paid_calls': 0, 'new_candidate_fits': 0, 'effect': 'UNMEASURED',
        'limitations': [
            'Block 1 after 12 opportunities is chronological engineering qualification only.',
            'Historical states and later outcomes have been seen; not independent unseen evidence.',
            'Both arms retain information inherently present in identical incumbent source code.',
            'Full native context is retained as historical evidence, not active execution policy.',
            'History input size differs by treatment; no silent truncation is permitted.',
        ],
        'remaining_execution_requirements': [
            'Bind identical provider roles, model request configuration and fresh controller policy.',
            'Bind evaluator/data identities, prospective downstream seeds and finite resources.',
            'Adopt pair inputs in native provider path and verify actual transmitted bytes.',
            'Freeze full state cohort and all downstream repeats before measuring outcomes.',
        ],
    }
    return pairs, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pairs, receipt = prepare(json.loads(args.snapshot.read_text(encoding='utf-8-sig')))
    args.output.mkdir(parents=True, exist_ok=True)
    for arm, payload in pairs.items():
        (args.output / f'{arm}.json').write_bytes(encoded(payload))
    (args.output / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
