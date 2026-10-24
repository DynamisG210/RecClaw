"""Exercise paired history intervention through the recorded native projection.

Read-only; stdout contains preparation evidence. No provider or FIT is called.
Retain keeps the entire native scientific_memory before normal role projection.
Reset starts empty. Both discard pending queues as executable controller state.
"""
import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path


def run(home):
    paths = json.loads((home.parent / 'ORIGINAL_RUNTIME_PATHS.json').read_text())
    cfg = json.loads((home / 'EVALUATOR_CONFIG.json').read_text())
    sys.path[:0] = [str(home.parent), str(home.parent / 'runtime'),
                   str(Path(cfg['task_root']) / 'controller_adapter'), *paths]
    from recclaw_core.research_line.interfaces import project_provider_context_view
    original = pickle.loads((home / 'search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes())
    context = original.state_before.context
    # Only current incumbent and executable parent binding are common facts.
    frontier = {k: v for k, v in context.frontier.items()
                if k.startswith('incumbent_') or k == 'lineage_parent_binding'}
    policy = {k: context.policy[k] for k in
              ('producer_token_allocation', 'mechanism_axis_targeting',
               'memory_retrieval_policy') if k in context.policy}
    goal = dict(context.frozen_goal)
    goal['research_objective'] = 'Improve the supplied MoRec/CDs incumbent using mechanism hypotheses.'
    results = {}
    for arm in ('retain', 'reset'):
        # Empty V2 containers preserve the same native projection path. A bare
        # {} selects the legacy fallback, confounding history with prompt schema.
        memory = (context.scientific_memory if arm == 'retain' else
                  {'global_memory': {}, 'by_role': {}})
        paired = dataclasses.replace(context, campaign_id='history-value-paired-probe',
            frontier=frontier, scientific_memory=memory, unresolved_questions={},
            policy=policy, frozen_goal=goal)
        results[arm] = {role: project_provider_context_view(paired.producer_view(role))
                       for role in ('mechanism_composer', 'lineage_refiner',
                                    'falsification_designer', 'frontier_architect')}
    coverage = {}
    for role, view in results['retain'].items():
        # latest_result is historical evidence, not part of the common incumbent.
        for arm in ('retain', 'reset'):
            projected = results[arm][role]
            latest = projected['state'].pop('latest_result', None)
            if latest is not None:
                projected['memory']['latest_result'] = latest
        recent = view['memory']['recent_experiments']
        index = recent['attempt_fields'].index('round')
        rounds = [row[index] for row in recent['attempts']]
        if rounds != list(range(1, 13)):
            raise ValueError(f'Historical opportunity omitted for {role}: {rounds}')
        reset_recent = results['reset'][role]['memory'].get('recent_experiments', {})
        if reset_recent.get('attempts'):
            raise ValueError('Reset contains prior experiment rows')
        for field in ('schema', 'objective', 'state', 'contract'):
            if view[field] != results['reset'][role][field]:
                raise ValueError(f'Common projected field changed: {role}/{field}')
        coverage[role] = rounds
    return {'stage': 'PAIRED_NATIVE_PROJECTION_EXECUTED_NOT_PROVIDER_ADOPTED',
            'paid_calls': 0, 'candidate_fits': 0, 'views': results,
            'retained_opportunities': coverage,
            'notes': ['Uses original native role projection; not all raw storage fields are prompt-visible.',
                      'Full incumbent source attachment and fresh finite budget still require runner binding.',
                      'This probe does not clone or launch a campaign.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    print(json.dumps(run(parser.parse_args().home), ensure_ascii=False))
