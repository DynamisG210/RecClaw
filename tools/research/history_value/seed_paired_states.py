"""Create isolated native continuation states from trusted historical checkpoints.

No campaign is launched. Pickles returned on stdout are for the original runtime.
The runner must separately bind fresh finite resource ledgers and physical locks.
"""
import argparse
import base64
import copy
import dataclasses
import hashlib
import json
import pickle
import sys
from pathlib import Path


def seed(home, campaign_prefix='history-value-b01-after12'):
    cfg = json.loads((home / 'EVALUATOR_CONFIG.json').read_text())
    sys.path[:0] = [str(home.parent), str(home.parent / 'runtime'),
                   str(Path(cfg['task_root']) / 'controller_adapter'),
                   *json.loads((home.parent / 'ORIGINAL_RUNTIME_PATHS.json').read_text())]
    campaign = home / 'search/campaign'
    old = pickle.loads((campaign / 'ROUND_13_CHECKPOINT.pkl').read_bytes()).state_before
    fresh = pickle.loads((campaign / 'ROUND_01_CHECKPOINT.pkl').read_bytes()).state_before
    frontier = {k: copy.deepcopy(v) for k, v in old.frontier.items()
                if k.startswith('incumbent_') or k == 'lineage_parent_binding'}
    index = frontier['incumbent_round_index']
    keys = [k for k, p in old.candidate_root_by_capability.items()
            if f'round-{index:04d}' in Path(p).parts]
    if len(keys) != 1:
        raise ValueError('Incumbent capability is ambiguous')
    key = keys[0]
    # Both arms receive the same pre-history policy, not a learned router from
    # one arm. Research knowledge is supplied only through memory state.
    policy = fresh.policy
    policy_view = copy.deepcopy(fresh.context.policy)
    budget = dict(old.context.budget, campaign_round_bound=15,
                  physical_attempts_per_logical_call_max=1,
                  implementer_logical_calls_max=1,
                  implementer_output_token_ceiling_each=20000)
    result = {'stage': 'ISOLATED_NATIVE_STATES_PREPARED_NOT_LAUNCHED',
              'paid_calls': 0, 'candidate_fits': 0, 'arms': {},
              'removed_per_arm': {'carryover_open_candidates': len(old.carryover_open_candidates),
                                  'deferred_innovation_backlog': len(old.deferred_innovation_backlog)},
              'shared_policy': dataclasses.asdict(policy)}
    states = {}
    for arm in ('retain', 'reset'):
        identity = campaign_prefix + '-' + arm
        memory = (copy.deepcopy(old.context.scientific_memory) if arm == 'retain'
                  else {'global_memory': {}, 'by_role': {}})
        context = dataclasses.replace(old.context, campaign_id=identity,
            frontier=copy.deepcopy(frontier), scientific_memory=memory,
            policy=copy.deepcopy(policy_view), budget=copy.deepcopy(budget),
            unresolved_questions=())
        state = dataclasses.replace(old, campaign_id=identity, context=context,
            frontier=copy.deepcopy(frontier), policy=policy,
            active_profile=dataclasses.replace(old.active_profile, campaign_id=identity),
            search_memory_head=(old.search_memory_head if arm == 'retain' else fresh.search_memory_head),
            carryover_proposals=(), carryover_open_candidates=(),
            qualified_execution_by_capability={key: copy.deepcopy(old.qualified_execution_by_capability[key])},
            candidate_root_by_capability={key: old.candidate_root_by_capability[key]},
            resource_profile_by_capability={}, deferred_innovation_backlog=(),
            last_round_result_digest=None)
        states[arm] = state
        raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        result['arms'][arm] = {'state_pickle_base64': base64.b64encode(raw).decode(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'next_round': state.next_round_index,
            'queued_candidates': len(state.carryover_open_candidates),
            'backlog': len(state.deferred_innovation_backlog),
            'parent_roots': dict(state.candidate_root_by_capability)}
    for field in ('frontier', 'policy', 'incumbent_observation',
                  'qualified_execution_by_capability', 'candidate_root_by_capability'):
        if getattr(states['retain'], field) != getattr(states['reset'], field):
            raise ValueError('Common state field differs: ' + field)
    result['common_native_fields_equal'] = True
    result['remaining_runner_binding'] = ['new allocation ledgers', 'shared physical admission',
                                         'new route config', 'native campaign state persistence']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    parser.add_argument('--campaign-prefix', default='history-value-b01-after12')
    args = parser.parse_args()
    print(json.dumps(seed(args.home, args.campaign_prefix)))
