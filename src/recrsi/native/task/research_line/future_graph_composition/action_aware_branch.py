"""Prospective branch-exhaustion semantics; not an experiment launcher."""
from contextlib import contextmanager
import inspect
import textwrap
from unittest.mock import patch

VERSION = 'graph-action-aware-branch-exhaustion-v1'
LIMIT = 2
CONTROLS = frozenset(('MATCHED_CONTROL', 'MECHANISM_OFF'))


def operation(bound):
    action = bound.get('research_action')
    value = 'DIRECT_IMPROVEMENT' if action is None else action['operation']
    if value not in CONTROLS | {'DIRECT_IMPROVEMENT'}:
        raise ValueError('Unknown bound research operation')
    return value


def choose_route(ledger, decisions, parents, root_id, root_value):
    parent, best, failed_improvements, controls, branch = root_id, root_value, 0, 0, 0
    by_source = {p['entrypoint_source_sha256']: cid for cid, p in parents.items()}
    by_id = {d['candidate_id']: d for d in decisions}
    for row in sorted(ledger.get('results', []), key=lambda r: r['available_index']):
        if not row.get('genuine'):
            continue
        cid = by_source[row['source_sha256']]
        decision = by_id[cid]
        if decision['version'] != VERSION:
            raise ValueError('New policy cannot reinterpret old branch decisions')
        action = operation({'research_action': {'operation': decision['research_operation']}})
        if decision['mode'] == 'DISCOVER':
            branch += 1
            parent, best, failed_improvements, controls = cid, row['primary_metric'], 0, 0
        elif row['primary_metric'] > best:
            # A control may also discover a better model. Retain the measured
            # frontier regardless of its intended action.
            parent, best, failed_improvements, controls = cid, row['primary_metric'], 0, 0
        elif action in CONTROLS:
            controls += 1
        else:
            failed_improvements += 1
    if branch == 0:
        reason = 'NO_MEASURED_BRANCH'
    elif failed_improvements >= LIMIT:
        reason = 'IMPROVEMENT_ATTEMPTS_EXHAUSTED'
    elif controls >= LIMIT:
        reason = 'CONTROL_ALLOWANCE_EXHAUSTED'
    else:
        reason = None
    discover = reason is not None
    return dict(version=VERSION, mode='DISCOVER' if discover else 'REFINE',
        parent_id=root_id if discover else parent, branch_index=branch + int(discover),
        branch_best=best, nonimproving_evaluations=failed_improvements + controls,
        failed_improvement_attempts=failed_improvements,
        nonimproving_control_evaluations=controls, branch_exit_reason=reason,
        effective_evaluations=ledger.get('genuine_count', 0))


@contextmanager
def action_aware_scope(branch_module):
    """Enter before original independent_scope in a fresh version only.

    Capture the action from the already validated native binding in the same
    decision write, not from free-form proposal narratives or a later sidecar.
    """
    source = textwrap.dedent(inspect.getsource(branch_module.adapter_class))
    replacements = {
        'bound = super()._bind_definition(spec, definition)':
            'bound = super()._bind_definition(spec, definition)\n            research_operation = operation(bound)',
        "if prior['parent_id'] != spec.closest_parent or prior['spec_digest'] != spec.digest:":
            "if (prior['parent_id'] != spec.closest_parent or prior['spec_digest'] != spec.digest\n                    or prior.get('version') != ACTION_POLICY_VERSION\n                    or prior.get('research_operation') != research_operation):",
        "route['spec_digest'] = spec.digest":
            "route['spec_digest'] = spec.digest\n                route['research_operation'] = research_operation",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Review changed native branch decision binding')
        source = source.replace(old, new)
    namespace = dict(branch_module.__dict__, choose_route=choose_route,
        operation=operation, ACTION_POLICY_VERSION=VERSION)
    exec(compile(source, __file__, 'exec'), namespace)
    with patch.object(branch_module, 'choose_route', choose_route), \
         patch.object(branch_module, 'adapter_class', namespace['adapter_class']):
        yield
