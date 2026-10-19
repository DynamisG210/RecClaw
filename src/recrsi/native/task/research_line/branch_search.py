"""Prospective branch allocation on the existing program/native campaign.

No launch or paid permission. Trusted genuine outcomes advance the experiment
quota. A branch keeps its best parent; two measured non-improving descendants
return construction to root. Global best and all outcomes remain archived.
"""
import json
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

VERSION = 'ngcf-independent-discovery-v1'
TARGET = 12
ATTEMPT_LIMIT = 36  # Incomplete if exhausted; never relabel as 12 evaluations.
STALL_LIMIT = 2


def choose_route(ledger, decisions, parents, root_id, root_value):
    """Replay only completed genuine evaluations, never failed proposals."""
    parent, best, stalled, branch = root_id, root_value, 0, 0
    by_source = {p['entrypoint_source_sha256']: cid for cid, p in parents.items()}
    decisions = {d['candidate_id']: d for d in decisions}
    for row in sorted(ledger.get('results', []), key=lambda r: r['available_index']):
        if not row.get('genuine'):
            continue
        cid = by_source[row['source_sha256']]
        decision = decisions[cid]
        if decision['mode'] == 'DISCOVER':
            branch += 1
            parent, best, stalled = cid, row['primary_metric'], 0
        elif row['primary_metric'] > best:
            parent, best, stalled = cid, row['primary_metric'], 0
        else:
            stalled += 1
    discover = branch == 0 or stalled >= STALL_LIMIT
    return dict(version=VERSION, mode='DISCOVER' if discover else 'REFINE',
                parent_id=root_id if discover else parent,
                branch_index=branch + int(discover),
                branch_best=best, nonimproving_evaluations=stalled,
                effective_evaluations=ledger.get('genuine_count', 0))


def remove_program_contents(value):
    """Discovery keeps observed outcomes but not incumbent source priming.

    Root code is added separately by the task adapter. This is an explicit
    prospective information intervention, not a claim that history is absent.
    """
    if isinstance(value, dict):
        return {k: remove_program_contents(v) for k, v in value.items()
                if k not in ('implementation', 'program_source', 'source_regions')}
    if isinstance(value, list):
        return [remove_program_contents(v) for v in value]
    return value


def adapter_class(base):
    class BranchAdapter(base):
        def _branch_route(self):
            path = Path(self.config['output']) / 'EVALUATION_LEDGER.json'
            ledger = json.loads(path.read_bytes()) if path.exists() else {}
            directory = Path(self.config['output']) / 'branch_decisions'
            records = [json.loads(p.read_bytes()) for p in directory.glob('*.json')]
            roots = [r for r in ledger.get('results', []) if r.get('root') and r.get('valid')]
            root_value = roots[0]['primary_metric'] if roots else None
            return choose_route(ledger, records, self._parents(), self.parent_id, root_value)

        def provider_context(self, context):
            value = super().provider_context(context)
            route = self._branch_route()
            task = value['knowledge_base']['search_space']
            task['branch_allocation'] = route
            task['construction'] = (
                'Construct one complete program from the designated construction parent. '
                'DISCOVER starts an autonomous full computation hypothesis from root; '
                'REFINE improves this measured branch. All architecture and objective '
                'changes remain available. Past failed combinations do not prohibit '
                'other combinations. No opponent program is provided. Global best is '
                'retained by the evaluator independently of this construction choice.')
            task['research_actions']['parents'] = [p for p in task['research_actions']['parents']
                                                   if p['candidate_id'] == route['parent_id']]
            # Supply the actual chosen branch source, including a branch below
            # the global best; do not implicitly substitute the global winner.
            task['branch_construction_parent'] = dict(
                candidate_id=route['parent_id'], source=self._parent_source(route['parent_id']))
            return value

        def resolve_confirmation(self, kind, primary_binding, context):
            if context.get('phase') == 'PROJECT_PROVIDER_PROPOSAL':
                route = self._branch_route()
                if primary_binding.get('parent_ref') != route['parent_id']:
                    return self._unsupported('BRANCH_PARENT_MISMATCH')
                self._current_branch_route = route
            return super().resolve_confirmation(kind, primary_binding, context)

        def _bind_definition(self, spec, definition):
            bound = super()._bind_definition(spec, definition)
            directory = Path(self.config['output']) / 'branch_decisions'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (spec.digest + '.json')
            if path.exists():
                # A prepared/replayed definition must retain its allocation.
                prior = json.loads(path.read_bytes())
                if prior['parent_id'] != spec.closest_parent or prior['spec_digest'] != spec.digest:
                    raise ValueError('Prepared branch allocation changed')
            else:
                route = dict(self._current_branch_route)
                route['candidate_id'] = 'cand-graph-' + spec.digest[:24]
                route['spec_digest'] = spec.digest
                path.write_text(json.dumps(route, sort_keys=True) + '\n', encoding='utf8')
            return bound
    return BranchAdapter


@contextmanager
def branch_scope():
    """Install before native composition; transport/worker recovery stays native."""
    import program_adapter
    import sequential_program
    from recclaw_core.research_line import standalone
    native_run = standalone.StandaloneResearchComposition.run
    native_projection = program_adapter.program_projection

    def projection(context):
        # Native evidence archival consumes the complete execution first. Only
        # the final provider view changes; never strip the archival interface.
        value = native_projection(context)
        task = value['knowledge_base']['search_space']
        route = task['branch_allocation']
        if route['mode'] == 'DISCOVER':
            value = remove_program_contents(value)
        value['knowledge_base']['search_space']['construction'] = (
            context['knowledge_base']['search_space']['construction'])
        return value

    def until_effective(composition, round_count=None):
        adapter = composition.campaign.search_space_adapter
        ledger_path = Path(adapter.config['output']) / 'EVALUATION_LEDGER.json'
        results = []
        while True:
            ledger = json.loads(ledger_path.read_bytes()) if ledger_path.exists() else {}
            if ledger.get('genuine_count', 0) >= TARGET:
                break
            before = composition.campaign.state.next_round_index
            if before > ATTEMPT_LIMIT:
                break  # Persist explicit shortfall below; never report complete.
            batch = native_run(composition, round_count=1)
            results.extend(batch)
            if composition.campaign.state.next_round_index == before:
                return tuple(results)  # Native unfinished boundary, no blind retry.
        status = dict(version=VERSION, target_effective_evaluations=TARGET,
                      effective_evaluations=ledger.get('genuine_count', 0),
                      status='COMPLETE' if ledger.get('genuine_count', 0) == TARGET
                      else 'INCOMPLETE_ATTEMPT_LIMIT',
                      attempts_closed=composition.campaign.state.next_round_index - 1)
        (ledger_path.parent / 'EFFECTIVE_SEARCH_STATUS.json').write_text(
            json.dumps(status, indent=2) + '\n', encoding='utf8')
        if status['status'] != 'COMPLETE':
            raise RuntimeError('INCOMPLETE_EFFECTIVE_SEARCH: ' + json.dumps(status))
        return tuple(results)

    with ExitStack() as stack:
        stack.enter_context(patch.object(program_adapter, 'ProgramAdapter',
                                         adapter_class(program_adapter.ProgramAdapter)))
        stack.enter_context(sequential_program.sequential_scope(
            schedule=sequential_program.ROLES * (ATTEMPT_LIMIT // 4)))
        stack.enter_context(patch.object(program_adapter, 'program_projection', projection))
        stack.enter_context(patch.object(standalone.StandaloneResearchComposition,
                                         'run', until_effective))
        yield


def branch_config(config):
    """Prospective caller must also declare matching paid/physical resource caps."""
    return replace(config, round_count=ATTEMPT_LIMIT,
                   observation_seed_schedule=(config.seed,) * ATTEMPT_LIMIT,
                   max_attempts_per_round=1, close_exhausted_no_metric_slot=True,
                   prebinding_token_ceiling_retry=False)
