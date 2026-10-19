"""Unreleased single-program opportunities on the existing native campaign.

The schedule is chosen before any proposal. Native campaign checkpoints, outcome
interpretation, source handoff and physical accounting still own each round.
This module neither runs a campaign nor grants an execution release.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from unittest.mock import patch

VERSION = 'ngcf-sequential-program-v1'
ROLES = ('mechanism_composer', 'lineage_refiner',
         'falsification_designer', 'frontier_architect')
SCHEDULE = ROLES * 3


@contextmanager
def sequential_scope(schedule=SCHEDULE):
    from recclaw_core.research_line import interfaces, producers, runtime, standalone, interpreter, provider
    from program_adapter import program_scope

    schedule = tuple(schedule)
    if not schedule or any(role not in ROLES for role in schedule):
        raise ValueError('Freeze a nonempty schedule of original Research roles')
    original_context_init = interfaces.ResearchContext.__post_init__
    original_roles = interfaces.research_producer_roles
    original_produce = runtime.produce_research_specs
    original_factory = standalone._round_input_factory

    def metadata(index):
        return dict(research_mode=VERSION, sequential_role_schedule=list(schedule),
                    sequential_opportunity=index,
                    producer_logical_calls=int(index <= len(schedule)))

    def initialize(context):
        budget = dict(context.budget)
        prior = budget.get('sequential_role_schedule')
        if prior is not None and tuple(prior) != schedule:
            raise ValueError('Cannot resume with a different role schedule')
        # dataclasses.replace is also the native next-round transition. Bind the
        # new role to that context identity; failed/replayed rounds keep theirs.
        budget.update(metadata(context.round_index))
        object.__setattr__(context, 'budget', budget)
        original_context_init(context)

    def roles(budget):
        if budget.get('research_mode') != VERSION:
            return original_roles(budget)
        if tuple(budget['sequential_role_schedule']) != schedule:
            raise ValueError('Prepared opportunity schedule drift')
        index = budget['sequential_opportunity']
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(schedule)+1:
            raise ValueError('No scheduled generation opportunity remains')
        if index == len(schedule)+1:
            return ()  # Native terminal successor still computes its identity.
        return (schedule[index - 1],)

    def produce(context, producer, bindings, **kwargs):
        expected = roles(context.budget)
        if not expected:
            raise ValueError('All frozen generation opportunities consumed')
        outcomes = original_produce(context, producer, bindings, **kwargs)
        if tuple(item.producer_role for item in outcomes) != expected:
            raise ValueError('Native generation changed the frozen role')
        # Native recovery derives scheduled roles from outcome provenance,
        # including rejected proposals and failed HTTP attempts.
        return tuple(replace(item, provenance={**dict(item.provenance),
                           **metadata(context.round_index)}) for item in outcomes)

    def factory(*args, **kwargs):
        native = original_factory(*args, **kwargs)
        def inputs(state):
            value = native(state)
            return replace(value, meta_research_inputs=None, budget_snapshot={**dict(value.budget_snapshot),
                           **metadata(state.context.round_index)})
        return inputs

    with ExitStack() as stack:
        stack.enter_context(program_scope())
        stack.enter_context(patch.object(interfaces.ResearchContext, '__post_init__', initialize))
        for module in (interfaces, producers, runtime, standalone, interpreter, provider):
            stack.enter_context(patch.object(module, 'research_producer_roles', roles))
        stack.enter_context(patch.object(runtime, 'produce_research_specs', produce))
        stack.enter_context(patch.object(standalone, '_round_input_factory', factory))
        yield
