"""Fixed explicit policy, with full outcome-conditioned scientific memory.

Portable form of the measured FrozenSearchPolicy condition. Like that condition,
this scoped composition runs one campaign per process: runtime hooks are restored
on exit, but must not overlap another campaign in the same process.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import fields, replace

from . import runtime


@contextmanager
def fixed_search_policy(composition):
    """Freeze four explicit policy consumers, not the agent's research context."""
    policy_type = type(composition.policy)
    sealed_policy = composition.manifest["policy"]["value"]
    initial_policy = policy_type(**{
        field.name: sealed_policy[field.name]
        for field in fields(policy_type)
        if field.init and field.name in sealed_policy
    })
    adapter = composition.campaign.search_space_adapter
    provider = composition.provider._producer
    native_call = provider._call
    native_acquire = runtime._acquire_innovation_spec
    native_ranking = runtime._search_ranking_inputs
    native_post_round = composition.campaign.post_round_state_transition

    def research_call(producer_role, context_view, **kwargs):
        projected = adapter.provider_context(context_view)
        projected["policy"] = initial_policy.to_dict()
        projected["producer_token_fraction"] = dict(
            initial_policy.producer_token_allocation
        )[context_view["producer_role"]]
        return native_call(producer_role, projected, **kwargs)

    def acquire(candidates, **kwargs):
        kwargs["research_policy"] = initial_policy
        return native_acquire(candidates, **kwargs)

    def ranking(context):
        executed_identities, _task, _effects = native_ranking(context)
        return executed_identities, None, {}

    def post_round(state, result, round_index, opportunity_ref):
        if native_post_round is not None:
            state = native_post_round(state, result, round_index, opportunity_ref)
        return replace(
            state, policy=initial_policy,
            context=replace(state.context, policy=initial_policy.to_dict()),
        )

    with ExitStack() as restore:
        for owner, name, replacement in (
            (provider, "_call", research_call),
            (runtime, "_acquire_innovation_spec", acquire),
            (runtime, "_search_ranking_inputs", ranking),
            (composition.campaign, "post_round_state_transition", post_round),
        ):
            original = getattr(owner, name)
            restore.callback(setattr, owner, name, original)
            setattr(owner, name, replacement)
        yield composition
