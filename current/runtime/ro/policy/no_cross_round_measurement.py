"""Experiment-only v161 control: freeze outcome-conditioned scientific inputs.

Attach once to one native E2 campaign in a dedicated process. All physical
state, deduplication, role coverage, within-round proposals and repair persist.
This is not a production policy or a general-purpose context filter.
"""
from dataclasses import fields, replace
from unittest.mock import patch
from contextlib import ExitStack

from recclaw_core.research_line import runtime
from recclaw_core.research_line.interfaces import ResearchContext


class NoCrossRoundMeasurement:
    def __init__(self, composition):
        self.composition = composition
        initial = composition.manifest["context"]["value"]
        self.initial_context = ResearchContext(**{
            f.name: initial[f.name] for f in fields(ResearchContext)
        })
        policy_type = type(composition.policy)
        initial_policy = composition.manifest["policy"]["value"]
        self.initial_policy = policy_type(**{
            f.name: initial_policy[f.name]
            for f in fields(policy_type) if f.init and f.name in initial_policy
        })
        self.adapter = composition.campaign.search_space_adapter
        self.native_provider_context = self.adapter.provider_context
        self.provider = composition.provider._producer
        self.native_provider_call = self.provider._call
        self.native_acquire = runtime._acquire_innovation_spec
        self.native_ranking = runtime._search_ranking_inputs
        self.native_feedback = runtime._selected_implementation_feedback
        self.native_post_round = composition.campaign.post_round_state_transition
        self._patches = ExitStack()

    def provider_context(self, context_view):
        role = context_view["producer_role"]
        # This is the last adapter boundary before the existing Provider renders
        # its prompt. Rebuild scientific fields from the original E2 context.
        current = self.native_provider_context(context_view)
        projected = self.native_provider_context(
            self.initial_context.producer_view(role)
        )
        for name in ("context_ref", "context_digest", "campaign_id", "round_index",
                     "protocol_ref", "protocol_digest", "budget"):
            projected[name] = current[name]
        if "research_portfolio" in context_view:
            projected["research_portfolio"] = context_view["research_portfolio"]
        projected["active_task_directive"] = None
        return projected

    def research_call(self, producer_role, context_view, **kwargs):
        # All native entry methods, including delegated transport resume, meet
        # here after adapter additions and before prompt construction.
        return self.native_provider_call(
            producer_role, self.provider_context(context_view), **kwargs
        )

    def acquire(self, candidates, **kwargs):
        # The native context is retained: its score-free role coverage and
        # candidate feasibility/identity checks continue unchanged.
        kwargs["research_policy"] = self.initial_policy
        return self.native_acquire(candidates, **kwargs)

    def ranking(self, context):
        executed_identities, _task, _effects = self.native_ranking(context)
        return executed_identities, None, {}

    def feedback(self, context, mechanism_change):
        # Current-candidate errors enter the separate native repair_context.
        # Retain initial comparison facts but no previous-candidate measurements.
        return self.native_feedback(
            self.initial_context if context is not None else None,
            mechanism_change,
        )

    def post_round(self, state, result, round_index, opportunity_ref):
        if self.native_post_round is not None:
            state = self.native_post_round(state, result, round_index, opportunity_ref)
        return replace(state, policy=self.initial_policy,
                       context=replace(state.context, policy=self.initial_policy.to_dict()))

    def install(self):
        for owner, name, replacement in (
            (self.provider, "_call", self.research_call),
            (runtime, "_acquire_innovation_spec", self.acquire),
            (runtime, "_search_ranking_inputs", self.ranking),
            (runtime, "_selected_implementation_feedback", self.feedback),
            (self.composition.campaign, "post_round_state_transition", self.post_round),
        ):
            self._patches.enter_context(patch.object(owner, name, replacement))
        return self

    def close(self):
        self._patches.close()


def install_composition_hook():
    """Install before the native CLI imports compose_standalone_campaign."""
    from recclaw_core.research_line import standalone
    native_compose = standalone.compose_standalone_campaign
    installed = []

    def compose(*args, **kwargs):
        composition = native_compose(*args, **kwargs)
        installed.append(NoCrossRoundMeasurement(composition).install())
        return composition

    standalone.compose_standalone_campaign = compose
    return installed
