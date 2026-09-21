"""Independent RQ4 interventions for the existing root-relative E2 adapter.

The original joint control remains a frozen dependency. These conditions change
separate consumers while retaining durable observations and native execution.
"""
from unittest.mock import patch

from no_cross_round_measurement import NoCrossRoundMeasurement, runtime


class NoResearchMemory(NoCrossRoundMeasurement):
    """Remove historical scientific readout; retain learned controller policy."""

    def provider_context(self, context_view):
        projected = super().provider_context(context_view)
        # Policy is a separate learned component, deliberately retained here.
        projected["policy"] = context_view["policy"]
        projected["producer_token_fraction"] = context_view["producer_token_fraction"]
        return projected

    def install(self):
        for owner, name, replacement in (
            (self.provider, "_call", self.research_call),
            (runtime, "_selected_implementation_feedback", self.feedback),
        ):
            self._patches.enter_context(patch.object(owner, name, replacement))
        return self


class FrozenSearchPolicy(NoCrossRoundMeasurement):
    """Freeze explicit acquisition/ranking adaptation; retain scientific readout."""

    def provider_context(self, context_view):
        projected = self.native_provider_context(context_view)
        projected["policy"] = self.initial_policy.to_dict()
        projected["producer_token_fraction"] = dict(
            self.initial_policy.producer_token_allocation
        )[context_view["producer_role"]]
        return projected

    def install(self):
        for owner, name, replacement in (
            (self.provider, "_call", self.research_call),
            (runtime, "_acquire_innovation_spec", self.acquire),
            (runtime, "_search_ranking_inputs", self.ranking),
            (self.composition.campaign, "post_round_state_transition", self.post_round),
        ):
            self._patches.enter_context(patch.object(owner, name, replacement))
        return self


def install_composition_hook(condition):
    from recclaw_core.research_line import standalone

    intervention = {
        "no_research_memory": NoResearchMemory,
        "frozen_search_policy": FrozenSearchPolicy,
    }[condition]
    native_compose = standalone.compose_standalone_campaign
    installed = []

    def compose(*args, **kwargs):
        composition = native_compose(*args, **kwargs)
        installed.append(intervention(composition).install())
        return composition

    standalone.compose_standalone_campaign = compose
    return installed
