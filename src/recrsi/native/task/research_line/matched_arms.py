"""Two prospective method arms on the same repaired graph runtime.

This module changes the permitted research action, not the task, data,
selector, evaluator, models or budgets. It never launches an experiment.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import research_actions as actions


class RootRelativeAdapter(actions.ResearchActionAdapter):
    """Expose measured history, but construct every child from the root."""

    def proposal_schema(self):
        schema = super().proposal_schema()
        fields = schema['properties']['proposals']['items']['properties']
        fields['closest_parent'] = {'const': self.parent_id, 'type': 'string'}
        fields['research_action'] = {'type': 'null'}
        return schema

    def task_context(self):
        context = super().task_context()
        context['construction'] = (
            'Construct every candidate from original_root. Measured candidates '
            'remain comparison evidence; their source is not the implementation '
            'starting point. Both fixed recipes and complete open mechanisms '
            'are available, with one unambiguous definition.'
        )
        context['research_actions'] = {
            **context['research_actions'],
            'operations': [],
            'optional': False,
            'candidate_cost': 'One original-root candidate opportunity',
            'parents_are_comparison_evidence_only': True,
        }
        return context

    def build_provider_research_producer(self, **kwargs):
        producer = super().build_provider_research_producer(**kwargs)
        producer._producer.proposal_template = (
            Path(__file__).parent / 'root_relative_producer.txt'
        ).read_text(encoding='utf8')
        return producer


@contextmanager
def matched_arm(adapter_module, policy_module, arm):
    if arm == 'root_relative':
        with patch.object(actions, 'ResearchActionAdapter', RootRelativeAdapter):
            with actions.research_actions(adapter_module, policy_module):
                yield
    elif arm == 'evidence_bound_parent':
        with actions.research_actions(adapter_module, policy_module):
            yield
    else:
        raise ValueError('Unknown frozen comparison arm')
