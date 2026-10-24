"""Bind task requirements before constructing the research producer."""
from contextlib import contextmanager
from unittest.mock import patch

INPUT_VERSION = 'graph-producer-task-requirements-v1'

@contextmanager
def task_proposal_requirements(adapter_module):
    original = adapter_module.GraphSearchAdapter.build_provider_research_producer
    def build(self, **kwargs):
        kwargs['proposal_protocol_requirements'] = adapter_module.REQUIREMENTS
        return original(self, **kwargs)
    with patch.object(adapter_module.GraphSearchAdapter, 'build_provider_research_producer', build):
        yield
