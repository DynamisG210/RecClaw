"""Prospective limits on existing native campaign; no new selector or scheduler."""
from contextlib import contextmanager
from dataclasses import replace
import os
from unittest.mock import patch

LIMITS=dict(native_rounds=36,effective_evaluations=12,selected_attempts_per_round=1,discovery_generations_per_round=1,training_admissions=36,reference_usd=7,active_seconds=None,worker_seconds=None,qualification_seconds=None,cumulative_gpu_seconds=None)

def bounded(config):
    return replace(config,round_count=36,observation_seed_schedule=(config.seed,)*36,
                   max_attempts_per_round=1,prebinding_token_ceiling_retry=False,
                   close_exhausted_no_metric_slot=True)

def bind(module):
    native=module.research_config
    module.research_config=lambda inputs:bounded(native(inputs))
    import graph_native_launch
    original=graph_native_launch.inspection
    def inspection(inputs,method):
        result=original(inputs,method);result['limits']=dict(LIMITS)
        result['evidence_class']=('USER_CONFIGURED_SOFTWARE_RUN' if inputs['evaluator']['phase']=='USER_RESEARCH' else 'FROZEN_GRAPH_SEARCH_VALIDATION')
        return result
    graph_native_launch.inspection=inspection

@contextmanager
def pilot_scope():
    from recclaw_core.research_line import standalone
    with patch.object(standalone,'MAX_DISCOVERY_GENERATIONS_PER_ROUND',1),patch.dict(os.environ,{'E2_SHARED_CONTROLLER_SLOT_CEILING':'36'}):
        yield
