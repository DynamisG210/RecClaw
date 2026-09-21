"""Explicit graph result handoff to the existing native NDCG@5 API."""
import math


PRIMARY_METRIC = 'ndcg@5'


def primary_observation(metrics):
    value = metrics[PRIMARY_METRIC]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Graph result needs a finite actual NDCG@5')
    return dict(name=PRIMARY_METRIC, value=float(value))


def execution_metrics(metrics):
    observation = primary_observation(metrics)
    return {**metrics, PRIMARY_METRIC: observation['value'],
            'primary_metric': observation, 'graph_raw_validation_metrics': dict(metrics)}


def baseline_source(*, metrics, **identity):
    from recclaw_core.research_line.single_round import ResearchBaselineSourceV1
    observation = primary_observation(metrics)
    return ResearchBaselineSourceV1.from_identity(
        frozen_ndcg_at_5=observation['value'], **identity)
