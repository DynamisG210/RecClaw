"""Prospective fixed-100 graph metric contract; no ranking or search replacement."""
VERSION = 'graph.fixed100.ndcg5.hr5.se-style.v1'


def require_binding(binding):
    evaluation = binding['evaluation']
    expected = 'sequential.fixed100.ndcg5.hr5.v1' if binding.get('adapter') == 'recbole-sequential-fixed100-v1' else VERSION
    if binding.get('metric_binding_version') != expected:
        raise ValueError('Explicit prospective @5 graph binding required')
    if (evaluation['primary_metric'] != 'NDCG@5' or evaluation['topk'] != [5, 10]
            or 'Hit' not in evaluation['metrics'] or 'NDCG' not in evaluation['metrics']
            or evaluation['candidate_universe'] != 'FIXED_100_UNIFORM'
            or evaluation.get('secondary_metric') != 'Hit@5'):
        raise ValueError('Fixed-100 NDCG@5 selection and Hit@5 reporting required')


def report_metrics(result):
    """HR is native user-level Hit, never multi-positive Recall."""
    if result.get('primary_metric') != 'NDCG@5' or result.get('test_accessed') is not False:
        raise ValueError('Require original @5-selected validation result')
    values = result['metrics']
    return {'NDCG@5':values['ndcg@5'],'HR@5':values['hit@5']}
