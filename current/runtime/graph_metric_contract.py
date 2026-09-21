"""Prospective full-sort graph metric contract; no ranking or search replacement."""
VERSION = 'graph.fullsort.ndcg5.hr5.v1'


def require_binding(binding):
    evaluation = binding['evaluation']
    if binding.get('metric_binding_version') != VERSION:
        raise ValueError('Explicit prospective @5 graph binding required')
    if (evaluation['primary_metric'] != 'NDCG@5' or evaluation['topk'] != [5, 10]
            or 'Hit' not in evaluation['metrics'] or 'NDCG' not in evaluation['metrics']
            or evaluation['candidate_universe'] != 'FULL_SORT'
            or evaluation.get('secondary_metric') != 'Hit@5'):
        raise ValueError('Full-sort NDCG@5 selection and Hit@5 reporting required')


def report_metrics(result):
    """HR is native user-level Hit, never multi-positive Recall."""
    if result.get('primary_metric') != 'NDCG@5' or result.get('test_accessed') is not False:
        raise ValueError('Require original @5-selected validation result')
    values = result['metrics']
    return {'NDCG@5':values['ndcg@5'],'HR@5':values['hit@5']}
