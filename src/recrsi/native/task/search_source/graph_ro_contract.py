"""Register the graph validation-only contract in the isolated core process."""
GRAPH_SPLIT = 'train/dev'
GRAPH_EVALUATOR = {
    'candidate_universe': 'FULL_SORT',
    'heldout_access': 'DENIED',
    'metric': 'NDCG@5',
    'nonfinite_policy': 'REJECT',
    'partition_role': 'DEVELOPMENT_VALIDATION',
    'source': 'BEST_VALID_RESULT',
}


def register_graph_contract():
    from recclaw_core.experiments.helix_abc_v1 import experiment_binding
    from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value
    contract = (GRAPH_SPLIT, canonical_value(GRAPH_EVALUATOR))
    if contract not in experiment_binding._SUPPORTED_EVALUATION_CONTRACTS:
        experiment_binding._SUPPORTED_EVALUATION_CONTRACTS += (contract,)
    return contract
