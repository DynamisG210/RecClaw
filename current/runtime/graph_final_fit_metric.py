"""Select the common @5 winner and prepare its unchanged-recipe final-fit binding."""
import copy
from graph_metric_contract import require_binding


def select_winner(ledger):
    # Stable arrival order preserves the common ledger's strict-improvement tie rule.
    rows=[row for row in ledger['results'] if row.get('valid') and row.get('admitted')]
    if not rows: raise ValueError('No admitted common result')
    for row in rows:
        evidence=row['evidence']
        if evidence.get('primary_metric')!='NDCG@5' or evidence.get('test_accessed') is not False:
            raise ValueError('Old-cutoff or heldout evidence cannot select a formal winner')
        if row['primary_metric']!=evidence['metrics']['ndcg@5']:
            raise ValueError('Common ledger primary is not measured NDCG@5')
    return max(rows,key=lambda row:row['primary_metric'])


def final_fit_binding(binding, winner, seed):
    require_binding(binding)
    if seed not in [61001,61002,61003]: raise ValueError('Original three final-fit seeds required')
    result=copy.deepcopy(binding)
    result.update(phase='FINAL_FIT',train_seed=seed,selected_source_sha256=winner['source_sha256'],status='PREPARED_OWNER_FREEZE_REQUIRED')
    result['selection_evidence']={'metric':'NDCG@5','value':winner['primary_metric'],'source_sha256':winner['source_sha256'],'refit_seed':seed}
    return result
