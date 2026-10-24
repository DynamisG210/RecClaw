"""Shared validation of a measured baseline before preparation and execution."""
import math


def validate_root_receipt(envelope, physical):
    evidence = envelope.get('artifacts', {}).get('native_evidence', {})
    if envelope.get('status') != 'SUCCESS' or evidence.get('termination') not in ('EARLY_STOP', 'EPOCH_LIMIT'):
        raise ValueError('Baseline requires a complete native_evidence receipt; run recrsi baseline')
    if evidence.get('unresolved_gpu_seconds') != 0:
        raise ValueError('Baseline has unresolved physical work')
    result = envelope['result']
    if result.get('role') != 'root' or not result.get('checkpoint_reload_verified'):
        raise ValueError('Baseline requires a verified root checkpoint')
    jobs = physical.get('jobs', {})
    if not jobs or physical.get('root_reference_cost') or any(j.get('finished_ns') is None for j in jobs.values()):
        raise ValueError('Baseline requires its settled physical ledger')
    cost = evidence.get('physical_cost', {})
    for key in ('gpu_seconds', 'active_seconds'):
        value = cost.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Baseline requires finite nonnegative ' + key)
    seconds = [j.get('gpu_seconds') for j in jobs.values()]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in seconds):
        raise ValueError('Invalid baseline GPU ledger')
    if sum(seconds) != cost['gpu_seconds'] or cost['active_seconds'] <= 0:
        raise ValueError('Baseline cost differs from its physical ledger')
    if cost['gpu_seconds'] == 0 and evidence.get('device') != 'cpu':
        raise ValueError('Zero GPU time requires an explicitly recorded CPU baseline')
    return cost
