"""Reference an unchanged fit through an explicitly proven default ABI migration."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_root(config):
    original_path = Path(config['root_result_source'])
    original = json.loads(original_path.read_text())
    proof_path = Path(config['root_equivalence_proof'])
    proof = json.loads(proof_path.read_text())
    expected = dict(source_sha256=proof['original_source_sha'],
        recipe_sha256=digest(config['training_config']),
        manifest_sha256=digest(config['search_manifest']), seed=config['train_seed'])
    if original['status'] != 'ok' or original['identity'] != expected:
        raise ValueError('Historical root does not match the paired task')
    for key in ('default_mask_and_rng_equivalent',
                'full_worker_default_weights_metrics_rng_equivalent'):
        if proof.get(key) is not True:
            raise ValueError('Missing default-path equivalence evidence: ' + key)
    if digest(config['parent_source']) != proof['aligned_source_sha']:
        raise ValueError('Aligned default parent differs from equivalence evidence')
    if digest(config['bert_worker']) != proof['aligned_worker_sha']:
        raise ValueError('Aligned worker differs from equivalence evidence')
    mapping = dict(basis='EXPLICIT_DEFAULT_ABI_EQUIVALENCE_NOT_RETRAINED',
        evidence_scope=proof['scope'], proof_ref=str(proof_path),
        proof_sha256=digest(proof_path), original_result_ref=str(original_path),
        original_result_sha256=digest(original_path),
        actual_trained_identity=original['identity'],
        aligned_default_source_sha256=proof['aligned_source_sha'],
        aligned_worker_sha256=proof['aligned_worker_sha'], physical_execution_repeated=False)
    # Keep original source, seed, metric and identity intact. Consumers attach
    # the mapping to the anchor, never pretend this was an aligned-worker fit.
    return original, mapping
