"""RO physical port to the common graph evaluator; no core metric-name alias.

The graph profile/AutoSpace assembly must bind this port and native graph recipe.
Uses the accepted isolated graph core's original NDCG@5 result API.
"""
import hashlib
import json
from pathlib import Path


def completed_worker_feedback(result, binding, binding_path):
    """Project only this worker's recorded training facts into native feedback."""
    telemetry = {key: result[key] for key in (
        'epochs_completed', 'max_epochs', 'patience', 'stop_reason',
    ) if key in result}
    if 'best_epoch' in result:
        telemetry['best_observed_epoch'] = result['best_epoch']
    stage = result.get('physical_stage', {})
    telemetry['phase_records_status'] = 'NO_TRAINING_OUTPUT'
    if stage.get('kind') == 'training' and stage.get('output'):
        curve = Path(stage['output']) / 'LEARNING_CURVE.jsonl'
        telemetry['phase_records_source'] = str(curve)
        if curve.is_file():
            raw = curve.read_bytes()
            records = []
            for line in raw.splitlines():
                row = json.loads(line)
                for phase, fields in (
                    ('TRAIN', {'loss': 'loss', 'train_seconds': 'wall_time_ms'}),
                    ('EVAL', {'score': 'valid_score', 'validation_seconds': 'wall_time_ms'}),
                ):
                    record = {'epoch': row['epoch'], 'phase': phase}
                    for source, target in fields.items():
                        if source in row:
                            record[target] = row[source] * 1000 if target == 'wall_time_ms' else row[source]
                    records.append(record)
            telemetry.update(phase_records=records, phase_records_status='RECORDED' if records else 'EMPTY',
                             phase_records_sha256=hashlib.sha256(raw).hexdigest())
        else:
            telemetry['phase_records_status'] = 'MISSING'
    configuration = {key: binding[key] for key in (
        'task_id', 'line', 'dataset', 'model_config', 'training', 'evaluation', 'train_seed',
    ) if key in binding}
    configuration.update(binding_ref=str(binding_path),
                         binding_sha256=hashlib.sha256(Path(binding_path).read_bytes()).hexdigest())
    return {'resource_telemetry': telemetry, 'worker_configuration': configuration}


class GraphROExecution:
    def __init__(self,config): self.config=config

    def __call__(self,**kwargs):
        from recclaw_core.experiments.helix_abc_v1.experiment_binding import ExperimentBindingV1
        from graph_native_adapter import evaluate_source
        from graph_ro_metric import execution_metrics
        from graph_ro_contract import register_graph_contract
        graph_split, graph_evaluator = register_graph_contract()
        recipe = kwargs['execution_recipe']
        if recipe.get('split') != graph_split or recipe.get('evaluator') != graph_evaluator:
            raise ValueError('Graph RO recipe must use full-sort dev evaluation with heldout denied')
        config=self.config
        binding=json.loads(Path(config['binding']).read_bytes())
        if kwargs['seed']!=config['train_seed'] or kwargs['epochs']!=binding['training']['epochs']:
            raise ValueError('RO recipe must preserve the common graph training seed/horizon')
        root=Path(kwargs['candidate_root']).resolve(strict=True)
        source=(root/'candidate.py').read_bytes().decode('utf-8')
        digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
        identity=ExperimentBindingV1.from_execution_recipe(kwargs['execution_recipe'],
            candidate_root=root,dataset_manifest_digest=config['dataset_manifest_digest'],
            seed=kwargs['seed'],epochs=kwargs['epochs'],timeout_seconds=kwargs['timeout_seconds'],
            execution_purpose=kwargs['execution_purpose'],resource_telemetry=kwargs['resource_telemetry'],
            watchdog_seconds=kwargs.get('watchdog_seconds'),prefix_contract_digest=None,
            run_id=str(kwargs['run_id']),round_id='graph:'+str(kwargs['run_id']),
            claim_id=binding['task_id']+'-search',permit_digest=digest(binding['task_id']+'-validation-only'),
            runtime_binding_digest=config['runtime_binding_digest'],runtime_release_digest=config['runtime_release_digest'],
            runner_abi='recclaw.external-comparison.graph-native-launch.v1',
            filesystem_capability_digest=digest('graph-search-test-denied'))
        envelope=evaluate_source(config,source);result=envelope['artifacts']['native_evidence']
        valid=envelope['artifacts']['valid']
        if valid and result.get('test_accessed') is not False:
            raise ValueError('Graph RO requires explicit validation-only worker evidence')
        outcome=dict(exit_status='SUCCESS' if valid else 'RUNTIME_FAILURE',
            experiment_binding=identity.canonical_dict(),binding_digest=identity.digest,
            experiment_binding_digest=identity.digest,experiment_binding_ref=identity.ref,
            execution_recipe_digest=identity.execution_recipe_digest,seed=identity.seed,
            wall_time_ms=round(result.get('wall_time_secs',0.)*1000),test_accessed=False,
            graph_native_evidence=result)
        outcome.update(completed_worker_feedback(result, binding, config['binding']))
        if valid:
            if binding['evaluation']['primary_metric'] != 'NDCG@5':
                raise ValueError('Graph RO requires the actual NDCG@5 contract')
            metrics = dict(result.get('metrics', {}))
            score = envelope['metrics']['ndcg_at_5']
            if metrics.get('ndcg@5') != score:
                raise ValueError('Graph ledger primary differs from actual worker NDCG@5')
            outcome['metrics']=execution_metrics(metrics)
            outcome['primary_metric']=binding['evaluation']['primary_metric']
        else:
            outcome.update(failure_phase=result.get('failure_stage','CANDIDATE_EXECUTION'),
                           failure_message=result.get('error','Graph candidate failed'))
        return outcome
