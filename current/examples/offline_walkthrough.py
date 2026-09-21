"""Read real interfaces with synthetic inputs; no Provider, dataset or training."""
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
sys.path[:0] = [str(RUNTIME), str(RUNTIME / 'ro'),
               str(RUNTIME / 'ro/policy'), str(RUNTIME / 'isolated_graph_core'),
               str(RUNTIME / 'isolated_graph_core/src')]

from jsonschema import Draft202012Validator
from recclaw_core.research_line.standalone import compose_standalone_campaign
from recclaw_core.experiments.helix_abc_v1.research_contracts import DISCOVERY_PRODUCERS
from graph_autospace import engine, response_schema
from graph_candidate_contract import qualify_source
from graph_ro_execution import completed_worker_feedback
from graph_search_adapter import GraphSearchAdapter
from learning_components import FrozenSearchPolicy

# Syntax-only interface fixture. It deliberately contains no model or training.
SOURCE = '''class FreshCandidateModel:
    def __init__(self): pass
    def calculate_loss(self, interaction): pass
    def predict(self, interaction): pass
    def full_sort_predict(self, interaction): pass
'''


def main():
    assert callable(compose_standalone_campaign)
    assert GraphSearchAdapter.implementation_allowed_files == ('candidate.py',)
    assert FrozenSearchPolicy.__name__ == 'FrozenSearchPolicy'
    for stage in ('genesis', 'critic', 'assembled', 'quality'):
        Draft202012Validator.check_schema(response_schema(stage))
    assembler = engine(SOURCE)
    assert len(assembler.REQUIRED_AXES) == 4
    assert qualify_source(SOURCE)['status'] == 'ELIGIBLE_SOURCE'
    try:
        qualify_source('import os\n' + SOURCE)
    except ValueError:
        pass
    else:
        raise AssertionError('Forbidden import accepted')
    with tempfile.TemporaryDirectory(prefix='recclaw-course-') as tmp:
        folder = Path(tmp)
        binding = {'task_id': 'SYNTHETIC_DEMO', 'train_seed': 123,
                   'training': {'epochs': 2}, 'evaluation': {'primary_metric': 'NDCG@5'}}
        binding_path = folder / 'binding.json'
        binding_path.write_text(json.dumps(binding), encoding='utf-8')
        rows = [{'epoch': 0, 'loss': 1.0, 'score': 0.1, 'train_seconds': 1.2,
                 'validation_seconds': 0.2},
                {'epoch': 1, 'loss': 0.8, 'score': 0.12, 'train_seconds': 1.1,
                 'validation_seconds': 0.2}]
        curve = folder / 'LEARNING_CURVE.jsonl'
        curve.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        result = {'epochs_completed': 2, 'best_epoch': 1, 'stop_reason': 'DEMO_ONLY',
                  'physical_stage': {'kind': 'training', 'output': str(folder)}}
        feedback = completed_worker_feedback(result, binding, binding_path)
        telemetry = feedback['resource_telemetry']
        assert telemetry['phase_records_status'] == 'RECORDED'
        assert len(telemetry['phase_records']) == 4
        assert telemetry['phase_records'][0]['wall_time_ms'] == 1200
        assert telemetry['best_observed_epoch'] == 1
        assert feedback['worker_configuration']['train_seed'] == 123
        curve.unlink()
        missing = completed_worker_feedback(result, binding, binding_path)['resource_telemetry']
        assert missing['phase_records_status'] == 'MISSING'
        assert 'phase_records' not in missing
    print(json.dumps({'status': 'PASS', 'data': 'SYNTHETIC_NOT_EXPERIMENTAL',
                      'roles': list(DISCOVERY_PRODUCERS),
                      'autospace_axes': list(assembler.REQUIRED_AXES),
                      'checks': ['core_and_adapter_import', 'four_schemas',
                                 'source_eligibility_positive_and_negative',
                                 'training_feedback_and_missingness'],
                      'provider_calls': 0, 'training_jobs': 0}, indent=2))


if __name__ == '__main__':
    main()
