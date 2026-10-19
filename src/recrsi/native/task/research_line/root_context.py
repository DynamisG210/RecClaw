"""Prospective graph RO input only; install outside the existing execute call."""
import hashlib
from contextlib import contextmanager
from unittest.mock import patch

INPUT_VERSION = 'graph-root-relative-comparators-v3'
SEMANTICS = (
    'Construct every SELECT/COMPOSE from original_root, not from the latest measured candidate. '
    'The latest candidate is comparison evidence only; retain its mechanism only by explicitly '
    'selecting it in this proposal. Compare declared intent, actual code, fixed worker configuration '
    'and observed results. A combined intervention or one score does not isolate its cause. '
    'Use only this task and its own history; no other environment or method supplies a winning answer.'
    'Separate original-root improvement from improvement over the incumbent frontier. '
    'The original-root score is original_root.development_metrics; it does not move. '
    'Native comparator_delta is relative to the incumbent used in that episode and may not be root-relative. '
    'A negative frontier delta does not imply a negative root delta. Name the comparator and its score '
    'when interpreting an observation; infer no root-relative failure or causal conclusion from frontier delta alone. '
)

@contextmanager
def graph_root_context(adapter_module):
    original = adapter_module.GraphSearchAdapter.provider_context

    def project(self, context):
        result = original(self, context)
        knowledge = dict(result['knowledge_base'])
        task = dict(knowledge['search_space'])
        latest = context.get('latest_completed_execution')
        # Existing native projection retains the complete latest execution, including
        # code/intent/configuration/measurement. Reference it rather than copy code twice.
        comparison = dict(
            input_version=INPUT_VERSION,
            construction_semantics=SEMANTICS,
            original_root=dict(
                parent_id=self.parent_id,
                files=[dict(path='candidate.py', content=self.parent)],
                source_sha256=hashlib.sha256(self.parent.encode('utf-8')).hexdigest(),
                primary_metric=task['primary_metric'],
                development_metrics=self.facts.get('selected_root_metrics'),
                reference_cost=self.facts.get('root_reference_cost'),
                fixed_worker_configuration=dict(
                    provenance='existing task facts and evaluator train_seed; not a new fit',
                    model_config=self.facts.get('original_model_config'),
                    training=self.facts.get('training'),
                    evaluation=self.facts.get('evaluation'),
                    train_seed=self.config.get('train_seed'),
                ),
            ),
            latest_completed_candidate=(dict(
                status='AVAILABLE',
                source_context_ref='state.latest_completed_execution',
                candidate_id=latest.get('candidate_id'),
                source_sha256=latest.get('entrypoint_source_sha256'),
                declared_mechanism=latest.get('proposed_mechanism'),
                evidence_fields=['implementation', 'proposed_mechanism', 'development_metrics',
                                 'worker_configuration', 'native_training', 'measured_cost'],
                semantics='Read the referenced native record alongside original_root. Missing fields remain unknown.',
            ) if latest is not None else dict(status='NO_COMPLETED_CANDIDATE', source_context_ref=None)),
        )
        task['root_relative_comparison'] = comparison
        knowledge['search_space'] = task
        result['knowledge_base'] = knowledge
        return result

    with patch.object(adapter_module.GraphSearchAdapter, 'provider_context', project):
        yield
