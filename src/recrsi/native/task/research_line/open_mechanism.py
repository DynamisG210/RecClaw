"""Candidate-local graph mechanisms; the accepted AutoSpace is never mutated.

Install only in a fresh task entry. Existing SELECT/COMPOSE requests continue
through the original adapter. No Provider, evaluator, policy or budget override.
"""
import copy
import hashlib
from score_binding import bind_source_contract, contract_digest
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from graph_search_adapter import GraphSearchAdapter as FixedGraphAdapter
from graph_autospace import digest

VERSION = 'ngcf-executable-definition-v3'
FIELDS = ('conceptual_primitive_ids', 'conceptual_operator_id',
          'conceptual_target_primitive_id', 'conceptual_replacement_primitive_id',
          'conceptual_custom_components')
MARKER = 'CANDIDATE_MECHANISM_DEFINITION_JSON:'
ASSETS = Path(__file__).resolve().parent


def mechanism_schema():
    text = {'type': 'string', 'minLength': 1}
    strings = {'type': 'array', 'items': text}
    def obj(properties):
        return dict(type='object', additionalProperties=False,
                    properties=properties, required=list(properties))
    return obj({
        'operations': dict(type='array', minItems=1, items=obj({
            'operation_id': text, 'target': text, 'replaces': text, 'definition': text})),
        'evidence_refs': strings,
        'revises_spec_ref': {'type': ['string', 'null']},
        'score_effect': obj({
            'parameterization': {'enum': ['UNCHANGED', 'FIXED', 'LEARNED']},
            'phase': {'enum': ['TRAIN_ONLY', 'INFERENCE_ONLY', 'TRAIN_AND_INFERENCE']}}),
    })


class OpenMechanismAdapter(FixedGraphAdapter):
    # Keep fixed-candidate identity unchanged. Open identities separately include
    # VERSION and the complete definition; no pretend registered primitive IDs.
    def provider_context(self, context):
        result = super().provider_context(context)
        task = result['knowledge_base']['search_space']
        if 'root_relative_comparison' in task:
            comparison = dict(task['root_relative_comparison'])
            comparison['construction_semantics'] = (
                self.task_context()['construction'] +
                ' Retain a historical mechanism through explicit fixed selection '
                'or a complete restatement in the candidate-local definition. '
                'Native comparator_delta is relative to the recorded incumbent; '
                'original-root metrics are a separate comparison. Use only this run own evidence.')
            task['root_relative_comparison'] = comparison
        from mechanism_evidence import augment_evidence
        return augment_evidence(result, self.config['output'])

    def task_context(self):
        result = super().task_context()
        result['construction'] = (
            'Original-root construction. SELECT/COMPOSE uses its complete fixed '
            'recipe; mechanism uses its complete candidate-local definition. '
            'Measured descendants are evidence, not implicitly inherited code.')
        result['candidate_mechanism_contract'] = dict(
            version=VERSION, operator='mechanism', parent_id=self.parent_id,
            knowledge_space_digest=self.conceptual_space.digest,
            knowledge_role='Structured references; not an exhaustive hypothesis universe',
            global_registry_mutation=False)
        return result

    def proposal_schema(self):
        schema = super().proposal_schema()
        item = schema['properties']['proposals']['items']
        item.pop('allOf')
        p = item['properties']
        p['resolution_facts']['properties']['required_dependencies']['items']['enum'] = list(self.available_dependencies)
        p['conceptual_operator_id']['enum'].append('mechanism')
        p['open_mechanism'] = {'anyOf': [{'type': 'null'}, mechanism_schema()]}
        # Closed Provider schema uses explicit null for fixed recipes. Route
        # consistency is checked by the selected binding, not schema conditionals.
        item['required'].append('open_mechanism')
        return schema

    def build_provider_research_producer(self, **kwargs):
        wrapped = super().build_provider_research_producer(**kwargs)
        wrapped._producer.proposal_template = (ASSETS / 'producer.txt').read_text(encoding='utf8')
        return wrapped

    def build_provider_implementer(self, **kwargs):
        wrapped = super().build_provider_implementer(**kwargs)
        wrapped._implementer.implementation_template = (ASSETS / 'implementer.txt').read_text(encoding='utf8')
        return wrapped

    def implementation_policy(self, policy):
        return replace(super().implementation_policy(policy),
                       prompt_digest=digest((ASSETS / 'implementer.txt').read_text(encoding='utf8')))

    def resolve_confirmation(self, kind, primary_binding, context):
        if context.get('phase') != 'PROJECT_PROVIDER_PROPOSAL':
            return super().resolve_confirmation(kind, primary_binding, context)
        draft = copy.deepcopy(dict(primary_binding))
        definition = draft.pop('open_mechanism', None)
        if draft.get('conceptual_operator_id') != 'mechanism':
            if definition is not None:
                return self._unsupported('Fixed recipe cannot silently discard open_mechanism')
            return super().resolve_confirmation(kind, draft, context)
        from jsonschema import Draft202012Validator, ValidationError
        from recclaw_core.research_line.producers import _project_result
        from recclaw_core.research_line.interfaces import ProducerOutcome
        from recclaw_core.research_line.search_space_adapter import ConfirmationResolutionV1 as Resolution, ConfirmationResolutionKindV1 as Kind
        try:
            if not isinstance(definition, dict) or draft.get('conceptual_primitive_ids'):
                raise ValueError('mechanism requires its complete local definition and no selected primitive IDs')
            # Native Provider normalizes JSON config before handing the draft to
            # this adapter; validate its original fixed-config wire equivalent.
            execution = draft['execution_contract']
            if 'config' in execution:
                if execution['config'] != {}:
                    raise ValueError('Graph external configuration must remain fixed')
                execution.pop('config')
                execution['config_json'] = '{}'
            Draft202012Validator(self.proposal_schema()).validate({
                'schema': 'recclaw.research-line.fresh-open-spec-proposal-response.v1',
                'proposals': [dict(draft, open_mechanism=definition)]})
            for key in FIELDS:
                draft.pop(key)
            if draft.get('idea_mode') == 'DIAGNOSIS_DRIVEN' and draft.get('observed_failure_mode') in (None, 'NOT_OBSERVED'):
                draft.update(idea_mode='FRONTIER_HYPOTHESIS', observed_failure_mode=None)
            execution = draft['execution_contract']
            draft['execution_contract'] = {k: execution[k] for k in ('capability_family', 'model', 'base_model_config')}
            draft['execution_contract']['config'] = {}
            ids=[operation['operation_id'] for operation in definition['operations']]
            if len(ids)!=len(set(ids)):
                raise ValueError('Operation identities must be unique within the definition')
            # Native OpenSpec already carries implementation_requirements through
            # serialization, selection, blind-builder, feedback and paid replay.
            # Bind the structured definition into that identity, not only metadata.
            canonical = json.dumps(definition, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            draft['implementation_requirements'].append(MARKER + canonical)
            spec, facts, source, program, companion = _project_result(
                context['producer_role'], draft, bindings=context['producer_bindings'],
                frozen_profile_ref=context['frozen_profile_ref'])
            spec = replace(spec, execution_contract={**spec.execution_contract, 'config': {}})
            binding = self._bind_definition(spec, definition)
            rc = context['research_context']
            outcome = ProducerOutcome(
                producer_role=context['producer_role'], context_ref=rc.context_ref,
                context_digest=rc.digest, spec=spec, resolution_facts=facts,
                source_proposal=source, source_mechanism_program=program,
                implementation_companion=companion,
                provenance=dict(status='PRODUCED', producer_role=context['producer_role'],
                    context_ref=rc.context_ref, context_digest=rc.digest,
                    spec_id=spec.spec_id, spec_digest=spec.digest,
                    candidate_mechanism_binding=binding))
            return Resolution(Kind.EXACT_BINDING, binding={'producer_outcome': outcome})
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            return self._unsupported('CANDIDATE_MECHANISM_PROJECTION:' + str(error))

    @staticmethod
    def _unsupported(reason):
        from recclaw_core.research_line.search_space_adapter import ConfirmationResolutionV1 as Resolution, ConfirmationResolutionKindV1 as Kind
        return Resolution(Kind.UNSUPPORTED, reason=reason)

    def _bind_definition(self, spec, definition):
        encoded = MARKER + json.dumps(definition, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        if encoded not in spec.implementation_requirements or spec.closest_parent != self.parent_id:
            raise ValueError('Complete mechanism definition must belong to this root-relative spec')
        return dict(version=VERSION, spec_ref=spec.spec_id, spec_digest=spec.digest,
                    parent_id=self.parent_id, knowledge_space_digest=self.conceptual_space.digest,
                    definition=copy.deepcopy(definition), definition_digest=digest(definition))

    def _prepare_native_innovation(self, outcome):
        bound = outcome.provenance.get('candidate_mechanism_binding')
        if bound is None:
            return super()._prepare_native_innovation(outcome)
        spec = outcome.spec
        if self._bind_definition(spec, bound['definition']) != bound:
            raise ValueError('Candidate mechanism binding drift')
        candidate_id = 'cand-graph-' + spec.digest[:24]
        axis = next(iter(outcome.resolution_facts.get('high_change_dimensions', ())), 'architecture')
        program = dict(schema='recclaw.external-comparison.graph-native-mechanism.v1',
            candidate_id=candidate_id, producer_role=outcome.producer_role,
            spec_ref=spec.spec_id, spec_digest=spec.digest,
            mechanism_id='graph-native-' + spec.digest[:24], mechanism_axis=axis,
            execution_contract=spec.execution_contract, mechanism_hypothesis=spec.hypothesis,
            mechanism_change=spec.mechanism_change, candidate_mechanism_binding=bound)
        semantic = digest({'adapter': self.adapter_id, 'program': program})
        program.update(semantic_identity_ref='graph-native-mechanism:' + candidate_id,
                       semantic_identity_digest=semantic)
        return dict(producer_outcome=outcome, compiler_binding=None, compiled_implementation=None,
            mechanism_program=program, semantic_identity_ref=program['semantic_identity_ref'],
            semantic_identity_digest=semantic, effective_identity=dict(
                effective_experiment_digest=digest({'adapter': self.adapter_id, 'candidate': semantic, 'task': self.task_id}),
                effective_family_digest=digest({'version': VERSION, 'parent_id': self.parent_id}), primitive_ids=()),
            candidate_id=candidate_id, mechanism_id=program['mechanism_id'], mechanism_axis=axis,
            parent_available=True)

    def _open_binding(self, program):
        bound = program['candidate_mechanism_binding']
        if (bound['version'] != VERSION or bound['spec_ref'] != program['spec_ref']
                or bound['spec_digest'] != program['spec_digest']
                or bound['parent_id'] != self.parent_id
                or bound['knowledge_space_digest'] != self.conceptual_space.digest
                or digest(bound['definition']) != bound['definition_digest']):
            raise ValueError('Candidate-local mechanism identity drift')
        return bound

    def _mechanism_score_request(self, program):
        bound = self._open_binding(program)
        effect = bound['definition']['score_effect']
        # Preserve the existing physical score-check ABI. This identifier denotes
        # the complete candidate scorer, NOT a primitive inserted into AutoSpace.
        entries = [] if effect['parameterization'] == 'UNCHANGED' else [dict(
            primitive_id='candidate-mechanism:' + bound['definition_digest'],
            score_parameterization=effect['parameterization'], effect_phase=effect['phase'])]
        return dict(selected_spec_digest=bound['spec_digest'], selected_primitive_ids=[], entries=entries)

    def adapt_implementation_request(self, request, *, mechanism_program):
        if 'candidate_mechanism_binding' not in mechanism_program:
            return super().adapt_implementation_request(request, mechanism_program=mechanism_program)
        bound = self._open_binding(mechanism_program)
        return dict(request, initial_program={'candidate.py': self.parent},
                    candidate_mechanism_recipe=bound,
                    graph_score_contract_request=self._mechanism_score_request(mechanism_program))

    def qualification_executor(self, package, *, research_spec, candidate_root,
                               fixture=None, unit_check=None, mechanism_program=None):
        if 'candidate_mechanism_binding' not in (mechanism_program or {}):
            return super().qualification_executor(package, research_spec=research_spec,
                candidate_root=candidate_root, fixture=fixture, unit_check=unit_check,
                mechanism_program=mechanism_program)
        return self._qualify_open(package, research_spec, candidate_root, mechanism_program)

    def _qualify_open(self, package, research_spec, candidate_root, mechanism_program):
        from graph_qualification import qualify_program
        from recclaw_core.experiments.helix_abc_v1.innovation_spine import InnovationSpineError
        from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import MechanicalQualificationRun
        from recclaw_core.experiments.helix_abc_v1.vnext_contracts import QualificationReceiptV1,QualificationStageV1,QualificationStatusV1,QualificationFailureClassV1,QualificationCheckStatusV1
        source=Path(candidate_root)/'candidate.py'
        if set(p.name for p in Path(candidate_root).iterdir() if p.is_file())!={'candidate.py'}:
            raise ValueError('Graph materializer must emit only candidate.py')
        try:
            bound=self._open_binding(mechanism_program)
            if self._bind_definition(research_spec,bound['definition']) != bound or package.research_spec_digest != research_spec.digest:
                raise ValueError('Complete mechanism/spec identity differs')
            required=self._mechanism_score_request(mechanism_program)
            score_contract=bind_source_contract(source.read_bytes(),required)
        except (ValueError,SyntaxError) as error:
            raise InnovationSpineError(failure_class='IMPLEMENTATION',reason_code='GRAPH_SCORE_CONTRACT_INVALID',
                message=str(error),details={'scoring_contract_status':'UNKNOWN_OR_INVALID'}) from error
        result=(self.qualify or qualify_program)({**self.config,'qualification_score_contract':score_contract,'candidate_mechanism_recipe':bound},source)
        if result['status']!='QUALIFIED':
            raise InnovationSpineError(failure_class='IMPLEMENTATION',reason_code='GRAPH_QUALIFICATION_FAILED',message=result.get('error','Graph qualification failed'),details=result)
        if result.get('score_contract_digest') != contract_digest(score_contract):
            raise RuntimeError('Qualification did not consume the selected scoring contract')
        checks=result.get('score_connection_checks',{})
        if [row['primitive_id'] for row in checks.get('entries',[])] != [row['primitive_id'] for row in score_contract['entries']]:
            raise RuntimeError('Qualification scoring coverage differs')
        binding=json.loads(Path(self.config['binding']).read_bytes())
        if (result['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest() or result['train_seed']!=self.config['train_seed']
                or result['catalog_mapping_sha256']!=binding['catalog_mapping_sha256'] or result['test_accessed'] is not False
                or result['validation_rows_accessed'] is not False or result['forward_backward_batches']!=1 or result['optimizer_steps']!=0
                or result['checkpoint_reload_verified'] is not True):
            raise RuntimeError('Graph qualification returned mismatched physical evidence')
        receipt=QualificationReceiptV1(candidate_package_ref=package.package_id,candidate_package_digest=package.digest,
            research_spec_ref=package.research_spec_ref,research_spec_digest=package.research_spec_digest,
            candidate_root_ref=package.candidate_root_ref,candidate_root_digest=package.candidate_root_digest,source_tree_digest=package.source_tree_digest,
            runtime_identity_ref=package.runtime_identity_ref,runtime_identity_digest=package.runtime_identity_digest,
            protocol_ref=research_spec.protocol_ref,protocol_digest=research_spec.protocol_digest,
            # Existing receipt ABI names its terminal stage ONE_EPOCH_SMOKE.
            # Physical evidence below explicitly records ONE_BATCH and zero steps.
            stage=QualificationStageV1.ONE_EPOCH_SMOKE,status=QualificationStatusV1.PASS,failure_class=QualificationFailureClassV1.NONE,
            static_result=QualificationCheckStatusV1.PASS,construction_result=QualificationCheckStatusV1.PASS,
            api_contract_result=QualificationCheckStatusV1.PASS,unit_result=QualificationCheckStatusV1.PASS,smoke_result=QualificationCheckStatusV1.PASS,
            failure_detail_ref=None,failure_detail_digest=None)
        return MechanicalQualificationRun(receipt=receipt,stage_observations=dict(physical_stage='GRAPH_TRAIN_ONLY_ONE_BATCH_BACKWARD_CHECKPOINT',
            legacy_stage_name_is_not_epoch_count=True,graph_qualification=result),failure_detail=None,smoke_executions=1)


@contextmanager
def open_mechanisms(adapter_module):
    """Inside existing root-context/requirements scopes, wrap fresh execute/resume."""
    with patch.object(adapter_module, 'GraphSearchAdapter', OpenMechanismAdapter):
        yield
