"""E2 native scientific ABI on E1's shared MAIN evaluator and qualification."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch


class MorecROExecution:
    def __init__(self, config):
        self.config = config

    def __call__(self, **kwargs):
        from e2_shared_contract import REQUIRED_PROGRAM_FILES
        from morec_main_evaluator import run_files
        from recclaw_core.experiments.helix_abc_v1.experiment_binding import ExperimentBindingV1
        root = Path(kwargs['candidate_root'])
        files = {name: (root/name).read_bytes().decode('utf-8') for name in REQUIRED_PROGRAM_FILES}
        if kwargs['seed'] != self.config['train_seed']:
            raise ValueError('RO candidate differs from the common training seed')
        digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
        binding = ExperimentBindingV1.from_execution_recipe(kwargs['execution_recipe'],
            candidate_root=root, dataset_manifest_digest=self.config['dataset_manifest_digest'],
            seed=kwargs['seed'], epochs=kwargs['epochs'], timeout_seconds=kwargs['timeout_seconds'],
            execution_purpose=kwargs['execution_purpose'], resource_telemetry=kwargs['resource_telemetry'],
            watchdog_seconds=kwargs.get('watchdog_seconds'), prefix_contract_digest=None,
            run_id=str(kwargs['run_id']), round_id=f"morec:{kwargs['run_id']}",
            claim_id='morec-cds-formal-rq1-search', permit_digest=digest('morec-cds-validation-only'),
            runtime_binding_digest=self.config['runtime_binding_digest'],
            runtime_release_digest=self.config['runtime_release_digest'],
            runner_abi='recclaw.external-comparison.e2-native-launch.v1',
            filesystem_capability_digest=digest('morec-cds-test-denied'))
        result = run_files(self.config, files)
        outcome = dict(exit_status='SUCCESS' if result['status']=='SUCCESS' else 'RUNTIME_FAILURE',
            experiment_binding=binding.canonical_dict(), binding_digest=binding.digest,
            experiment_binding_digest=binding.digest, experiment_binding_ref=binding.ref,
            execution_recipe_digest=binding.execution_recipe_digest, seed=binding.seed,
            wall_time_ms=round(result['wall_time_secs']*1000), test_accessed=False,
            morec_native_evidence=result)
        if result.get('test_accessed') is not False:
            raise ValueError('Common result lacks validation-only identity')
        if result['status']=='SUCCESS':
            # This deployed core uses the explicit primary NDCG@5 scalar ABI.
            outcome['metrics']=dict(**{'ndcg@5':result['metrics']['ndcg@5']},
                e2_raw_validation_metrics=result['metrics'], e2_controller_metric='NDCG@5')
        else:
            outcome.update(failure_phase=result.get('failure_stage','CANDIDATE_EXECUTION'),
                           failure_message=result.get('error','Native candidate failed'))
        return outcome


def search_adapter(config, conceptual_space):
    import e2_ro_search_space_adapter as native
    from ro_shared_controller_adapter import E2SharedQualifier
    from morec_main_evaluator import run_qualification

    class MainAdapter(native.E2RoSearchSpaceAdapter):
        def __init__(self):
            # The native loader already checked this fresh MAIN space. The old
            # eight-repeat package wrapper is not this run's experiment protocol.
            self.task_root=Path(config['task_root']).resolve(strict=True)
            self.initial_root=self.task_root/'initial_code'
            self.qualifier=E2SharedQualifier(self.initial_root)
            self.conceptual_space=conceptual_space

        def _proposal_schema(self):
            resources=Path(native.__file__).resolve().parent
            schema=copy.deepcopy(json.loads((resources/'e2_proposal_response.schema.json').read_text()))
            proposal=schema['properties']['proposals']['items']['properties']
            primitive_ids=sorted(self.conceptual_space.primitives)
            nullable_primitive_ids=[None,*primitive_ids]
            proposal['conceptual_primitive_ids']['items']['enum']=primitive_ids
            proposal['conceptual_target_primitive_id']['enum']=nullable_primitive_ids
            proposal['conceptual_replacement_primitive_id']['enum']=nullable_primitive_ids
            proposal['conceptual_operator_id']['enum']=[
                item['operator_id'] for item in self.conceptual_space.grammar['operators']
            ]
            custom=self.conceptual_space.grammar.get('composition_policy',{}).get('custom_synthesis',{})
            custom_schema=proposal['conceptual_custom_components']
            custom_schema['maxItems']=int(custom.get('max_components',0)) if custom.get('enabled') else 0
            custom_props=custom_schema['items']['allOf'][1]['properties']
            custom_props['family_id']={'const':self.conceptual_space.family_id}
            custom_props['slot_id']['enum']=list(custom.get('allowed_slots',())) if custom.get('enabled') else []
            return schema

        def build_provider_research_producer(self, **kwargs):
            from recclaw_core.research_line.provider import ProviderResearchProducer

            kwargs.setdefault('allowed_frozen_profile_kinds',self.supported_frozen_profile_kinds)
            resources=Path(native.__file__).resolve().parent
            return native.E2ProviderContextAdapter(
                ProviderResearchProducer(
                    **kwargs,
                    proposal_template_source=resources/'e2_producer_prompt.txt',
                    proposal_schema_source=self._proposal_schema(),
                    proposal_schema_path=resources/'e2_proposal_response.schema.json',
                ),
                self,
            )

        def qualify_materialized_candidate(self, *args, **kwargs):
            class ProcessPort:
                def __getattr__(self,name): return getattr(subprocess,name)
                def run(self,command,**options):
                    return run_qualification(config,command,**options)
            # Qualification science/receipt remain native. Its old parent-wall
            # accounting is replaced by the shared physical-worker accounting.
            with patch.object(native,'subprocess',ProcessPort()), \
                    patch.object(native,'gpu_fit_budget_path',lambda *a,**k:None), \
                    patch.object(native,'gpu_fit_budget_consumed',lambda *a,**k:0.), \
                    patch.object(native,'record_gpu_fit_seconds',lambda *a,**k:None):
                return super().qualify_materialized_candidate(*args,**kwargs)
    return MainAdapter()
