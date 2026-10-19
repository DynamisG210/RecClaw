"""Task-only graph adapter; native binding, materializer and feedback stay native."""
import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from graph_native_methods import GraphNativeMethods,GraphProviderContextAdapter,GraphImplementerContextAdapter
from graph_autospace import digest,engine,METHOD
from graph_score_connection import score_request,bind_source_contract,contract_digest

ROOT=Path(__file__).resolve().parent
REQUIREMENTS=('general collaborative filtering','offline top-n evaluation','train-only fitting','full-sort development NDCG@5','test access denied')
DEPENDENCIES=('python-stdlib','torch','pytorch','numpy','scipy','recbole')
CONSTRUCTION='Complete root-relative SELECT/COMPOSE. Every candidate starts from the supplied original root; retain a measured mechanism only by explicitly selecting it. Historical positive and negative outcomes remain research evidence.'

@contextmanager
def graph_materializer_abi():
    """Accept one exact task-owned path, scoped to this isolated RO process."""
    from recclaw_core.experiments.helix_abc_v1 import innovation_spine as native
    original=native._validated_entrypoint
    def entrypoint(value,*,allowed_files):
        if value=='candidate:FreshCandidateModel' and allowed_files==('candidate.py',): return value
        return original(value,allowed_files=allowed_files)
    with patch.object(native,'_CANDIDATE_PREFIXES',native._CANDIDATE_PREFIXES+('candidate.py',)),patch.object(native,'_validated_entrypoint',entrypoint):
        yield

class GraphSearchAdapter(GraphNativeMethods):
    adapter_id='recclaw.search-space-adapter.graph-model-loss.v1'
    supported_frozen_profile_kinds=('OFFLINE_TOPN',)
    implementation_allowed_files=('candidate.py',)
    available_dependencies=DEPENDENCIES
    protocol_requirements=REQUIREMENTS
    compatibility_requirements=REQUIREMENTS

    def __init__(self,config,launch,facts,conceptual_space,*,qualify=None):
        self.config=config;self.launch=launch;self.facts=dict(facts,dataset=json.loads(Path(config['binding']).read_bytes())['dataset']);self.task_id=facts['task_id']
        self.conceptual_space=conceptual_space
        self.parent=Path(config['parent_source']).read_bytes().decode('utf-8')
        self.parent_id=self.task_id+':original-root'
        self.qualify=qualify

    def task_context(self):
        return dict(task=self.facts,primary_metric='NDCG@5',candidate_universe=self.facts['evaluation']['candidate_universe'],test_access='DENIED',
            allowed_program_files=['candidate.py'],entrypoint='candidate:FreshCandidateModel',
            construction=CONSTRUCTION,parent_id=self.parent_id,conceptual_search_space=self.conceptual_space.provider_projection())

    def provider_context(self,context):
        result=dict(context);knowledge=dict(result.get('knowledge_base',{}));knowledge['search_space']=self.task_context();result['knowledge_base']=knowledge
        result['frozen_goal']={**dict(result.get('frozen_goal',{})),'metric':'NDCG@5','direction':'maximize'}
        result['budget']={**dict(result.get('budget',{})),'dataset':self.facts['dataset'],'candidate_universe':self.facts['evaluation']['candidate_universe'],'heldout_access':'DENIED'}
        return result

    def adapt_initial_context(self,context):
        projected=self.provider_context(context.producer_view('frontier_architect'))
        return replace(context,knowledge_base=projected['knowledge_base'],frozen_goal=projected['frozen_goal'],budget=projected['budget'])

    def adapt_innovation_inputs(self,inputs):
        # Physical qualification/resources belong to the common graph evaluator.
        return replace(inputs,resource_admission_required=False,resource_probe=None,resource_probe_parent=None)

    def adapt_resolver_environment(self,environment):
        return {**dict(environment),'protocol_requirements':REQUIREMENTS,'available_dependencies':DEPENDENCIES}

    def bind_conceptual_proposal(self,**kwargs):
        if kwargs.get('custom_components') or kwargs.get('target_primitive_id') or kwargs.get('replacement_primitive_id'):
            raise ValueError('This fixed graph method uses SELECT/COMPOSE only')
        ids=engine(self.parent).canonicalize_selection(tuple(kwargs['primitive_ids']),self.conceptual_space.primitives)
        return super().bind_conceptual_proposal(**dict(kwargs,primitive_ids=ids))

    def resolve_confirmation(self,kind,primary_binding,context):
        if context.get('phase')=='PROJECT_PROVIDER_PROPOSAL':
            from recclaw_core.research_line.search_space_adapter import ConfirmationResolutionV1,ConfirmationResolutionKindV1
            draft=primary_binding
            execution=draft.get('execution_contract',{})
            if draft.get('closest_parent')!=self.parent_id or execution.get('config_json','{}')!='{}':
                return ConfirmationResolutionV1(ConfirmationResolutionKindV1.UNSUPPORTED,reason='Graph proposal must use original root and fixed external config')
        return super().resolve_confirmation(kind,primary_binding,context)

    def proposal_schema(self):
        schema=json.loads((ROOT/'resources/graph_proposal.schema.json').read_bytes())
        p=schema['properties']['proposals']['items']['properties']
        p['compatibility_requirements']['items']['enum']=list(self.compatibility_requirements)
        p['conceptual_primitive_ids']['items']['enum']=sorted(self.conceptual_space.primitives)
        p['closest_parent']={'const':self.parent_id,'type':'string'}
        p['conceptual_operator_id']['enum']=[row['operator_id'] for row in self.conceptual_space.grammar['operators']]
        return schema

    def build_provider_research_producer(self,**kwargs):
        from recclaw_core.research_line.provider import ProviderResearchProducer
        kwargs.setdefault('allowed_frozen_profile_kinds',self.supported_frozen_profile_kinds)
        return GraphProviderContextAdapter(ProviderResearchProducer(**kwargs,
            proposal_template_source=ROOT/'resources/graph_producer.txt',proposal_schema_source=self.proposal_schema(),
            proposal_schema_path=ROOT/'resources/graph_proposal.schema.json'),self)

    def build_provider_implementer(self,**kwargs):
        kwargs.setdefault('implementation_output_token_ceiling_per_physical_call',20000)
        from recclaw_core.research_line.provider import ProviderImplementerGateway
        return GraphImplementerContextAdapter(ProviderImplementerGateway(**kwargs,
            implementation_template_source=ROOT/'resources/graph_implementer.txt',
            implementation_schema_source=ROOT/'resources/graph_implementation.schema.json',
            implementation_schema_path=ROOT/'resources/graph_implementation.schema.json',bl_icf_implementation_appendix_source=None),self)

    def implementation_context(self,request):
        return {**dict(request),'graph_task':self.task_context(),'initial_program':{'candidate.py':self.parent}}

    def adapt_implementation_request(self,request,*,mechanism_program):
        bound=self._trusted_conceptual_binding(mechanism_program)
        primitives=[self.conceptual_space.primitives[p].record for p in bound.primitive_ids]
        required=score_request(primitives,bound.proposal_digest,bound.primitive_ids)
        return {**dict(request),'initial_program':{'candidate.py':self.parent},
            'graph_score_contract_request':required,'autospace_recipe':dict(
            construction=CONSTRUCTION,binding=bound.as_dict(),primitives=primitives)}

    def canonicalize_implementation_response(self,response,request):
        # Never rewrite a generated graph mechanism or silently substitute root.
        return response

    def implementation_policy(self,policy):
        from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
        return replace(policy,allowed_files=('candidate.py',),prompt_digest=sha256_digest((ROOT/'resources/graph_implementer.txt').read_text()))

    def qualify_materialized_candidate(self,materialized,candidate_root,research_spec,*,mechanism_program=None):
        return self.qualification_executor(materialized.package,research_spec=research_spec,
            candidate_root=candidate_root,mechanism_program=mechanism_program)

    def qualification_executor(self,package,*,research_spec,candidate_root,fixture=None,unit_check=None,mechanism_program=None):
        from graph_qualification import qualify_program
        from recclaw_core.experiments.helix_abc_v1.innovation_spine import InnovationSpineError
        from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import MechanicalQualificationRun
        from recclaw_core.experiments.helix_abc_v1.vnext_contracts import QualificationReceiptV1,QualificationStageV1,QualificationStatusV1,QualificationFailureClassV1,QualificationCheckStatusV1
        source=Path(candidate_root)/'candidate.py'
        if set(p.name for p in Path(candidate_root).iterdir() if p.is_file())!={'candidate.py'}:
            raise ValueError('Graph materializer must emit only candidate.py')
        try:
            bound=self._trusted_conceptual_binding(mechanism_program)
            if bound.proposal_digest != research_spec.digest or package.research_spec_digest != research_spec.digest:
                raise ValueError('Selected score contract/spec identity differs')
            required=score_request([self.conceptual_space.primitives[p].record for p in bound.primitive_ids],
                                   research_spec.digest,bound.primitive_ids)
            score_contract=bind_source_contract(source.read_bytes(),required)
        except (ValueError,SyntaxError) as error:
            raise InnovationSpineError(failure_class='IMPLEMENTATION',reason_code='GRAPH_SCORE_CONTRACT_INVALID',
                message=str(error),details={'scoring_contract_status':'UNKNOWN_OR_INVALID'}) from error
        result=(self.qualify or qualify_program)({**self.config,'qualification_score_contract':score_contract},source)
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

    def describe(self): return json.dumps(self.task_context(),sort_keys=True)
