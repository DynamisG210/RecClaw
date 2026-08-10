from __future__ import annotations

import sys
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import COMMON_EVALUATOR
from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    RecBoleQualificationFixture,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import CapabilityKindV1
from recclaw_core.research_line.bootstrap import bootstrap_search_pool
from recclaw_core.research_line.producers import produce_research_specs
from recclaw_core.research_line.runtime import (
    InnovationRuntimeInputs,
    activate_staged_innovation,
    run_research_round,
)


_HELIX_TEST_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "helix_abc_v1"
sys.path.insert(0, str(_HELIX_TEST_ROOT))

from test_runtime import (  # noqa: E402
    _bindings,
    _context,
    _environment,
    _incumbent,
    _innovation_inputs,
    _router,
    _runner,
)
from test_vnext_local_orchestration import _open_draft  # noqa: E402


def _provider_open_draft(
    producer_role: str,
    *,
    capability_family: str,
    base_model_config: str,
    model: str = "FreshCandidateModel",
) -> dict[str, object]:
    draft = _open_draft(producer_role=producer_role)
    draft.update(
        idea_mode="FRONTIER_HYPOTHESIS",
        research_question=f"Can {producer_role} move the BL-ICF frontier?",
        observed_failure_mode=None,
        closest_parent=base_model_config,
        minimal_testable_wedge="Implement one candidate-local mechanism.",
        causal_chain=[
            "candidate-local mechanism changes representation",
            "changed representation changes pairwise scores",
        ],
        discriminative_predictions=[
            "mechanism-on changes scores",
            "mechanism-off restores parent behavior",
        ],
        mechanism_off_definition="Disable the candidate-local mechanism.",
        resource_hypothesis="One bounded implementation and qualification fits.",
        realization_mode="NON_NESTED",
        execution_contract={
            "capability_family": capability_family,
            "model": model,
            "base_model_config": base_model_config,
            "config": {},
        },
    )
    return draft


def test_provider_shape_open_specs_reach_innovation_and_next_fresh_search(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:open-provider-path")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    bindings = _bindings(context, profile)

    def producer(role: str, _view: dict[str, object]) -> dict[str, object]:
        return _provider_open_draft(
            role,
            capability_family="OPEN_INTERACTION_CUSTOM",
            base_model_config="BPR",
            model="GWaveOneInteractionGate",
        )

    projected = produce_research_specs(context, producer, bindings)
    assert len(projected) == len(DISCOVERY_PRODUCERS)
    assert all(outcome.source_proposal is None for outcome in projected)

    calls: list[dict[str, object]] = []
    result = run_research_round(
        context=context,
        active_profile=profile,
        producer=producer,
        producer_bindings=bindings,
        resolver_environment=_environment(profile),
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="activate and search the admitted OpenSpec",
        innovation_inputs=_innovation_inputs(tmp_path, profile),
    )

    assert len(calls) == 1
    assert result.search_acquisition is not None
    assert result.innovation is not None and result.innovation.activation_ready
    assert result.innovation.selected_outcome.spec is not None
    assert result.innovation.selected_outcome.source_proposal is None
    assert result.innovation.search_candidate is not None
    assert result.innovation.search_candidate.feature_evidence.compile_valid is True
    next_profile, successor, proposal, qualified_execution = (
        activate_staged_innovation(result)
    )
    assert len(next_profile.entries) == 67
    assert successor.active_profile_ref == next_profile.profile_ref
    assert proposal.mechanism_id.startswith("OPEN_")
    assert qualified_execution["model"] != ""

    round_two = run_research_round(
        context=successor,
        active_profile=next_profile,
        producer=lambda role, _view: _provider_open_draft(
            role,
            capability_family="OPEN_INTERACTION_CUSTOM",
            base_model_config="BPR",
            model="GWaveOneInteractionGate",
        ),
        producer_bindings=_bindings(successor, next_profile),
        resolver_environment=_environment(next_profile),
        carryover_proposals=(),
        carryover_open_candidates=(proposal,),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=result.interpretation.policy_successor,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner(calls, seed=54305),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54305",
        next_discriminative_test="test the admitted OpenSpec realization",
        qualified_execution_by_capability={
            result.innovation.capability.capability_id: qualified_execution
        },
    )
    assert len(calls) == 2
    assert round_two.search_acquisition is not None
    assert round_two.search_acquisition.selected_binding.proposal == proposal
    assert round_two.execution_recipe["model"] == "GWaveOneInteractionGate"


def test_lightgcn_execution_contract_is_not_qualified_as_bpr(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:open-lightgcn")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    implementer_policy = fresh_r1._shared_policy(
        sha256_digest({"prompt": "family-neutral"}),
        sha256_digest({"tool_policy": "local-only"}),
        execution_contract={
            "capability_family": "OPEN_LIGHTGCN_CUSTOM",
            "model": "FreshCandidateModel",
            "base_model_config": "LightGCN",
            "config": {},
        },
    )
    source = """import torch
from torch import nn
from recbole.model.general_recommender.lightgcn import LightGCN


class FreshCandidateModel(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.open_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self):
        users, items = super().forward()
        scale = torch.tanh(self.open_residual_scale)
        return (
            users + scale * self.user_embedding.weight,
            items + scale * self.item_embedding.weight,
        )

    def calculate_loss(self, interaction):
        return super().calculate_loss(interaction)

    def predict(self, interaction):
        return super().predict(interaction)

    def full_sort_predict(self, interaction):
        return super().full_sort_predict(interaction)
"""
    implementation = {
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "files": [
            {"path": "recclaw_ext/__init__.py", "content": "# candidate package\n"},
            {"path": "recclaw_ext/candidate.py", "content": source},
        ],
        "implementation_summary": "LightGCN residual custom model fixture.",
    }

    def fixture_factory(
        _policy: object,
        attempt: int,
        _root: Path,
    ) -> RecBoleQualificationFixture:
        base = fresh_r1._qualification_fixture(
            Path(__file__).resolve().parents[2],
            seed=20260731 + attempt,
            root=tmp_path / f"qualification-{attempt}",
            base_model_config="LightGCN",
        )
        return RecBoleQualificationFixture(
            project_root=base.project_root,
            recbole_root=base.recbole_root,
            data_path=base.data_path,
            dataset=base.dataset,
            base_model_config="LightGCN",
            seed=base.seed,
            checkpoint_dir=base.checkpoint_dir,
            runtime_identity_ref=implementer_policy.runtime_identity_ref,
            runtime_identity_digest=implementer_policy.runtime_identity_digest,
        )

    innovation_inputs = InnovationRuntimeInputs(
        implementer=lambda _request: implementation,
        policy=implementer_policy,
        candidate_parent=tmp_path / "innovation-lightgcn",
        fixture_factory=fixture_factory,
        unit_check_factory=lambda _policy: fresh_r1._shared_behavioral_unit_check(
            {}, base_model_config="LightGCN"
        ),
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="open-lightgcn-v1",
        registry_version="open-lightgcn-registry-v1",
        predecessor_registry_ref="registry:fixed-66",
        predecessor_registry_digest=sha256_digest({"registry": "fixed-66"}),
        profile_version="open-lightgcn-profile-v2",
        fresh_campaign_id="campaign:open-lightgcn:fresh",
    )
    result = run_research_round(
        context=context,
        active_profile=profile,
        producer=lambda role, _view: _provider_open_draft(
            role,
            capability_family="OPEN_LIGHTGCN_CUSTOM",
            base_model_config="LightGCN",
        ),
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner([]),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="activate the LightGCN custom capability",
        innovation_inputs=innovation_inputs,
    )

    assert result.innovation is not None and result.innovation.activation_ready
    assert result.innovation.qualified_execution["base_model_config"] == "LightGCN"
    assert result.innovation.qualified_execution["model"] == "FreshCandidateModel"
