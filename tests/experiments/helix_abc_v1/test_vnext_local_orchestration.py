from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import (
    NEXT_FRESH_CAMPAIGN,
    CapabilityAdmissionError,
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    EpisodeClosureStatusV1,
    EpisodeEvidenceClassV1,
    EpisodeMemoryLaneV1,
    FrozenComparisonIdentityV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
    RecBoleQualificationFixture,
    ResearchFailureClassV1,
    SharedImplementerPolicy,
    VNextContractError,
    admit_local_qualification,
    build_local_next_fresh_profile,
    build_shared_implementer_request,
    candidate_tree_identity,
    close_local_qualification_diagnostic,
    frozen_search_bindings,
    frozen_search_resolver_environment,
    qualify_local_innovation_candidate,
    resolve_candidate_proposal_v4,
    resolve_open_producer_draft,
    runtime_release_digest,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DiscoveryCreditV1,
    MatchedControlPlanV1,
    ProposalIntentV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DATA_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "innovation_spine"
    / "data"
)
RUNTIME_REF = "runtime:recclaw-frozen-recbole"


def _digest(label: str) -> str:
    return sha256_digest({"g_wave_1_integrated_fixture": label})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _bindings(*, label: str) -> dict[str, Any]:
    return frozen_search_bindings(
        context_ref=f"context:g-wave-1-{label}",
        context_digest=_digest(f"{label}-context"),
    )


def _environment() -> dict[str, Any]:
    return frozen_search_resolver_environment(
        available_dependencies=("recbole-runtime",),
        budget_limits={"implementation_units": 2},
    )


def _open_draft(*, producer_role: str) -> dict[str, Any]:
    return {
        "producer_role": producer_role,
        "hypothesis": (
            "An interaction-conditioned feature gate changes pairwise ranking "
            "beyond every exact current-profile mechanism."
        ),
        "mechanism_change": (
            "Insert learned user-item interaction gates before pairwise scoring."
        ),
        "competing_explanation": (
            "Any apparent effect is only additional parameter capacity."
        ),
        "matched_control_requirement": (
            "Use a parameter-matched ungated pairwise model."
        ),
        "implementation_requirements": (
            "RecBole general recommender interface",
            "candidate-local package",
        ),
        "expected_evidence": (
            "interaction-gate construction",
            "matched gate ablation",
        ),
        "falsifier": (
            "A parameter-matched ungated model reproduces the full signature."
        ),
        "compatibility_requirements": (
            "general collaborative filtering",
            "pairwise input",
        ),
        "high_change_justification": (
            "The learned interaction structure is not one of the exact 66 "
            "current-profile mechanisms."
        ),
        "current_profile_expressibility_claim": "NOT_EXPRESSIBLE",
        "resolution_facts": {
            "requested_current_semantics_digest": None,
            "capability_diff": (
                "learned user-item interaction gate",
            ),
            "high_change_dimensions": (
                "INTERACTION_STRUCTURE",
                "MODEL_STRUCTURE",
            ),
            "required_dependencies": ("recbole-runtime",),
            "required_budget": {"implementation_units": 1},
        },
    }


def _policy() -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/wave1_interaction_gate.py",),
        dependency_identity_ref="dependencies:g-wave-1-recbole",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
        prompt_digest=_digest("shared-implementer-prompt"),
        tool_policy_digest=_digest("local-write-only-policy"),
        implementation_token_ceiling=4096,
    )


def _positive_response() -> dict[str, Any]:
    source = """import torch
from torch import nn

from recbole.model.general_recommender.bpr import BPR
from recbole.model.init import xavier_normal_initialization


class GWaveOneInteractionGate(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.user_gate = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_gate = nn.Linear(self.embedding_size, self.embedding_size)
        self.user_gate.apply(xavier_normal_initialization)
        self.item_gate.apply(xavier_normal_initialization)

    def _score(self, user_e, item_e):
        gate = torch.sigmoid(
            self.user_gate(user_e) + self.item_gate(item_e)
        )
        return torch.mul(user_e * gate, item_e).sum(dim=-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        positive = interaction[self.ITEM_ID]
        negative = interaction[self.NEG_ITEM_ID]
        user_e = self.user_embedding(user)
        positive_e = self.item_embedding(positive)
        negative_e = self.item_embedding(negative)
        return self.loss(
            self._score(user_e, positive_e),
            self._score(user_e, negative_e),
        )

    def predict(self, interaction):
        user_e = self.user_embedding(interaction[self.USER_ID])
        item_e = self.item_embedding(interaction[self.ITEM_ID])
        return self._score(user_e, item_e)

    def full_sort_predict(self, interaction):
        user_e = self.user_embedding(interaction[self.USER_ID])[:, None, :]
        item_e = self.item_embedding.weight[None, :, :]
        return self._score(user_e, item_e).reshape(-1)
"""
    return {
        "entrypoint": (
            "recclaw_ext.wave1_interaction_gate:GWaveOneInteractionGate"
        ),
        "files": [
            {
                "content": source,
                "path": "recclaw_ext/wave1_interaction_gate.py",
            }
        ],
        "implementation_summary": (
            "Fresh interaction-gated complete-model fixture."
        ),
    }


def _negative_response() -> dict[str, Any]:
    source = """from recbole.model.general_recommender.bpr import BPR
from recbole.utils import InputType


class GWaveOneInvalidGate(BPR):
    input_type = InputType.POINTWISE
"""
    return {
        "entrypoint": (
            "recclaw_ext.wave1_interaction_gate:GWaveOneInvalidGate"
        ),
        "files": [
            {
                "content": source,
                "path": "recclaw_ext/wave1_interaction_gate.py",
            }
        ],
        "implementation_summary": (
            "Intentional interface-negative interaction-gate fixture."
        ),
    }


def _fixture(
    tmp_path: Path,
    *,
    label: str,
) -> RecBoleQualificationFixture:
    return RecBoleQualificationFixture(
        project_root=ROOT,
        recbole_root=_recbole_root(),
        data_path=FIXTURE_DATA_ROOT,
        dataset="mini",
        base_model_config="BPR",
        seed=20260731,
        checkpoint_dir=tmp_path / "qualification" / label / "checkpoints",
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
    )


def _unit_check(model: object, config: object, dataset: object) -> None:
    from recbole.utils import ModelType

    assert config["MODEL_TYPE"] is ModelType.GENERAL
    assert hasattr(model, "user_gate")
    assert hasattr(model, "item_gate")
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert dataset.item_num > 1


def _candidate_root(
    tmp_path: Path,
    *,
    blind_candidate_id: str,
    label: str,
) -> tuple[Path, str]:
    parent = tmp_path / f"g-wave-1-{label}"
    parent.mkdir()
    return (
        parent / blind_candidate_id,
        f"candidate-root:g-wave-1:{label}:{blind_candidate_id}",
    )


def _predecessor_entries(
    environment: dict[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    by_semantics = {
        entry["semantics_digest"]: entry
        for entry in environment["current_capabilities"]
    }
    return tuple(
        (
            by_semantics[mechanism.mechanism_semantics_digest][
                "capability_ref"
            ],
            by_semantics[mechanism.mechanism_semantics_digest][
                "capability_digest"
            ],
            mechanism.entrypoint,
        )
        for mechanism in executable_mechanisms()
    )


def _comparison_identity(
    *,
    label: str,
    spec: Any,
    package: Any,
    comparator_entry: tuple[str, str, str],
) -> FrozenComparisonIdentityV1:
    return FrozenComparisonIdentityV1(
        campaign_id="g-wave-1-engineering-intake",
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        executable_capability_ref=package.package_id,
        executable_capability_digest=package.digest,
        executable_profile_ref=spec.current_profile_ref,
        executable_profile_digest=spec.current_profile_digest,
        experiment_binding_ref=f"qualification-binding:g-wave-1:{label}",
        experiment_binding_digest=_digest(f"{label}-binding"),
        comparator_ref=comparator_entry[0],
        comparator_digest=comparator_entry[1],
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
    )


def _search_ready_proposal(
    *,
    protocol_digest: str,
) -> CandidateProposalV4:
    mechanism = executable_mechanisms()[0]
    candidate_id = "cand-g-wave-1-search-ready"
    control = MatchedControlPlanV1(
        mechanism_question_digest=_digest("search-ready-question"),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest=protocol_digest,
        changed_axis=mechanism.mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    utility = SearchUtilityFeaturesV1(
        runnable_probability=0.8,
        useful_signal=0.7,
        frontier_potential=0.6,
        information_gain=0.7,
        cost=0.2,
        blocker_risk=0.1,
    )
    evidence = RouterFeatureEvidenceV1(
        compile_valid=True,
        handler_available=True,
        materializer_available=True,
        blocker_rate=0.0,
        semantic_duplicate=False,
        parent_available=True,
        mechanism_depth=len(mechanism.operator_ids),
        estimated_cost=0.2,
        llm_diagnostic=utility,
    )
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id="producer-g-wave-1-mechanism-composer",
        producer_role="mechanism_composer",
        proposal_intent=ProposalIntentV1.DISCOVERY,
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        candidate_label="Exact current-profile control",
        mechanism_hypothesis="The exact frozen mechanism remains executable.",
        competing_hypothesis="The control is neutral under the frozen protocol.",
        predicted_outcome_signature="No new capability is required.",
        failure_mode="The exact mechanism is unavailable.",
        utility_features=utility,
        feature_evidence=evidence,
        matched_control_plan=control,
        discriminative_plan=None,
        parent_candidate_id=None,
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


def test_candidate_proposal_projection_stays_search_ready() -> None:
    bindings = _bindings(label="search-ready")
    environment = _environment()
    spec, resolution = resolve_candidate_proposal_v4(
        _search_ready_proposal(
            protocol_digest=str(bindings["protocol_digest"])
        ),
        bindings=bindings,
        environment=environment,
        required_dependencies=("recbole-runtime",),
        required_budget={"implementation_units": 1},
    )

    assert spec.current_profile_expressibility_claim.value == "EXPRESSIBLE"
    assert (
        resolution.resolution
        is CapabilityResolutionResultV1.SEARCH_READY
    )
    assert resolution.current_profile_match is True
    assert resolution.catalog_fallback_used is False
    assert resolution.no_silent_fallback is True


def test_g_wave1_complete_integrated_intake(tmp_path: Path) -> None:
    environment = _environment()
    bindings_a = _bindings(label="source-a")
    bindings_b = _bindings(label="source-b")
    spec, resolution = resolve_open_producer_draft(
        _open_draft(producer_role="mechanism_composer"),
        bindings=bindings_a,
        environment=environment,
    )
    alternate_spec, alternate_resolution = resolve_open_producer_draft(
        _open_draft(producer_role="frontier_architect"),
        bindings=bindings_b,
        environment=environment,
    )

    assert len(environment["current_capabilities"]) == 66
    for observed in (resolution, alternate_resolution):
        assert (
            observed.resolution
            is CapabilityResolutionResultV1.INNOVATION_REQUIRED
        )
        assert observed.current_profile_match is False
        assert observed.resolved_current_capability_ref is None
        assert observed.resolved_current_capability_digest is None
        assert observed.catalog_fallback_used is False
        assert observed.no_silent_fallback is True
        assert observed.capability_diff

    policy = _policy()
    request = build_shared_implementer_request(spec, policy=policy)
    alternate_request = build_shared_implementer_request(
        alternate_spec,
        policy=policy,
    )
    request_bytes = canonical_json_bytes(request)
    assert request == alternate_request
    assert b"producer" not in request_bytes.lower()
    assert b"context" not in request_bytes.lower()
    assert b"origin" not in request_bytes.lower()
    assert b"outcome" not in request_bytes.lower()

    blind_candidate_id = str(request["blind_candidate_id"])
    positive_root, positive_root_ref = _candidate_root(
        tmp_path,
        blind_candidate_id=blind_candidate_id,
        label="positive",
    )
    negative_root, negative_root_ref = _candidate_root(
        tmp_path,
        blind_candidate_id=blind_candidate_id,
        label="negative",
    )
    assert not positive_root.exists()
    assert not negative_root.exists()
    current_campaign = {
        "profile_ref": spec.current_profile_ref,
        "profile_digest": spec.current_profile_digest,
        "slate_ref": "slate:g-wave-1-current",
        "slate_digest": _digest("current-slate"),
    }
    current_campaign_before = canonical_json_bytes(current_campaign)

    positive_materialized, positive_run = (
        qualify_local_innovation_candidate(
            spec,
            policy=policy,
            implementation_response=_positive_response(),
            candidate_root=positive_root,
            candidate_root_ref=positive_root_ref,
            fixture=_fixture(tmp_path, label="positive"),
            unit_check=_unit_check,
        )
    )
    negative_materialized, negative_run = (
        qualify_local_innovation_candidate(
            spec,
            policy=policy,
            implementation_response=_negative_response(),
            candidate_root=negative_root,
            candidate_root_ref=negative_root_ref,
            fixture=_fixture(tmp_path, label="negative"),
            unit_check=_unit_check,
        )
    )

    for root, root_ref, materialized in (
        (positive_root, positive_root_ref, positive_materialized),
        (negative_root, negative_root_ref, negative_materialized),
    ):
        manifest = snapshot_candidate_tree(root)
        assert tuple(row["path"] for row in manifest) == policy.allowed_files
        assert all(not (root / row["path"]).is_symlink() for row in manifest)
        assert manifest == tuple(
            materialized.implementation_receipt["written_files"]
        )
        source_digest, root_digest = candidate_tree_identity(
            root,
            candidate_root_ref=root_ref,
        )
        assert source_digest == materialized.package.source_tree_digest
        assert root_digest == materialized.package.candidate_root_digest
        assert materialized.package.digest == sha256_digest(
            materialized.package.canonical_dict()
        )

    assert positive_run.receipt.status is QualificationStatusV1.PASS
    assert positive_run.receipt.stage is QualificationStageV1.ONE_EPOCH_SMOKE
    assert positive_run.smoke_executions == 1
    assert positive_run.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert positive_run.receipt.mechanism_belief_authority == "NONE"
    assert positive_run.receipt.current_campaign_effect_evidence is False
    assert positive_run.stage_observations["CONSTRUCTION"]["model_class"] == (
        "GWaveOneInteractionGate"
    )
    api_observation = positive_run.stage_observations["API_CONTRACT"]
    assert api_observation["calculate_loss_scalars"] == 1
    assert api_observation["predict_values"] > 0
    assert (
        api_observation["full_sort_values"]
        > api_observation["predict_values"]
    )

    assert negative_run.receipt.status is QualificationStatusV1.FAIL
    assert negative_run.receipt.stage is QualificationStageV1.API_CONTRACT
    assert (
        negative_run.receipt.failure_class
        is QualificationFailureClassV1.INTERFACE
    )
    assert (
        negative_run.receipt.static_result
        is QualificationCheckStatusV1.PASS
    )
    assert (
        negative_run.receipt.construction_result
        is QualificationCheckStatusV1.PASS
    )
    assert (
        negative_run.receipt.api_contract_result
        is QualificationCheckStatusV1.FAIL
    )
    assert (
        negative_run.receipt.unit_result
        is QualificationCheckStatusV1.NOT_RUN
    )
    assert (
        negative_run.receipt.smoke_result
        is QualificationCheckStatusV1.NOT_RUN
    )
    assert negative_run.smoke_executions == 0

    predecessor_entries = _predecessor_entries(environment)
    assert len(predecessor_entries) == 66
    comparison = _comparison_identity(
        label="negative-interface",
        spec=spec,
        package=negative_materialized.package,
        comparator_entry=predecessor_entries[0],
    )
    diagnostic = close_local_qualification_diagnostic(
        comparison,
        negative_run,
        failure_class=ResearchFailureClassV1.INTERFACE,
    )
    assert diagnostic.evidence_class is EpisodeEvidenceClassV1.ENGINEERING_ONLY
    assert diagnostic.memory_lane is EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC
    assert diagnostic.mechanism_memory_allowed is False
    assert diagnostic.engineering_diagnostic_allowed is True
    assert diagnostic.episode_ref is None
    assert diagnostic.outcome_ref is None
    assert diagnostic.identity_result is EpisodeClosureStatusV1.PASS
    assert diagnostic.protocol_result is EpisodeClosureStatusV1.PASS
    assert diagnostic.package_result is EpisodeClosureStatusV1.PASS
    assert diagnostic.interface_result is EpisodeClosureStatusV1.FAIL
    assert diagnostic.execution_result is EpisodeClosureStatusV1.NOT_RUN
    assert diagnostic.outcome_result is EpisodeClosureStatusV1.NOT_RUN
    assert diagnostic.comparator_result is EpisodeClosureStatusV1.NOT_RUN
    assert diagnostic.evidence_result is EpisodeClosureStatusV1.NOT_RUN

    with pytest.raises(
        VNextContractError,
        match="passing qualification without outcome",
    ):
        close_local_qualification_diagnostic(
            _comparison_identity(
                label="positive-no-outcome",
                spec=spec,
                package=positive_materialized.package,
                comparator_entry=predecessor_entries[0],
            ),
            positive_run,
            failure_class=ResearchFailureClassV1.NONE,
        )
    with pytest.raises(
        CapabilityAdmissionError,
        match="all-stage passing qualification receipt",
    ):
        admit_local_qualification(
            spec,
            negative_materialized,
            negative_run,
            capability_kind=CapabilityKindV1.COMPLETE_MODEL,
            capability_version="1.0.0",
            semantic_identity_ref="semantics:g-wave-1-negative",
            semantic_identity_digest=_digest("negative-semantics"),
            registry_version="g-wave-1-invalid",
            predecessor_registry_ref="registry:g-wave-1-current",
            predecessor_registry_digest=_digest("current-registry"),
        )

    capability, registry = admit_local_qualification(
        spec,
        positive_materialized,
        positive_run,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantics:g-wave-1-interaction-gate",
        semantic_identity_digest=_digest("interaction-gate-semantics"),
        registry_version="g-wave-1-intake",
        predecessor_registry_ref="registry:g-wave-1-current",
        predecessor_registry_digest=_digest("current-registry"),
    )
    build_manifest, next_profile, profile_receipt = (
        build_local_next_fresh_profile(
            registry,
            profile_version="g-wave-1-next",
            predecessor_profile_ref=spec.current_profile_ref,
            predecessor_profile_digest=spec.current_profile_digest,
            current_campaign_slate_ref=str(current_campaign["slate_ref"]),
            current_campaign_slate_digest=str(
                current_campaign["slate_digest"]
            ),
            predecessor_executable_entries=predecessor_entries,
            compatibility_requirements=spec.compatibility_requirements,
        )
    )
    rebuilt = build_local_next_fresh_profile(
        registry,
        profile_version="g-wave-1-next",
        predecessor_profile_ref=spec.current_profile_ref,
        predecessor_profile_digest=spec.current_profile_digest,
        current_campaign_slate_ref=str(current_campaign["slate_ref"]),
        current_campaign_slate_digest=str(current_campaign["slate_digest"]),
        predecessor_executable_entries=tuple(reversed(predecessor_entries)),
        compatibility_requirements=tuple(
            reversed(spec.compatibility_requirements)
        ),
    )

    assert rebuilt == (build_manifest, next_profile, profile_receipt)
    assert capability.current_campaign_ineligible is True
    assert capability.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert next_profile.current_campaign_eligible is False
    assert next_profile.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert len(next_profile.executable_entries) == 67
    assert any(
        entry[0] == capability.capability_id
        for entry in next_profile.executable_entries
    )
    assert profile_receipt.current_profile_unchanged is True
    assert profile_receipt.deterministic_rebuild is True
    assert profile_receipt.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert canonical_json_bytes(current_campaign) == current_campaign_before

    intake = {
        "activation_boundary": NEXT_FRESH_CAMPAIGN,
        "build_manifest_ref": build_manifest.manifest_id,
        "build_manifest_digest": build_manifest.digest,
        "diagnostic_closure_ref": diagnostic.closure_id,
        "diagnostic_closure_digest": diagnostic.digest,
        "innovation_resolution_ref": resolution.resolution_id,
        "innovation_resolution_digest": resolution.digest,
        "negative_qualification_receipt_ref": (
            negative_run.receipt.receipt_id
        ),
        "negative_qualification_receipt_digest": (
            negative_run.receipt.digest
        ),
        "next_profile_ref": next_profile.profile_id,
        "next_profile_digest": next_profile.digest,
        "open_spec_ref": spec.spec_id,
        "open_spec_digest": spec.digest,
        "positive_qualification_receipt_ref": (
            positive_run.receipt.receipt_id
        ),
        "positive_qualification_receipt_digest": (
            positive_run.receipt.digest
        ),
        "profile_build_receipt_ref": profile_receipt.receipt_id,
        "profile_build_receipt_digest": profile_receipt.digest,
        "qualified_capability_ref": capability.capability_id,
        "qualified_capability_digest": capability.digest,
        "registry_ref": registry.registry_id,
        "registry_digest": registry.digest,
        "schema": "recclaw.g-wave-1.integrated-intake.v1",
    }
    intake_bytes = canonical_json_bytes(intake)
    assert b"outcome" not in intake_bytes.lower()
    assert b"score" not in intake_bytes.lower()
    assert b"mechanism_belief" not in intake_bytes.lower()
    assert b"episode_ref" not in intake_bytes.lower()
    print("G_WAVE_1_INTAKE=" + intake_bytes.decode("utf-8"))
