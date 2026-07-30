from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    CapabilityAdmissionError,
    VersionedCapabilityRegistry,
    admit_qualified_capability,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (
    MechanicalRecBoleAdapterV1,
    RecBoleQualificationFixture,
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    SharedImplementerPolicy,
    build_shared_implementer_request,
    materialize_candidate_package,
)
from recclaw_core.experiments.helix_abc_v1.next_fresh_profile import (
    NextFreshProfileBuildManifest,
    build_next_fresh_profile,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (
    runtime_release_digest,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    CapabilityKindV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "innovation_spine"
    / "data"
)
RUNTIME_REF = "runtime:recclaw-frozen-recbole"
PROTOCOL_REF = "protocol:bc-wave-1-local-general-cf"


def _digest(label: str) -> str:
    return sha256_digest({"bc_wave_1_fixture": label})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _spec(*, label: str) -> OpenResearchSpecV1:
    positive = label == "positive"
    return OpenResearchSpecV1(
        hypothesis=(
            "A fresh candidate-local pairwise model can enter the next profile."
            if positive
            else "A pointwise declaration must fail the pairwise interface gate."
        ),
        mechanism_change=(
            "Provide a fresh complete-model entrypoint outside the fixed catalog."
            if positive
            else "Provide an intentionally interface-invalid local entrypoint."
        ),
        competing_explanation=(
            "Qualification demonstrates executability, not recommendation effect."
        ),
        matched_control_requirement=(
            "Use the same deterministic RecBole fixture and mechanical qualifier."
        ),
        implementation_requirements=(
            "Candidate-local GeneralRecommender entrypoint.",
            "Frozen pairwise interaction interface.",
        ),
        expected_evidence=(
            "Development-only identity-closed qualification receipt.",
        ),
        falsifier=(
            "Reject construction, API, unit, or one-epoch qualification failure."
        ),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
        protocol_ref=PROTOCOL_REF,
        protocol_digest=_digest("protocol"),
        context_ref=f"context:bc-wave-1-{label}",
        context_digest=_digest(f"{label}-context"),
        current_profile_ref="profile:bc-current-campaign",
        current_profile_digest=_digest("current-profile"),
        producer_role="mechanism_composer",
        high_change_justification=(
            "The candidate-local complete model is outside the fixed catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _policy(*, label: str) -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=(f"recclaw_ext/{label}_candidate.py",),
        dependency_identity_ref="dependencies:bc-wave-1-recbole",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
        prompt_digest=_digest("shared-implementer-prompt"),
        tool_policy_digest=_digest("local-write-only-policy"),
        implementation_token_ceiling=4096,
    )


def _implementation_response(*, label: str) -> dict[str, Any]:
    class_name = (
        "BCWaveOnePositiveModel"
        if label == "positive"
        else "BCWaveOneInvalidModel"
    )
    imports = "from recbole.model.general_recommender.bpr import BPR\n"
    class_body = "    pass\n"
    if label == "negative":
        imports += "from recbole.utils import InputType\n"
        class_body = "    input_type = InputType.POINTWISE\n"
    return {
        "entrypoint": f"recclaw_ext.{label}_candidate:{class_name}",
        "files": [
            {
                "path": f"recclaw_ext/{label}_candidate.py",
                "content": (
                    imports
                    + "\n"
                    + f"class {class_name}(BPR):\n"
                    + class_body
                ),
            }
        ],
        "implementation_summary": (
            f"Fresh deterministic {label} BC Wave 1 fixture."
        ),
    }


def _materialize(
    tmp_path: Path,
    *,
    label: str,
):
    spec = _spec(label=label)
    policy = _policy(label=label)
    request = build_shared_implementer_request(spec, policy=policy)
    candidate_parent = tmp_path / "bc-wave-1"
    candidate_parent.mkdir(exist_ok=True)
    candidate_root = candidate_parent / str(request["blind_candidate_id"])
    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=_implementation_response(label=label),
        candidate_root=candidate_root,
        candidate_root_ref=(
            f"candidate-root:bc-wave-1:{request['blind_candidate_id']}"
        ),
    )
    return spec, request, materialized, candidate_root


def _qualification_fixture(
    tmp_path: Path,
    *,
    label: str,
) -> RecBoleQualificationFixture:
    return RecBoleQualificationFixture(
        project_root=ROOT,
        recbole_root=_recbole_root(),
        data_path=DATA_ROOT,
        dataset="mini",
        base_model_config="BPR",
        seed=20260731,
        checkpoint_dir=tmp_path / "qualification" / label / "checkpoints",
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
    )


def _shared_unit_check(
    model: object,
    config: object,
    dataset: object,
) -> None:
    from recbole.utils import ModelType

    assert config["MODEL_TYPE"] is ModelType.GENERAL
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert dataset.item_num > 1


def _assert_development_only(result: Any) -> None:
    assert result.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert result.receipt.mechanism_belief_authority == "NONE"
    assert result.receipt.current_campaign_effect_evidence is False


def test_bc_wave1_complete_local_r1_engineering_vertical(
    tmp_path: Path,
) -> None:
    (
        positive_spec,
        positive_request,
        positive_materialized,
        positive_root,
    ) = _materialize(tmp_path, label="positive")
    (
        negative_spec,
        negative_request,
        negative_materialized,
        negative_root,
    ) = _materialize(tmp_path, label="negative")
    positive_before = snapshot_candidate_tree(positive_root)
    negative_before = snapshot_candidate_tree(negative_root)
    current_campaign = {
        "profile_ref": positive_spec.current_profile_ref,
        "profile_digest": positive_spec.current_profile_digest,
        "slate_ref": "slate:bc-current-campaign",
        "slate_digest": _digest("current-slate"),
    }
    current_campaign_before = canonical_json_bytes(current_campaign)

    qualifier = MechanicalRecBoleAdapterV1()
    positive_run = qualifier.qualify(
        positive_materialized.package,
        research_spec=positive_spec,
        candidate_root=positive_root,
        fixture=_qualification_fixture(tmp_path, label="positive"),
        unit_check=_shared_unit_check,
    )
    negative_run = qualifier.qualify(
        negative_materialized.package,
        research_spec=negative_spec,
        candidate_root=negative_root,
        fixture=_qualification_fixture(tmp_path, label="negative"),
        unit_check=_shared_unit_check,
    )

    assert positive_request["schema"] == (
        "recclaw.shared-implementer-request.v1"
    )
    assert negative_request["schema"] == (
        "recclaw.shared-implementer-request.v1"
    )
    assert positive_request["blind_candidate_id"] != (
        negative_request["blind_candidate_id"]
    )
    assert positive_run.receipt.status is QualificationStatusV1.PASS
    assert positive_run.receipt.stage is (
        QualificationStageV1.ONE_EPOCH_SMOKE
    )
    assert positive_run.smoke_executions == 1
    assert (
        positive_run.stage_observations["ONE_EPOCH_SMOKE"][
            "metric_values_excluded_from_qualification"
        ]
        is True
    )
    assert snapshot_candidate_tree(positive_root) == positive_before
    _assert_development_only(positive_run)

    assert negative_run.receipt.status is QualificationStatusV1.FAIL
    assert negative_run.receipt.stage is QualificationStageV1.API_CONTRACT
    assert negative_run.receipt.failure_class is (
        QualificationFailureClassV1.INTERFACE
    )
    assert negative_run.receipt.static_result is (
        QualificationCheckStatusV1.PASS
    )
    assert negative_run.receipt.construction_result is (
        QualificationCheckStatusV1.PASS
    )
    assert negative_run.receipt.api_contract_result is (
        QualificationCheckStatusV1.FAIL
    )
    assert negative_run.receipt.unit_result is (
        QualificationCheckStatusV1.NOT_RUN
    )
    assert negative_run.receipt.smoke_result is (
        QualificationCheckStatusV1.NOT_RUN
    )
    assert negative_run.smoke_executions == 0
    assert snapshot_candidate_tree(negative_root) == negative_before
    _assert_development_only(negative_run)
    with pytest.raises(CapabilityAdmissionError, match="all-stage passing"):
        admit_qualified_capability(
            negative_spec,
            negative_materialized.package,
            negative_run.receipt,
            capability_kind=CapabilityKindV1.COMPLETE_MODEL,
            capability_version="1.0.0",
            semantic_identity_ref="semantics:bc-wave-1-negative",
            semantic_identity_digest=_digest("negative-semantics"),
        )

    capability = admit_qualified_capability(
        positive_spec,
        positive_materialized.package,
        positive_run.receipt,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantics:bc-wave-1-positive",
        semantic_identity_digest=_digest("positive-semantics"),
    )
    registry = VersionedCapabilityRegistry.build(
        registry_version="bc-wave-1-intake",
        predecessor_registry_ref="registry:bc-current-campaign",
        predecessor_registry_digest=_digest("current-registry"),
        protocol_ref=positive_spec.protocol_ref,
        protocol_digest=positive_spec.protocol_digest,
        capabilities=(capability,),
    )
    build_manifest = NextFreshProfileBuildManifest(
        profile_version="bc-wave-1-next",
        predecessor_profile_ref=positive_spec.current_profile_ref,
        predecessor_profile_digest=positive_spec.current_profile_digest,
        current_campaign_profile_ref=positive_spec.current_profile_ref,
        current_campaign_profile_digest=(
            positive_spec.current_profile_digest
        ),
        current_campaign_slate_ref=str(current_campaign["slate_ref"]),
        current_campaign_slate_digest=str(
            current_campaign["slate_digest"]
        ),
        predecessor_executable_entries=(
            (
                "capability:bc-current-bpr",
                _digest("current-bpr"),
                "recbole.model.general_recommender.bpr:BPR",
            ),
        ),
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        registry_version=registry.registry_version,
        protocol_ref=positive_spec.protocol_ref,
        protocol_digest=positive_spec.protocol_digest,
        compatibility_requirements=(
            positive_spec.compatibility_requirements
        ),
    )
    next_profile, profile_receipt = build_next_fresh_profile(
        build_manifest,
        registry,
    )

    assert capability.current_campaign_ineligible is True
    assert capability.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert next_profile.current_campaign_eligible is False
    assert next_profile.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert profile_receipt.current_profile_unchanged is True
    assert profile_receipt.deterministic_rebuild is True
    assert profile_receipt.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert canonical_json_bytes(current_campaign) == current_campaign_before

    intake = {
        "activation_boundary": NEXT_FRESH_CAMPAIGN,
        "build_manifest_ref": build_manifest.manifest_id,
        "build_manifest_digest": build_manifest.digest,
        "current_campaign_profile_ref": (
            positive_spec.current_profile_ref
        ),
        "current_campaign_profile_digest": (
            positive_spec.current_profile_digest
        ),
        "current_campaign_slate_ref": current_campaign["slate_ref"],
        "current_campaign_slate_digest": current_campaign["slate_digest"],
        "negative_candidate_package_ref": (
            negative_materialized.package.package_id
        ),
        "negative_candidate_package_digest": (
            negative_materialized.package.digest
        ),
        "negative_qualification_receipt_ref": (
            negative_run.receipt.receipt_id
        ),
        "negative_qualification_receipt_digest": (
            negative_run.receipt.digest
        ),
        "next_profile_ref": next_profile.profile_id,
        "next_profile_digest": next_profile.digest,
        "positive_candidate_package_ref": (
            positive_materialized.package.package_id
        ),
        "positive_candidate_package_digest": (
            positive_materialized.package.digest
        ),
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
        "schema": "recclaw.bc-wave-1.local-r1-intake.v1",
    }
    intake_bytes = canonical_json_bytes(intake)
    assert b"outcome" not in intake_bytes.lower()
    assert b"score" not in intake_bytes.lower()
    assert b"mechanism_belief" not in intake_bytes.lower()
    print("BC_WAVE_1_INTAKE=" + intake_bytes.decode("utf-8"))
