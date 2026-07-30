from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import (
    NEXT_FRESH_CAMPAIGN,
    CapabilityAdmissionError,
    CapabilityKindV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
    RecBoleQualificationFixture,
    SharedImplementerPolicy,
    admit_local_qualification,
    build_local_next_fresh_profile,
    build_shared_implementer_request,
    qualify_local_innovation_candidate,
    runtime_release_digest,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DATA_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "innovation_spine"
    / "data"
)
RUNTIME_REF = "runtime:recclaw-frozen-recbole"
PROTOCOL_REF = "protocol:g-bc-wave-1-local-general-cf"


def _digest(label: str) -> str:
    return sha256_digest({"g_bc_wave_1_fixture": label})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _spec(*, label: str) -> OpenResearchSpecV1:
    positive = label == "positive"
    return OpenResearchSpecV1(
        hypothesis=(
            "A candidate-local pairwise model can enter the next fresh profile."
            if positive
            else "A pointwise declaration must fail the pairwise interface gate."
        ),
        mechanism_change=(
            "Provide a complete-model entrypoint outside the fixed catalog."
            if positive
            else "Provide an intentionally interface-invalid local entrypoint."
        ),
        competing_explanation=(
            "Qualification demonstrates executability, not recommendation effect."
        ),
        matched_control_requirement=(
            "Use the same deterministic RecBole fixture and qualifier."
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
        context_ref=f"context:g-bc-wave-1-{label}",
        context_digest=_digest(f"{label}-context"),
        current_profile_ref="profile:g-bc-current-campaign",
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
        allowed_files=(f"recclaw_ext/{label}_g_candidate.py",),
        dependency_identity_ref="dependencies:g-bc-wave-1-recbole",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
        prompt_digest=_digest("shared-implementer-prompt"),
        tool_policy_digest=_digest("local-write-only-policy"),
        implementation_token_ceiling=4096,
    )


def _response(*, label: str) -> dict[str, Any]:
    class_name = (
        "GBCWaveOnePositiveModel"
        if label == "positive"
        else "GBCWaveOneInvalidModel"
    )
    imports = "from recbole.model.general_recommender.bpr import BPR\n"
    class_body = "    pass\n"
    if label == "negative":
        imports += "from recbole.utils import InputType\n"
        class_body = "    input_type = InputType.POINTWISE\n"
    return {
        "entrypoint": f"recclaw_ext.{label}_g_candidate:{class_name}",
        "files": [
            {
                "content": (
                    imports
                    + "\n"
                    + f"class {class_name}(BPR):\n"
                    + class_body
                ),
                "path": f"recclaw_ext/{label}_g_candidate.py",
            }
        ],
        "implementation_summary": (
            f"Local no-Provider {label} G integration fixture."
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
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert dataset.item_num > 1


def _candidate_root(
    tmp_path: Path,
    spec: OpenResearchSpecV1,
    policy: SharedImplementerPolicy,
    *,
    label: str,
) -> tuple[Path, str]:
    request = build_shared_implementer_request(spec, policy=policy)
    parent = tmp_path / "g-bc-wave-1"
    parent.mkdir(exist_ok=True)
    blind_candidate_id = str(request["blind_candidate_id"])
    return (
        parent / blind_candidate_id,
        f"candidate-root:g-bc-wave-1:{label}:{blind_candidate_id}",
    )


def test_g_bc_wave1_complete_local_next_profile_intake(
    tmp_path: Path,
) -> None:
    positive_spec = _spec(label="positive")
    negative_spec = _spec(label="negative")
    positive_policy = _policy(label="positive")
    negative_policy = _policy(label="negative")
    positive_root, positive_root_ref = _candidate_root(
        tmp_path,
        positive_spec,
        positive_policy,
        label="positive",
    )
    negative_root, negative_root_ref = _candidate_root(
        tmp_path,
        negative_spec,
        negative_policy,
        label="negative",
    )
    current_campaign = {
        "profile_ref": positive_spec.current_profile_ref,
        "profile_digest": positive_spec.current_profile_digest,
        "slate_ref": "slate:g-bc-current-campaign",
        "slate_digest": _digest("current-slate"),
    }
    current_campaign_before = canonical_json_bytes(current_campaign)

    positive_materialized, positive_run = (
        qualify_local_innovation_candidate(
            positive_spec,
            policy=positive_policy,
            implementation_response=_response(label="positive"),
            candidate_root=positive_root,
            candidate_root_ref=positive_root_ref,
            fixture=_fixture(tmp_path, label="positive"),
            unit_check=_unit_check,
        )
    )
    negative_materialized, negative_run = (
        qualify_local_innovation_candidate(
            negative_spec,
            policy=negative_policy,
            implementation_response=_response(label="negative"),
            candidate_root=negative_root,
            candidate_root_ref=negative_root_ref,
            fixture=_fixture(tmp_path, label="negative"),
            unit_check=_unit_check,
        )
    )

    assert positive_run.receipt.status is QualificationStatusV1.PASS
    assert positive_run.receipt.stage is QualificationStageV1.ONE_EPOCH_SMOKE
    assert positive_run.smoke_executions == 1
    assert positive_run.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert positive_run.receipt.mechanism_belief_authority == "NONE"
    assert positive_run.receipt.current_campaign_effect_evidence is False

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
    assert negative_run.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert negative_run.receipt.mechanism_belief_authority == "NONE"
    assert negative_run.receipt.current_campaign_effect_evidence is False
    with pytest.raises(
        CapabilityAdmissionError,
        match="all-stage passing qualification receipt",
    ):
        admit_local_qualification(
            negative_spec,
            negative_materialized,
            negative_run,
            capability_kind=CapabilityKindV1.COMPLETE_MODEL,
            capability_version="1.0.0",
            semantic_identity_ref="semantics:g-bc-wave-1-negative",
            semantic_identity_digest=_digest("negative-semantics"),
            registry_version="g-bc-wave-1-invalid",
            predecessor_registry_ref="registry:g-bc-current-campaign",
            predecessor_registry_digest=_digest("current-registry"),
        )

    capability, registry = admit_local_qualification(
        positive_spec,
        positive_materialized,
        positive_run,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantics:g-bc-wave-1-positive",
        semantic_identity_digest=_digest("positive-semantics"),
        registry_version="g-bc-wave-1-intake",
        predecessor_registry_ref="registry:g-bc-current-campaign",
        predecessor_registry_digest=_digest("current-registry"),
    )
    manifest, next_profile, profile_receipt = (
        build_local_next_fresh_profile(
            registry,
            profile_version="g-bc-wave-1-next",
            predecessor_profile_ref=positive_spec.current_profile_ref,
            predecessor_profile_digest=positive_spec.current_profile_digest,
            current_campaign_slate_ref=str(current_campaign["slate_ref"]),
            current_campaign_slate_digest=str(
                current_campaign["slate_digest"]
            ),
            predecessor_executable_entries=(
                (
                    "capability:g-bc-current-bpr",
                    _digest("current-bpr"),
                    "recbole.model.general_recommender.bpr:BPR",
                ),
            ),
            compatibility_requirements=(
                positive_spec.compatibility_requirements
            ),
        )
    )
    rebuilt = build_local_next_fresh_profile(
        registry,
        profile_version="g-bc-wave-1-next",
        predecessor_profile_ref=positive_spec.current_profile_ref,
        predecessor_profile_digest=positive_spec.current_profile_digest,
        current_campaign_slate_ref=str(current_campaign["slate_ref"]),
        current_campaign_slate_digest=str(current_campaign["slate_digest"]),
        predecessor_executable_entries=(
            (
                "capability:g-bc-current-bpr",
                _digest("current-bpr"),
                "recbole.model.general_recommender.bpr:BPR",
            ),
        ),
        compatibility_requirements=positive_spec.compatibility_requirements,
    )

    assert rebuilt == (manifest, next_profile, profile_receipt)
    assert capability.current_campaign_ineligible is True
    assert capability.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert next_profile.current_campaign_eligible is False
    assert next_profile.activation_boundary == NEXT_FRESH_CAMPAIGN
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
        "build_manifest_ref": manifest.manifest_id,
        "build_manifest_digest": manifest.digest,
        "current_campaign_profile_ref": positive_spec.current_profile_ref,
        "current_campaign_profile_digest": (
            positive_spec.current_profile_digest
        ),
        "negative_qualification_receipt_ref": (
            negative_run.receipt.receipt_id
        ),
        "negative_qualification_receipt_digest": (
            negative_run.receipt.digest
        ),
        "next_profile_ref": next_profile.profile_id,
        "next_profile_digest": next_profile.digest,
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
        "schema": "recclaw.g-bc-wave-1.local-intake.v1",
    }
    intake_bytes = canonical_json_bytes(intake)
    assert b"outcome" not in intake_bytes.lower()
    assert b"score" not in intake_bytes.lower()
    assert b"mechanism_belief" not in intake_bytes.lower()
    print("G_BC_WAVE_1_INTAKE=" + intake_bytes.decode("utf-8"))
