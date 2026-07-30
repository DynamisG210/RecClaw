from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import (
    CapabilityAdmissionError,
    CapabilityKindV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationFailureClassV1,
    QualificationStageV1,
    QualificationStatusV1,
    RecBoleQualificationFixture,
    SharedImplementerPolicy,
    admit_local_qualification,
    build_shared_implementer_request,
    qualify_local_innovation_candidate,
    runtime_release_digest,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DATA_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "innovation_spine"
    / "data"
)
RUNTIME_REF = "runtime:recclaw-frozen-recbole"


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _recbole_root() -> Path:
    import recbole

    return Path(recbole.__file__).resolve().parents[1]


def _spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis=(
            "A local candidate outside the frozen catalog should satisfy the "
            "shared RecBole qualification contract."
        ),
        mechanism_change="Introduce one candidate-local recommender model.",
        competing_explanation=(
            "A pass may only reflect the inherited BPR substrate."
        ),
        matched_control_requirement=(
            "Use the frozen BPR development configuration."
        ),
        implementation_requirements=(
            "Provide one GeneralRecommender entrypoint.",
            "Use the standard pairwise interaction contract.",
        ),
        expected_evidence=(
            "Identity-closed qualification receipt.",
            "One-epoch development smoke completion.",
        ),
        falsifier="Reject candidates that fail the shared qualifier.",
        compatibility_requirements=(
            "Frozen local general-CF protocol.",
            "RecBole 1.2.1 runtime.",
        ),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("protocol"),
        context_ref="context:local-integration-fixture",
        context_digest=_digest("context"),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=_digest("current-profile"),
        producer_role="mechanism_composer",
        high_change_justification=(
            "The fixture represents a candidate outside the frozen catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _policy() -> SharedImplementerPolicy:
    return SharedImplementerPolicy(
        allowed_files=("recclaw_ext/local_integration_fixture.py",),
        dependency_identity_ref="dependencies:recbole-bpr-fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref=RUNTIME_REF,
        runtime_identity_digest=runtime_release_digest(),
        prompt_digest=_digest("prompt"),
        tool_policy_digest=_digest("tool-policy"),
        implementation_token_ceiling=4096,
    )


def _response(*, api_invalid: bool) -> dict[str, Any]:
    imports = "from recbole.model.general_recommender.bpr import BPR\n"
    body = "    pass\n"
    class_name = "LocalIntegrationModel"
    if api_invalid:
        imports += "from recbole.utils import InputType\n"
        body = "    input_type = InputType.POINTWISE\n"
        class_name = "InvalidLocalIntegrationModel"
    return {
        "entrypoint": (
            f"recclaw_ext.local_integration_fixture:{class_name}"
        ),
        "files": [
            {
                "content": (
                    imports
                    + "\n"
                    + f"class {class_name}(BPR):\n"
                    + body
                ),
                "path": "recclaw_ext/local_integration_fixture.py",
            }
        ],
        "implementation_summary": "Local no-Provider integration fixture.",
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
        seed=20260730,
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
    parent = tmp_path / label
    parent.mkdir()
    blind_candidate_id = str(request["blind_candidate_id"])
    return (
        parent / blind_candidate_id,
        f"candidate-root:{label}:{blind_candidate_id}",
    )


def test_local_orchestration_reaches_registry_without_provider_or_outcome(
    tmp_path: Path,
) -> None:
    spec = _spec()
    policy = _policy()
    candidate_root, candidate_root_ref = _candidate_root(
        tmp_path,
        spec,
        policy,
        label="passing",
    )

    materialized, qualification = qualify_local_innovation_candidate(
        spec,
        policy=policy,
        implementation_response=_response(api_invalid=False),
        candidate_root=candidate_root,
        candidate_root_ref=candidate_root_ref,
        fixture=_fixture(tmp_path, label="passing"),
        unit_check=_unit_check,
    )
    capability, registry = admit_local_qualification(
        spec,
        materialized,
        qualification,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantic:local-integration-model",
        semantic_identity_digest=_digest("semantic-identity"),
        registry_version="local-integration-v1",
    )

    assert qualification.receipt.status is QualificationStatusV1.PASS
    assert qualification.receipt.evidence_class == "DEVELOPMENT_ONLY"
    assert qualification.receipt.current_campaign_effect_evidence is False
    assert capability.current_campaign_ineligible is True
    assert registry.capabilities == (capability,)
    assert registry.protocol_digest == spec.protocol_digest


def test_local_orchestration_preserves_failed_qualification_and_blocks_admission(
    tmp_path: Path,
) -> None:
    spec = _spec()
    policy = _policy()
    candidate_root, candidate_root_ref = _candidate_root(
        tmp_path,
        spec,
        policy,
        label="api-invalid",
    )

    materialized, qualification = qualify_local_innovation_candidate(
        spec,
        policy=policy,
        implementation_response=_response(api_invalid=True),
        candidate_root=candidate_root,
        candidate_root_ref=candidate_root_ref,
        fixture=_fixture(tmp_path, label="api-invalid"),
        unit_check=_unit_check,
    )

    assert qualification.receipt.status is QualificationStatusV1.FAIL
    assert qualification.receipt.stage is QualificationStageV1.API_CONTRACT
    assert (
        qualification.receipt.failure_class
        is QualificationFailureClassV1.INTERFACE
    )
    assert qualification.smoke_executions == 0
    with pytest.raises(
        CapabilityAdmissionError,
        match="all-stage passing qualification receipt",
    ):
        admit_local_qualification(
            spec,
            materialized,
            qualification,
            capability_kind=CapabilityKindV1.COMPLETE_MODEL,
            capability_version="1.0.0",
            semantic_identity_ref="semantic:invalid-local-model",
            semantic_identity_digest=_digest("invalid-semantic-identity"),
            registry_version="local-integration-invalid-v1",
        )
