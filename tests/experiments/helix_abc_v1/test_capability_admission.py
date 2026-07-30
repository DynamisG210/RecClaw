from __future__ import annotations

from dataclasses import replace

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    CapabilityAdmissionError,
    CapabilityRegistryError,
    VersionedCapabilityRegistry,
    admit_qualified_capability,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    CandidatePackageV1,
    CapabilityKindV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)


def _digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def _spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="A local candidate can satisfy the shared RecBole contract.",
        mechanism_change="Provide a candidate-local complete model.",
        competing_explanation="Qualification may reflect executability only.",
        matched_control_requirement="Use the frozen RecBole development setup.",
        implementation_requirements=("GeneralRecommender entrypoint.",),
        expected_evidence=("Development-only qualification receipt.",),
        falsifier="Reject any non-passing mechanical qualification.",
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
        protocol_ref="protocol:local-general-cf-development",
        protocol_digest=_digest("protocol"),
        context_ref="context:capability-admission-fixture",
        context_digest=_digest("context"),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=_digest("profile"),
        producer_role="mechanism_composer",
        high_change_justification="The candidate is outside the fixed catalog.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _package(spec: OpenResearchSpecV1) -> CandidatePackageV1:
    return CandidatePackageV1(
        research_spec_ref=spec.spec_id,
        research_spec_digest=spec.digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        source_tree_digest=_digest("source-tree"),
        candidate_root_ref="candidate-root:admission-fixture",
        candidate_root_digest=_digest("candidate-root"),
        executable_entrypoint="recclaw_ext.candidate:CandidateModel",
        allowed_files=("recclaw_ext/candidate.py",),
        dependency_identity_ref="dependencies:recbole-fixture",
        dependency_identity_digest=_digest("dependencies"),
        runtime_identity_ref="runtime:recbole-fixture",
        runtime_identity_digest=_digest("runtime"),
        implementation_receipt_ref="implementation:blind-fixture",
        implementation_receipt_digest=_digest("implementation"),
        origin_blind_projection_digest=_digest("blind-projection"),
    )


def _passing_receipt(
    spec: OpenResearchSpecV1,
    package: CandidatePackageV1,
) -> QualificationReceiptV1:
    return QualificationReceiptV1(
        candidate_package_ref=package.package_id,
        candidate_package_digest=package.digest,
        research_spec_ref=spec.spec_id,
        research_spec_digest=spec.digest,
        candidate_root_ref=package.candidate_root_ref,
        candidate_root_digest=package.candidate_root_digest,
        source_tree_digest=package.source_tree_digest,
        runtime_identity_ref=package.runtime_identity_ref,
        runtime_identity_digest=package.runtime_identity_digest,
        protocol_ref=package.protocol_ref,
        protocol_digest=package.protocol_digest,
        stage=QualificationStageV1.ONE_EPOCH_SMOKE,
        status=QualificationStatusV1.PASS,
        failure_class=QualificationFailureClassV1.NONE,
        static_result=QualificationCheckStatusV1.PASS,
        construction_result=QualificationCheckStatusV1.PASS,
        api_contract_result=QualificationCheckStatusV1.PASS,
        unit_result=QualificationCheckStatusV1.PASS,
        smoke_result=QualificationCheckStatusV1.PASS,
        failure_detail_ref=None,
        failure_detail_digest=None,
    )


def _admit(
    spec: OpenResearchSpecV1,
    package: CandidatePackageV1,
    receipt: QualificationReceiptV1,
    *,
    semantic_ref: str = "semantics:candidate-model",
    semantic_digest: str | None = None,
):
    return admit_qualified_capability(
        spec,
        package,
        receipt,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref=semantic_ref,
        semantic_identity_digest=(
            semantic_digest or _digest("candidate-semantics")
        ),
    )


def test_all_stage_pass_admits_next_fresh_executability_only() -> None:
    spec = _spec()
    package = _package(spec)
    receipt = _passing_receipt(spec, package)

    capability = _admit(spec, package, receipt)
    rendered = capability.canonical_bytes().decode("utf-8").lower()

    assert capability.candidate_package_ref == package.package_id
    assert capability.candidate_package_digest == package.digest
    assert capability.source_tree_digest == package.source_tree_digest
    assert capability.qualification_receipt_ref == receipt.receipt_id
    assert capability.qualification_receipt_digest == receipt.digest
    assert capability.protocol_ref == spec.protocol_ref
    assert capability.protocol_digest == spec.protocol_digest
    assert capability.executable_entrypoint == package.executable_entrypoint
    assert capability.compatibility_requirements == (
        "general collaborative filtering",
        "pairwise input",
    )
    assert capability.current_campaign_ineligible is True
    assert capability.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert "score" not in rendered
    assert "outcome" not in rendered
    assert "mechanism_conclusion" not in rendered


def test_failed_receipt_with_not_run_suffix_is_rejected() -> None:
    spec = _spec()
    package = _package(spec)
    receipt = replace(
        _passing_receipt(spec, package),
        stage=QualificationStageV1.API_CONTRACT,
        status=QualificationStatusV1.FAIL,
        failure_class=QualificationFailureClassV1.INTERFACE,
        api_contract_result=QualificationCheckStatusV1.FAIL,
        unit_result=QualificationCheckStatusV1.NOT_RUN,
        smoke_result=QualificationCheckStatusV1.NOT_RUN,
        failure_detail_ref="qualification-failure:api-contract",
        failure_detail_digest=_digest("api-failure"),
    )

    with pytest.raises(
        CapabilityAdmissionError,
        match="all-stage passing",
    ):
        _admit(spec, package, receipt)


@pytest.mark.parametrize(
    "receipt_mutation",
    (
        {"candidate_package_digest": _digest("wrong-package")},
        {"source_tree_digest": _digest("wrong-tree")},
        {"runtime_identity_digest": _digest("wrong-runtime")},
    ),
)
def test_identity_drift_is_rejected(
    receipt_mutation: dict[str, str],
) -> None:
    spec = _spec()
    package = _package(spec)
    receipt = replace(
        _passing_receipt(spec, package),
        **receipt_mutation,
    )

    with pytest.raises(CapabilityAdmissionError, match="identity mismatch"):
        _admit(spec, package, receipt)


def test_package_protocol_drift_is_rejected() -> None:
    spec = _spec()
    package = replace(
        _package(spec),
        protocol_digest=_digest("other-protocol"),
    )
    receipt = _passing_receipt(spec, package)

    with pytest.raises(CapabilityAdmissionError, match="incompatible"):
        _admit(spec, package, receipt)


def test_registry_bytes_are_order_independent_and_content_addressed() -> None:
    spec = _spec()
    package = _package(spec)
    receipt = _passing_receipt(spec, package)
    capability_a = _admit(spec, package, receipt)
    capability_b = _admit(
        spec,
        package,
        receipt,
        semantic_ref="semantics:second-candidate",
        semantic_digest=_digest("second-semantics"),
    )
    inputs = {
        "registry_version": "2026-07-30.rc0",
        "predecessor_registry_ref": "registry:fixed-space-v2",
        "predecessor_registry_digest": _digest("predecessor-registry"),
        "protocol_ref": spec.protocol_ref,
        "protocol_digest": spec.protocol_digest,
    }

    registry_ab = VersionedCapabilityRegistry.build(
        **inputs,
        capabilities=(capability_a, capability_b),
    )
    registry_ba = VersionedCapabilityRegistry.build(
        **inputs,
        capabilities=(capability_b, capability_a),
    )
    registry_with_exact_duplicate = VersionedCapabilityRegistry.build(
        **inputs,
        capabilities=(capability_b, capability_a, capability_a),
    )

    assert registry_ab.canonical_bytes() == registry_ba.canonical_bytes()
    assert (
        registry_ab.canonical_bytes()
        == registry_with_exact_duplicate.canonical_bytes()
    )
    assert registry_ab.digest == bytes_sha256(registry_ab.canonical_bytes())
    assert registry_ab.registry_id.endswith(registry_ab.digest)
    manifest = registry_ab.canonical_dict()
    assert manifest["registry_version"] == "2026-07-30.rc0"
    assert (
        manifest["predecessor_registry_ref"]
        == "registry:fixed-space-v2"
    )
    assert (
        manifest["predecessor_registry_digest"]
        == _digest("predecessor-registry")
    )
    assert manifest["protocol_ref"] == spec.protocol_ref
    assert manifest["protocol_digest"] == spec.protocol_digest


def test_registry_rejects_protocol_drift_and_semantic_version_conflict() -> None:
    spec = _spec()
    package = _package(spec)
    receipt = _passing_receipt(spec, package)
    capability = _admit(spec, package, receipt)

    with pytest.raises(CapabilityRegistryError, match="protocol mismatch"):
        VersionedCapabilityRegistry.build(
            registry_version="2026-07-30.rc0",
            predecessor_registry_ref=None,
            predecessor_registry_digest=None,
            protocol_ref=spec.protocol_ref,
            protocol_digest=_digest("wrong-registry-protocol"),
            capabilities=(capability,),
        )

    conflicting = replace(
        capability,
        candidate_package_digest=_digest("conflicting-package"),
    )
    with pytest.raises(
        CapabilityRegistryError,
        match="conflicting duplicate semantic/version ID",
    ):
        VersionedCapabilityRegistry.build(
            registry_version="2026-07-30.rc0",
            predecessor_registry_ref=None,
            predecessor_registry_digest=None,
            protocol_ref=spec.protocol_ref,
            protocol_digest=spec.protocol_digest,
            capabilities=(capability, conflicting),
        )


def test_registry_fails_closed_on_conflicting_capability_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    package = _package(spec)
    receipt = _passing_receipt(spec, package)
    capability_a = _admit(spec, package, receipt)
    capability_b = _admit(
        spec,
        package,
        receipt,
        semantic_ref="semantics:second-candidate",
        semantic_digest=_digest("second-semantics"),
    )
    monkeypatch.setattr(
        QualifiedCapabilityV1,
        "capability_id",
        property(lambda _capability: "recclaw-qualified-capability-v1:collision"),
    )

    with pytest.raises(
        CapabilityRegistryError,
        match="conflicting duplicate capability ID",
    ):
        VersionedCapabilityRegistry.build(
            registry_version="2026-07-30.rc0",
            predecessor_registry_ref=None,
            predecessor_registry_digest=None,
            protocol_ref=spec.protocol_ref,
            protocol_digest=spec.protocol_digest,
            capabilities=(capability_a, capability_b),
        )
