from __future__ import annotations

import sys
import unittest
from dataclasses import is_dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import recclaw_core.experiments.helix_abc_v1 as public_api  # noqa: E402
from recclaw_core.experiments.helix_abc_v1 import vnext_contracts  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    AcquisitionDecisionV1,
    AcquisitionDispositionV1,
    AcquisitionStageV1,
    CandidatePackageV1,
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    CapabilityResolutionV1,
    CurrentProfileExpressibilityV1,
    EpisodeEvidenceClassV1,
    ExecutableProfileVNext,
    NEXT_FRESH_CAMPAIGN,
    OpenResearchSpecV1,
    ProfileBuildReceiptV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
    VNextContractError,
)


def digest(label: str) -> str:
    return sha256_digest({"fixture": label})


def open_spec() -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="A gated propagation path improves sparse-user signal.",
        mechanism_change="Add a learned complete-model propagation gate.",
        competing_explanation="Any gain comes only from extra parameter capacity.",
        matched_control_requirement="Match parameter count with a disabled gate.",
        implementation_requirements=(
            "RecBole 1.2.1 general recommender interface",
            "candidate-local package",
        ),
        expected_evidence=(
            "matched comparator outcome",
            "one-epoch construction and loss smoke",
        ),
        falsifier="No matched improvement or the disabled gate explains the outcome.",
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
        protocol_ref="protocol:ml1m-general-cf:v1",
        protocol_digest=digest("protocol"),
        context_ref="context:campaign-r0",
        context_digest=digest("context"),
        current_profile_ref="profile:fixed-space-v2",
        current_profile_digest=digest("current-profile"),
        producer_role="frontier_architect",
        high_change_justification=(
            "The complete propagation model is outside the exact 66-program catalog."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def candidate_package(spec: OpenResearchSpecV1) -> CandidatePackageV1:
    return CandidatePackageV1(
        research_spec_ref=spec.spec_id,
        research_spec_digest=spec.digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        source_tree_digest=digest("candidate-source-tree"),
        candidate_root_ref="candidate-root:blind-gated-propagation",
        candidate_root_digest=digest("candidate-root"),
        executable_entrypoint="recclaw_ext.models.gated_graph:GatedGraph",
        allowed_files=(
            "tests/candidates/test_gated_graph.py",
            "recclaw_ext/models/gated_graph.py",
        ),
        dependency_identity_ref="dependencies:recbole-runtime-v1",
        dependency_identity_digest=digest("dependencies"),
        runtime_identity_ref="runtime:recbole-1.2.1-py3.10",
        runtime_identity_digest=digest("runtime"),
        implementation_receipt_ref="implementation-receipt:blind-call-1",
        implementation_receipt_digest=digest("implementation-receipt"),
        origin_blind_projection_digest=digest("origin-blind-projection"),
    )


def passing_receipt(
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
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
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


def qualified_capability(
    package: CandidatePackageV1,
    receipt: QualificationReceiptV1,
) -> QualifiedCapabilityV1:
    return QualifiedCapabilityV1(
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantics:gated-propagation",
        semantic_identity_digest=digest("capability-semantics"),
        executable_entrypoint=package.executable_entrypoint,
        candidate_package_ref=package.package_id,
        candidate_package_digest=package.digest,
        source_tree_digest=package.source_tree_digest,
        qualification_receipt_ref=receipt.receipt_id,
        qualification_receipt_digest=receipt.digest,
        qualification_stage=QualificationStageV1.ONE_EPOCH_SMOKE,
        qualification_status=QualificationStatusV1.PASS,
        protocol_ref=package.protocol_ref,
        protocol_digest=package.protocol_digest,
        compatibility_requirements=("pairwise input", "standard RecBole trainer"),
        predecessor_capability_ref=None,
        predecessor_capability_digest=None,
        current_campaign_ineligible=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


def executable_profile(
    capability: QualifiedCapabilityV1,
) -> ExecutableProfileVNext:
    return ExecutableProfileVNext(
        profile_version="vnext-rc0-fixture",
        predecessor_profile_ref="profile:fixed-space-v2",
        predecessor_profile_digest=digest("current-profile"),
        registry_ref="registry:qualified-capabilities:v1",
        registry_digest=digest("registry"),
        executable_entries=(
            (
                capability.capability_id,
                capability.digest,
                capability.executable_entrypoint,
            ),
            (
                "capability:existing-bpr",
                digest("existing-bpr"),
                "recbole.model.general_recommender.bpr:BPR",
            ),
        ),
        protocol_ref=capability.protocol_ref,
        protocol_digest=capability.protocol_digest,
        compatibility_requirements=("general collaborative filtering", "pairwise input"),
        current_campaign_eligible=False,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


def research_episode(
    capability: QualifiedCapabilityV1,
    profile: ExecutableProfileVNext,
    receipt: QualificationReceiptV1,
) -> TypedResearchEpisodeV1:
    return TypedResearchEpisodeV1(
        campaign_id="fresh-campaign-r2",
        context_ref="context:fresh-campaign-r2-round-1",
        context_digest=digest("episode-context"),
        hypothesis="The gated path improves sparse-user signal.",
        executable_capability_ref=capability.capability_id,
        executable_capability_digest=capability.digest,
        executable_profile_ref=profile.profile_id,
        executable_profile_digest=profile.digest,
        experiment_binding_ref="experiment-binding:r2-round-1",
        experiment_binding_digest=digest("experiment-binding"),
        comparator_ref="capability:matched-disabled-gate",
        comparator_digest=digest("matched-comparator"),
        outcome_ref="outcome:r2-round-1",
        outcome_digest=digest("outcome"),
        cost_ref="cost:r2-round-1",
        cost_digest=digest("cost"),
        protocol_ref=capability.protocol_ref,
        protocol_digest=capability.protocol_digest,
        evidence_class=EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT,
        experiment_executed=True,
        mechanism_interpretation="Signal is consistent with the gated path.",
        competing_explanation="Extra capacity remains a bounded alternative.",
        failure_class=ResearchFailureClassV1.NONE,
        mechanism_negative_evidence=False,
        next_discriminative_test="Repeat with the parameter-matched disabled gate.",
        qualification_receipt_ref=receipt.receipt_id,
        qualification_receipt_digest=receipt.digest,
        qualification_evidence_used_as_scientific=False,
    )


def idea_decision(spec: OpenResearchSpecV1) -> AcquisitionDecisionV1:
    return AcquisitionDecisionV1(
        stage=AcquisitionStageV1.IDEA,
        subject_kind="OPEN_RESEARCH_SPEC",
        subject_ref=spec.spec_id,
        subject_digest=spec.digest,
        policy_ref="policy:static-idea-v1",
        policy_digest=digest("idea-policy"),
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        feature_schema_ref="schema:idea-features-v1",
        feature_schema_digest=digest("idea-features"),
        feature_snapshot_digest=digest("idea-feature-snapshot"),
        budget_schema_ref="schema:idea-budget-v1",
        budget_schema_digest=digest("idea-budget"),
        budget_snapshot_digest=digest("idea-budget-snapshot"),
        eligibility_schema_ref="schema:idea-eligibility-v1",
        eligibility_schema_digest=digest("idea-eligibility"),
        eligibility_snapshot_digest=digest("idea-eligibility-snapshot"),
        disposition=AcquisitionDispositionV1.SELECT,
        reason_codes=("HIGH_CHANGE_FEASIBLE", "OUTSIDE_CURRENT_PROFILE"),
        cross_domain_schema_reuse=False,
    )


def profile_receipt(
    capability: QualifiedCapabilityV1,
    profile: ExecutableProfileVNext,
) -> ProfileBuildReceiptV1:
    return ProfileBuildReceiptV1(
        predecessor_profile_ref=profile.predecessor_profile_ref,
        predecessor_profile_hash=profile.predecessor_profile_digest,
        registry_ref=profile.registry_ref,
        registry_digest=profile.registry_digest,
        qualified_registry_refs=(
            ("registry-entry:gated-propagation", capability.digest),
        ),
        new_profile_ref=profile.profile_id,
        new_profile_hash=profile.digest,
        build_policy_ref="profile-builder:deterministic-v1",
        build_policy_digest=digest("profile-builder"),
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
        current_profile_unchanged=True,
        deterministic_rebuild=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


class VNextRC0ContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = open_spec()
        self.package = candidate_package(self.spec)
        self.qualification = passing_receipt(self.spec, self.package)
        self.capability = qualified_capability(self.package, self.qualification)
        self.profile = executable_profile(self.capability)
        self.episode = research_episode(
            self.capability,
            self.profile,
            self.qualification,
        )
        self.acquisition = idea_decision(self.spec)
        self.profile_build = profile_receipt(self.capability, self.profile)
        self.resolution = CapabilityResolutionV1(
            research_spec_ref=self.spec.spec_id,
            research_spec_digest=self.spec.digest,
            current_profile_ref=self.spec.current_profile_ref,
            current_profile_digest=self.spec.current_profile_digest,
            resolution=CapabilityResolutionResultV1.INNOVATION_REQUIRED,
            current_profile_match=False,
            resolved_current_capability_ref=None,
            resolved_current_capability_digest=None,
            capability_diff=("complete gated propagation model",),
            protocol_compatible=True,
            dependency_compatible=True,
            budget_compatible=True,
            reason_codes=("OUTSIDE_EXACT_CATALOG", "PROTOCOL_COMPATIBLE"),
            no_silent_fallback=True,
            catalog_fallback_used=False,
        )

    def test_all_nine_contracts_have_schema_bound_canonical_identity(self) -> None:
        contracts = (
            self.spec,
            self.resolution,
            self.package,
            self.qualification,
            self.capability,
            self.episode,
            self.profile,
            self.acquisition,
            self.profile_build,
        )
        self.assertEqual(len(contracts), 9)
        for contract in contracts:
            self.assertEqual(
                contract.canonical_bytes(),
                canonical_json_bytes(contract.canonical_dict()),
            )
            self.assertEqual(contract.digest, sha256_digest(contract.canonical_dict()))
            self.assertEqual(contract.record_id.rsplit(":", 1)[1], contract.digest)
            self.assertEqual(contract.canonical_dict()["schema"], contract.schema)

        reordered = replace(
            self.spec,
            implementation_requirements=tuple(
                reversed(self.spec.implementation_requirements)
            ),
            compatibility_requirements=tuple(
                reversed(self.spec.compatibility_requirements)
            ),
        )
        self.assertEqual(reordered.digest, self.spec.digest)
        self.assertNotEqual(
            replace(self.spec, hypothesis="A different hypothesis.").digest,
            self.spec.digest,
        )

    def test_public_package_exports_exact_rc0_contract_surface(self) -> None:
        contract_names = (
            "OpenResearchSpecV1",
            "CapabilityResolutionV1",
            "CandidatePackageV1",
            "QualificationReceiptV1",
            "QualifiedCapabilityV1",
            "TypedResearchEpisodeV1",
            "ExecutableProfileVNext",
            "AcquisitionDecisionV1",
            "ProfileBuildReceiptV1",
        )
        self.assertEqual(
            {
                name
                for name, value in vars(vnext_contracts).items()
                if isinstance(value, type)
                and value.__module__ == vnext_contracts.__name__
                and is_dataclass(value)
            },
            set(contract_names),
        )
        for contract_name in contract_names:
            self.assertIn(contract_name, public_api.__all__)
            self.assertIs(
                getattr(public_api, contract_name),
                globals()[contract_name],
            )

    def test_resolver_domain_is_exact_and_catalog_fallback_is_impossible(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CapabilityResolutionResultV1),
            (
                "SEARCH_READY",
                "INNOVATION_REQUIRED",
                "DEFERRED_PROTOCOL_CHANGE",
                "UNSUPPORTED",
                "INVALID_SPEC",
            ),
        )
        search_ready = replace(
            self.resolution,
            resolution=CapabilityResolutionResultV1.SEARCH_READY,
            current_profile_match=True,
            resolved_current_capability_ref="capability:existing-bpr",
            resolved_current_capability_digest=digest("existing-bpr"),
            capability_diff=(),
            reason_codes=("EXACT_PROFILE_MATCH",),
        )
        self.assertTrue(search_ready.current_profile_match)
        deferred = replace(
            self.resolution,
            resolution=CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE,
            capability_diff=("time split evaluation",),
            protocol_compatible=False,
            reason_codes=("REQUIRES_TIME_SPLIT",),
        )
        unsupported = replace(
            self.resolution,
            resolution=CapabilityResolutionResultV1.UNSUPPORTED,
            dependency_compatible=False,
            reason_codes=("DEPENDENCY_UNAVAILABLE",),
        )
        invalid = replace(
            self.resolution,
            resolution=CapabilityResolutionResultV1.INVALID_SPEC,
            capability_diff=(),
            reason_codes=("MISSING_FALSIFIER",),
        )
        self.assertIs(
            deferred.resolution,
            CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE,
        )
        self.assertIs(unsupported.resolution, CapabilityResolutionResultV1.UNSUPPORTED)
        self.assertIs(invalid.resolution, CapabilityResolutionResultV1.INVALID_SPEC)
        with self.assertRaisesRegex(VNextContractError, "silently fall back"):
            replace(self.resolution, catalog_fallback_used=True)
        with self.assertRaisesRegex(VNextContractError, "substitute a catalog"):
            replace(
                self.resolution,
                resolved_current_capability_ref="capability:nearest-fixed-entry",
                resolved_current_capability_digest=digest("nearest-fixed-entry"),
            )

    def test_candidate_package_binds_blind_root_tree_runtime_and_protocol(self) -> None:
        self.assertEqual(
            self.package.allowed_files,
            tuple(sorted(self.package.allowed_files)),
        )
        self.assertEqual(self.package.research_spec_digest, self.spec.digest)
        self.assertEqual(self.package.protocol_digest, self.spec.protocol_digest)
        with self.assertRaises(ValueError):
            replace(self.package, allowed_files=("/tmp/outside.py",))

    def test_qualification_receipt_is_development_only_and_never_mechanism_negative(self) -> None:
        self.assertNotIn(
            "MECHANISM",
            tuple(item.value for item in QualificationFailureClassV1),
        )
        self.assertEqual(self.qualification.evidence_class, "DEVELOPMENT_ONLY")
        self.assertEqual(self.qualification.mechanism_belief_authority, "NONE")
        failed = replace(
            self.qualification,
            stage=QualificationStageV1.API_CONTRACT,
            status=QualificationStatusV1.FAIL,
            failure_class=QualificationFailureClassV1.INTERFACE,
            api_contract_result=QualificationCheckStatusV1.FAIL,
            unit_result=QualificationCheckStatusV1.NOT_RUN,
            smoke_result=QualificationCheckStatusV1.NOT_RUN,
            failure_detail_ref="qualification-failure:missing-input-type",
            failure_detail_digest=digest("qualification-failure"),
        )
        self.assertIs(failed.failure_class, QualificationFailureClassV1.INTERFACE)
        with self.assertRaisesRegex(VNextContractError, "ordered PASS prefix"):
            replace(failed, construction_result=QualificationCheckStatusV1.NOT_RUN)
        with self.assertRaisesRegex(VNextContractError, "development-only"):
            replace(self.qualification, current_campaign_effect_evidence=True)

    def test_complete_models_are_capabilities_and_activate_only_next_campaign(self) -> None:
        self.assertIs(self.capability.capability_kind, CapabilityKindV1.COMPLETE_MODEL)
        self.assertTrue(self.capability.current_campaign_ineligible)
        self.assertEqual(self.capability.activation_boundary, NEXT_FRESH_CAMPAIGN)
        with self.assertRaisesRegex(VNextContractError, "current campaign"):
            replace(self.capability, current_campaign_ineligible=False)

    def test_episode_separates_engineering_failure_from_mechanism_evidence(self) -> None:
        engineering = replace(
            self.episode,
            comparator_ref=None,
            comparator_digest=None,
            evidence_class=EpisodeEvidenceClassV1.ENGINEERING_ONLY,
            experiment_executed=False,
            mechanism_interpretation="NOT_ADJUDICATED",
            failure_class=ResearchFailureClassV1.INTERFACE,
            mechanism_negative_evidence=False,
        )
        self.assertIs(engineering.evidence_class, EpisodeEvidenceClassV1.ENGINEERING_ONLY)
        with self.assertRaisesRegex(VNextContractError, "cannot become mechanism"):
            replace(engineering, mechanism_negative_evidence=True)
        with self.assertRaisesRegex(VNextContractError, "real compared experiment"):
            replace(
                self.episode,
                experiment_executed=False,
                failure_class=ResearchFailureClassV1.MECHANISM,
                mechanism_negative_evidence=True,
            )
        mechanism_negative = replace(
            self.episode,
            failure_class=ResearchFailureClassV1.MECHANISM,
            mechanism_negative_evidence=True,
            mechanism_interpretation="The gated mechanism failed under the matched test.",
        )
        self.assertTrue(mechanism_negative.experiment_executed)
        with self.assertRaisesRegex(VNextContractError, "cannot be used as scientific"):
            replace(self.episode, qualification_evidence_used_as_scientific=True)

    def test_profile_and_build_receipt_bind_registry_predecessor_and_fresh_activation(
        self,
    ) -> None:
        self.assertEqual(
            self.profile.executable_entries,
            tuple(sorted(self.profile.executable_entries)),
        )
        self.assertFalse(self.profile.current_campaign_eligible)
        self.assertTrue(self.profile_build.current_profile_unchanged)
        self.assertTrue(self.profile_build.deterministic_rebuild)
        self.assertEqual(
            self.profile_build.activation_boundary,
            NEXT_FRESH_CAMPAIGN,
        )
        self.assertEqual(len(self.profile_build.build_input_digest), 64)
        with self.assertRaisesRegex(VNextContractError, "next fresh campaign"):
            replace(self.profile, current_campaign_eligible=True)
        with self.assertRaisesRegex(VNextContractError, "preserve the current profile"):
            replace(self.profile_build, current_profile_unchanged=False)

    def test_acquisition_explicitly_separates_idea_and_experiment_domains(self) -> None:
        experiment = replace(
            self.acquisition,
            stage=AcquisitionStageV1.EXPERIMENT,
            subject_kind="EXECUTABLE_CAPABILITY",
            subject_ref=self.capability.capability_id,
            subject_digest=self.capability.digest,
            policy_ref="policy:static-experiment-v1",
            policy_digest=digest("experiment-policy"),
            feature_schema_ref="schema:experiment-features-v1",
            feature_schema_digest=digest("experiment-features"),
            feature_snapshot_digest=digest("experiment-feature-snapshot"),
            budget_schema_ref="schema:experiment-budget-v1",
            budget_schema_digest=digest("experiment-budget"),
            budget_snapshot_digest=digest("experiment-budget-snapshot"),
            eligibility_schema_ref="schema:experiment-eligibility-v1",
            eligibility_schema_digest=digest("experiment-eligibility"),
            eligibility_snapshot_digest=digest("experiment-eligibility-snapshot"),
        )
        self.assertNotEqual(
            self.acquisition.feature_schema_digest,
            experiment.feature_schema_digest,
        )
        with self.assertRaisesRegex(VNextContractError, "OPEN_RESEARCH_SPEC"):
            replace(self.acquisition, subject_kind="EXECUTABLE_CAPABILITY")
        with self.assertRaisesRegex(VNextContractError, "may not reuse"):
            replace(self.acquisition, cross_domain_schema_reuse=True)


if __name__ == "__main__":
    unittest.main()
