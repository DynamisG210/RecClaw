from __future__ import annotations

from dataclasses import replace

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    EpisodeEvidenceClassV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)


def digest(label: str) -> str:
    return sha256_digest({"scientific_episode_fixture": label})


def comparison_identity() -> FrozenComparisonIdentityV1:
    return FrozenComparisonIdentityV1(
        campaign_id="fresh-campaign-r2",
        context_ref="context:fresh-campaign-r2-round-1",
        context_digest=digest("context"),
        executable_capability_ref="capability:gated-propagation:v1",
        executable_capability_digest=digest("capability"),
        executable_profile_ref="profile:fresh-r2:v1",
        executable_profile_digest=digest("profile"),
        experiment_binding_ref="experiment-binding:fresh-r2-round-1",
        experiment_binding_digest=digest("experiment-binding"),
        comparator_ref="capability:matched-disabled-gate:v1",
        comparator_digest=digest("comparator"),
        protocol_ref="protocol:ml1m-general-cf:v1",
        protocol_digest=digest("protocol"),
    )


def research_episode(
    identity: FrozenComparisonIdentityV1,
    *,
    failure_class: ResearchFailureClassV1 = ResearchFailureClassV1.NONE,
    evidence_class: EpisodeEvidenceClassV1 = (
        EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT
    ),
) -> TypedResearchEpisodeV1:
    mechanism_negative = failure_class is ResearchFailureClassV1.MECHANISM
    inconclusive = failure_class is ResearchFailureClassV1.INCONCLUSIVE
    return TypedResearchEpisodeV1(
        campaign_id=identity.campaign_id,
        context_ref=identity.context_ref,
        context_digest=identity.context_digest,
        hypothesis="A gated propagation path improves sparse-user signal.",
        executable_capability_ref=identity.executable_capability_ref,
        executable_capability_digest=identity.executable_capability_digest,
        executable_profile_ref=identity.executable_profile_ref,
        executable_profile_digest=identity.executable_profile_digest,
        experiment_binding_ref=identity.experiment_binding_ref,
        experiment_binding_digest=identity.experiment_binding_digest,
        comparator_ref=identity.comparator_ref,
        comparator_digest=identity.comparator_digest,
        outcome_ref="outcome:fresh-r2-round-1",
        outcome_digest=digest("outcome"),
        cost_ref="cost:fresh-r2-round-1",
        cost_digest=digest("cost"),
        protocol_ref=identity.protocol_ref,
        protocol_digest=identity.protocol_digest,
        evidence_class=evidence_class,
        experiment_executed=True,
        mechanism_interpretation=(
            "NOT_ADJUDICATED"
            if inconclusive
            else (
                "The matched outcome contradicts the gated-path mechanism."
                if mechanism_negative
                else "The matched outcome supports the gated-path mechanism."
            )
        ),
        competing_explanation="Extra capacity remains a bounded alternative.",
        failure_class=failure_class,
        mechanism_negative_evidence=mechanism_negative,
        next_discriminative_test="Repeat with a parameter-matched disabled gate.",
        qualification_receipt_ref=None,
        qualification_receipt_digest=None,
        qualification_evidence_used_as_scientific=False,
    )


def implementation_failure_receipt() -> QualificationReceiptV1:
    return QualificationReceiptV1(
        candidate_package_ref="candidate-package:gated-propagation:v1",
        candidate_package_digest=digest("candidate-package"),
        research_spec_ref="research-spec:gated-propagation:v1",
        research_spec_digest=digest("research-spec"),
        candidate_root_ref="candidate-root:gated-propagation:v1",
        candidate_root_digest=digest("candidate-root"),
        source_tree_digest=digest("source-tree"),
        runtime_identity_ref="runtime:recbole-1.2.1",
        runtime_identity_digest=digest("runtime"),
        protocol_ref="protocol:ml1m-general-cf:v1",
        protocol_digest=digest("protocol"),
        stage=QualificationStageV1.STATIC_VALIDATION,
        status=QualificationStatusV1.FAIL,
        failure_class=QualificationFailureClassV1.IMPLEMENTATION,
        static_result=QualificationCheckStatusV1.FAIL,
        construction_result=QualificationCheckStatusV1.NOT_RUN,
        api_contract_result=QualificationCheckStatusV1.NOT_RUN,
        unit_result=QualificationCheckStatusV1.NOT_RUN,
        smoke_result=QualificationCheckStatusV1.NOT_RUN,
        failure_detail_ref="qualification-failure:compile-error:v1",
        failure_detail_digest=digest("qualification-implementation-failure"),
    )


def passing_qualification_receipt() -> QualificationReceiptV1:
    return replace(
        implementation_failure_receipt(),
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


def scientific_closure(
    failure_class: ResearchFailureClassV1,
) -> ScientificEpisodeClosureV1:
    identity = comparison_identity()
    episode = research_episode(
        identity,
        failure_class=failure_class,
        evidence_class=(
            EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT
            if failure_class is ResearchFailureClassV1.INCONCLUSIVE
            else EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT
        ),
    )
    return close_scientific_episode(
        comparison_identity=identity,
        failure_class=failure_class,
        episode=episode,
        observed_outcome_ref=episode.outcome_ref,
        observed_outcome_digest=episode.outcome_digest,
    )


def diagnostic_closure(
    failure_class: ResearchFailureClassV1,
    *,
    qualification_receipt: QualificationReceiptV1 | None = None,
) -> ScientificEpisodeClosureV1:
    return close_scientific_episode(
        comparison_identity=comparison_identity(),
        failure_class=failure_class,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        qualification_receipt=qualification_receipt,
        failure_detail_ref=f"diagnostic:{failure_class.value.lower()}:v1",
        failure_detail_digest=digest(f"{failure_class.value.lower()}-detail"),
    )


def canonical_closure_fixtures() -> dict[str, ScientificEpisodeClosureV1]:
    return {
        "identity_drift": diagnostic_closure(
            ResearchFailureClassV1.IDENTITY_DRIFT
        ),
        "implementation_failure": diagnostic_closure(
            ResearchFailureClassV1.IMPLEMENTATION,
            qualification_receipt=implementation_failure_receipt(),
        ),
        "inconclusive": scientific_closure(
            ResearchFailureClassV1.INCONCLUSIVE
        ),
        "interface_failure": diagnostic_closure(
            ResearchFailureClassV1.INTERFACE
        ),
        "mechanism_negative": scientific_closure(
            ResearchFailureClassV1.MECHANISM
        ),
        "missing_outcome": diagnostic_closure(
            ResearchFailureClassV1.OUTCOME_MISSING
        ),
        "package_failure": diagnostic_closure(
            ResearchFailureClassV1.PACKAGE
        ),
        "protocol_failure": diagnostic_closure(
            ResearchFailureClassV1.PROTOCOL
        ),
        "provider_failure": diagnostic_closure(
            ResearchFailureClassV1.PROVIDER
        ),
        "resource_failure": diagnostic_closure(
            ResearchFailureClassV1.RESOURCE
        ),
        "runtime_failure": diagnostic_closure(
            ResearchFailureClassV1.RUNTIME
        ),
        "success": scientific_closure(ResearchFailureClassV1.NONE),
    }


def identity_drifted_episode() -> TypedResearchEpisodeV1:
    identity = comparison_identity()
    return replace(
        research_episode(identity),
        experiment_binding_digest=digest("drifted-experiment-binding"),
    )
