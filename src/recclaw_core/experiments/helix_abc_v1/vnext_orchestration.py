"""Thin local orchestration for accepted Research Line vNext components.

This module only composes the RC0 contracts and accepted Innovation Spine
entrypoints.  It deliberately stops after deterministic Next Fresh Profile
construction and engineering diagnostics, before Provider execution or any
outcome-bearing scientific closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .capability_admission import (
    VersionedCapabilityRegistry,
    admit_qualified_capability,
)
from .innovation_recbole_adapter import (
    MechanicalQualificationRun,
    MechanicalRecBoleAdapterV1,
    RecBoleQualificationFixture,
)
from .innovation_spine import (
    MaterializedCandidate,
    SharedImplementerPolicy,
    materialize_candidate_package,
)
from .next_fresh_profile import (
    NextFreshProfileBuildManifest,
    build_next_fresh_profile,
)
from .open_spec import (
    project_candidate_proposal_v4,
    project_open_producer_draft,
    resolve_capability,
)
from .research_contracts import CandidateProposalV4
from .scientific_episode import (
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
    close_scientific_episode,
)
from .vnext_contracts import (
    CapabilityResolutionV1,
    CapabilityKindV1,
    ExecutableProfileVNext,
    OpenResearchSpecV1,
    ProfileBuildReceiptV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
    ResearchFailureClassV1,
    VNextContractError,
)


def resolve_candidate_proposal_v4(
    proposal: CandidateProposalV4,
    *,
    bindings: Mapping[str, Any],
    environment: Mapping[str, Any],
    required_dependencies: Sequence[str] = (),
    required_budget: Mapping[str, int] | None = None,
) -> tuple[OpenResearchSpecV1, CapabilityResolutionV1]:
    """Project and resolve one existing frozen-profile Producer proposal."""

    spec, facts = project_candidate_proposal_v4(
        proposal,
        bindings=bindings,
        required_dependencies=required_dependencies,
        required_budget=required_budget,
    )
    return spec, resolve_capability(
        spec,
        resolution_facts=facts,
        environment=environment,
    )


def resolve_open_producer_draft(
    draft: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> tuple[OpenResearchSpecV1, CapabilityResolutionV1]:
    """Project and resolve one high-change draft from a frozen Producer role."""

    spec, facts = project_open_producer_draft(draft, bindings=bindings)
    return spec, resolve_capability(
        spec,
        resolution_facts=facts,
        environment=environment,
    )


def qualify_local_innovation_candidate(
    spec: OpenResearchSpecV1,
    *,
    policy: SharedImplementerPolicy,
    implementation_response: Mapping[str, Any],
    candidate_root: Path,
    candidate_root_ref: str,
    fixture: RecBoleQualificationFixture,
    unit_check: Callable[[Any, Any, Any], None],
) -> tuple[MaterializedCandidate, MechanicalQualificationRun]:
    """Materialize and qualify one already-produced implementation response.

    The caller supplies the implementation response directly.  No Provider,
    experiment outcome, current-profile mutation, or scientific interpretation
    is reachable through this boundary.
    """

    materialized = materialize_candidate_package(
        spec,
        policy=policy,
        implementation_response=implementation_response,
        candidate_root=candidate_root,
        candidate_root_ref=candidate_root_ref,
    )
    qualification = MechanicalRecBoleAdapterV1().qualify(
        materialized.package,
        research_spec=spec,
        candidate_root=candidate_root,
        fixture=fixture,
        unit_check=unit_check,
    )
    return materialized, qualification


def admit_local_qualification(
    spec: OpenResearchSpecV1,
    materialized: MaterializedCandidate,
    qualification: MechanicalQualificationRun,
    *,
    capability_kind: CapabilityKindV1,
    capability_version: str,
    semantic_identity_ref: str,
    semantic_identity_digest: str,
    registry_version: str,
    predecessor_registry_ref: str,
    predecessor_registry_digest: str,
) -> tuple[QualifiedCapabilityV1, VersionedCapabilityRegistry]:
    """Admit a passing receipt and build its next registry version."""

    capability = admit_qualified_capability(
        spec,
        materialized.package,
        qualification.receipt,
        capability_kind=capability_kind,
        capability_version=capability_version,
        semantic_identity_ref=semantic_identity_ref,
        semantic_identity_digest=semantic_identity_digest,
        predecessor=None,
    )
    registry = VersionedCapabilityRegistry.build(
        registry_version=registry_version,
        predecessor_registry_ref=predecessor_registry_ref,
        predecessor_registry_digest=predecessor_registry_digest,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        capabilities=(capability,),
    )
    return capability, registry


def build_local_next_fresh_profile(
    registry: VersionedCapabilityRegistry,
    *,
    profile_version: str,
    predecessor_profile_ref: str,
    predecessor_profile_digest: str,
    current_campaign_slate_ref: str,
    current_campaign_slate_digest: str,
    predecessor_executable_entries: Iterable[tuple[str, str, str]],
    compatibility_requirements: Iterable[str],
) -> tuple[
    NextFreshProfileBuildManifest,
    ExecutableProfileVNext,
    ProfileBuildReceiptV1,
]:
    """Build a successor bound to the unchanged current-profile identity."""

    manifest = NextFreshProfileBuildManifest(
        profile_version=profile_version,
        predecessor_profile_ref=predecessor_profile_ref,
        predecessor_profile_digest=predecessor_profile_digest,
        current_campaign_profile_ref=predecessor_profile_ref,
        current_campaign_profile_digest=predecessor_profile_digest,
        current_campaign_slate_ref=current_campaign_slate_ref,
        current_campaign_slate_digest=current_campaign_slate_digest,
        predecessor_executable_entries=tuple(
            predecessor_executable_entries
        ),
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        registry_version=registry.registry_version,
        protocol_ref=registry.protocol_ref,
        protocol_digest=registry.protocol_digest,
        compatibility_requirements=tuple(compatibility_requirements),
    )
    profile, receipt = build_next_fresh_profile(manifest, registry)
    return manifest, profile, receipt


def close_local_qualification_diagnostic(
    comparison_identity: FrozenComparisonIdentityV1,
    qualification: MechanicalQualificationRun,
    *,
    failure_class: ResearchFailureClassV1,
) -> ScientificEpisodeClosureV1:
    """Route one failed qualification only to D0 engineering diagnostics."""

    receipt = qualification.receipt
    if receipt.status is not QualificationStatusV1.FAIL:
        raise VNextContractError(
            "passing qualification without outcome creates no Episode or "
            "scientific closure"
        )
    if receipt.failure_class.value != failure_class.value:
        raise VNextContractError(
            "qualification and diagnostic failure classes must match"
        )
    return close_scientific_episode(
        comparison_identity=comparison_identity,
        failure_class=failure_class,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        qualification_receipt=receipt,
        failure_detail_ref=receipt.failure_detail_ref,
        failure_detail_digest=receipt.failure_detail_digest,
    )


__all__ = [
    "admit_local_qualification",
    "build_local_next_fresh_profile",
    "close_local_qualification_diagnostic",
    "qualify_local_innovation_candidate",
    "resolve_candidate_proposal_v4",
    "resolve_open_producer_draft",
]
