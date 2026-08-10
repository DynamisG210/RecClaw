"""Thin local orchestration for accepted Research Line vNext components.

This module only composes the RC0 contracts and accepted Innovation Spine
entrypoints.  It deliberately stops after deterministic Next Fresh Profile
construction and engineering diagnostics, before Provider execution or any
outcome-bearing scientific closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

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
from .research_capability import StrongStaticRouterV1
from .research_contracts import CandidateProposalV4
from .search_adapter import (
    ExperimentAcquisitionResultV1,
    SearchExecutableProfileV1,
    SearchProfileEntryOriginV1,
    activate_next_fresh_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    route_frozen_experiment_slate,
)
from .scientific_episode import (
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
    close_scientific_episode,
)
from .vnext_contracts import (
    CapabilityResolutionV1,
    CapabilityResolutionResultV1,
    CapabilityKindV1,
    ExecutableProfileVNext,
    OpenResearchSpecV1,
    ProfileBuildReceiptV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
    ResearchFailureClassV1,
    VNextContractError,
)
if TYPE_CHECKING:
    from recclaw_core.research_line.interfaces import ProducerOutcome


def resolve_producer_outcomes(
    outcomes: Iterable[ProducerOutcome],
    *,
    environment: Mapping[str, Any],
) -> tuple[tuple[ProducerOutcome, CapabilityResolutionV1 | None], ...]:
    """Resolve successful Producer outcomes without changing typed failures."""

    resolved = []
    for outcome in outcomes:
        if outcome.spec is None:
            resolved.append((outcome, None))
            continue
        resolved.append(
            (
                outcome,
                resolve_capability(
                    outcome.spec,
                    resolution_facts=outcome.resolution_facts,
                    environment=environment,
                ),
            )
        )
    return tuple(resolved)


def route_current_search_outcomes(
    outcomes: Sequence[tuple[ProducerOutcome, CapabilityResolutionV1]],
    *,
    active_profile: SearchExecutableProfileV1,
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None = None,
) -> ExperimentAcquisitionResultV1:
    """Bind one non-empty current-profile Search Candidate Pool and route it."""

    if not outcomes:
        raise VNextContractError(
            "current Search routing requires a non-empty Candidate Pool"
        )

    bindings = []
    for outcome, resolution in outcomes:
        if outcome.spec is None or outcome.source_proposal is None:
            raise VNextContractError(
                "current Search routing requires successful Producer proposals"
            )
        if resolution.resolution is not CapabilityResolutionResultV1.SEARCH_READY:
            raise VNextContractError(
                "current Search Candidate Pool accepts SEARCH_READY outcomes only"
            )
        if (
            resolution.research_spec_ref != outcome.spec.spec_id
            or resolution.research_spec_digest != outcome.spec.digest
            or resolution.current_profile_ref != active_profile.profile_ref
            or resolution.current_profile_digest != active_profile.profile_digest
            or resolution.resolved_current_capability_ref is None
        ):
            raise VNextContractError(
                "SEARCH_READY outcome is not bound to the active capability"
            )
        bindings.append(
            bind_search_candidate(
                profile=active_profile,
                proposal=outcome.source_proposal,
                capability_ref=resolution.resolved_current_capability_ref,
            )
        )

    slate = freeze_experiment_slate(
        profile=active_profile,
        bindings=tuple(bindings),
        budget_snapshot=budget_snapshot,
    )
    return route_frozen_experiment_slate(
        profile=active_profile,
        slate=slate,
        router=router,
        policy_projection=policy_projection,
    )


def route_next_fresh_search(
    *,
    predecessor: SearchExecutableProfileV1,
    next_profile: ExecutableProfileVNext,
    registry: VersionedCapabilityRegistry,
    fresh_campaign_id: str,
    proposals: Sequence[CandidateProposalV4],
    capability_refs: Sequence[str],
    budget_snapshot: Mapping[str, Any],
    router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None = None,
) -> tuple[SearchExecutableProfileV1, ExperimentAcquisitionResultV1]:
    """Activate, bind, and route a qualified successor in a fresh campaign."""

    if not proposals or len(proposals) != len(capability_refs):
        raise VNextContractError(
            "next-fresh Search routing requires aligned non-empty proposals and capability refs"
        )

    active_profile = activate_next_fresh_search_profile(
        predecessor=predecessor,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=fresh_campaign_id,
    )
    bindings = tuple(
        bind_search_candidate(
            profile=active_profile,
            proposal=proposal,
            capability_ref=capability_ref,
        )
        for proposal, capability_ref in zip(
            proposals,
            capability_refs,
            strict=True,
        )
    )
    if not any(
        binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
        for binding in bindings
    ):
        raise VNextContractError(
            "next-fresh Search slate must bind a qualified registry capability"
        )
    slate = freeze_experiment_slate(
        profile=active_profile,
        bindings=bindings,
        budget_snapshot=budget_snapshot,
    )
    result = route_frozen_experiment_slate(
        profile=active_profile,
        slate=slate,
        router=router,
        policy_projection=policy_projection,
    )
    return active_profile, result


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
        allow_optional_unit_check=True,
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
    "resolve_producer_outcomes",
    "resolve_candidate_proposal_v4",
    "resolve_open_producer_draft",
    "route_current_search_outcomes",
    "route_next_fresh_search",
]
