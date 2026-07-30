"""Thin local orchestration for accepted Research Line vNext components.

This module only composes the RC0 contracts and accepted Innovation Spine
entrypoints.  It deliberately stops after deterministic Next Fresh Profile
construction and before Provider execution or scientific outcome closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
from .vnext_contracts import (
    CapabilityKindV1,
    ExecutableProfileVNext,
    OpenResearchSpecV1,
    ProfileBuildReceiptV1,
    QualifiedCapabilityV1,
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


__all__ = [
    "admit_local_qualification",
    "build_local_next_fresh_profile",
    "qualify_local_innovation_candidate",
]
