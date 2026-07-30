"""Thin local orchestration for accepted Research Line vNext components.

This module only composes the RC0 contracts and accepted Innovation Spine
entrypoints.  It deliberately stops before Next Fresh Profile construction
and scientific outcome closure, whose owner implementations are not part of
the accepted integration baseline yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

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
from .vnext_contracts import (
    CapabilityKindV1,
    OpenResearchSpecV1,
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
) -> tuple[QualifiedCapabilityV1, VersionedCapabilityRegistry]:
    """Admit a passing receipt and build the initial registry version."""

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
        predecessor_registry_ref=None,
        predecessor_registry_digest=None,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        capabilities=(capability,),
    )
    return capability, registry


__all__ = [
    "admit_local_qualification",
    "qualify_local_innovation_candidate",
]
