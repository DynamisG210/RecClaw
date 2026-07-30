"""Deterministic admission and manifest data for qualified capabilities.

Qualification establishes next-fresh executability only.  This module does
not attach outcome metrics, mechanism conclusions, or current-campaign
authority to an admitted capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

from .canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from .vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    CandidatePackageV1,
    CapabilityKindV1,
    OpenResearchSpecV1,
    QualificationCheckStatusV1,
    QualificationFailureClassV1,
    QualificationReceiptV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)


class CapabilityAdmissionError(ValueError):
    """Raised when qualification cannot authorize next-fresh execution."""


class CapabilityRegistryError(ValueError):
    """Raised when registry manifest inputs are inconsistent."""


def _nonempty(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise CapabilityRegistryError(
            f"{field_name} must be a normalized non-empty string"
        )
    return value


def _validate_receipt_identity(
    spec: OpenResearchSpecV1,
    package: CandidatePackageV1,
    receipt: QualificationReceiptV1,
) -> None:
    expected = {
        "candidate_package_ref": package.package_id,
        "candidate_package_digest": package.digest,
        "research_spec_ref": spec.spec_id,
        "research_spec_digest": spec.digest,
        "candidate_root_ref": package.candidate_root_ref,
        "candidate_root_digest": package.candidate_root_digest,
        "source_tree_digest": package.source_tree_digest,
        "protocol_ref": package.protocol_ref,
        "protocol_digest": package.protocol_digest,
        "runtime_identity_ref": package.runtime_identity_ref,
        "runtime_identity_digest": package.runtime_identity_digest,
    }
    mismatches = tuple(
        field_name
        for field_name, expected_value in expected.items()
        if getattr(receipt, field_name) != expected_value
    )
    if mismatches:
        raise CapabilityAdmissionError(
            "qualification identity mismatch: " + ",".join(mismatches)
        )
    if (
        package.research_spec_ref != spec.spec_id
        or package.research_spec_digest != spec.digest
    ):
        raise CapabilityAdmissionError(
            "candidate package does not bind the supplied research spec"
        )
    if (
        package.protocol_ref != spec.protocol_ref
        or package.protocol_digest != spec.protocol_digest
    ):
        raise CapabilityAdmissionError(
            "candidate package is incompatible with the research protocol"
        )


def _validate_passing_receipt(receipt: QualificationReceiptV1) -> None:
    stage_results = (
        receipt.static_result,
        receipt.construction_result,
        receipt.api_contract_result,
        receipt.unit_result,
        receipt.smoke_result,
    )
    if (
        receipt.stage is not QualificationStageV1.ONE_EPOCH_SMOKE
        or receipt.status is not QualificationStatusV1.PASS
        or receipt.failure_class is not QualificationFailureClassV1.NONE
        or any(
            result is not QualificationCheckStatusV1.PASS
            for result in stage_results
        )
    ):
        raise CapabilityAdmissionError(
            "admission requires an all-stage passing qualification receipt"
        )


def admit_qualified_capability(
    spec: OpenResearchSpecV1,
    package: CandidatePackageV1,
    receipt: QualificationReceiptV1,
    *,
    capability_kind: CapabilityKindV1,
    capability_version: str,
    semantic_identity_ref: str,
    semantic_identity_digest: str,
    predecessor: QualifiedCapabilityV1 | None = None,
) -> QualifiedCapabilityV1:
    """Admit one exactly bound package for the next fresh campaign."""

    if not isinstance(spec, OpenResearchSpecV1):
        raise CapabilityAdmissionError("spec must be OpenResearchSpecV1")
    if not isinstance(package, CandidatePackageV1):
        raise CapabilityAdmissionError("package must be CandidatePackageV1")
    if not isinstance(receipt, QualificationReceiptV1):
        raise CapabilityAdmissionError(
            "receipt must be QualificationReceiptV1"
        )
    if not isinstance(capability_kind, CapabilityKindV1):
        raise CapabilityAdmissionError(
            "capability_kind is outside the RC0 capability domain"
        )
    _validate_passing_receipt(receipt)
    _validate_receipt_identity(spec, package, receipt)
    if predecessor is not None:
        if not isinstance(predecessor, QualifiedCapabilityV1):
            raise CapabilityAdmissionError(
                "predecessor must be QualifiedCapabilityV1"
            )
        if (
            predecessor.semantic_identity_ref != semantic_identity_ref
            or predecessor.semantic_identity_digest != semantic_identity_digest
        ):
            raise CapabilityAdmissionError(
                "predecessor must share the admitted semantic identity"
            )
        if (
            predecessor.protocol_ref != package.protocol_ref
            or predecessor.protocol_digest != package.protocol_digest
        ):
            raise CapabilityAdmissionError(
                "predecessor protocol is incompatible with the admitted package"
            )

    return QualifiedCapabilityV1(
        capability_kind=capability_kind,
        capability_version=capability_version,
        semantic_identity_ref=semantic_identity_ref,
        semantic_identity_digest=semantic_identity_digest,
        executable_entrypoint=package.executable_entrypoint,
        candidate_package_ref=package.package_id,
        candidate_package_digest=package.digest,
        source_tree_digest=package.source_tree_digest,
        qualification_receipt_ref=receipt.receipt_id,
        qualification_receipt_digest=receipt.digest,
        qualification_stage=receipt.stage,
        qualification_status=receipt.status,
        protocol_ref=package.protocol_ref,
        protocol_digest=package.protocol_digest,
        compatibility_requirements=spec.compatibility_requirements,
        predecessor_capability_ref=(
            predecessor.capability_id if predecessor is not None else None
        ),
        predecessor_capability_digest=(
            predecessor.digest if predecessor is not None else None
        ),
        current_campaign_ineligible=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


@dataclass(frozen=True, slots=True)
class VersionedCapabilityRegistry:
    """Canonical, content-addressed registry manifest with no mutable service."""

    registry_version: str
    predecessor_registry_ref: str | None
    predecessor_registry_digest: str | None
    protocol_ref: str
    protocol_digest: str
    capabilities: tuple[QualifiedCapabilityV1, ...]

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.versioned-capability-registry.v1"
    )
    identity_namespace: ClassVar[str] = (
        "recclaw-versioned-capability-registry-v1"
    )

    def __post_init__(self) -> None:
        _nonempty(self.registry_version, field_name="registry_version")
        _nonempty(self.protocol_ref, field_name="protocol_ref")
        protocol_digest = validate_sha256(
            self.protocol_digest,
            field_name="protocol_digest",
        )
        object.__setattr__(self, "protocol_digest", protocol_digest)
        if (self.predecessor_registry_ref is None) != (
            self.predecessor_registry_digest is None
        ):
            raise CapabilityRegistryError(
                "predecessor registry ref and digest must be paired"
            )
        if self.predecessor_registry_ref is not None:
            _nonempty(
                self.predecessor_registry_ref,
                field_name="predecessor_registry_ref",
            )
            predecessor_digest = validate_sha256(
                str(self.predecessor_registry_digest),
                field_name="predecessor_registry_digest",
            )
            object.__setattr__(
                self,
                "predecessor_registry_digest",
                predecessor_digest,
            )
        if not isinstance(self.capabilities, (tuple, list)):
            raise CapabilityRegistryError(
                "capabilities must be a finite capability sequence"
            )

        by_capability_id: dict[str, QualifiedCapabilityV1] = {}
        by_semantic_version: dict[
            tuple[str, str], QualifiedCapabilityV1
        ] = {}
        for capability in self.capabilities:
            if not isinstance(capability, QualifiedCapabilityV1):
                raise CapabilityRegistryError(
                    "registry entries must be QualifiedCapabilityV1"
                )
            if (
                capability.protocol_ref != self.protocol_ref
                or capability.protocol_digest != protocol_digest
            ):
                raise CapabilityRegistryError(
                    "registry capability protocol mismatch"
                )
            existing_id = by_capability_id.get(capability.capability_id)
            if (
                existing_id is not None
                and existing_id.digest != capability.digest
            ):
                raise CapabilityRegistryError(
                    "conflicting duplicate capability ID"
                )
            by_capability_id[capability.capability_id] = capability

            semantic_version = (
                capability.semantic_identity_ref,
                capability.capability_version,
            )
            existing_version = by_semantic_version.get(semantic_version)
            if (
                existing_version is not None
                and existing_version.capability_id != capability.capability_id
            ):
                raise CapabilityRegistryError(
                    "conflicting duplicate semantic/version ID"
                )
            by_semantic_version[semantic_version] = capability

        normalized = tuple(
            sorted(
                by_capability_id.values(),
                key=lambda capability: capability.capability_id,
            )
        )
        object.__setattr__(self, "capabilities", normalized)

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "registry_version": self.registry_version,
                "predecessor_registry_ref": self.predecessor_registry_ref,
                "predecessor_registry_digest": (
                    self.predecessor_registry_digest
                ),
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
                "capabilities": tuple(
                    capability.canonical_dict()
                    for capability in self.capabilities
                ),
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def registry_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())

    @classmethod
    def build(
        cls,
        *,
        registry_version: str,
        predecessor_registry_ref: str | None,
        predecessor_registry_digest: str | None,
        protocol_ref: str,
        protocol_digest: str,
        capabilities: Iterable[QualifiedCapabilityV1],
    ) -> VersionedCapabilityRegistry:
        return cls(
            registry_version=registry_version,
            predecessor_registry_ref=predecessor_registry_ref,
            predecessor_registry_digest=predecessor_registry_digest,
            protocol_ref=protocol_ref,
            protocol_digest=protocol_digest,
            capabilities=tuple(capabilities),
        )


__all__ = [
    "CapabilityAdmissionError",
    "CapabilityRegistryError",
    "VersionedCapabilityRegistry",
    "admit_qualified_capability",
]
