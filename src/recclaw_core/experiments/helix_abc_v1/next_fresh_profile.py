"""Deterministic, next-fresh-only executable profile construction."""

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
from .capability_admission import VersionedCapabilityRegistry
from .vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    ExecutableProfileVNext,
    ProfileBuildReceiptV1,
)


class NextFreshProfileBuildError(ValueError):
    """Raised when a profile build input is incompatible or identity-drifted."""


def _normalized_string(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise NextFreshProfileBuildError(
            f"{field_name} must be a normalized non-empty string"
        )
    return value


def _normalized_entrypoint(value: str) -> str:
    normalized = _normalized_string(
        value,
        field_name="executable_entrypoint",
    )
    module_name, separator, attribute = normalized.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute
        or any(character.isspace() for character in normalized)
    ):
        raise NextFreshProfileBuildError(
            "executable entrypoints must use module.path:Attribute"
        )
    return normalized


def _normalized_entries(
    values: Iterable[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], ...]:
    by_ref: dict[str, tuple[str, str, str]] = {}
    try:
        entries = tuple(values)
    except TypeError as error:
        raise NextFreshProfileBuildError(
            "predecessor entries must be a finite iterable"
        ) from error
    for index, value in enumerate(entries):
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise NextFreshProfileBuildError(
                f"predecessor entry {index} must be a ref/digest/entrypoint tuple"
            )
        ref = _normalized_string(str(value[0]), field_name="capability_ref")
        digest = validate_sha256(
            str(value[1]),
            field_name="capability_digest",
        )
        entrypoint = _normalized_entrypoint(str(value[2]))
        entry = (ref, digest, entrypoint)
        existing = by_ref.get(ref)
        if existing is not None and existing != entry:
            raise NextFreshProfileBuildError(
                "conflicting predecessor executable capability ref"
            )
        by_ref[ref] = entry
    return tuple(sorted(by_ref.values(), key=lambda item: item[0]))


def _normalized_requirements(values: Iterable[str]) -> tuple[str, ...]:
    try:
        requirements = tuple(
            _normalized_string(
                str(value),
                field_name="compatibility_requirement",
            )
            for value in values
        )
    except TypeError as error:
        raise NextFreshProfileBuildError(
            "compatibility requirements must be a finite iterable"
        ) from error
    if not requirements or len(set(requirements)) != len(requirements):
        raise NextFreshProfileBuildError(
            "compatibility requirements must be unique and non-empty"
        )
    return tuple(sorted(requirements))


@dataclass(frozen=True, slots=True)
class NextFreshProfileBuildManifest:
    """Content-addressed build inputs, including the untouched campaign view."""

    profile_version: str
    predecessor_profile_ref: str
    predecessor_profile_digest: str
    current_campaign_profile_ref: str
    current_campaign_profile_digest: str
    current_campaign_slate_ref: str
    current_campaign_slate_digest: str
    predecessor_executable_entries: tuple[tuple[str, str, str], ...]
    registry_ref: str
    registry_digest: str
    registry_version: str
    protocol_ref: str
    protocol_digest: str
    compatibility_requirements: tuple[str, ...]

    schema: ClassVar[str] = (
        "recclaw.research-line.vnext.next-fresh-profile-build-manifest.v1"
    )
    identity_namespace: ClassVar[str] = (
        "recclaw-next-fresh-profile-build-manifest-v1"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "profile_version",
            "predecessor_profile_ref",
            "current_campaign_profile_ref",
            "current_campaign_slate_ref",
            "registry_ref",
            "registry_version",
            "protocol_ref",
        ):
            _normalized_string(
                getattr(self, field_name),
                field_name=field_name,
            )
        for field_name in (
            "predecessor_profile_digest",
            "current_campaign_profile_digest",
            "current_campaign_slate_digest",
            "registry_digest",
            "protocol_digest",
        ):
            normalized = validate_sha256(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(self, field_name, normalized)
        if (
            self.predecessor_profile_ref
            != self.current_campaign_profile_ref
            or self.predecessor_profile_digest
            != self.current_campaign_profile_digest
        ):
            raise NextFreshProfileBuildError(
                "predecessor identity must equal the untouched current profile"
            )
        object.__setattr__(
            self,
            "predecessor_executable_entries",
            _normalized_entries(self.predecessor_executable_entries),
        )
        object.__setattr__(
            self,
            "compatibility_requirements",
            _normalized_requirements(self.compatibility_requirements),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "profile_version": self.profile_version,
                "predecessor_profile_ref": self.predecessor_profile_ref,
                "predecessor_profile_digest": (
                    self.predecessor_profile_digest
                ),
                "current_campaign_profile_ref": (
                    self.current_campaign_profile_ref
                ),
                "current_campaign_profile_digest": (
                    self.current_campaign_profile_digest
                ),
                "current_campaign_slate_ref": (
                    self.current_campaign_slate_ref
                ),
                "current_campaign_slate_digest": (
                    self.current_campaign_slate_digest
                ),
                "predecessor_executable_entries": (
                    self.predecessor_executable_entries
                ),
                "registry_ref": self.registry_ref,
                "registry_digest": self.registry_digest,
                "registry_version": self.registry_version,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
                "compatibility_requirements": (
                    self.compatibility_requirements
                ),
                "activation_boundary": NEXT_FRESH_CAMPAIGN,
            }
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_dict())

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def manifest_id(self) -> str:
        return content_id(self.identity_namespace, self.canonical_dict())


def build_next_fresh_profile(
    manifest: NextFreshProfileBuildManifest,
    registry: VersionedCapabilityRegistry,
) -> tuple[ExecutableProfileVNext, ProfileBuildReceiptV1]:
    """Build one deterministic successor without touching campaign state."""

    if not isinstance(manifest, NextFreshProfileBuildManifest):
        raise NextFreshProfileBuildError(
            "manifest must be NextFreshProfileBuildManifest"
        )
    if not isinstance(registry, VersionedCapabilityRegistry):
        raise NextFreshProfileBuildError(
            "registry must be VersionedCapabilityRegistry"
        )
    if (
        manifest.registry_ref != registry.registry_id
        or manifest.registry_digest != registry.digest
    ):
        raise NextFreshProfileBuildError("registry identity drift")
    if manifest.registry_version != registry.registry_version:
        raise NextFreshProfileBuildError("registry version drift")
    if (
        manifest.protocol_ref != registry.protocol_ref
        or manifest.protocol_digest != registry.protocol_digest
    ):
        raise NextFreshProfileBuildError("registry protocol drift")

    entries = {
        ref: (ref, digest, entrypoint)
        for ref, digest, entrypoint in manifest.predecessor_executable_entries
    }
    qualified_registry_refs: list[tuple[str, str]] = []
    for capability in registry.capabilities:
        if (
            capability.protocol_ref != manifest.protocol_ref
            or capability.protocol_digest != manifest.protocol_digest
            or capability.compatibility_requirements
            != manifest.compatibility_requirements
        ):
            raise NextFreshProfileBuildError(
                "qualified capability is incompatible with the next profile"
            )
        entry = (
            capability.capability_id,
            capability.digest,
            capability.executable_entrypoint,
        )
        existing = entries.get(capability.capability_id)
        if existing is not None and existing != entry:
            raise NextFreshProfileBuildError(
                "qualified capability conflicts with predecessor entry"
            )
        entries[capability.capability_id] = entry
        qualified_registry_refs.append(
            (capability.capability_id, capability.digest)
        )

    profile = ExecutableProfileVNext(
        profile_version=manifest.profile_version,
        predecessor_profile_ref=manifest.predecessor_profile_ref,
        predecessor_profile_digest=manifest.predecessor_profile_digest,
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        executable_entries=tuple(entries.values()),
        protocol_ref=manifest.protocol_ref,
        protocol_digest=manifest.protocol_digest,
        compatibility_requirements=manifest.compatibility_requirements,
        current_campaign_eligible=False,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )
    receipt = ProfileBuildReceiptV1(
        predecessor_profile_ref=manifest.predecessor_profile_ref,
        predecessor_profile_hash=manifest.predecessor_profile_digest,
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        qualified_registry_refs=tuple(qualified_registry_refs),
        new_profile_ref=profile.profile_id,
        new_profile_hash=profile.digest,
        build_policy_ref=manifest.manifest_id,
        build_policy_digest=manifest.digest,
        protocol_ref=manifest.protocol_ref,
        protocol_digest=manifest.protocol_digest,
        current_profile_unchanged=True,
        deterministic_rebuild=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )
    return profile, receipt


__all__ = [
    "NextFreshProfileBuildError",
    "NextFreshProfileBuildManifest",
    "build_next_fresh_profile",
]
