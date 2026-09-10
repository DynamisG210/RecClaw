"""Canonical executable-package identity for one qualified OpenSpec."""

from __future__ import annotations

import hashlib
from importlib import resources
from typing import Any, Mapping

from recclaw_core.mechanism_space.canonical import load_json_bytes

from .canonical import canonical_value, content_id, sha256_digest, validate_sha256
from .vnext_contracts import OpenResearchSpecV1


class RealizationIdentityError(ValueError):
    """Raised when implementation provenance is not bound to its OpenSpec."""


_BL_ICF_RESOURCE_PACKAGE = "recclaw_core.search_spaces.bl_icf_v1.resources"
_BL_ICF_SEARCH_SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM = (
    "SHA256_UTF8_LF_TERMINATED_ID_LIST_V1"
)


def bl_icf_ordered_primitive_ids_conformance() -> dict[str, Any]:
    """Return the stable audit projection of the canonical registered-ID order."""

    registry = load_json_bytes(
        resources.files(_BL_ICF_RESOURCE_PACKAGE)
        .joinpath("primitive_registry_v1.json")
        .read_bytes()
    )
    primitive_ids = tuple(
        str(primitive["primitive_id"])
        for axis in registry["axes"]
        for primitive in axis["primitives"]
    )
    payload = ("\n".join(primitive_ids) + "\n").encode("utf-8")
    return canonical_value(
        {
            "ordered_primitive_ids_count": len(primitive_ids),
            "ordered_primitive_ids_digest_algorithm": (
                ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM
            ),
            "ordered_primitive_ids_digest": hashlib.sha256(payload).hexdigest(),
        }
    )


def bl_icf_search_space_conformance(
    *,
    space_identity: Mapping[str, Any],
    profile_ref: Mapping[str, Any],
    fixed_fallback: bool = False,
) -> dict[str, Any]:
    """Return the exact public cross-arm BL-ICF conformance projection."""

    identity = canonical_value(dict(space_identity))
    profile = canonical_value(dict(profile_ref))
    if not isinstance(fixed_fallback, bool):
        raise RealizationIdentityError("fixed_fallback must be a boolean")
    ordered = bl_icf_ordered_primitive_ids_conformance()
    return canonical_value(
        {
            "search_space_id": _normalized_string(
                str(identity.get("search_space_id", "")),
                field_name="space_identity.search_space_id",
            ),
            "search_space_digest": validate_sha256(
                str(identity.get("search_space_digest", "")),
                field_name="space_identity.search_space_digest",
            ),
            **ordered,
            "profile_ref": profile,
            "fixed_fallback": fixed_fallback,
        }
    )


def mechanism_semantic_identity_ref(
    *,
    search_space_id: str,
    candidate_id: str,
) -> str:
    """Return the compiler-owned semantic namespace for one mechanism family."""

    space_id = _normalized_string(search_space_id, field_name="search_space_id")
    compiled_id = _normalized_string(candidate_id, field_name="candidate_id")
    if space_id == _BL_ICF_SEARCH_SPACE_ID:
        return f"bl-icf-mechanism:{compiled_id}"
    return f"mechanism:{space_id}:{compiled_id}"


def search_space_conformance(
    *,
    space_identity: Mapping[str, Any],
    profile_ref: Mapping[str, Any],
    fixed_fallback: bool = False,
) -> dict[str, Any]:
    """Derive package conformance from the compiler-selected mechanism space."""

    identity = canonical_value(dict(space_identity))
    search_space_id = _normalized_string(
        str(identity.get("search_space_id", "")),
        field_name="space_identity.search_space_id",
    )
    if search_space_id == _BL_ICF_SEARCH_SPACE_ID:
        return bl_icf_search_space_conformance(
            space_identity=identity,
            profile_ref=profile_ref,
            fixed_fallback=fixed_fallback,
        )
    if not isinstance(fixed_fallback, bool):
        raise RealizationIdentityError("fixed_fallback must be a boolean")
    from recclaw_core.mechanism_space import prompt_projection

    projection = prompt_projection(search_space_id)
    primitive_ids = tuple(
        str(primitive["primitive_id"])
        for axis in projection["axes"]
        for primitive in axis["primitives"]
    )
    payload = ("\n".join(primitive_ids) + "\n").encode("utf-8")
    return canonical_value(
        {
            "search_space_id": search_space_id,
            "search_space_digest": validate_sha256(
                str(identity.get("search_space_digest", "")),
                field_name="space_identity.search_space_digest",
            ),
            "ordered_primitive_ids_count": len(primitive_ids),
            "ordered_primitive_ids_digest_algorithm": (
                ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM
            ),
            "ordered_primitive_ids_digest": hashlib.sha256(payload).hexdigest(),
            "profile_ref": canonical_value(dict(profile_ref)),
            "fixed_fallback": fixed_fallback,
        }
    )


def bl_icf_scientific_mechanism_program(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a BL-ICF program onto its scientific mechanism semantics.

    ``source_ownership_mode`` selects how an already-declared mechanism is
    materialized.  It remains part of the exact program identity, but must not
    turn two mechanically different realizations into two research mechanisms.
    """

    projection = canonical_value(dict(program))
    for operator in projection.get("program_payload", {}).get(
        "architecture_operators", ()
    ):
        parameters = operator.get("parameters")
        if isinstance(parameters, dict):
            parameters.pop("source_ownership_mode", None)
    return canonical_value(projection)


def _normalized_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RealizationIdentityError(
            f"{field_name} must be a normalized non-empty string"
        )
    return value


def _normalized_entrypoint(value: str) -> str:
    normalized = _normalized_string(value, field_name="executable_entrypoint")
    module_name, separator, attribute = normalized.partition(":")
    if (
        separator != ":"
        or not module_name
        or not attribute
        or any(character.isspace() for character in normalized)
    ):
        raise RealizationIdentityError(
            "executable_entrypoint must use module.path:Attribute"
        )
    return normalized


def open_spec_realization_identity(
    spec: OpenResearchSpecV1,
    *,
    candidate_package_ref: str,
    candidate_package_digest: str,
    candidate_root_ref: str,
    candidate_root_digest: str,
    source_tree_digest: str,
    executable_entrypoint: str,
    execution_contract: Mapping[str, Any],
) -> tuple[str, str]:
    """Return package realization ref/digest, never mechanism semantics."""

    if not isinstance(spec, OpenResearchSpecV1):
        raise RealizationIdentityError(
            "realization identity requires OpenResearchSpecV1"
        )
    package_ref = _normalized_string(
        candidate_package_ref,
        field_name="candidate_package_ref",
    )
    root_ref = _normalized_string(
        candidate_root_ref,
        field_name="candidate_root_ref",
    )
    package_digest = validate_sha256(
        candidate_package_digest,
        field_name="candidate_package_digest",
    )
    root_digest = validate_sha256(
        candidate_root_digest,
        field_name="candidate_root_digest",
    )
    source_digest = validate_sha256(
        source_tree_digest,
        field_name="source_tree_digest",
    )
    entrypoint = _normalized_entrypoint(executable_entrypoint)
    if spec.execution_contract is None:
        raise RealizationIdentityError(
            "realization identity requires a resolver execution_contract"
        )
    if not isinstance(execution_contract, Mapping):
        raise RealizationIdentityError("execution_contract must be a mapping")
    normalized_contract = canonical_value(dict(execution_contract))
    if normalized_contract != canonical_value(dict(spec.execution_contract)):
        raise RealizationIdentityError(
            "realization execution_contract is not bound to its spec"
        )
    identity = canonical_value(
        {
            "schema": "recclaw-open-spec-realization-semantics-e0-v1",
            "research_spec_digest": spec.digest,
            "candidate_package_ref": package_ref,
            "candidate_package_digest": package_digest,
            "candidate_root_ref": root_ref,
            "candidate_root_digest": root_digest,
            "source_tree_digest": source_digest,
            "executable_entrypoint": entrypoint,
            "execution_contract": normalized_contract,
        }
    )
    digest = sha256_digest(identity)
    return content_id("recclaw-open-spec-realization-v1", identity), digest


__all__ = [
    "ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM",
    "RealizationIdentityError",
    "bl_icf_ordered_primitive_ids_conformance",
    "bl_icf_scientific_mechanism_program",
    "bl_icf_search_space_conformance",
    "mechanism_semantic_identity_ref",
    "open_spec_realization_identity",
    "search_space_conformance",
]
