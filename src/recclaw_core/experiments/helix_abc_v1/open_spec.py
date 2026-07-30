"""A0 Open Research Spec producer projection and deterministic resolution.

The frozen v1 Producers retain their scientific roles.  This module only
projects their output into the RC0 ``OpenResearchSpecV1`` contract and resolves
the requested capability against frozen, caller-supplied profile/protocol
facts.  It performs no Provider call, implementation, qualification, registry
mutation, campaign mutation, or experiment routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space.canonical import deep_thaw

from .campaign_runtime import (
    bl_icf_executable_profile_v2,
    executable_mechanisms,
    execution_recipe_for_program,
)
from .canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from .research_contracts import CandidateProposalV4, DISCOVERY_PRODUCERS
from .vnext_contracts import (
    CapabilityResolutionResultV1,
    CapabilityResolutionV1,
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
)


class OpenSpecProjectionError(ValueError):
    """Raised when A0 producer or resolver inputs are structurally invalid."""


HIGH_CHANGE_DIMENSIONS = (
    "COMPOSITE_MECHANISM",
    "CORE_OBJECTIVE",
    "CORE_RELATION",
    "CORE_REPRESENTATION",
    "CUSTOM_EXECUTABLE_CAPABILITY",
    "INTERACTION_STRUCTURE",
    "MODEL_STRUCTURE",
    "PROPAGATION_MECHANISM",
    "TRAINING_PROCEDURE",
)
_HIGH_CHANGE_DIMENSION_SET = frozenset(HIGH_CHANGE_DIMENSIONS)
_PROTOCOL_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "campaign_development_protocol_v1.json"
)
_FROZEN_PROTOCOL_REQUIREMENTS = (
    "frozen dataset and split",
    "full-sort NDCG@10",
    "general collaborative filtering",
    "offline top-n evaluation",
    "pairwise input",
    "train-only fitting",
)


def _nonempty(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OpenSpecProjectionError(
            f"{field_name} must be a non-empty, whitespace-normalized string"
        )
    return value


def _sorted_unique_strings(
    values: Sequence[Any],
    *,
    field_name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise OpenSpecProjectionError(f"{field_name} must be a sequence")
    normalized = tuple(
        _nonempty(str(value), field_name=f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if not allow_empty and not normalized:
        raise OpenSpecProjectionError(f"{field_name} must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise OpenSpecProjectionError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _reason_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() else "_"
        for character in value.upper()
    )
    normalized = "_".join(part for part in token.split("_") if part)
    if not normalized:
        raise OpenSpecProjectionError("reason-code source must be non-empty")
    return normalized


def _normalize_budget(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise OpenSpecProjectionError(f"{field_name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, amount in sorted(value.items(), key=lambda item: str(item[0])):
        name = _nonempty(str(key), field_name=f"{field_name}.key")
        if (
            not isinstance(amount, int)
            or isinstance(amount, bool)
            or amount < 0
        ):
            raise OpenSpecProjectionError(
                f"{field_name}.{name} must be a non-negative integer"
            )
        normalized[name] = amount
    return normalized


def _normalize_resolution_facts(
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(facts, Mapping):
        raise OpenSpecProjectionError("resolution_facts must be a mapping")
    expected = {
        "requested_current_semantics_digest",
        "capability_diff",
        "high_change_dimensions",
        "required_dependencies",
        "required_budget",
    }
    if set(facts) != expected:
        raise OpenSpecProjectionError(
            "resolution_facts must contain exactly the frozen A0 fact fields"
        )
    requested = facts["requested_current_semantics_digest"]
    if requested is not None:
        requested = validate_sha256(
            str(requested),
            field_name="requested_current_semantics_digest",
        )
    dimensions = _sorted_unique_strings(
        facts["high_change_dimensions"],
        field_name="high_change_dimensions",
    )
    return canonical_value(
        {
            "requested_current_semantics_digest": requested,
            "capability_diff": _sorted_unique_strings(
                facts["capability_diff"],
                field_name="capability_diff",
            ),
            "high_change_dimensions": dimensions,
            "required_dependencies": _sorted_unique_strings(
                facts["required_dependencies"],
                field_name="required_dependencies",
            ),
            "required_budget": _normalize_budget(
                facts["required_budget"],
                field_name="required_budget",
            ),
        }
    )


def _normalize_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bindings, Mapping):
        raise OpenSpecProjectionError("bindings must be a mapping")
    expected = {
        "protocol_ref",
        "protocol_digest",
        "context_ref",
        "context_digest",
        "current_profile_ref",
        "current_profile_digest",
        "implementation_requirements",
        "compatibility_requirements",
    }
    if set(bindings) != expected:
        raise OpenSpecProjectionError(
            "bindings must contain exactly the frozen OpenSpec binding fields"
        )
    normalized = {
        "protocol_ref": _nonempty(
            bindings["protocol_ref"], field_name="protocol_ref"
        ),
        "protocol_digest": validate_sha256(
            str(bindings["protocol_digest"]), field_name="protocol_digest"
        ),
        "context_ref": _nonempty(
            bindings["context_ref"], field_name="context_ref"
        ),
        "context_digest": validate_sha256(
            str(bindings["context_digest"]), field_name="context_digest"
        ),
        "current_profile_ref": _nonempty(
            bindings["current_profile_ref"], field_name="current_profile_ref"
        ),
        "current_profile_digest": validate_sha256(
            str(bindings["current_profile_digest"]),
            field_name="current_profile_digest",
        ),
        "implementation_requirements": _sorted_unique_strings(
            bindings["implementation_requirements"],
            field_name="implementation_requirements",
            allow_empty=False,
        ),
        "compatibility_requirements": _sorted_unique_strings(
            bindings["compatibility_requirements"],
            field_name="compatibility_requirements",
            allow_empty=False,
        ),
    }
    return canonical_value(normalized)


def frozen_search_bindings(
    *,
    context_ref: str,
    context_digest: str,
    implementation_requirements: Sequence[str] = (
        "RecBole general recommender interface",
        "candidate-local package",
    ),
    compatibility_requirements: Sequence[str] = (
        "general collaborative filtering",
        "pairwise input",
    ),
) -> dict[str, Any]:
    """Return immutable-value bindings to the frozen 66-entry Search profile."""

    profile = bl_icf_executable_profile_v2()
    protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    return _normalize_bindings(
        {
            "protocol_ref": str(protocol["protocol_id"]),
            "protocol_digest": sha256_digest(protocol),
            "context_ref": context_ref,
            "context_digest": context_digest,
            "current_profile_ref": str(profile["profile_id"]),
            "current_profile_digest": str(profile["profile_digest"]),
            "implementation_requirements": tuple(
                implementation_requirements
            ),
            "compatibility_requirements": tuple(
                compatibility_requirements
            ),
        }
    )


def frozen_search_resolver_environment(
    *,
    available_dependencies: Sequence[str] = (),
    budget_limits: Mapping[str, int] | None = None,
    protocol_requirements: Sequence[str] = _FROZEN_PROTOCOL_REQUIREMENTS,
) -> dict[str, Any]:
    """Project the frozen Search profile into deterministic Resolver facts."""

    profile = bl_icf_executable_profile_v2()
    protocol = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    capabilities = []
    for mechanism in executable_mechanisms():
        capability_ref = (
            f"{profile['profile_id']}:mechanism:{mechanism.mechanism_id}"
        )
        capability_digest = sha256_digest(
            {
                "profile_ref": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "mechanism_id": mechanism.mechanism_id,
                "mechanism_semantics_digest": (
                    mechanism.mechanism_semantics_digest
                ),
            }
        )
        capabilities.append(
            {
                "capability_ref": capability_ref,
                "capability_digest": capability_digest,
                "semantics_digest": mechanism.mechanism_semantics_digest,
            }
        )
    return canonical_value(
        {
            "current_profile_ref": profile["profile_id"],
            "current_profile_digest": profile["profile_digest"],
            "current_capabilities": capabilities,
            "protocol_ref": protocol["protocol_id"],
            "protocol_digest": sha256_digest(protocol),
            "protocol_requirements": _sorted_unique_strings(
                protocol_requirements,
                field_name="protocol_requirements",
            ),
            "available_dependencies": _sorted_unique_strings(
                available_dependencies,
                field_name="available_dependencies",
            ),
            "budget_limits": _normalize_budget(
                budget_limits or {},
                field_name="budget_limits",
            ),
        }
    )


def _matched_control_requirement(proposal: CandidateProposalV4) -> str:
    return canonical_json_bytes(proposal.matched_control_plan.to_dict()).decode(
        "utf-8"
    )


def _expected_evidence(proposal: CandidateProposalV4) -> tuple[str, ...]:
    evidence = [
        proposal.predicted_outcome_signature,
        f"matched control status: {proposal.matched_control_plan.plan_status}",
    ]
    if proposal.discriminative_plan is not None:
        evidence.append(proposal.discriminative_plan.next_decision_rule)
    return tuple(evidence)


def project_candidate_proposal_v4(
    proposal: CandidateProposalV4,
    *,
    bindings: Mapping[str, Any],
    required_dependencies: Sequence[str] = (),
    required_budget: Mapping[str, int] | None = None,
) -> tuple[OpenResearchSpecV1, dict[str, Any]]:
    """Mechanically project one existing Producer proposal without role rewriting."""

    if not isinstance(proposal, CandidateProposalV4):
        raise OpenSpecProjectionError(
            "existing Producer projection requires CandidateProposalV4"
        )
    normalized_bindings = _normalize_bindings(bindings)
    recipe = execution_recipe_for_program(deep_thaw(proposal.mechanism_program))
    if recipe["mechanism_id"] != proposal.mechanism_id:
        raise OpenSpecProjectionError(
            "proposal mechanism identity does not match the frozen profile"
        )
    falsifier = (
        proposal.discriminative_plan.falsifier
        if proposal.discriminative_plan is not None
        else proposal.failure_mode
    )
    spec = OpenResearchSpecV1(
        hypothesis=proposal.mechanism_hypothesis,
        mechanism_change=(
            f"{proposal.mechanism_axis}: {proposal.candidate_label} "
            f"[{proposal.mechanism_id}]"
        ),
        competing_explanation=proposal.competing_hypothesis,
        matched_control_requirement=_matched_control_requirement(proposal),
        implementation_requirements=tuple(
            normalized_bindings["implementation_requirements"]
        ),
        expected_evidence=_expected_evidence(proposal),
        falsifier=falsifier,
        compatibility_requirements=tuple(
            normalized_bindings["compatibility_requirements"]
        ),
        protocol_ref=normalized_bindings["protocol_ref"],
        protocol_digest=normalized_bindings["protocol_digest"],
        context_ref=normalized_bindings["context_ref"],
        context_digest=normalized_bindings["context_digest"],
        current_profile_ref=normalized_bindings["current_profile_ref"],
        current_profile_digest=normalized_bindings[
            "current_profile_digest"
        ],
        producer_role=proposal.producer_role,
        high_change_justification=(
            "Not claimed: this Producer output is an exact frozen-profile "
            f"mechanism ({proposal.mechanism_id})."
        ),
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.EXPRESSIBLE
        ),
    )
    facts = _normalize_resolution_facts(
        {
            "requested_current_semantics_digest": recipe[
                "mechanism_semantics_digest"
            ],
            "capability_diff": (),
            "high_change_dimensions": (),
            "required_dependencies": tuple(required_dependencies),
            "required_budget": dict(required_budget or {}),
        }
    )
    return spec, facts


def project_open_producer_draft(
    draft: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
) -> tuple[OpenResearchSpecV1, dict[str, Any]]:
    """Project a high-change draft from any of the same four Producer roles."""

    if not isinstance(draft, Mapping):
        raise OpenSpecProjectionError("open Producer draft must be a mapping")
    expected = {
        "producer_role",
        "hypothesis",
        "mechanism_change",
        "competing_explanation",
        "matched_control_requirement",
        "implementation_requirements",
        "expected_evidence",
        "falsifier",
        "compatibility_requirements",
        "high_change_justification",
        "current_profile_expressibility_claim",
        "resolution_facts",
    }
    if set(draft) != expected:
        raise OpenSpecProjectionError(
            "open Producer draft must contain exactly the A0 projection fields"
        )
    normalized_bindings = _normalize_bindings(bindings)
    role = str(draft["producer_role"])
    if role not in DISCOVERY_PRODUCERS:
        raise OpenSpecProjectionError(
            "open Producer draft must retain one frozen Producer role"
        )
    try:
        expressibility = CurrentProfileExpressibilityV1(
            draft["current_profile_expressibility_claim"]
        )
    except ValueError as error:
        raise OpenSpecProjectionError(
            "current_profile_expressibility_claim is invalid"
        ) from error
    implementation_requirements = _sorted_unique_strings(
        draft["implementation_requirements"],
        field_name="implementation_requirements",
    )
    if not implementation_requirements:
        implementation_requirements = tuple(
            normalized_bindings["implementation_requirements"]
        )
    compatibility_requirements = _sorted_unique_strings(
        draft["compatibility_requirements"],
        field_name="compatibility_requirements",
    )
    if not compatibility_requirements:
        compatibility_requirements = tuple(
            normalized_bindings["compatibility_requirements"]
        )
    spec = OpenResearchSpecV1(
        hypothesis=draft["hypothesis"],
        mechanism_change=draft["mechanism_change"],
        competing_explanation=draft["competing_explanation"],
        matched_control_requirement=draft[
            "matched_control_requirement"
        ],
        implementation_requirements=implementation_requirements,
        expected_evidence=_sorted_unique_strings(
            draft["expected_evidence"],
            field_name="expected_evidence",
            allow_empty=False,
        ),
        falsifier=draft["falsifier"],
        compatibility_requirements=compatibility_requirements,
        protocol_ref=normalized_bindings["protocol_ref"],
        protocol_digest=normalized_bindings["protocol_digest"],
        context_ref=normalized_bindings["context_ref"],
        context_digest=normalized_bindings["context_digest"],
        current_profile_ref=normalized_bindings["current_profile_ref"],
        current_profile_digest=normalized_bindings[
            "current_profile_digest"
        ],
        producer_role=role,
        high_change_justification=draft["high_change_justification"],
        current_profile_expressibility_claim=expressibility,
    )
    return spec, _normalize_resolution_facts(draft["resolution_facts"])


def _coerce_open_spec(
    value: OpenResearchSpecV1 | Mapping[str, Any],
) -> OpenResearchSpecV1:
    if isinstance(value, OpenResearchSpecV1):
        return value
    if not isinstance(value, Mapping):
        raise OpenSpecProjectionError(
            "research_spec must be OpenResearchSpecV1 or a mapping"
        )
    payload = dict(value)
    for field_name in (
        "implementation_requirements",
        "expected_evidence",
        "compatibility_requirements",
    ):
        if field_name in payload:
            payload[field_name] = _sorted_unique_strings(
                payload[field_name],
                field_name=field_name,
                allow_empty=False,
            )
    if "current_profile_expressibility_claim" in payload:
        payload["current_profile_expressibility_claim"] = (
            CurrentProfileExpressibilityV1(
                payload["current_profile_expressibility_claim"]
            )
        )
    return OpenResearchSpecV1(**payload)


def _normalize_environment(
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        raise OpenSpecProjectionError("resolver environment must be a mapping")
    expected = {
        "current_profile_ref",
        "current_profile_digest",
        "current_capabilities",
        "protocol_ref",
        "protocol_digest",
        "protocol_requirements",
        "available_dependencies",
        "budget_limits",
    }
    if set(environment) != expected:
        raise OpenSpecProjectionError(
            "resolver environment must contain exactly the A0 environment fields"
        )
    capabilities = []
    seen_semantics: set[str] = set()
    for index, entry in enumerate(environment["current_capabilities"]):
        if not isinstance(entry, Mapping) or set(entry) != {
            "capability_ref",
            "capability_digest",
            "semantics_digest",
        }:
            raise OpenSpecProjectionError(
                f"current_capabilities[{index}] has an invalid shape"
            )
        semantics = validate_sha256(
            str(entry["semantics_digest"]),
            field_name=f"current_capabilities[{index}].semantics_digest",
        )
        if semantics in seen_semantics:
            raise OpenSpecProjectionError(
                "current capability semantics must be unique"
            )
        seen_semantics.add(semantics)
        capabilities.append(
            {
                "capability_ref": _nonempty(
                    entry["capability_ref"],
                    field_name=f"current_capabilities[{index}].capability_ref",
                ),
                "capability_digest": validate_sha256(
                    str(entry["capability_digest"]),
                    field_name=(
                        f"current_capabilities[{index}].capability_digest"
                    ),
                ),
                "semantics_digest": semantics,
            }
        )
    capabilities.sort(key=lambda item: item["semantics_digest"])
    return canonical_value(
        {
            "current_profile_ref": _nonempty(
                environment["current_profile_ref"],
                field_name="current_profile_ref",
            ),
            "current_profile_digest": validate_sha256(
                str(environment["current_profile_digest"]),
                field_name="current_profile_digest",
            ),
            "current_capabilities": capabilities,
            "protocol_ref": _nonempty(
                environment["protocol_ref"], field_name="protocol_ref"
            ),
            "protocol_digest": validate_sha256(
                str(environment["protocol_digest"]),
                field_name="protocol_digest",
            ),
            "protocol_requirements": _sorted_unique_strings(
                environment["protocol_requirements"],
                field_name="protocol_requirements",
            ),
            "available_dependencies": _sorted_unique_strings(
                environment["available_dependencies"],
                field_name="available_dependencies",
            ),
            "budget_limits": _normalize_budget(
                environment["budget_limits"],
                field_name="budget_limits",
            ),
        }
    )


def _raw_spec_identity(value: Any) -> tuple[str, str]:
    try:
        payload = canonical_value(value)
    except (TypeError, ValueError):
        payload = {"invalid_python_type": type(value).__name__}
    digest = sha256_digest(
        {
            "schema": "recclaw.research-line.vnext.invalid-open-spec.v1",
            "payload": payload,
        }
    )
    return f"invalid-open-research-spec:{digest}", digest


def _resolution(
    *,
    spec_ref: str,
    spec_digest: str,
    environment: Mapping[str, Any],
    result: CapabilityResolutionResultV1,
    capability_diff: Sequence[str],
    protocol_compatible: bool,
    dependency_compatible: bool,
    budget_compatible: bool,
    reason_codes: Sequence[str],
    matched_capability: Mapping[str, Any] | None = None,
) -> CapabilityResolutionV1:
    return CapabilityResolutionV1(
        research_spec_ref=spec_ref,
        research_spec_digest=spec_digest,
        current_profile_ref=environment["current_profile_ref"],
        current_profile_digest=environment["current_profile_digest"],
        resolution=result,
        current_profile_match=matched_capability is not None,
        resolved_current_capability_ref=(
            matched_capability["capability_ref"]
            if matched_capability is not None
            else None
        ),
        resolved_current_capability_digest=(
            matched_capability["capability_digest"]
            if matched_capability is not None
            else None
        ),
        capability_diff=tuple(capability_diff),
        protocol_compatible=protocol_compatible,
        dependency_compatible=dependency_compatible,
        budget_compatible=budget_compatible,
        reason_codes=tuple(reason_codes),
        no_silent_fallback=True,
        catalog_fallback_used=False,
    )


def resolve_capability(
    research_spec: OpenResearchSpecV1 | Mapping[str, Any],
    *,
    resolution_facts: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> CapabilityResolutionV1:
    """Resolve one spec with fixed, precedence-ordered deterministic rules."""

    normalized_environment = _normalize_environment(environment)
    try:
        spec = _coerce_open_spec(research_spec)
    except (TypeError, ValueError, KeyError):
        spec_ref, spec_digest = _raw_spec_identity(research_spec)
        return _resolution(
            spec_ref=spec_ref,
            spec_digest=spec_digest,
            environment=normalized_environment,
            result=CapabilityResolutionResultV1.INVALID_SPEC,
            capability_diff=(),
            protocol_compatible=False,
            dependency_compatible=False,
            budget_compatible=False,
            reason_codes=("OPEN_SPEC_CONTRACT_INVALID",),
        )
    try:
        facts = _normalize_resolution_facts(resolution_facts)
    except (TypeError, ValueError, KeyError):
        return _resolution(
            spec_ref=spec.spec_id,
            spec_digest=spec.digest,
            environment=normalized_environment,
            result=CapabilityResolutionResultV1.INVALID_SPEC,
            capability_diff=(),
            protocol_compatible=False,
            dependency_compatible=False,
            budget_compatible=False,
            reason_codes=("RESOLUTION_FACTS_INVALID",),
        )

    requested_semantics = facts["requested_current_semantics_digest"]
    capability_by_semantics = {
        entry["semantics_digest"]: entry
        for entry in normalized_environment["current_capabilities"]
    }
    matched = (
        capability_by_semantics.get(requested_semantics)
        if requested_semantics is not None
        else None
    )
    capability_diff = tuple(facts["capability_diff"])
    dimensions = tuple(facts["high_change_dimensions"])
    claim = spec.current_profile_expressibility_claim

    invalid_reasons: list[str] = []
    if spec.current_profile_ref != normalized_environment["current_profile_ref"] or (
        spec.current_profile_digest
        != normalized_environment["current_profile_digest"]
    ):
        invalid_reasons.append("CURRENT_PROFILE_IDENTITY_MISMATCH")
    if claim is CurrentProfileExpressibilityV1.UNRESOLVED:
        invalid_reasons.append("EXPRESSIBILITY_CLAIM_UNRESOLVED")
    if claim is CurrentProfileExpressibilityV1.EXPRESSIBLE:
        if requested_semantics is None:
            invalid_reasons.append("EXACT_SEMANTICS_IDENTITY_MISSING")
        elif matched is None:
            invalid_reasons.append("EXPRESSIBILITY_CLAIM_UNVERIFIED")
        if capability_diff or dimensions:
            invalid_reasons.append("EXPRESSIBLE_SPEC_HAS_CAPABILITY_DIFF")
    if claim is CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE:
        if matched is not None:
            invalid_reasons.append("OUTSIDE_PROFILE_CLAIM_CONTRADICTS_EXACT_MATCH")
        if not capability_diff:
            invalid_reasons.append("CAPABILITY_DIFF_MISSING")
        if not dimensions:
            invalid_reasons.append("HIGH_CHANGE_DIMENSION_MISSING")
        unknown_dimensions = sorted(
            set(dimensions) - _HIGH_CHANGE_DIMENSION_SET
        )
        invalid_reasons.extend(
            f"HIGH_CHANGE_DIMENSION_INVALID:{_reason_token(item)}"
            for item in unknown_dimensions
        )
    if invalid_reasons:
        return _resolution(
            spec_ref=spec.spec_id,
            spec_digest=spec.digest,
            environment=normalized_environment,
            result=CapabilityResolutionResultV1.INVALID_SPEC,
            capability_diff=capability_diff,
            protocol_compatible=False,
            dependency_compatible=False,
            budget_compatible=False,
            reason_codes=invalid_reasons,
        )

    protocol_missing = sorted(
        set(spec.compatibility_requirements)
        - set(normalized_environment["protocol_requirements"])
    )
    protocol_compatible = (
        spec.protocol_ref == normalized_environment["protocol_ref"]
        and spec.protocol_digest
        == normalized_environment["protocol_digest"]
        and not protocol_missing
    )
    if not protocol_compatible:
        reasons = []
        if (
            spec.protocol_ref != normalized_environment["protocol_ref"]
            or spec.protocol_digest
            != normalized_environment["protocol_digest"]
        ):
            reasons.append("PROTOCOL_IDENTITY_CHANGE_REQUIRED")
        reasons.extend(
            f"PROTOCOL_REQUIREMENT_UNAVAILABLE:{_reason_token(item)}"
            for item in protocol_missing
        )
        return _resolution(
            spec_ref=spec.spec_id,
            spec_digest=spec.digest,
            environment=normalized_environment,
            result=(
                CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE
            ),
            capability_diff=capability_diff,
            protocol_compatible=False,
            dependency_compatible=True,
            budget_compatible=True,
            reason_codes=reasons,
        )

    missing_dependencies = sorted(
        set(facts["required_dependencies"])
        - set(normalized_environment["available_dependencies"])
    )
    dependency_compatible = not missing_dependencies
    budget_reasons = []
    for name, required in facts["required_budget"].items():
        if name not in normalized_environment["budget_limits"]:
            budget_reasons.append(
                f"BUDGET_DIMENSION_UNAVAILABLE:{_reason_token(name)}"
            )
        elif required > normalized_environment["budget_limits"][name]:
            budget_reasons.append(f"BUDGET_EXCEEDED:{_reason_token(name)}")
    budget_compatible = not budget_reasons
    if not dependency_compatible or not budget_compatible:
        reasons = [
            f"DEPENDENCY_UNAVAILABLE:{_reason_token(item)}"
            for item in missing_dependencies
        ]
        reasons.extend(budget_reasons)
        return _resolution(
            spec_ref=spec.spec_id,
            spec_digest=spec.digest,
            environment=normalized_environment,
            result=CapabilityResolutionResultV1.UNSUPPORTED,
            capability_diff=capability_diff,
            protocol_compatible=True,
            dependency_compatible=dependency_compatible,
            budget_compatible=budget_compatible,
            reason_codes=reasons,
        )

    if matched is not None:
        return _resolution(
            spec_ref=spec.spec_id,
            spec_digest=spec.digest,
            environment=normalized_environment,
            result=CapabilityResolutionResultV1.SEARCH_READY,
            capability_diff=(),
            protocol_compatible=True,
            dependency_compatible=True,
            budget_compatible=True,
            reason_codes=("EXACT_CURRENT_PROFILE_MATCH",),
            matched_capability=matched,
        )

    return _resolution(
        spec_ref=spec.spec_id,
        spec_digest=spec.digest,
        environment=normalized_environment,
        result=CapabilityResolutionResultV1.INNOVATION_REQUIRED,
        capability_diff=capability_diff,
        protocol_compatible=True,
        dependency_compatible=True,
        budget_compatible=True,
        reason_codes=(
            "HIGH_CHANGE_MECHANISM_ELIGIBLE",
            "OUTSIDE_CURRENT_PROFILE",
            "NEXT_FRESH_CAMPAIGN_ONLY",
        ),
    )


__all__ = [
    "HIGH_CHANGE_DIMENSIONS",
    "OpenSpecProjectionError",
    "frozen_search_bindings",
    "frozen_search_resolver_environment",
    "project_candidate_proposal_v4",
    "project_open_producer_draft",
    "resolve_capability",
]
