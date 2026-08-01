"""Exact Prefreeze V4 Provider/local response-contract split.

This module is deliberately narrow.  It removes the six frozen V1
``uniqueItems: true`` keywords from the Provider-facing schema and enforces the
same six constraints locally before any OpenSpec projection or downstream use.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import jsonschema

from .canonical import canonical_json_bytes, canonical_value, sha256_digest


V1_UNIQUE_SCHEMA_PATHS = (
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "compatibility_requirements",
        "uniqueItems",
    ),
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "expected_evidence",
        "uniqueItems",
    ),
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "implementation_requirements",
        "uniqueItems",
    ),
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "resolution_facts",
        "properties",
        "capability_diff",
        "uniqueItems",
    ),
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "resolution_facts",
        "properties",
        "high_change_dimensions",
        "uniqueItems",
    ),
    (
        "properties",
        "proposals",
        "items",
        "properties",
        "resolution_facts",
        "properties",
        "required_dependencies",
        "uniqueItems",
    ),
)

V4_LOCAL_ARRAY_PATHS = (
    ("compatibility_requirements",),
    ("expected_evidence",),
    ("implementation_requirements",),
    ("resolution_facts", "capability_diff"),
    ("resolution_facts", "high_change_dimensions"),
    ("resolution_facts", "required_dependencies"),
)

V4_UNIQUENESS_CONTRACT = {
    "schema": "recclaw.research-line.v4-local-array-uniqueness-contract.v1",
    "comparison_semantics": "CANONICAL_JSON_DEEP_VALUE_EQUALITY",
    "provider_schema_derivation": "DELETE_EXACT_SIX_UNIQUE_ITEMS_TRUE_ONLY",
    "provider_schema_validation_order": 1,
    "local_uniqueness_validation_order": 2,
    "open_spec_projection_minimum_order": 3,
    "local_failure_class": "LOCAL_ARRAY_UNIQUENESS_CONTRACT_FAILURE",
    "local_failure_retry_eligible": False,
    "manual_patch": "FORBIDDEN",
    "successful_response_selection": "FORBIDDEN",
    "candidate_admission_on_failure": "FORBIDDEN",
    "mechanism_negative_evidence": False,
    "schema_paths": [list(path) for path in V1_UNIQUE_SCHEMA_PATHS],
    "response_paths_relative_to_each_proposal": [
        list(path) for path in V4_LOCAL_ARRAY_PATHS
    ],
}
V4_UNIQUENESS_CONTRACT_DIGEST = sha256_digest(V4_UNIQUENESS_CONTRACT)


class V4LocalUniquenessError(ValueError):
    """Terminal local response-contract failure for one frozen array path."""

    def __init__(self, *, proposal_index: int, path: Sequence[str], reason: str):
        self.proposal_index = proposal_index
        self.path = tuple(path)
        self.reason = reason
        super().__init__(
            f"proposal[{proposal_index}].{'.'.join(path)}: {reason}"
        )


def derive_v4_provider_schema(v1_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the six explicitly frozen unsupported Provider keywords."""

    derived = deepcopy(canonical_value(v1_schema))
    removed: list[tuple[str, ...]] = []
    for path in V1_UNIQUE_SCHEMA_PATHS:
        parent: Any = derived
        for part in path[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                raise ValueError(f"V1 schema path is missing: {'/'.join(path)}")
            parent = parent[part]
        if not isinstance(parent, dict) or parent.get(path[-1]) is not True:
            raise ValueError(
                f"V1 schema path is not uniqueItems=true: {'/'.join(path)}"
            )
        del parent[path[-1]]
        removed.append(path)
    if tuple(removed) != V1_UNIQUE_SCHEMA_PATHS:
        raise ValueError("V4 Provider schema removal set changed")
    return derived


def validate_v4_schema_derivation(
    *,
    v1_schema: Mapping[str, Any],
    v4_provider_schema: Mapping[str, Any],
) -> None:
    """Fail unless V4 equals V1 minus exactly the six frozen keywords."""

    expected = derive_v4_provider_schema(v1_schema)
    if canonical_value(v4_provider_schema) != canonical_value(expected):
        raise ValueError("V4 Provider schema is not the exact mechanical derivation")
    jsonschema.validators.validator_for(expected).check_schema(expected)


def _array_at_path(
    proposal: Mapping[str, Any],
    *,
    proposal_index: int,
    path: Sequence[str],
) -> list[Any]:
    current: Any = proposal
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise V4LocalUniquenessError(
                proposal_index=proposal_index,
                path=path,
                reason="MISSING_FIELD",
            )
        current = current[part]
    if not isinstance(current, list):
        raise V4LocalUniquenessError(
            proposal_index=proposal_index,
            path=path,
            reason="NOT_ARRAY",
        )
    return current


def validate_v4_local_uniqueness(response: Mapping[str, Any]) -> None:
    """Enforce deep-value uniqueness at all six exact response array paths."""

    proposals = response.get("proposals") if isinstance(response, Mapping) else None
    if not isinstance(proposals, list):
        raise V4LocalUniquenessError(
            proposal_index=-1,
            path=("proposals",),
            reason="MISSING_OR_NOT_ARRAY",
        )
    for proposal_index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise V4LocalUniquenessError(
                proposal_index=proposal_index,
                path=(),
                reason="PROPOSAL_NOT_OBJECT",
            )
        for path in V4_LOCAL_ARRAY_PATHS:
            values = _array_at_path(
                proposal,
                proposal_index=proposal_index,
                path=path,
            )
            identities = [
                canonical_json_bytes(canonical_value(value)) for value in values
            ]
            if len(set(identities)) != len(identities):
                raise V4LocalUniquenessError(
                    proposal_index=proposal_index,
                    path=path,
                    reason="DUPLICATE_CANONICAL_VALUE",
                )


def validate_v4_response_contract(
    response: Mapping[str, Any],
    *,
    provider_schema: Mapping[str, Any],
) -> None:
    """Run complete strict Provider schema, then local uniqueness validation."""

    jsonschema.validate(canonical_value(response), canonical_value(provider_schema))
    validate_v4_local_uniqueness(response)
