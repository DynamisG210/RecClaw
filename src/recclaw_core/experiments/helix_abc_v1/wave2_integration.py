"""G-owned, provider-free boundaries for Wave 2 integration preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    validate_sha256,
)


WAVE1_ACCEPTED_COMMIT = "4d493939bb1e118ae9c99c89e8b73077078c3ab8"
WAVE1_ACCEPTED_TREE = "3fd503618a664025e8f7a4c751ffdb646787a3d7"
PREFREEZE_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-manifest.v1"
DRY_RUN_RECEIPT_SCHEMA = "recclaw.research-line.r1-r2-dry-run-receipt.v1"
OWNER_INTAKE_SCHEMA = "recclaw.research-line.wave2-owner-intake.v1"

_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
_ENTRYPOINT_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*"
)


class Wave2OwnerLaneV1(str, Enum):
    """Owner lanes whose implementations remain outside G."""

    E0_SEARCH_ADAPTER = "E0_SEARCH_ADAPTER"
    D1_SCIENTIFIC_EPISODE_ADAPTER = "D1_SCIENTIFIC_EPISODE_ADAPTER"
    F0_OPEN_META_INTERFACE = "F0_OPEN_META_INTERFACE"


class Wave2IntegrationError(ValueError):
    """Raised when a Wave 2 integration boundary is not fail-closed."""


class Wave2OwnerCorrectionRequired(Wave2IntegrationError):
    """Minimal owner-facing correction for an incompatible public boundary."""

    def __init__(
        self,
        *,
        lane: Wave2OwnerLaneV1,
        field: str,
        expected: str,
        observed: str,
    ) -> None:
        self.lane = lane
        self.field = field
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"{lane.value} correction required for {field}: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _git_object(value: str, *, field_name: str) -> str:
    normalized = str(value)
    if not _GIT_OBJECT_RE.fullmatch(normalized):
        raise Wave2IntegrationError(
            f"{field_name} must be a lowercase 40-character Git object id"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Wave2OwnerIntakeV1:
    """Accepted owner identity needed for a mechanical G integration."""

    lane: Wave2OwnerLaneV1
    accepted_commit: str
    parent_commit: str
    owner_file_manifest_sha256: str
    targeted_tests_receipt_sha256: str
    structure_lint_receipt_sha256: str
    public_entrypoint: str
    schema: str = OWNER_INTAKE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OWNER_INTAKE_SCHEMA:
            raise Wave2IntegrationError("unsupported Wave 2 owner intake schema")
        object.__setattr__(
            self,
            "accepted_commit",
            _git_object(self.accepted_commit, field_name="accepted_commit"),
        )
        object.__setattr__(
            self,
            "parent_commit",
            _git_object(self.parent_commit, field_name="parent_commit"),
        )
        for field_name in (
            "owner_file_manifest_sha256",
            "targeted_tests_receipt_sha256",
            "structure_lint_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        if not _ENTRYPOINT_RE.fullmatch(self.public_entrypoint):
            raise Wave2IntegrationError(
                "public_entrypoint must use canonical module:attribute syntax"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "lane": self.lane,
                "accepted_commit": self.accepted_commit,
                "parent_commit": self.parent_commit,
                "owner_file_manifest_sha256": self.owner_file_manifest_sha256,
                "targeted_tests_receipt_sha256": self.targeted_tests_receipt_sha256,
                "structure_lint_receipt_sha256": self.structure_lint_receipt_sha256,
                "public_entrypoint": self.public_entrypoint,
            }
        )


@dataclass(frozen=True, slots=True)
class Wave2IntegrationHarnessV1:
    """Pure descriptor for accepted owner ports; it executes no owner logic."""

    intakes: tuple[Wave2OwnerIntakeV1, ...] = ()

    def __post_init__(self) -> None:
        lanes = tuple(intake.lane for intake in self.intakes)
        if len(lanes) != len(set(lanes)):
            raise Wave2IntegrationError("each Wave 2 owner lane may be attached once")

    @property
    def missing_lanes(self) -> tuple[Wave2OwnerLaneV1, ...]:
        attached = {intake.lane for intake in self.intakes}
        return tuple(lane for lane in Wave2OwnerLaneV1 if lane not in attached)

    @property
    def ready(self) -> bool:
        return not self.missing_lanes

    def attach(
        self,
        intake: Wave2OwnerIntakeV1,
        *,
        observed_entrypoint: str,
    ) -> "Wave2IntegrationHarnessV1":
        if observed_entrypoint != intake.public_entrypoint:
            raise Wave2OwnerCorrectionRequired(
                lane=intake.lane,
                field="public_entrypoint",
                expected=intake.public_entrypoint,
                observed=observed_entrypoint,
            )
        return Wave2IntegrationHarnessV1(
            intakes=tuple(
                sorted(
                    (*self.intakes, intake),
                    key=lambda item: item.lane.value,
                )
            )
        )

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": "recclaw.research-line.wave2-integration-harness.v1",
                "intakes": [intake.canonical_dict() for intake in self.intakes],
                "missing_lanes": self.missing_lanes,
                "ready": self.ready,
            }
        )


_PREFREEZE_SHAPE: dict[str, Any] = {
    "schema": None,
    "wave1_base": {"commit": None, "tree": None},
    "source_identity": {
        "commit": None,
        "tree_digest": None,
        "schema_ref": None,
        "schema_digest": None,
    },
    "runtime_identity": {
        "runtime_ref": None,
        "runtime_digest": None,
        "dependency_lock_ref": None,
        "dependency_lock_digest": None,
    },
    "provider_identity": {
        "endpoint_ref": None,
        "endpoint_digest": None,
        "model_ref": None,
        "model_digest": None,
        "credential_identity_digest": None,
    },
    "shared_proposal_call": {
        "granularity": None,
        "token_budget": None,
        "call_count": None,
        "failure_rule_ref": None,
        "failure_rule_digest": None,
        "prompt_ref": None,
        "prompt_digest": None,
        "tool_ref": None,
        "tool_digest": None,
        "response_contract_ref": None,
        "response_contract_digest": None,
        "no_retry": None,
        "proposal_budget_per_side": None,
    },
    "shared_implementation": {
        "implementer_ref": None,
        "implementer_digest": None,
        "qualifier_ref": None,
        "qualifier_digest": None,
        "manual_candidate_patch_forbidden": None,
        "qualification_evidence_class": None,
    },
    "r1_identity": {
        "lineage_ref": None,
        "lineage_digest": None,
        "seed_ref": None,
        "seed_digest": None,
        "outcome_namespace_ref": None,
        "outcome_namespace_digest": None,
        "memory_namespace_ref": None,
        "memory_namespace_digest": None,
        "root_ref": None,
        "root_digest": None,
        "db_ref": None,
        "db_digest": None,
        "side_a_identity_ref": None,
        "side_a_identity_digest": None,
        "side_b_identity_ref": None,
        "side_b_identity_digest": None,
    },
    "r2_identity": {
        "lineage_ref": None,
        "lineage_digest": None,
        "seed_ref": None,
        "seed_digest": None,
        "outcome_namespace_ref": None,
        "outcome_namespace_digest": None,
        "memory_namespace_ref": None,
        "memory_namespace_digest": None,
        "root_ref": None,
        "root_digest": None,
        "db_ref": None,
        "db_digest": None,
    },
    "evidence_policy": {
        "held_out_absent": None,
        "missingness_policy_ref": None,
        "missingness_policy_digest": None,
        "analysis_plan_ref": None,
        "analysis_plan_digest": None,
    },
    "r1_gate": {
        "requirements": {
            "minimum_fresh_specs": None,
            "minimum_producer_roles": None,
            "minimum_qualified": None,
            "minimum_real_mechanism_changes": None,
            "accepted_change_kinds": None,
        },
        "result_slots": {
            "fresh_spec_receipt_refs": None,
            "producer_role_receipt_refs": None,
            "qualified_capability_receipt_refs": None,
            "real_mechanism_change_receipt_refs": None,
        },
    },
}


def _validate_shape(
    value: Any,
    shape: Mapping[str, Any],
    *,
    path: str = "",
) -> None:
    if not isinstance(value, Mapping):
        raise Wave2IntegrationError(f"{path or 'manifest'} must be an object")
    expected = set(shape)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Wave2IntegrationError(
            f"{path or 'manifest'} keys mismatch: missing={missing}, extra={extra}"
        )
    for key, child_shape in shape.items():
        if isinstance(child_shape, Mapping):
            child_path = f"{path}.{key}" if path else key
            _validate_shape(value[key], child_shape, path=child_path)


def _leaf_paths(
    shape: Mapping[str, Any],
    *,
    prefix: str = "",
) -> tuple[str, ...]:
    paths: list[str] = []
    for key, child in shape.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            paths.extend(_leaf_paths(child, prefix=path))
        else:
            paths.append(path)
    return tuple(paths)


def _value_at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


def prefreeze_missing_fields(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return unresolved manifest leaves after first validating its exact shape."""

    _validate_shape(payload, _PREFREEZE_SHAPE)
    return tuple(
        path
        for path in _leaf_paths(_PREFREEZE_SHAPE)
        if _value_at(payload, path) is None
    )


_DIGEST_PATHS = tuple(
    path
    for path in _leaf_paths(_PREFREEZE_SHAPE)
    if path.endswith("_digest")
)
_REF_PATHS = tuple(
    path
    for path in _leaf_paths(_PREFREEZE_SHAPE)
    if path.endswith("_ref")
)
_ISOLATION_REF_PAIRS = (
    ("r1_identity.lineage_ref", "r2_identity.lineage_ref"),
    ("r1_identity.seed_ref", "r2_identity.seed_ref"),
    ("r1_identity.outcome_namespace_ref", "r2_identity.outcome_namespace_ref"),
    ("r1_identity.memory_namespace_ref", "r2_identity.memory_namespace_ref"),
    ("r1_identity.root_ref", "r2_identity.root_ref"),
    ("r1_identity.db_ref", "r2_identity.db_ref"),
)


@dataclass(frozen=True, slots=True)
class FrozenR1R2PrefreezeManifestV1:
    """Complete, canonical pre-outcome identity admitted by the dry-run launcher."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = canonical_value(self.payload)
        _validate_shape(normalized, _PREFREEZE_SHAPE)
        missing = prefreeze_missing_fields(normalized)
        if missing:
            raise Wave2IntegrationError(
                "prefreeze manifest has unresolved required fields: "
                + ", ".join(missing)
            )
        if normalized["schema"] != PREFREEZE_SCHEMA:
            raise Wave2IntegrationError("unsupported prefreeze manifest schema")
        if normalized["wave1_base"] != {
            "commit": WAVE1_ACCEPTED_COMMIT,
            "tree": WAVE1_ACCEPTED_TREE,
        }:
            raise Wave2IntegrationError("Wave 1 accepted base identity mismatch")
        _git_object(
            normalized["source_identity"]["commit"],
            field_name="source_identity.commit",
        )
        for path in _DIGEST_PATHS:
            validate_sha256(_value_at(normalized, path), field_name=path)
        for path in _REF_PATHS:
            value = _value_at(normalized, path)
            if not isinstance(value, str) or not value.strip():
                raise Wave2IntegrationError(f"{path} must be a non-empty identity ref")
        fixed_values = {
            "shared_proposal_call.no_retry": True,
            "shared_proposal_call.proposal_budget_per_side": 8,
            "shared_implementation.manual_candidate_patch_forbidden": True,
            "shared_implementation.qualification_evidence_class": "DEVELOPMENT_ONLY",
            "evidence_policy.held_out_absent": True,
            "r1_gate.requirements.minimum_fresh_specs": 4,
            "r1_gate.requirements.minimum_producer_roles": 2,
            "r1_gate.requirements.minimum_qualified": 2,
            "r1_gate.requirements.minimum_real_mechanism_changes": 1,
            "r1_gate.requirements.accepted_change_kinds": [
                "STRUCTURAL",
                "INTERACTION",
                "PROPAGATION",
            ],
        }
        for path, expected in fixed_values.items():
            observed = _value_at(normalized, path)
            if observed != expected:
                raise Wave2IntegrationError(
                    f"{path} must remain frozen at {expected!r}"
                )
        for field_name in ("granularity", "failure_rule_ref"):
            if not str(normalized["shared_proposal_call"][field_name]).strip():
                raise Wave2IntegrationError(
                    f"shared_proposal_call.{field_name} must be non-empty"
                )
        for field_name in ("token_budget", "call_count"):
            observed = normalized["shared_proposal_call"][field_name]
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed <= 0
            ):
                raise Wave2IntegrationError(
                    f"shared_proposal_call.{field_name} must be a positive integer"
                )
        for r1_path, r2_path in _ISOLATION_REF_PAIRS:
            if _value_at(normalized, r1_path) == _value_at(normalized, r2_path):
                raise Wave2IntegrationError(
                    f"{r1_path} and {r2_path} must be isolated"
                )
        side_refs = (
            normalized["r1_identity"]["side_a_identity_ref"],
            normalized["r1_identity"]["side_b_identity_ref"],
        )
        if len(set(side_refs)) != 2:
            raise Wave2IntegrationError("R1 A/B fresh identities must differ")
        for slot, values in normalized["r1_gate"]["result_slots"].items():
            if not isinstance(values, list) or values:
                raise Wave2IntegrationError(
                    f"pre-outcome {slot} must stay empty; only real R1 receipts "
                    "may populate the post-run gate"
                )
        object.__setattr__(self, "payload", normalized)

    @property
    def digest(self) -> str:
        return bytes_sha256(canonical_json_bytes(self.payload))

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(self.payload)


def load_prefreeze_manifest(
    path: Path,
    *,
    expected_digest: str,
) -> FrozenR1R2PrefreezeManifestV1:
    """Load exactly one canonical manifest and bind it to its expected digest."""

    expected = validate_sha256(expected_digest, field_name="expected_digest")
    raw = path.read_bytes()
    if bytes_sha256(raw) != expected:
        raise Wave2IntegrationError("prefreeze manifest byte digest mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Wave2IntegrationError(
            "prefreeze manifest is not valid UTF-8 JSON"
        ) from exc
    if canonical_json_bytes(payload) != raw:
        raise Wave2IntegrationError(
            "prefreeze manifest bytes are not canonical and digest-stable"
        )
    return FrozenR1R2PrefreezeManifestV1(payload=payload)


def dry_run_r1_r2_launcher(
    path: Path,
    *,
    expected_digest: str,
) -> dict[str, Any]:
    """Validate launch identities without invoking a Provider or an experiment."""

    manifest = load_prefreeze_manifest(path, expected_digest=expected_digest)
    return canonical_value(
        {
            "schema": DRY_RUN_RECEIPT_SCHEMA,
            "mode": "LOCAL_DETERMINISTIC_DRY_RUN",
            "manifest_ref": path.name,
            "manifest_digest": manifest.digest,
            "wave1_base_commit": WAVE1_ACCEPTED_COMMIT,
            "r1_r2_isolation_verified": True,
            "provider_calls": 0,
            "experiment_runs": 0,
            "outcomes_consumed": 0,
            "r1_gate_result_slots": "UNPOPULATED_REAL_R1_RECEIPTS_ONLY",
            "launch_authorized": False,
        }
    )
