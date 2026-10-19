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
    sha256_digest,
    validate_sha256,
)


WAVE1_ACCEPTED_COMMIT = "4d493939bb1e118ae9c99c89e8b73077078c3ab8"
WAVE1_ACCEPTED_TREE = "3fd503618a664025e8f7a4c751ffdb646787a3d7"
WAVE2_ACCEPTED_COMMIT = "56d3156f53d17d4d850368ce7a093fa368957d81"
WAVE2_ACCEPTED_TREE = "30b239b7ac85cd8de176d4b719b327ba02b96e0b"
WAVE2_ACCEPTED_TREE_ARCHIVE_SHA256 = (
    "f25d901a5493399c247f2e205a834c485ab52a32756cfc6fa97b97fe1d021ba7"
)
WAVE2_GATE_RECEIPT_SHA256 = (
    "fcb431c0ae8738f243ba10021770a2c507bee2b6d8454df622c2dc05700332eb"
)
PREFREEZE_SCHEMA = "recclaw.research-line.r1-r2-prefreeze-manifest.v3"
DRY_RUN_RECEIPT_SCHEMA = "recclaw.research-line.r1-r2-dry-run-receipt.v3"
BLOCKED_RECEIPT_SCHEMA = "recclaw.research-line.prefreeze-blocked-receipt.v2"
READY_RECEIPT_SCHEMA = "recclaw.research-line.r1-prefreeze-ready-receipt.v1"
OWNER_INTAKE_SCHEMA = "recclaw.research-line.wave2-owner-intake.v1"
GPT_5_4_MODEL_DIGEST = (
    "568d98474c084e840c6ddf03e03aa9ce82b577fef2912412546cfca2b278d99b"
)

_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}")
_ENTRYPOINT_RE = re.compile(
    r"[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*"
)
_UNRESOLVED_IDENTITY_MARKERS = frozenset(
    {"NULL", "TBD", "UNKNOWN", "UNRESOLVED", "UNSET"}
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


D1_ACCEPTED_INTAKE = Wave2OwnerIntakeV1(
    lane=Wave2OwnerLaneV1.D1_SCIENTIFIC_EPISODE_ADAPTER,
    accepted_commit="9d4937f48a3f37bdc031888bb5fb19222b25de96",
    parent_commit="5c6b064fa90f6d495e226831f0ec86fc9e89d391",
    owner_file_manifest_sha256=(
        "943d602d30bf4087742ee3463b0fabb7612bebbc1ffafa3e6ac818ec6811ffeb"
    ),
    targeted_tests_receipt_sha256=(
        "1ab03383dd7451ae2a2ae9042c606bf3430decf1aca0c4e286425e957c7e3d43"
    ),
    structure_lint_receipt_sha256=(
        "4ba243c0f613c873c8a3a2e3a37e427a36bf58206688ac984585d4e14e95a5fa"
    ),
    public_entrypoint=(
        "recclaw_core.experiments.helix_abc_v1."
        "scientific_episode_adapter:project_episode_to_mechanism_belief"
    ),
)

F0_ACCEPTED_INTAKE = Wave2OwnerIntakeV1(
    lane=Wave2OwnerLaneV1.F0_OPEN_META_INTERFACE,
    accepted_commit="ea348c3d301672d8fe090fe76c14e1a7241862da",
    parent_commit=WAVE1_ACCEPTED_COMMIT,
    owner_file_manifest_sha256=(
        "f0003649814872b4140f9880368646d423d3c79978f36040170a43c4435decb2"
    ),
    targeted_tests_receipt_sha256=(
        "cc506f417fa6ea6a2a1d874cb32a354d470cafb00fa143d197b74c2223cb02ca"
    ),
    structure_lint_receipt_sha256=(
        "04c2cb88d3a8acd984a37ca8dbd48317e9fac0a908335e76d9318bc5559e428c"
    ),
    public_entrypoint=(
        "recclaw_core.experiments.helix_abc_v1.open_meta:__all__"
    ),
)

E0_ACCEPTED_INTAKE = Wave2OwnerIntakeV1(
    lane=Wave2OwnerLaneV1.E0_SEARCH_ADAPTER,
    accepted_commit="a07bd2ccf1effb11702a738668db0af7d83c47d2",
    parent_commit=WAVE1_ACCEPTED_COMMIT,
    owner_file_manifest_sha256=(
        "8f7934c997a4fa71816b0b2af2f9f8df9af989ab3199164db485bccdc87d5ab0"
    ),
    targeted_tests_receipt_sha256=(
        "9e8cd269d3f4379f1759cdc3adde93d36fd252d0288a9400135cd28a19beeded"
    ),
    structure_lint_receipt_sha256=(
        "96cdfb1a519760db31125b29c81d296a04d5c8ee6fc18eb60d9f3fbe9cabdd01"
    ),
    public_entrypoint=(
        "recclaw_core.experiments.helix_abc_v1.search_adapter:__all__"
    ),
)


def accepted_wave2_harness() -> Wave2IntegrationHarnessV1:
    """Return the complete accepted E0/D1/F0 mechanical intake harness."""

    harness = Wave2IntegrationHarnessV1()
    for intake in (
        E0_ACCEPTED_INTAKE,
        D1_ACCEPTED_INTAKE,
        F0_ACCEPTED_INTAKE,
    ):
        harness = harness.attach(
            intake,
            observed_entrypoint=intake.public_entrypoint,
        )
    return harness


_PREFREEZE_SHAPE: dict[str, Any] = {
    "schema": None,
    "accepted_wave2": {
        "commit": None,
        "git_tree": None,
        "tree_archive_sha256": None,
        "gate_receipt_sha256": None,
    },
    "runtime_identity": {
        "runtime_ref": None,
        "runtime_digest": None,
        "dependency_lock_ref": None,
        "dependency_lock_digest": None,
    },
    "provider_identity": {
        "call_entrypoint_ref": None,
        "call_entrypoint_digest": None,
        "endpoint_ref": None,
        "endpoint_digest": None,
        "endpoint_support_status": None,
        "release_ref": None,
        "release_digest": None,
        "model_name": None,
        "model_digest": None,
        "returned_model": None,
        "authentication_status": None,
        "credential_config_digest": None,
        "credential_identity_digest": None,
        "credential_identity_present": None,
        "schema_probe_receipt_ref": None,
        "schema_probe_receipt_digest": None,
        "schema_probe_status": None,
        "probe_call_count": None,
    },
    "open_spec_contract": {
        "contract_ref": None,
        "contract_digest": None,
        "producer_adapter_ref": None,
        "producer_adapter_digest": None,
        "resolver_ref": None,
        "resolver_digest": None,
        "producer_roles": None,
        "resolution_results": None,
        "required_resolution": None,
        "fixed_catalog_candidate_allowed": None,
        "producer_rewrite_forbidden": None,
    },
    "shared_proposal_call": {
        "granularity": None,
        "token_budget": None,
        "call_count": None,
        "expected_proposals_per_call": None,
        "call_schedule_ref": None,
        "call_schedule_digest": None,
        "failure_rule_ref": None,
        "failure_rule_digest": None,
        "content_not_json_action": None,
        "replacement_call": None,
        "schema_relaxation": None,
        "successful_response_selection": None,
        "prompt_ref": None,
        "prompt_digest": None,
        "tool_ref": None,
        "tool_digest": None,
        "response_contract_ref": None,
        "response_contract_digest": None,
        "no_retry": None,
        "retry_count": None,
        "proposal_budget_per_side": None,
        "shared_call_contract_digest": None,
        "side_a_call_contract_digest": None,
        "side_b_call_contract_digest": None,
    },
    "shared_implementation": {
        "implementer_ref": None,
        "implementer_digest": None,
        "qualifier_ref": None,
        "qualifier_digest": None,
        "origin_blind_projection_ref": None,
        "origin_blind_projection_digest": None,
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
        "candidate_namespace_ref": None,
        "candidate_namespace_digest": None,
        "package_namespace_ref": None,
        "package_namespace_digest": None,
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
        "candidate_namespace_ref": None,
        "candidate_namespace_digest": None,
        "package_namespace_ref": None,
        "package_namespace_digest": None,
    },
    "evidence_policy": {
        "held_out_absent": None,
        "missingness_policy_ref": None,
        "missingness_policy_digest": None,
        "threshold_policy_ref": None,
        "threshold_policy_digest": None,
        "analysis_plan_ref": None,
        "analysis_plan_digest": None,
        "qualification_gate_ref": None,
        "qualification_gate_digest": None,
    },
    "r1_gate": {
        "requirements": {
            "scope": None,
            "minimum_fresh_specs_per_side": None,
            "minimum_producer_roles_per_side": None,
            "minimum_qualified_per_side": None,
            "minimum_real_mechanism_changes": None,
            "accepted_change_kinds": None,
            "negative_fixture_required": None,
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


def proposal_call_contract_digest(payload: Mapping[str, Any]) -> str:
    """Bind every provider/call field that must be identical across R1 A/B."""

    provider = payload["provider_identity"]
    proposal = payload["shared_proposal_call"]
    return sha256_digest(
        {
            "provider": {
                "credential_identity_digest": provider[
                    "credential_identity_digest"
                ],
                "endpoint_digest": provider["endpoint_digest"],
                "model_digest": provider["model_digest"],
                "model_name": provider["model_name"],
                "release_digest": provider["release_digest"],
            },
            "proposal_call": {
                "call_count": proposal["call_count"],
                "call_schedule_digest": proposal["call_schedule_digest"],
                "content_not_json_action": proposal[
                    "content_not_json_action"
                ],
                "expected_proposals_per_call": proposal[
                    "expected_proposals_per_call"
                ],
                "failure_rule_digest": proposal["failure_rule_digest"],
                "granularity": proposal["granularity"],
                "no_retry": proposal["no_retry"],
                "prompt_digest": proposal["prompt_digest"],
                "proposal_budget_per_side": proposal[
                    "proposal_budget_per_side"
                ],
                "replacement_call": proposal["replacement_call"],
                "response_contract_digest": proposal[
                    "response_contract_digest"
                ],
                "retry_count": proposal["retry_count"],
                "schema_relaxation": proposal["schema_relaxation"],
                "successful_response_selection": proposal[
                    "successful_response_selection"
                ],
                "token_budget": proposal["token_budget"],
                "tool_digest": proposal["tool_digest"],
            },
        }
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
    (
        "r1_identity.candidate_namespace_ref",
        "r2_identity.candidate_namespace_ref",
    ),
    (
        "r1_identity.package_namespace_ref",
        "r2_identity.package_namespace_ref",
    ),
)
_ISOLATION_DIGEST_PAIRS = tuple(
    (left.replace("_ref", "_digest"), right.replace("_ref", "_digest"))
    for left, right in _ISOLATION_REF_PAIRS
)


@dataclass(frozen=True, slots=True)
class FrozenR1R2PrefreezeManifestV3:
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
        if normalized["accepted_wave2"] != {
            "commit": WAVE2_ACCEPTED_COMMIT,
            "git_tree": WAVE2_ACCEPTED_TREE,
            "tree_archive_sha256": WAVE2_ACCEPTED_TREE_ARCHIVE_SHA256,
            "gate_receipt_sha256": WAVE2_GATE_RECEIPT_SHA256,
        }:
            raise Wave2IntegrationError("accepted Wave 2 source identity mismatch")
        _git_object(
            normalized["accepted_wave2"]["commit"],
            field_name="accepted_wave2.commit",
        )
        _git_object(
            normalized["accepted_wave2"]["git_tree"],
            field_name="accepted_wave2.git_tree",
        )
        for path in _DIGEST_PATHS:
            validate_sha256(_value_at(normalized, path), field_name=path)
        for path in _REF_PATHS:
            value = _value_at(normalized, path)
            if not isinstance(value, str) or not value.strip():
                raise Wave2IntegrationError(f"{path} must be a non-empty identity ref")
            if value.strip().upper() in _UNRESOLVED_IDENTITY_MARKERS:
                raise Wave2IntegrationError(
                    f"{path} must not use an unresolved identity marker"
                )
        fixed_values = {
            "provider_identity.endpoint_support_status": (
                "VERIFIED_EXACT_GPT_5_4_FRESH_OPEN_SPEC_SCHEMA"
            ),
            "provider_identity.model_name": "gpt-5.4",
            "provider_identity.model_digest": GPT_5_4_MODEL_DIGEST,
            "provider_identity.returned_model": "gpt-5.4",
            "provider_identity.authentication_status": "VERIFIED",
            "provider_identity.credential_identity_present": True,
            "provider_identity.schema_probe_status": "PASS",
            "provider_identity.probe_call_count": 1,
            "open_spec_contract.producer_roles": [
                "mechanism_composer",
                "lineage_refiner",
                "falsification_designer",
                "frontier_architect",
            ],
            "open_spec_contract.resolution_results": [
                "SEARCH_READY",
                "INNOVATION_REQUIRED",
                "DEFERRED_PROTOCOL_CHANGE",
                "UNSUPPORTED",
                "INVALID_SPEC",
            ],
            "open_spec_contract.required_resolution": "INNOVATION_REQUIRED",
            "open_spec_contract.fixed_catalog_candidate_allowed": False,
            "open_spec_contract.producer_rewrite_forbidden": True,
            "shared_proposal_call.granularity": (
                "ONE_PREASSIGNED_PRODUCER_ROLE_ONE_PROPOSAL_PER_CALL"
            ),
            "shared_proposal_call.token_budget": 6000,
            "shared_proposal_call.call_count": 8,
            "shared_proposal_call.expected_proposals_per_call": 1,
            "shared_proposal_call.no_retry": True,
            "shared_proposal_call.retry_count": 0,
            "shared_proposal_call.proposal_budget_per_side": 8,
            "shared_proposal_call.content_not_json_action": (
                "TERMINAL_CONSUME_PREASSIGNED_SLOT"
            ),
            "shared_proposal_call.replacement_call": "FORBIDDEN",
            "shared_proposal_call.schema_relaxation": "FORBIDDEN",
            "shared_proposal_call.successful_response_selection": "FORBIDDEN",
            "shared_implementation.manual_candidate_patch_forbidden": True,
            "shared_implementation.qualification_evidence_class": "DEVELOPMENT_ONLY",
            "evidence_policy.held_out_absent": True,
            "r1_gate.requirements.scope": (
                "PER_SIDE_EXCEPT_REAL_MECHANISM_OVERALL"
            ),
            "r1_gate.requirements.minimum_fresh_specs_per_side": 4,
            "r1_gate.requirements.minimum_producer_roles_per_side": 2,
            "r1_gate.requirements.minimum_qualified_per_side": 2,
            "r1_gate.requirements.minimum_real_mechanism_changes": 1,
            "r1_gate.requirements.accepted_change_kinds": [
                "STRUCTURAL",
                "INTERACTION",
                "PROPAGATION",
            ],
            "r1_gate.requirements.negative_fixture_required": True,
        }
        for path, expected in fixed_values.items():
            observed = _value_at(normalized, path)
            if observed != expected:
                raise Wave2IntegrationError(
                    f"{path} must remain frozen at {expected!r}"
                )
        for r1_path, r2_path in (
            *_ISOLATION_REF_PAIRS,
            *_ISOLATION_DIGEST_PAIRS,
        ):
            if _value_at(normalized, r1_path) == _value_at(normalized, r2_path):
                raise Wave2IntegrationError(
                    f"{r1_path} and {r2_path} must be isolated"
                )
        shared_call_digest = normalized["shared_proposal_call"][
            "shared_call_contract_digest"
        ]
        expected_call_digest = proposal_call_contract_digest(normalized)
        if shared_call_digest != expected_call_digest:
            raise Wave2IntegrationError(
                "shared proposal call contract digest does not bind the "
                "frozen provider and call fields"
            )
        side_call_digests = (
            normalized["shared_proposal_call"]["side_a_call_contract_digest"],
            normalized["shared_proposal_call"]["side_b_call_contract_digest"],
        )
        if side_call_digests != (shared_call_digest, shared_call_digest):
            raise Wave2IntegrationError(
                "R1 A/B model, granularity, token budget, call count, prompt, "
                "tool, response contract, and failure rule must be identical"
            )
        side_refs = (
            normalized["r1_identity"]["side_a_identity_ref"],
            normalized["r1_identity"]["side_b_identity_ref"],
        )
        if len(set(side_refs)) != 2:
            raise Wave2IntegrationError("R1 A/B fresh identities must differ")
        side_digests = (
            normalized["r1_identity"]["side_a_identity_digest"],
            normalized["r1_identity"]["side_b_identity_digest"],
        )
        if len(set(side_digests)) != 2:
            raise Wave2IntegrationError(
                "R1 A/B fresh identity digests must differ"
            )
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
) -> FrozenR1R2PrefreezeManifestV3:
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
    manifest = FrozenR1R2PrefreezeManifestV3(payload=payload)
    repo_root = path.resolve().parents[3]
    _verify_prefreeze_resource_bindings(
        repo_root=repo_root,
        payload=manifest.payload,
    )
    return manifest


def _repo_resource(
    *,
    repo_root: Path,
    ref: str,
) -> tuple[Path, tuple[str, ...]]:
    prefix = "repo:"
    if not ref.startswith(prefix):
        raise Wave2IntegrationError(
            f"prefreeze local resource ref must start with {prefix!r}"
        )
    relative, marker, fragment = ref[len(prefix) :].partition("#")
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise Wave2IntegrationError(
            "prefreeze resource ref escapes the repository"
        ) from exc
    if not candidate.is_file():
        raise Wave2IntegrationError(
            f"prefreeze resource ref does not exist: {relative}"
        )
    fragment_parts = tuple(
        part for part in fragment.split("/") if part
    ) if marker else ()
    return candidate, fragment_parts


def _verify_repo_digest_binding(
    *,
    repo_root: Path,
    ref: str,
    digest: str,
) -> None:
    resource, fragment = _repo_resource(repo_root=repo_root, ref=ref)
    if not fragment:
        observed = bytes_sha256(resource.read_bytes())
    else:
        try:
            value: Any = json.loads(resource.read_bytes())
            for part in fragment:
                value = value[part]
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            raise Wave2IntegrationError(
                f"prefreeze JSON resource fragment is invalid: {ref}"
            ) from exc
        observed = sha256_digest(value)
    if observed != digest:
        raise Wave2IntegrationError(
            f"prefreeze resource digest mismatch for {ref}"
        )


def _verify_prefreeze_resource_bindings(
    *,
    repo_root: Path,
    payload: Mapping[str, Any],
) -> None:
    raw_or_fragment_pairs = (
        ("runtime_identity.runtime_ref", "runtime_identity.runtime_digest"),
        (
            "runtime_identity.dependency_lock_ref",
            "runtime_identity.dependency_lock_digest",
        ),
        (
            "shared_proposal_call.call_schedule_ref",
            "shared_proposal_call.call_schedule_digest",
        ),
        (
            "shared_proposal_call.failure_rule_ref",
            "shared_proposal_call.failure_rule_digest",
        ),
        (
            "shared_proposal_call.prompt_ref",
            "shared_proposal_call.prompt_digest",
        ),
        (
            "shared_proposal_call.tool_ref",
            "shared_proposal_call.tool_digest",
        ),
        (
            "shared_proposal_call.response_contract_ref",
            "shared_proposal_call.response_contract_digest",
        ),
        (
            "provider_identity.schema_probe_receipt_ref",
            "provider_identity.schema_probe_receipt_digest",
        ),
        (
            "evidence_policy.missingness_policy_ref",
            "evidence_policy.missingness_policy_digest",
        ),
        (
            "evidence_policy.threshold_policy_ref",
            "evidence_policy.threshold_policy_digest",
        ),
        (
            "evidence_policy.analysis_plan_ref",
            "evidence_policy.analysis_plan_digest",
        ),
        (
            "evidence_policy.qualification_gate_ref",
            "evidence_policy.qualification_gate_digest",
        ),
    )
    identity_names = (
        "lineage",
        "seed",
        "outcome_namespace",
        "memory_namespace",
        "root",
        "db",
        "candidate_namespace",
        "package_namespace",
    )
    identity_pairs = tuple(
        (
            f"{phase}_identity.{name}_ref",
            f"{phase}_identity.{name}_digest",
        )
        for phase in ("r1", "r2")
        for name in identity_names
    ) + (
        (
            "r1_identity.side_a_identity_ref",
            "r1_identity.side_a_identity_digest",
        ),
        (
            "r1_identity.side_b_identity_ref",
            "r1_identity.side_b_identity_digest",
        ),
    )
    for ref_path, digest_path in (*raw_or_fragment_pairs, *identity_pairs):
        _verify_repo_digest_binding(
            repo_root=repo_root,
            ref=_value_at(payload, ref_path),
            digest=_value_at(payload, digest_path),
        )

    release_path, release_fragment = _repo_resource(
        repo_root=repo_root,
        ref=payload["provider_identity"]["release_ref"],
    )
    if release_fragment:
        raise Wave2IntegrationError(
            "provider release ref must bind the complete release resource"
        )
    try:
        release = json.loads(release_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Wave2IntegrationError(
            "provider release resource is invalid"
        ) from exc
    release_digest = release.get("release_digest")
    release_preimage = dict(release)
    release_preimage.pop("release_digest", None)
    if (
        release_digest != payload["provider_identity"]["release_digest"]
        or sha256_digest(release_preimage) != release_digest
        or release.get("endpoint_digest")
        != payload["provider_identity"]["endpoint_digest"]
        or release.get("model") != "gpt-5.4"
        or release.get("response_schema_digest")
        != payload["shared_proposal_call"]["response_contract_digest"]
        or release.get("retry_count") != 0
        or release.get("request_mode") != "SINGLE_JSON_SCHEMA_NO_TOOLS"
    ):
        raise Wave2IntegrationError(
            "provider release resource differs from the frozen call contract"
        )

    probe_path, probe_fragment = _repo_resource(
        repo_root=repo_root,
        ref=payload["provider_identity"]["schema_probe_receipt_ref"],
    )
    if probe_fragment:
        raise Wave2IntegrationError(
            "provider probe receipt ref must bind the complete receipt"
        )
    probe = json.loads(probe_path.read_bytes())
    required_probe_values = {
        "status": "PASS",
        "physical_provider_calls": 1,
        "retry_count": 0,
        "authentication_status": "VERIFIED",
        "model_requested": "gpt-5.4",
        "returned_model": "gpt-5.4",
        "endpoint_digest": payload["provider_identity"]["endpoint_digest"],
        "credential_config_digest": payload["provider_identity"][
            "credential_config_digest"
        ],
        "credential_identity_digest": payload["provider_identity"][
            "credential_identity_digest"
        ],
        "provider_release_digest": payload["provider_identity"][
            "release_digest"
        ],
        "response_schema_digest": payload["shared_proposal_call"][
            "response_contract_digest"
        ],
        "sensitive_values_persisted": False,
        "sensitive_headers_persisted": False,
        "research_candidates_generated": 0,
        "open_specs_projected": 0,
        "resolver_calls": 0,
        "candidate_roots_created": 0,
        "candidate_admissions": 0,
        "training_runs": 0,
        "outcomes_consumed": 0,
        "held_out_reads": 0,
    }
    for field_name, expected_value in required_probe_values.items():
        if probe.get(field_name) != expected_value:
            raise Wave2IntegrationError(
                f"provider probe receipt does not prove {field_name}"
            )
    if probe.get("blocked_fields") != []:
        raise Wave2IntegrationError(
            "provider probe receipt retains unresolved fields"
        )


# Import compatibility for the accepted G validator name.  The manifest schema
# itself is v3 and the v3 class above is the only implementation.
FrozenR1R2PrefreezeManifestV2 = FrozenR1R2PrefreezeManifestV3


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
            "accepted_wave2_commit": WAVE2_ACCEPTED_COMMIT,
            "r1_r2_isolation_verified": True,
            "provider_calls": 0,
            "gpu_runs": 0,
            "experiment_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
            "roots_created": 0,
            "databases_created": 0,
            "r1_gate_result_slots": "UNPOPULATED_REAL_R1_RECEIPTS_ONLY",
            "launch_authorized": False,
        }
    )


def r1_prefreeze_ready_receipt(
    path: Path,
    *,
    expected_digest: str,
) -> dict[str, Any]:
    """Emit the sole precondition for an independent R1 worker launch."""

    manifest = load_prefreeze_manifest(path, expected_digest=expected_digest)
    provider = manifest.payload["provider_identity"]
    proposal = manifest.payload["shared_proposal_call"]
    return canonical_value(
        {
            "schema": READY_RECEIPT_SCHEMA,
            "status": "R1_PREFREEZE_READY",
            "manifest_ref": path.name,
            "manifest_digest": manifest.digest,
            "accepted_wave2_commit": WAVE2_ACCEPTED_COMMIT,
            "accepted_wave2_tree": WAVE2_ACCEPTED_TREE,
            "provider_probe_receipt_ref": provider[
                "schema_probe_receipt_ref"
            ],
            "provider_probe_receipt_digest": provider[
                "schema_probe_receipt_digest"
            ],
            "provider_probe_status": provider["schema_probe_status"],
            "provider_probe_call_count": provider["probe_call_count"],
            "model": provider["model_name"],
            "returned_model": provider["returned_model"],
            "shared_call_contract_digest": proposal[
                "shared_call_contract_digest"
            ],
            "side_a_call_contract_digest": proposal[
                "side_a_call_contract_digest"
            ],
            "side_b_call_contract_digest": proposal[
                "side_b_call_contract_digest"
            ],
            "r1_worker_launch_authorized": True,
            "r2_launch_authorized": False,
            "training_started": False,
            "candidate_admission_performed": False,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
            "claim_ceiling": "DEVELOPMENT_ONLY",
            "authorization_scope": (
                "INDEPENDENT_R1_WORKER_MAY_START_EXACT_FROZEN_R1_ONLY"
            ),
        }
    )
