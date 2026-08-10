"""Research-only, identity-bound Portfolio profile source.

The source is a frozen bundle of pre-outcome inputs.  It is deliberately
small: E owns PortfolioCandidateProfileV2 validation and this module only
binds those inputs to the current resolved Search bindings, projects bounded
prior memory, and exposes a stable source identity for prepared-round resume.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)

from .portfolio import PortfolioCandidateV2
from .portfolio_profile_builder import (
    PortfolioCandidateProfileV2,
    build_portfolio_candidate_profile_v2,
)


PROFILE_SOURCE_SCHEMA = "recclaw.research-line.profile-source.v1"
PROFILE_SOURCE_SCHEMA_VERSION = "v1"


class ResearchProfileSourceError(ValueError):
    """Raised when a profile source cannot produce a complete handoff."""


_HISTORY_KEYS = {
    "prior_attempts": frozenset(
        {
            "round",
            "round_index",
            "candidate_id",
            "semantic_digest",
            "candidate_semantic_digest",
            "family_id",
            "parent_id",
            "parent_candidate_id",
            "compute_pattern",
            "sealed",
            "sealed_valid_seal",
            "sealed_resource_admitted",
            "attempt_ref",
            "attempt_digest",
            "source_digest",
        }
    ),
    "family_history": frozenset(
        {
            "round",
            "round_index",
            "family_id",
            "candidate_id",
            "semantic_digest",
            "candidate_semantic_digest",
            "stable",
            "stable_delta",
            "source_digest",
        }
    ),
    "parent_history": frozenset(
        {
            "round",
            "round_index",
            "parent_id",
            "parent_candidate_id",
            "stable",
            "stable_validation",
            "stable_lineage_risk",
            "stable_delta",
            "source_digest",
        }
    ),
    "frontier_history": frozenset(
        {
            "round",
            "round_index",
            "family_id",
            "candidate_id",
            "semantic_digest",
            "candidate_semantic_digest",
            "stable",
            "stable_frontier_gain",
            "source_digest",
        }
    ),
}


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ResearchProfileSourceError(
            f"{name} must be a normalized non-empty string"
        )
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchProfileSourceError(f"{name} must be a mapping")
    try:
        return dict(canonical_value(dict(value)))
    except (TypeError, ValueError) as error:
        raise ResearchProfileSourceError(f"{name} is not canonicalizable") from error


def _optional_mapping(value: Any, *, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return canonical_value(_mapping(value, name=name))


def _source_payload(
    *,
    schema_version: str,
    source_ref: str,
    records: Mapping[str, Mapping[str, Any]] | None = None,
    pre_round_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if (records is None) == (pre_round_policies is None):
        raise ResearchProfileSourceError(
            "source payload must contain exactly one source form"
        )
    body = (
        {"records": records}
        if pre_round_policies is None
        else {"pre_round_policies": pre_round_policies}
    )
    return canonical_value(
        {
            "schema": PROFILE_SOURCE_SCHEMA,
            "schema_version": schema_version,
            "source_ref": source_ref,
            **body,
        }
    )


_POLICY_SELECTOR_FIELDS = (
    "producer_role",
    "capability_ref",
    "capability_digest",
    "family_id",
)


def _policy_selector(value: Mapping[str, Any], *, name: str) -> dict[str, str]:
    policy = _mapping(value, name=name)
    selector = {
        "producer_role": _text(
            policy.get("producer_role"),
            name=f"{name}.producer_role",
        ),
    }
    capability_ref = policy.get("capability_ref")
    capability_digest = policy.get("capability_digest")
    family_id = policy.get("family_id")
    if capability_ref is not None:
        selector["capability_ref"] = _text(
            capability_ref,
            name=f"{name}.capability_ref",
        )
        if capability_digest is not None:
            selector["capability_digest"] = validate_sha256(
                capability_digest,
                field_name=f"{name}.capability_digest",
            )
    elif capability_digest is not None:
        raise ResearchProfileSourceError(
            f"{name}.capability_digest requires capability_ref"
        )
    if family_id is not None:
        selector["family_id"] = _text(
            family_id,
            name=f"{name}.family_id",
        )
    return canonical_value(selector)


def _policy_key(selector: Mapping[str, Any]) -> str:
    return sha256_digest(selector)


def _reject_future_candidate_fields(value: Any, *, name: str) -> None:
    """Reject candidate-bound inputs in a pre-round policy template."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in {"candidate_id", "candidate_identity", "candidate_ids"}:
                raise ResearchProfileSourceError(
                    f"{name} cannot contain future candidate identity field {key}"
                )
            _reject_future_candidate_fields(item, name=f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_future_candidate_fields(item, name=f"{name}[{index}]")


def _resource_compute_pattern(value: Any, *, name: str) -> str:
    """Return the identity-bound compute pattern from resource evidence."""

    profile = _mapping(value, name=name)
    prediction = _mapping(profile.get("prediction"), name=f"{name}.prediction")
    nested = prediction.get("compute_pattern")
    top_level = profile.get("compute_pattern")
    if nested is None and top_level is None:
        raise ResearchProfileSourceError(
            f"{name} requires an explicit prediction.compute_pattern"
        )
    if nested is not None:
        nested = _text(nested, name=f"{name}.prediction.compute_pattern")
    if top_level is not None:
        top_level = _text(top_level, name=f"{name}.compute_pattern")
    if nested is not None and top_level is not None and nested != top_level:
        raise ResearchProfileSourceError(
            f"{name} has conflicting compute_pattern evidence"
        )
    return nested or top_level


def _policy_resource_profile(
    value: Any,
    *,
    selector: Mapping[str, str],
    name: str,
    allow_candidate_id: bool = False,
) -> dict[str, Any]:
    profile = _mapping(value, name=name)
    if not allow_candidate_id:
        _reject_future_candidate_fields(profile, name=name)
    capability_ref = selector.get("capability_ref")
    if capability_ref is not None and profile.get("candidate_ref") != capability_ref:
        raise ResearchProfileSourceError(
            f"{name}.candidate_ref does not match the stable capability selector"
        )
    validate_sha256(profile.get("profile_digest"), field_name=f"{name}.profile_digest")
    source_digest = validate_sha256(
        profile.get("candidate_source_sha256"),
        field_name=f"{name}.candidate_source_sha256",
    )
    profile["candidate_source_sha256"] = source_digest
    package_digest = profile.get("candidate_package_digest")
    if package_digest is not None:
        profile["candidate_package_digest"] = validate_sha256(
            package_digest,
            field_name=f"{name}.candidate_package_digest",
        )
    prediction = _mapping(profile.get("prediction"), name=f"{name}.prediction")
    prediction_identity = _mapping(
        prediction.get("identity"),
        name=f"{name}.prediction.identity",
    )
    if prediction_identity.get("candidate_ref") not in (None, capability_ref):
        raise ResearchProfileSourceError(
            f"{name}.prediction.identity.candidate_ref drift"
        )
    if not allow_candidate_id:
        prediction_identity.pop("candidate_id", None)
    prediction_identity.pop("candidate_ref", None)
    prediction["identity"] = prediction_identity
    _resource_compute_pattern(profile, name=name)
    profile["prediction"] = prediction
    return canonical_value(profile)


def _normalize_pre_round_policy(
    value: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    policy = _mapping(value, name=name)
    for field_name in ("candidate_id", "candidate_identity", "candidate_ids"):
        if field_name in policy:
            raise ResearchProfileSourceError(
                f"{name} cannot contain future candidate identity field {field_name}"
            )
    _reject_future_candidate_fields(policy.get("qualified_execution"), name=f"{name}.qualified_execution")
    _reject_future_candidate_fields(policy.get("exploration_mandate"), name=f"{name}.exploration_mandate")
    _reject_future_candidate_fields(policy.get("task_or_exploration_mandate"), name=f"{name}.task_or_exploration_mandate")
    selector = _policy_selector(policy, name=name)
    compute_pattern = policy.get("compute_pattern")
    if compute_pattern is not None:
        compute_pattern = _text(
            compute_pattern,
            name=f"{name}.compute_pattern",
        )
    calibration_prior = policy.get("calibration_prior")
    if not isinstance(calibration_prior, Mapping):
        raise ResearchProfileSourceError(
            f"{name}.calibration_prior is required and must be explicit"
        )
    resource_profile = None
    if policy.get("resource_profile") is not None:
        if selector.get("capability_ref") is None:
            raise ResearchProfileSourceError(
                f"{name}.resource_profile requires a fixed capability_ref selector"
            )
        resource_profile = _policy_resource_profile(
            policy.get("resource_profile"),
            selector=selector,
            name=f"{name}.resource_profile",
        )
        prediction_pattern = _resource_compute_pattern(
            resource_profile,
            name=f"{name}.resource_profile",
        )
        if compute_pattern is not None and prediction_pattern != compute_pattern:
            raise ResearchProfileSourceError(
                f"{name}.compute_pattern does not match resource evidence"
            )
    mandate = policy.get("exploration_mandate")
    alias_mandate = policy.get("task_or_exploration_mandate")
    if mandate is not None and alias_mandate is not None:
        if canonical_value(mandate) != canonical_value(alias_mandate):
            raise ResearchProfileSourceError(
                f"{name} provides conflicting exploration mandate fields"
            )
    if mandate is None:
        mandate = alias_mandate
    if mandate is not None:
        mandate = _mapping(mandate, name=f"{name}.exploration_mandate")
        mandate_role = mandate.get("producer_role")
        if mandate_role != selector["producer_role"]:
            raise ResearchProfileSourceError(
                f"{name}.exploration_mandate producer_role drift"
            )
    if policy.get("qualified_execution") is not None:
        raise ResearchProfileSourceError(
            f"{name}.qualified_execution cannot be embedded in a pre-round policy"
        )
    if policy.get("candidate_root_path") is not None:
        raise ResearchProfileSourceError(
            f"{name}.candidate_root_path cannot be embedded in a pre-round policy"
        )
    normalized = {
        **selector,
        "calibration_prior": calibration_value(calibration_prior),
        "exploration_mandate": (
            canonical_value(mandate) if mandate is not None else None
        ),
        "resource_profile": resource_profile,
    }
    if "parent_id" in policy:
        parent_id = policy["parent_id"]
        if parent_id is not None:
            parent_id = _text(parent_id, name=f"{name}.parent_id")
        normalized["parent_id"] = parent_id
    for field_name in (
        "prior_attempts",
        "family_history",
        "parent_history",
        "frontier_history",
    ):
        if field_name in policy:
            normalized[field_name] = canonical_value(policy[field_name])
    if compute_pattern is not None:
        normalized["compute_pattern"] = compute_pattern
    return canonical_value(normalized)


def _normalize_pre_round_policies(value: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        raw_values = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
    else:
        raise ResearchProfileSourceError(
            "pre_round_policies must be a mapping or sequence"
        )
    normalized: dict[str, Mapping[str, Any]] = {}
    for index, raw_policy in enumerate(raw_values):
        policy = _normalize_pre_round_policy(
            raw_policy,
            name=f"pre_round_policies[{index}]",
        )
        key = _policy_key(policy)
        if key in normalized:
            raise ResearchProfileSourceError(
                "pre_round_policies contain duplicate stable selector coverage"
            )
        normalized[key] = policy
    if not normalized:
        raise ResearchProfileSourceError(
            "pre_round_policies must contain at least one policy"
        )
    return normalized


def lineage_identity_digest(candidate: PortfolioCandidateV2) -> str:
    """Digest the identity fields that must not drift across the handoff."""

    if not isinstance(candidate, PortfolioCandidateV2):
        raise ResearchProfileSourceError("candidate must be PortfolioCandidateV2")
    return sha256_digest(
        {
            "candidate_id": candidate.candidate_id,
            "semantic_digest": candidate.semantic_digest,
            "family_id": candidate.family_id,
            "parent_id": candidate.parent_id,
            "compute_pattern": candidate.compute_pattern,
        }
    )


@dataclass(frozen=True, slots=True)
class ResearchProfileRecordV1:
    """One complete E profile plus its physical admission inputs."""

    portfolio_profile: PortfolioCandidateProfileV2
    qualified_execution: Mapping[str, Any] | None
    candidate_root_path: str | None
    resource_profile: Mapping[str, Any]
    calibration_prior: Mapping[str, Any]
    task_or_exploration_mandate: Mapping[str, Any] | None
    durable_evidence_digests: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_profile, PortfolioCandidateProfileV2):
            raise ResearchProfileSourceError(
                "portfolio_profile must be PortfolioCandidateProfileV2"
            )
        object.__setattr__(
            self,
            "qualified_execution",
            _optional_mapping(
                self.qualified_execution,
                name="qualified_execution",
            ),
        )
        root = self.candidate_root_path
        if root is not None:
            if not isinstance(root, (str, Path)) or not str(root).strip():
                raise ResearchProfileSourceError(
                    "candidate_root_path must be a non-empty path"
                )
            object.__setattr__(self, "candidate_root_path", str(Path(root).resolve()))
        object.__setattr__(
            self,
            "resource_profile",
            canonical_value(
                _mapping(self.resource_profile, name="resource_profile")
            ),
        )
        object.__setattr__(
            self,
            "calibration_prior",
            canonical_value(
                _mapping(self.calibration_prior, name="calibration_prior")
            ),
        )
        object.__setattr__(
            self,
            "task_or_exploration_mandate",
            _optional_mapping(
                self.task_or_exploration_mandate,
                name="task_or_exploration_mandate",
            ),
        )
        evidence_digests = self.durable_evidence_digests
        if evidence_digests is not None:
            if not isinstance(evidence_digests, Mapping):
                raise ResearchProfileSourceError(
                    "durable_evidence_digests must be a mapping"
                )
            normalized_digests: dict[str, str] = {}
            for key, value in evidence_digests.items():
                normalized_digests[_text(str(key), name="evidence digest key")] = (
                    validate_sha256(
                        value,
                        field_name=f"durable_evidence_digests.{key}",
                    )
                )
            object.__setattr__(
                self,
                "durable_evidence_digests",
                canonical_value(normalized_digests),
            )

    @property
    def profile_digest(self) -> str:
        return self.portfolio_profile.profile_digest

    @property
    def lineage_identity_digest(self) -> str:
        return lineage_identity_digest(self.portfolio_profile.candidate)


def _durable_evidence_digests(
    *,
    qualified_execution: Mapping[str, Any] | None,
    candidate_root_path: str | None,
    resource_profile: Mapping[str, Any],
) -> dict[str, str]:
    digests = {
        "resource_profile": sha256_digest(resource_profile),
    }
    if qualified_execution is not None:
        digests["qualified_execution"] = sha256_digest(qualified_execution)
    if candidate_root_path is not None:
        digests["candidate_root_path"] = sha256_digest(
            {"candidate_root_path": str(Path(candidate_root_path).resolve())}
        )
    return canonical_value(digests)


@dataclass(frozen=True, slots=True)
class ResearchProfileSourceV1:
    """Stable, immutable Research-side source for current-round profiles."""

    schema_version: str
    source_ref: str
    source_digest: str
    records: Mapping[str, Mapping[str, Any]]
    pre_round_policies: Mapping[str, Mapping[str, Any]] | None = None

    schema = PROFILE_SOURCE_SCHEMA

    def __post_init__(self) -> None:
        schema_version = _text(self.schema_version, name="schema_version")
        source_ref = _text(self.source_ref, name="source_ref")
        records_raw = _mapping(self.records, name="records")
        records: dict[str, Mapping[str, Any]] = {}
        for raw_candidate_id, raw_record in records_raw.items():
            candidate_id = _text(raw_candidate_id, name="records.candidate_id")
            record = _mapping(raw_record, name=f"records[{candidate_id}]")
            identity = record.get("candidate_identity")
            if identity is not None:
                identity_map = _mapping(
                    identity,
                    name=f"records[{candidate_id}].candidate_identity",
                )
                identity_candidate_id = identity_map.get("candidate_id")
                if identity_candidate_id is not None and identity_candidate_id != candidate_id:
                    raise ResearchProfileSourceError(
                        "profile source candidate_identity candidate_id drift"
                    )
                record["candidate_identity"] = identity_map
            records[candidate_id] = canonical_value(record)
        policies = None
        if self.pre_round_policies is not None:
            policies = _normalize_pre_round_policies(self.pre_round_policies)
            if records:
                raise ResearchProfileSourceError(
                    "exact records and pre-round policies are mutually exclusive"
                )
        expected_digest = sha256_digest(
            _source_payload(
                schema_version=schema_version,
                source_ref=source_ref,
                records=records if policies is None else None,
                pre_round_policies=policies,
            )
        )
        try:
            supplied_digest = validate_sha256(
                self.source_digest,
                field_name="source_digest",
            )
        except (TypeError, ValueError) as error:
            raise ResearchProfileSourceError(
                "source_digest must be a SHA-256 digest"
            ) from error
        if supplied_digest != expected_digest:
            raise ResearchProfileSourceError(
                "source_digest does not match the frozen profile source bundle"
            )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "source_ref", source_ref)
        object.__setattr__(self, "source_digest", supplied_digest)
        object.__setattr__(self, "records", canonical_value(records))
        object.__setattr__(
            self,
            "pre_round_policies",
            canonical_value(policies) if policies is not None else None,
        )

    @classmethod
    def from_records(
        cls,
        *,
        source_ref: str,
        records: Mapping[str, Mapping[str, Any]],
        schema_version: str = PROFILE_SOURCE_SCHEMA_VERSION,
    ) -> "ResearchProfileSourceV1":
        normalized_records = _mapping(records, name="records")
        normalized_records = {
            _text(key, name="records.candidate_id"): canonical_value(
                _mapping(value, name=f"records[{key}]")
            )
            for key, value in normalized_records.items()
        }
        payload = _source_payload(
            schema_version=_text(schema_version, name="schema_version"),
            source_ref=_text(source_ref, name="source_ref"),
            records=normalized_records,
        )
        return cls(
            schema_version=schema_version,
            source_ref=source_ref,
            source_digest=sha256_digest(payload),
            records=normalized_records,
        )

    @classmethod
    def from_pre_round_policies(
        cls,
        *,
        source_ref: str,
        policies: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
        schema_version: str = PROFILE_SOURCE_SCHEMA_VERSION,
    ) -> "ResearchProfileSourceV1":
        normalized_policies = _normalize_pre_round_policies(policies)
        payload = _source_payload(
            schema_version=_text(schema_version, name="schema_version"),
            source_ref=_text(source_ref, name="source_ref"),
            pre_round_policies=normalized_policies,
        )
        return cls(
            schema_version=schema_version,
            source_ref=source_ref,
            source_digest=sha256_digest(payload),
            records={},
            pre_round_policies=normalized_policies,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchProfileSourceV1":
        payload = _mapping(value, name="profile_source")
        if payload.get("schema") != PROFILE_SOURCE_SCHEMA:
            raise ResearchProfileSourceError("profile source schema drift")
        policies = payload.get("pre_round_policies")
        if policies is not None and payload.get("records") not in (None, {}):
            raise ResearchProfileSourceError(
                "profile source cannot contain both exact records and pre-round policies"
            )
        return cls(
            schema_version=payload.get("schema_version"),
            source_ref=payload.get("source_ref"),
            source_digest=payload.get("source_digest"),
            records=payload.get("records", {}) if policies is None else {},
            pre_round_policies=policies,
        )

    @property
    def identity(self) -> dict[str, str]:
        self._assert_sealed()
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "source_ref": self.source_ref,
            "source_digest": self.source_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        self._assert_sealed()
        source_form = (
            {"records": self.records}
            if self.pre_round_policies is None
            else {"pre_round_policies": self.pre_round_policies}
        )
        return canonical_value(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "source_ref": self.source_ref,
                "source_digest": self.source_digest,
                **source_form,
            }
        )

    def _assert_sealed(self) -> None:
        """Reject mutation of nested record containers after construction."""

        try:
            expected_digest = sha256_digest(
                _source_payload(
                    schema_version=self.schema_version,
                    source_ref=self.source_ref,
                    records=(
                        self.records
                        if self.pre_round_policies is None
                        else None
                    ),
                    pre_round_policies=self.pre_round_policies,
                )
            )
        except (TypeError, ValueError) as error:
            raise ResearchProfileSourceError(
                "profile source bundle is no longer canonical after sealing"
            ) from error
        if expected_digest != self.source_digest:
            raise ResearchProfileSourceError(
                "profile source bundle was mutated after sealing"
            )

    @staticmethod
    def _context_mapping(context: Any) -> dict[str, Any]:
        if isinstance(context, Mapping):
            raw = dict(context)
        else:
            raw = {
                "campaign_id": getattr(context, "campaign_id", None),
                "round_index": getattr(context, "round_index", None),
                "active_profile_ref": getattr(context, "active_profile_ref", None),
                "active_profile_digest": getattr(
                    context,
                    "active_profile_digest",
                    None,
                ),
                "protocol_ref": getattr(context, "protocol_ref", None),
                "protocol_digest": getattr(context, "protocol_digest", None),
                "scientific_memory": getattr(context, "scientific_memory", {}),
            }
        current_context = {
            key: raw[key]
            for key in (
                "campaign_id",
                "round_index",
                "active_profile_ref",
                "active_profile_digest",
                "protocol_ref",
                "protocol_digest",
            )
            if raw.get(key) is not None
        }
        if "round_index" not in current_context:
            current_context["round_index"] = raw.get("round")
        return canonical_value(current_context)

    @staticmethod
    def _scientific_memory(context: Any) -> Mapping[str, Any]:
        if isinstance(context, Mapping):
            memory = context.get("scientific_memory", {})
        else:
            memory = getattr(context, "scientific_memory", {})
        return memory if isinstance(memory, Mapping) else {}

    @classmethod
    def _task_queue(cls, context: Any) -> Mapping[str, Any]:
        memory = cls._scientific_memory(context)
        global_memory = memory.get("global_memory")
        if not isinstance(global_memory, Mapping):
            global_memory = {}
        queue = global_memory.get("task_queue", memory.get("task_queue"))
        if queue is None:
            return {"schema": "recclaw.research-line.task-queue.v2", "tasks": ()}
        to_dict = getattr(queue, "to_dict", None)
        if callable(to_dict):
            queue = to_dict()
        return canonical_value(_mapping(queue, name="task_queue"))

    @staticmethod
    def _history_rows(
        value: Any,
        *,
        field_name: str,
        cutoff: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if isinstance(value, Mapping):
            nested = value.get(field_name, value.get("observations", ()))
            value = nested
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            return ()
        allowed = _HISTORY_KEYS[field_name]
        projected: list[Mapping[str, Any]] = []
        for raw in value:
            if not isinstance(raw, Mapping):
                continue
            round_value = raw.get("round_index", raw.get("round"))
            if isinstance(round_value, bool) or not isinstance(round_value, int):
                continue
            if round_value < 1 or round_value >= cutoff:
                continue
            # The allow-list is intentionally a projection, not a new profile
            # validator.  E remains the authority for field semantics.
            row = {
                str(key): canonical_value(item)
                for key, item in raw.items()
                if str(key).strip().lower() in allowed
            }
            if "round_index" not in row and "round" not in row:
                continue
            projected.append(canonical_value(row))
        return tuple(projected)

    @classmethod
    def _context_history(
        cls,
        context: Any,
        *,
        field_name: str,
        cutoff: int,
    ) -> tuple[Mapping[str, Any], ...]:
        memory = cls._scientific_memory(context)
        global_memory = memory.get("global_memory")
        global_memory = global_memory if isinstance(global_memory, Mapping) else {}
        containers = (
            global_memory,
            memory,
        )
        values: list[Any] = []
        aliases = (
            field_name,
            f"portfolio_{field_name}",
            f"profile_{field_name}",
        )
        for container in containers:
            for alias in aliases:
                if alias in container:
                    values.append(container[alias])
        result: list[Mapping[str, Any]] = []
        for value in values:
            result.extend(
                cls._history_rows(
                    value,
                    field_name=field_name,
                    cutoff=cutoff,
                )
            )
        return tuple(result)

    @staticmethod
    def _binding_map(binding: Any) -> Mapping[str, Any]:
        for method_name in ("canonical_dict", "to_dict"):
            method = getattr(binding, method_name, None)
            if callable(method):
                value = method()
                if isinstance(value, Mapping):
                    value = canonical_value(dict(value))
                    proposal = value.get("proposal")
                    if isinstance(proposal, Mapping) and "producer_role" not in proposal:
                        spec = proposal.get("spec")
                        if isinstance(spec, Mapping) and spec.get("producer_role"):
                            normalized_proposal = dict(proposal)
                            normalized_proposal["producer_role"] = spec[
                                "producer_role"
                            ]
                            value["proposal"] = canonical_value(
                                normalized_proposal
                            )
                    return canonical_value(value)
        return canonical_value(_mapping(binding, name="binding"))

    @staticmethod
    def _record_resolution(
        binding: Any,
        resolutions: Sequence[tuple[Any, Any]],
    ) -> Any:
        candidate_id = getattr(getattr(binding, "proposal", None), "candidate_id", None)
        proposal = getattr(binding, "proposal", None)
        proposal_spec = getattr(proposal, "spec", None)
        proposal_spec_digest = getattr(proposal_spec, "digest", None)
        identity_matches: list[Any] = []
        capability_matches: list[Any] = []
        for outcome, resolution in resolutions:
            if resolution is None:
                continue
            source = getattr(outcome, "source_proposal", None)
            if getattr(source, "candidate_id", None) == candidate_id:
                identity_matches.append(resolution)
                continue
            outcome_spec = getattr(outcome, "spec", None)
            if (
                proposal_spec_digest is not None
                and getattr(outcome_spec, "digest", None) == proposal_spec_digest
            ):
                identity_matches.append(resolution)
                continue
            if getattr(resolution, "resolved_current_capability_ref", None) == getattr(
                binding,
                "capability_ref",
                None,
            ):
                capability_matches.append(resolution)
        matches = identity_matches or capability_matches
        if len(matches) != 1:
            raise ResearchProfileSourceError(
                f"profile source requires exactly one resolution for {candidate_id}"
            )
        return matches[0]

    @staticmethod
    def _resolution_map(resolution: Any) -> Mapping[str, Any]:
        for method_name in ("to_dict", "canonical_dict"):
            method = getattr(resolution, method_name, None)
            if callable(method):
                value = method()
                if isinstance(value, Mapping):
                    return canonical_value(dict(value))
        return canonical_value(_mapping(resolution, name="resolution"))

    @staticmethod
    def _candidate_identity(
        record: Mapping[str, Any],
        *,
        binding: Any,
    ) -> dict[str, Any]:
        identity = _mapping(record.get("candidate_identity", {}), name="candidate_identity")
        proposal = getattr(binding, "proposal", None)
        defaults = {
            "candidate_id": getattr(proposal, "candidate_id", None),
            "semantic_digest": getattr(binding, "mechanism_semantics_digest", None),
            "family_id": getattr(proposal, "mechanism_axis", None),
            "parent_id": getattr(proposal, "parent_candidate_id", None),
            "resource_candidate_ref": getattr(binding, "capability_ref", None),
        }
        for key, value in defaults.items():
            if value is not None:
                identity.setdefault(key, value)
        if "candidate_package_digest" not in identity:
            package_digest = getattr(proposal, "candidate_package_digest", None)
            if package_digest is not None:
                identity["candidate_package_digest"] = package_digest
        if "candidate_source_sha256" not in identity:
            source_digest = getattr(proposal, "source_tree_digest", None)
            if source_digest is not None:
                identity["candidate_source_sha256"] = source_digest
        return canonical_value(identity)

    @staticmethod
    def _context_prior(
        record: Mapping[str, Any],
        *,
        context: Any,
        current_round: int,
        field_name: str,
    ) -> tuple[Mapping[str, Any], ...]:
        explicit = record.get(field_name, ())
        if isinstance(explicit, (str, bytes)) or not isinstance(explicit, Sequence):
            explicit = ()
        return tuple(explicit) + ResearchProfileSourceV1._context_history(
            context,
            field_name=field_name,
            cutoff=current_round,
        )

    @staticmethod
    def _binding_role(binding: Any) -> str:
        proposal = getattr(binding, "proposal", None)
        role = getattr(proposal, "producer_role", None)
        if role is None:
            role = getattr(getattr(proposal, "spec", None), "producer_role", None)
        return _text(role, name="binding.proposal.producer_role")

    @staticmethod
    def _binding_family(binding: Any) -> str:
        proposal = getattr(binding, "proposal", None)
        family = getattr(proposal, "mechanism_axis", None)
        if family is None:
            family = getattr(getattr(proposal, "spec", None), "mechanism_axis", None)
        return _text(family, name="binding.proposal.mechanism_axis")

    @staticmethod
    def _candidate_parent(binding: Any) -> str | None:
        proposal = getattr(binding, "proposal", None)
        parent = getattr(proposal, "parent_candidate_id", None)
        if parent is None:
            parent = getattr(getattr(proposal, "spec", None), "parent_candidate_id", None)
        if parent is None:
            return None
        return _text(parent, name="binding.proposal.parent_candidate_id")

    @staticmethod
    def _actual_candidate_digests(
        binding: Any,
        qualified_execution: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        proposal = getattr(binding, "proposal", None)
        proposal_package_digest = getattr(
            proposal,
            "candidate_package_digest",
            None,
        )
        proposal_source_digest = getattr(
            proposal,
            "candidate_source_sha256",
            None,
        )
        if proposal_source_digest is None:
            proposal_source_digest = getattr(proposal, "source_tree_digest", None)
        execution = qualified_execution or {}
        execution_package_digest = None
        execution_source_digest = None
        if isinstance(execution, Mapping):
            execution_package_digest = execution.get("candidate_package_digest")
            execution_source_digest = execution.get(
                "candidate_source_sha256",
                execution.get(
                    "candidate_source_tree_digest",
                    execution.get("source_tree_digest"),
                )
            )
        for field_name, proposal_value, execution_value in (
            (
                "candidate_package_digest",
                proposal_package_digest,
                execution_package_digest,
            ),
            (
                "candidate_source_sha256",
                proposal_source_digest,
                execution_source_digest,
            ),
        ):
            if proposal_value is not None and execution_value is not None:
                try:
                    proposal_digest = validate_sha256(
                        proposal_value,
                        field_name=field_name,
                    )
                    execution_digest = validate_sha256(
                        execution_value,
                        field_name=f"qualified_execution.{field_name}",
                    )
                except (TypeError, ValueError) as error:
                    raise ResearchProfileSourceError(
                        f"actual {field_name} is not a SHA-256 digest"
                    ) from error
                if proposal_digest != execution_digest:
                    raise ResearchProfileSourceError(
                        f"qualified execution {field_name} drift"
                    )
        package_digest = proposal_package_digest or execution_package_digest
        source_digest = proposal_source_digest or execution_source_digest
        for field_name, value in (
            ("candidate_package_digest", package_digest),
            ("candidate_source_sha256", source_digest),
        ):
            if value is not None:
                try:
                    value = validate_sha256(value, field_name=field_name)
                except (TypeError, ValueError) as error:
                    raise ResearchProfileSourceError(
                        f"actual {field_name} is not a SHA-256 digest"
                    ) from error
            if field_name == "candidate_package_digest":
                package_digest = value
            else:
                source_digest = value
        return package_digest, source_digest

    @classmethod
    def _rebound_resource_profile(
        cls,
        resource_profile: Mapping[str, Any],
        *,
        binding: Any,
        qualified_execution: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        resource = canonical_value(dict(resource_profile))
        capability_ref = getattr(binding, "capability_ref", None)
        if resource.get("candidate_ref") != capability_ref:
            raise ResearchProfileSourceError(
                "resource profile capability identity drift"
            )
        actual_package, actual_source = cls._actual_candidate_digests(
            binding,
            qualified_execution,
        )
        for field_name, actual in (
            ("candidate_package_digest", actual_package),
            ("candidate_source_sha256", actual_source),
        ):
            supplied = resource.get(field_name)
            if supplied is not None and actual is not None and supplied != actual:
                raise ResearchProfileSourceError(
                    f"resource profile {field_name} drift"
                )
            if supplied is None and actual is not None:
                resource[field_name] = actual
        prediction = _mapping(resource.get("prediction"), name="resource_profile.prediction")
        prediction_identity = _mapping(
            prediction.get("identity"),
            name="resource_profile.prediction.identity",
        )
        actual_identity = {
            "candidate_id": getattr(getattr(binding, "proposal", None), "candidate_id", None),
            "candidate_ref": capability_ref,
            "semantic_digest": getattr(binding, "mechanism_semantics_digest", None),
        }
        if any(
            not isinstance(value, str) or not value
            for value in actual_identity.values()
        ):
            raise ResearchProfileSourceError(
                "binding lacks the identity required to rebind resource evidence"
            )
        if actual_package is not None:
            actual_identity["candidate_package_digest"] = actual_package
        elif resource.get("candidate_package_digest") is not None:
            actual_identity["candidate_package_digest"] = resource[
                "candidate_package_digest"
            ]
        actual_identity["candidate_source_sha256"] = resource[
            "candidate_source_sha256"
        ]
        for field_name, actual in actual_identity.items():
            supplied = prediction_identity.get(field_name)
            if supplied is not None and supplied != actual:
                raise ResearchProfileSourceError(
                    f"resource prediction identity {field_name} drift"
                )
            prediction_identity[field_name] = actual
        prediction["identity"] = canonical_value(prediction_identity)
        resource["prediction"] = prediction
        return canonical_value(resource)

    def _materialize_pre_round_records(
        self,
        *,
        context: Any,
        resolutions: Sequence[tuple[Any, Any]],
        bindings: Sequence[Any],
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]],
        candidate_root_by_capability: Mapping[str, str | Path],
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        policies = self.pre_round_policies
        if not policies:
            raise ResearchProfileSourceError("pre-round policy source is empty")
        qualified_map = dict(qualified_execution_by_capability)
        root_map = dict(candidate_root_by_capability)
        resource_map = dict(resource_profile_by_capability)
        materialized: dict[str, Mapping[str, Any]] = {}
        for binding in bindings:
            role = self._binding_role(binding)
            family = self._binding_family(binding)
            capability_ref = _text(
                getattr(binding, "capability_ref", None),
                name="binding.capability_ref",
            )
            capability_digest = validate_sha256(
                getattr(binding, "capability_digest", None),
                field_name="binding.capability_digest",
            )
            matches = [
                (key, policy)
                for key, policy in policies.items()
                if policy["producer_role"] == role
                and (
                    policy.get("capability_ref") is None
                    or policy["capability_ref"] == capability_ref
                )
                and (
                    policy.get("capability_digest") is None
                    or policy["capability_digest"] == capability_digest
                )
                and (
                    policy.get("family_id") is None
                    or policy["family_id"] == family
                )
            ]
            if len(matches) != 1:
                raise ResearchProfileSourceError(
                    "pre-round policy coverage is missing or ambiguous for "
                    f"role={role}, capability_ref={capability_ref}, family={family}"
                )
            _policy_key_value, policy = matches[0]
            proposal = getattr(binding, "proposal", None)
            candidate_id = getattr(proposal, "candidate_id", None)
            candidate_id = _text(candidate_id, name="binding.proposal.candidate_id")
            if candidate_id in materialized:
                raise ResearchProfileSourceError(
                    "pre-round materialization contains duplicate candidate identity"
                )
            parent_id = self._candidate_parent(binding)
            expected_parent = policy.get("parent_id")
            if "parent_id" in policy and expected_parent != parent_id:
                raise ResearchProfileSourceError(
                    "pre-round policy parent identity drift"
                )
            resolution = self._record_resolution(binding, resolutions)
            resolution_ref = getattr(
                resolution,
                "resolved_current_capability_ref",
                None,
            )
            if resolution_ref is not None and resolution_ref != capability_ref:
                raise ResearchProfileSourceError(
                    "resolution capability identity drift"
                )
            origin = getattr(getattr(binding, "entry_origin", None), "value", None)
            policy_resource = policy.get("resource_profile")
            if origin == "FIXED_66":
                qualified_execution = None
                candidate_root_path = None
                if capability_ref in qualified_map or capability_ref in root_map:
                    raise ResearchProfileSourceError(
                        "FIXED_66 cannot consume qualified execution/root evidence"
                    )
                durable_resource = resource_map.get(capability_ref)
            elif origin == "QUALIFIED_REGISTRY":
                if capability_ref not in qualified_map:
                    raise ResearchProfileSourceError(
                        "qualified binding lacks durable capability-keyed execution evidence"
                    )
                if capability_ref not in root_map:
                    raise ResearchProfileSourceError(
                        "qualified binding lacks durable capability-keyed root evidence"
                    )
                if capability_ref not in resource_map:
                    raise ResearchProfileSourceError(
                        "qualified binding lacks durable capability-keyed resource evidence"
                    )
                qualified_execution = qualified_map[capability_ref]
                if not isinstance(qualified_execution, Mapping) or not qualified_execution:
                    raise ResearchProfileSourceError(
                        "durable qualified execution evidence must be non-empty"
                    )
                candidate_root_path = root_map[capability_ref]
                if not isinstance(candidate_root_path, (str, Path)) or not str(
                    candidate_root_path
                ).strip():
                    raise ResearchProfileSourceError(
                        "durable candidate root evidence must be a non-empty path"
                    )
                candidate_root_path = str(Path(candidate_root_path).resolve())
                durable_resource = resource_map[capability_ref]
                if policy_resource is not None:
                    raise ResearchProfileSourceError(
                        "qualified pre-round policy cannot embed resource evidence"
                    )
            else:
                raise ResearchProfileSourceError("unknown binding entry origin")
            rebound_resources: list[dict[str, Any]] = []
            if policy_resource is not None:
                rebound_resources.append(
                    self._rebound_resource_profile(
                        policy_resource,
                        binding=binding,
                        qualified_execution=qualified_execution,
                    )
                )
            if durable_resource is not None:
                durable_selector = {
                    "capability_ref": capability_ref,
                }
                normalized_durable_resource = _policy_resource_profile(
                    durable_resource,
                    selector=durable_selector,
                    name="durable resource profile",
                    allow_candidate_id=True,
                )
                rebound_resources.append(
                    self._rebound_resource_profile(
                        normalized_durable_resource,
                        binding=binding,
                        qualified_execution=qualified_execution,
                    )
                )
            if not rebound_resources:
                raise ResearchProfileSourceError(
                    "formal profile source lacks an explicit resource evidence record"
                )
            if len(rebound_resources) > 1 and any(
                item != rebound_resources[0] for item in rebound_resources[1:]
            ):
                raise ResearchProfileSourceError(
                    "policy and durable resource evidence identity drift"
                )
            resource_profile = rebound_resources[0]
            mandate = policy.get("exploration_mandate")
            if mandate is not None:
                mandate = canonical_value(dict(mandate))
                if mandate.get("producer_role") != role:
                    raise ResearchProfileSourceError(
                        "pre-round exploration mandate producer_role drift"
                    )
            package_digest, source_digest = self._actual_candidate_digests(
                binding,
                qualified_execution,
            )
            if package_digest is None:
                package_digest = resource_profile.get("candidate_package_digest")
            source_digest = source_digest or resource_profile.get(
                "candidate_source_sha256"
            )
            identity = {
                "candidate_id": candidate_id,
                "semantic_digest": getattr(binding, "mechanism_semantics_digest", None),
                "family_id": family,
                "parent_id": parent_id,
                "resource_candidate_ref": capability_ref,
                "binding_digest": sha256_digest(self._binding_map(binding)),
                "resolution_digest": sha256_digest(self._resolution_map(resolution)),
            }
            if package_digest is not None:
                identity["candidate_package_digest"] = package_digest
            if source_digest is not None:
                identity["candidate_source_sha256"] = source_digest
            resource_compute_pattern = _resource_compute_pattern(
                resource_profile,
                name="rebound resource profile",
            )
            policy_compute_pattern = policy.get("compute_pattern")
            if (
                policy_compute_pattern is not None
                and policy_compute_pattern != resource_compute_pattern
            ):
                raise ResearchProfileSourceError(
                    "pre-round policy compute_pattern does not match resource evidence"
                )
            identity["compute_pattern"] = resource_compute_pattern
            record = {
                "candidate_identity": identity,
                "calibration_prior": policy["calibration_prior"],
                "exploration_mandate": mandate,
                "qualified_execution": qualified_execution,
                "candidate_root_path": candidate_root_path,
                "resource_profile": resource_profile,
                "durable_evidence_digests": _durable_evidence_digests(
                    qualified_execution=qualified_execution,
                    candidate_root_path=(
                        str(candidate_root_path)
                        if candidate_root_path is not None
                        else None
                    ),
                    resource_profile=resource_profile,
                ),
            }
            for field_name in (
                "prior_attempts",
                "family_history",
                "parent_history",
                "frontier_history",
            ):
                if field_name in policy:
                    record[field_name] = policy[field_name]
            materialized[candidate_id] = canonical_value(record)
        if len(materialized) != len(bindings):
            raise ResearchProfileSourceError(
                "pre-round policy materialization did not cover every binding"
            )
        return materialized

    def build_profiles(
        self,
        *,
        context: Any,
        active_profile: Any,
        resolutions: Sequence[tuple[Any, Any]],
        search_bindings: Sequence[Any],
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
        candidate_root_by_capability: Mapping[str, str | Path] | None = None,
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[ResearchProfileRecordV1, ...]:
        """Build one complete profile per frozen binding after resolution."""

        del active_profile  # Profile identity is carried by the binding/context.
        self._assert_sealed()
        bindings = tuple(search_bindings)
        expected_ids = tuple(
            getattr(getattr(binding, "proposal", None), "candidate_id", None)
            for binding in bindings
        )
        if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in expected_ids):
            raise ResearchProfileSourceError("search binding lacks candidate identity")
        if self.pre_round_policies is None:
            records = self.records
            if set(records) != set(expected_ids):
                missing = sorted(set(expected_ids) - set(records))
                extra = sorted(set(records) - set(expected_ids))
                detail = []
                if missing:
                    detail.append("missing=" + ",".join(missing))
                if extra:
                    detail.append("extra=" + ",".join(extra))
                raise ResearchProfileSourceError(
                    "profile source coverage is not exact (" + "; ".join(detail) + ")"
                )
        else:
            records = self._materialize_pre_round_records(
                context=context,
                resolutions=resolutions,
                bindings=bindings,
                qualified_execution_by_capability=(
                    qualified_execution_by_capability or {}
                ),
                candidate_root_by_capability=(candidate_root_by_capability or {}),
                resource_profile_by_capability=(
                    resource_profile_by_capability or {}
                ),
            )
        context_map = self._context_mapping(context)
        current_round = context_map.get("round_index")
        if isinstance(current_round, bool) or not isinstance(current_round, int):
            raise ResearchProfileSourceError("context round_index is required")
        queue = self._task_queue(context)
        built: list[ResearchProfileRecordV1] = []
        for binding in bindings:
            candidate_id = getattr(binding.proposal, "candidate_id", None)
            record = records[candidate_id]
            resolution = self._record_resolution(binding, resolutions)
            binding_map = self._binding_map(binding)
            resolution_map = self._resolution_map(resolution)
            identity = self._candidate_identity(record, binding=binding)
            identity.setdefault("binding_digest", sha256_digest(binding_map))
            identity.setdefault("resolution_digest", sha256_digest(resolution_map))
            calibration_prior = record.get("calibration_prior")
            if calibration_prior is None:
                memory = self._scientific_memory(context)
                global_memory = memory.get("global_memory")
                global_memory = global_memory if isinstance(global_memory, Mapping) else {}
                calibration_prior = global_memory.get("calibration_prior")
            if not isinstance(calibration_prior, Mapping):
                raise ResearchProfileSourceError(
                    f"profile source lacks explicit calibration_prior for {candidate_id}"
                )
            mandate = record.get("task_or_exploration_mandate")
            if mandate is None:
                mandate = record.get("exploration_mandate")
            if mandate is None and self.pre_round_policies is None:
                memory = self._scientific_memory(context)
                global_memory = memory.get("global_memory")
                global_memory = global_memory if isinstance(global_memory, Mapping) else {}
                mandates = global_memory.get("exploration_mandates", {})
                if isinstance(mandates, Mapping):
                    mandate = mandates.get(candidate_id)
            resource_profile = record.get("resource_profile")
            if not isinstance(resource_profile, Mapping):
                raise ResearchProfileSourceError(
                    f"profile source lacks resource_profile for {candidate_id}"
                )
            try:
                profile = build_portfolio_candidate_profile_v2(
                    current_context=context_map,
                    candidate_identity=identity,
                    binding=binding_map,
                    resolution=resolution_map,
                    task_queue=queue,
                    exploration_mandate=mandate,
                    prior_attempts=self._context_prior(
                        record,
                        context=context,
                        current_round=current_round,
                        field_name="prior_attempts",
                    ),
                    calibration_prior=calibration_prior,
                    family_history=self._context_prior(
                        record,
                        context=context,
                        current_round=current_round,
                        field_name="family_history",
                    ),
                    parent_history=self._context_prior(
                        record,
                        context=context,
                        current_round=current_round,
                        field_name="parent_history",
                    ),
                    frontier_history=self._context_prior(
                        record,
                        context=context,
                        current_round=current_round,
                        field_name="frontier_history",
                    ),
                    resource_profile=resource_profile,
                    temporal_cutoff_round=current_round,
                )
            except (TypeError, ValueError) as error:
                raise ResearchProfileSourceError(
                    f"E profile builder rejected {candidate_id}: {error}"
                ) from error
            built.append(
                ResearchProfileRecordV1(
                    portfolio_profile=profile,
                    qualified_execution=record.get("qualified_execution"),
                    candidate_root_path=record.get("candidate_root_path"),
                    resource_profile=resource_profile,
                    calibration_prior=calibration_value(calibration_prior),
                    task_or_exploration_mandate=mandate,
                    durable_evidence_digests=record.get(
                        "durable_evidence_digests"
                    ),
                )
            )
        return tuple(built)


def calibration_value(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Canonicalize an explicit calibration prior without adding defaults."""

    return canonical_value(_mapping(value, name="calibration_prior"))


__all__ = [
    "PROFILE_SOURCE_SCHEMA",
    "PROFILE_SOURCE_SCHEMA_VERSION",
    "ResearchProfileRecordV1",
    "ResearchProfileSourceError",
    "ResearchProfileSourceV1",
    "calibration_value",
    "lineage_identity_digest",
]
