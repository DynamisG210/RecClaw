"""Prospective, outcome-blind comparison of the accepted Research Line policies.

This module is deliberately small.  It projects one byte-identical frozen
OpenSpec pool into the already-defined static, accepted F1, and accepted Q3
selection semantics.  It also freezes the arm manifests and computes the four
predeclared DEVELOPMENT_ONLY metrics without turning missingness into an
effect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import (
    bytes_sha256,
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from .open_meta import (
    AcquisitionDispositionV1,
    IdeaBudgetV1,
    IdeaCandidateIdentityV1,
    IdeaPolicyInputV1,
    STATIC_IDEA_POLICY_DIGEST_V1,
    STATIC_IDEA_POLICY_REF_V1,
    run_static_idea_policy,
)
from .open_meta_q3 import build_q3_acquisition_manifest
from .vnext_contracts import RealizationClassV1, RealizationTypingV1


POLICY_ORDER = ("STATIC", "CURRENT_F1", "OUTCOME_AWARE")
MECHANISM_STATES = frozenset(
    {
        "NOT_ASSESSED",
        "INACTIVE",
        "ACTIVE_SUPPORTED",
        "ACTIVE_CONTRADICTED",
        "NON_IDENTIFIABLE",
    }
)
MECHANISM_IDENTIFIABLE_STATES = frozenset(
    {"INACTIVE", "ACTIVE_SUPPORTED", "ACTIVE_CONTRADICTED"}
)


class ProspectivePolicyComparisonError(RuntimeError):
    """A frozen comparison identity, authority, or denominator drifted."""


REALIZATION_CONTRACT_SCHEMA = (
    "recclaw.research-line.vnext.shared-realization-contract.v1"
)
SHARED_REALIZATION_POOL_SCHEMA = (
    "recclaw.research-line.q5-shared-realization-pool.v1"
)
SHARED_OUTCOME_LEDGER_SCHEMA = (
    "recclaw.research-line.q5-shared-package-seed-outcome-ledger.v1"
)
REALIZATION_CLASSES = frozenset(item.value for item in RealizationClassV1)
REALIZATION_TYPINGS = frozenset(item.value for item in RealizationTypingV1)
REALIZATION_EQUIVALENCE_TOLERANCE = 1e-8


def _require_identity_fields(
    value: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    for field in fields:
        observed = value.get(field)
        if field.endswith("_digest"):
            try:
                validate_sha256(observed, field_name=f"{label}.{field}")
            except Exception as error:
                raise ProspectivePolicyComparisonError(
                    f"{label}.{field} is not a SHA-256 digest"
                ) from error
        elif not isinstance(observed, str) or not observed or observed != observed.strip():
            raise ProspectivePolicyComparisonError(
                f"{label}.{field} must be a normalized non-empty reference"
            )


def build_shared_realization_contract(
    *,
    research_spec_ref: str,
    research_spec_digest: str,
    candidate_package_ref: str,
    candidate_package_digest: str,
    source_tree_ref: str,
    source_tree_digest: str,
    equivalence_ref: str,
    equivalence_digest: str,
    protocol_ref: str,
    protocol_digest: str,
    realization_class: RealizationClassV1 | str,
) -> dict[str, Any]:
    """Bind one executable realization to the exact pre-outcome identity."""

    try:
        realization_class = RealizationClassV1(realization_class).value
    except ValueError as error:
        raise ProspectivePolicyComparisonError(
            f"unsupported realization_class: {realization_class}"
        ) from error
    payload = canonical_value(
        {
            "schema": REALIZATION_CONTRACT_SCHEMA,
            "research_spec_ref": research_spec_ref,
            "research_spec_digest": research_spec_digest,
            "candidate_package_ref": candidate_package_ref,
            "candidate_package_digest": candidate_package_digest,
            "source_tree_ref": source_tree_ref,
            "source_tree_digest": source_tree_digest,
            "equivalence_ref": equivalence_ref,
            "equivalence_digest": equivalence_digest,
            "protocol_ref": protocol_ref,
            "protocol_digest": protocol_digest,
            "realization_class": realization_class,
        }
    )
    _require_identity_fields(
        payload,
        (
            "research_spec_ref",
            "research_spec_digest",
            "candidate_package_ref",
            "candidate_package_digest",
            "source_tree_ref",
            "source_tree_digest",
            "equivalence_ref",
            "equivalence_digest",
            "protocol_ref",
            "protocol_digest",
        ),
        label="realization",
    )
    digest = sha256_digest(payload)
    return canonical_value(
        {
            **payload,
            "realization_ref": f"recclaw-realization-contract-v1:{digest}",
            "realization_digest": digest,
        }
    )


def _validate_realization_contract(
    contract: Mapping[str, Any], *, expected_protocol_digest: str | None = None
) -> dict[str, Any]:
    if contract.get("schema") != REALIZATION_CONTRACT_SCHEMA:
        raise ProspectivePolicyComparisonError("realization contract schema drift")
    required = (
        "research_spec_ref",
        "research_spec_digest",
        "candidate_package_ref",
        "candidate_package_digest",
        "source_tree_ref",
        "source_tree_digest",
        "equivalence_ref",
        "equivalence_digest",
        "protocol_ref",
        "protocol_digest",
        "realization_class",
        "realization_ref",
        "realization_digest",
    )
    missing = [field for field in required if field not in contract]
    if missing:
        raise ProspectivePolicyComparisonError(
            "realization contract missing fields: " + ", ".join(missing)
        )
    payload = {
        key: contract[key]
        for key in (
            "research_spec_ref",
            "research_spec_digest",
            "candidate_package_ref",
            "candidate_package_digest",
            "source_tree_ref",
            "source_tree_digest",
            "equivalence_ref",
            "equivalence_digest",
            "protocol_ref",
            "protocol_digest",
            "realization_class",
        )
    }
    normalized = build_shared_realization_contract(**payload)
    if (
        contract.get("realization_digest") != normalized["realization_digest"]
        or contract.get("realization_ref") != normalized["realization_ref"]
    ):
        raise ProspectivePolicyComparisonError(
            "realization contract digest or reference does not match its bindings"
        )
    if expected_protocol_digest is not None and contract["protocol_digest"] != expected_protocol_digest:
        raise ProspectivePolicyComparisonError("realization protocol digest drift")
    return normalized


def bind_observation_to_realization(
    observation: Mapping[str, Any],
    realization: Mapping[str, Any],
    *,
    observation_kind: str,
) -> dict[str, Any]:
    """Reject receipts/outcomes that do not name the current realization."""

    contract = _validate_realization_contract(realization)
    required_bindings = {
        "realization_digest": contract["realization_digest"],
        "candidate_package_digest": contract["candidate_package_digest"],
        "source_tree_digest": contract["source_tree_digest"],
        "research_spec_digest": contract["research_spec_digest"],
        "protocol_digest": contract["protocol_digest"],
    }
    for field, expected in required_bindings.items():
        if observation.get(field) != expected:
            raise ProspectivePolicyComparisonError(
                f"{observation_kind} is not bound to the current realization: {field}"
            )
    return canonical_value(
        {
            "observation_kind": observation_kind,
            "realization_ref": contract["realization_ref"],
            "realization_digest": contract["realization_digest"],
            "candidate_package_digest": contract["candidate_package_digest"],
            "source_tree_digest": contract["source_tree_digest"],
            "research_spec_digest": contract["research_spec_digest"],
            "protocol_digest": contract["protocol_digest"],
        }
    )


def _selected_candidate_ids(selection: Mapping[str, Any]) -> tuple[str, ...]:
    records = selection.get("candidates")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        selected = tuple(
            str(row["candidate_id"])
            for row in records
            if isinstance(row, Mapping) and row.get("selected") is True
        )
        if selected:
            return tuple(sorted(set(selected)))
    raw = selection.get("selected_candidate_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(sorted({str(value) for value in raw}))
    if selection.get("selected_candidate_id") is not None:
        return (str(selection["selected_candidate_id"]),)
    raise ProspectivePolicyComparisonError("policy selection has no selected candidates")


def build_shared_realization_pool(
    *,
    policy_selections: Mapping[str, Mapping[str, Any]],
    realizations: Sequence[Mapping[str, Any]],
    protocol_ref: str,
    protocol_digest: str,
) -> dict[str, Any]:
    """Deduplicate implementation/package work while retaining policy attribution."""

    validate_sha256(protocol_digest, field_name="protocol_digest")
    attribution_ids = {
        str(policy): _selected_candidate_ids(selection)
        for policy, selection in policy_selections.items()
    }
    selected_ids = {item for values in attribution_ids.values() for item in values}
    by_spec: dict[str, dict[str, Any]] = {}
    for raw in realizations:
        contract = raw.get("contract", raw)
        normalized = _validate_realization_contract(
            contract, expected_protocol_digest=protocol_digest
        )
        spec_digest = normalized["research_spec_digest"]
        if spec_digest not in selected_ids:
            raise ProspectivePolicyComparisonError(
                "shared realization pool contains an unselected OpenSpec"
            )
        previous = by_spec.get(spec_digest)
        if previous is not None and previous["realization_digest"] != normalized["realization_digest"]:
            raise ProspectivePolicyComparisonError(
                "one OpenSpec has more than one physical realization"
            )
        by_spec[spec_digest] = normalized
    missing = selected_ids - set(by_spec)
    if missing:
        raise ProspectivePolicyComparisonError(
            "selected OpenSpec has no shared realization: " + ", ".join(sorted(missing))
        )
    realization_attribution = {
        policy: tuple(
            sorted(by_spec[spec_digest]["realization_digest"] for spec_digest in ids)
        )
        for policy, ids in attribution_ids.items()
    }
    payload = canonical_value(
        {
            "schema": SHARED_REALIZATION_POOL_SCHEMA,
            "protocol_ref": protocol_ref,
            "protocol_digest": protocol_digest,
            "selected_open_spec_digests": tuple(sorted(selected_ids)),
            "realization_count": len(by_spec),
            "implementations_per_open_spec": 1,
            "realizations": tuple(
                by_spec[key] for key in sorted(by_spec)
            ),
            "policy_attribution": realization_attribution,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "pool_digest": sha256_digest(payload)}


def build_shared_package_seed_outcome_ledger(
    *,
    realization_pool: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Emit one outcome per package×seed and keep policy attribution separate."""

    contracts = {
        value["realization_digest"]: _validate_realization_contract(value)
        for value in realization_pool["realizations"]
    }
    by_package = {
        value["candidate_package_digest"]: value
        for value in contracts.values()
    }
    ledger: dict[str, dict[str, Any]] = {}
    for raw in outcomes:
        if any(key in raw for key in ("policy", "policy_name", "arm")):
            raise ProspectivePolicyComparisonError(
                "outcome ledger must not duplicate policy-specific physical outcomes"
            )
        package_digest = raw.get("candidate_package_digest")
        contract = by_package.get(package_digest)
        if contract is None:
            raise ProspectivePolicyComparisonError(
                "outcome names a package outside the shared realization pool"
            )
        bind_observation_to_realization(
            raw, contract, observation_kind="package_seed_outcome"
        )
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ProspectivePolicyComparisonError("package×seed outcome seed is invalid")
        key = f"{package_digest}:{seed}"
        if key in ledger:
            raise ProspectivePolicyComparisonError(
                "package×seed outcome was emitted more than once"
            )
        ledger[key] = canonical_value({**raw, "outcome_key": key})
    policy_attribution: dict[str, tuple[str, ...]] = {}
    for policy, realization_digests in realization_pool["policy_attribution"].items():
        keys = []
        for realization_digest in realization_digests:
            contract = contracts[realization_digest]
            keys.extend(
                key
                for key, value in ledger.items()
                if value["candidate_package_digest"]
                == contract["candidate_package_digest"]
            )
        policy_attribution[str(policy)] = tuple(sorted(set(keys)))
    payload = canonical_value(
        {
            "schema": SHARED_OUTCOME_LEDGER_SCHEMA,
            "realization_pool_digest": realization_pool.get("pool_digest"),
            "outcome_count": len(ledger),
            "package_seed_uniqueness": "ONE_OUTCOME_PER_PACKAGE_PER_SEED",
            "outcomes": tuple(ledger[key] for key in sorted(ledger)),
            "policy_attribution": policy_attribution,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "ledger_digest": sha256_digest(payload)}


def _probe_status_pass(value: Any) -> bool:
    if isinstance(value, Mapping):
        return value.get("status") == "PASS" or value.get("equivalent") is True
    return value is True or value == "PASS"


def _probe_delta(behavior: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if behavior.get(name) is not None:
            try:
                return float(behavior[name])
            except (TypeError, ValueError):
                return None
    return None


def _protocol_equivalence_pass(
    behavior: Mapping[str, Any], realization: Mapping[str, Any]
) -> bool:
    evidence = (
        behavior.get("mechanism_off_protocol_equivalence")
        or behavior.get("protocol_equivalence")
        or realization.get("protocol_equivalence")
    )
    if isinstance(evidence, Mapping):
        required = ("dataset", "evaluator", "seed", "optimizer", "batch")
        return all(_probe_status_pass(evidence.get(name)) for name in required)
    return _probe_status_pass(evidence)


def classify_realization_authority(
    *,
    realization: Mapping[str, Any],
    qualification: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Route only parent-equivalent nested realizations to mechanism authority."""

    realization_class = str(realization.get("realization_class", ""))
    if realization_class not in REALIZATION_CLASSES:
        raise ProspectivePolicyComparisonError("realization_class is not frozen")
    declared_typing = str(
        realization.get("realization_typing")
        or realization.get("realization_type")
        or (
            RealizationTypingV1.EFFECT_ONLY_NON_NESTED.value
            if realization.get("realization_mode") == "NON_NESTED"
            or realization_class == RealizationClassV1.NEW_CANDIDATE.value
            else RealizationTypingV1.NESTED_MECHANISM.value
        )
    )
    if declared_typing not in REALIZATION_TYPINGS:
        raise ProspectivePolicyComparisonError("realization_typing is not frozen")
    behavior = dict(qualification.get("behavioral_evidence") or {})
    admitted = admission.get("status") == "RESOURCE_ADMITTED"
    qualified = qualification.get("status") in {"QUALIFICATION_PASS", "PASS"}
    active = (
        behavior.get("probe_status") == "PASS_STRUCTURAL_BEHAVIOR_ACTIVE"
        and bool(behavior.get("overridden_behavioral_methods"))
        and max(
            _probe_delta(behavior, ("behavioral_loss_max_abs_delta",)) or 0.0,
            _probe_delta(behavior, ("behavioral_score_max_abs_delta",)) or 0.0,
        )
        > 0.0
    )
    reason = ""
    mechanism_state = "NOT_ASSESSED"
    mechanism_allowed = False
    if declared_typing == RealizationTypingV1.EFFECT_ONLY_NON_NESTED.value:
        reason = "NON_NESTED_REALIZATION_EFFECT_ONLY"
    elif not admitted:
        reason = "RESOURCE_FAILURE_HAS_NO_MECHANISM_AUTHORITY"
    elif not qualified:
        reason = "QUALIFICATION_FAILURE_HAS_NO_MECHANISM_AUTHORITY"
    else:
        parent_pass = _probe_status_pass(
            behavior.get("mechanism_off_parent_equivalence")
            or behavior.get("parent_equivalence")
            or realization.get("parent_equivalence")
        )
        deltas = {
            name: _probe_delta(behavior, aliases)
            for name, aliases in {
                "loss": ("mechanism_off_loss_max_abs_delta", "mechanism_off_loss_abs_delta"),
                "predict": ("mechanism_off_predict_max_abs_delta", "mechanism_off_predict_abs_delta"),
                "full_sort": ("mechanism_off_full_sort_max_abs_delta", "mechanism_off_full_sort_abs_delta"),
                "gradients": ("mechanism_off_gradients_max_abs_delta", "mechanism_off_gradients_abs_delta"),
                "checkpoint_load": ("mechanism_off_checkpoint_load_max_abs_delta", "mechanism_off_checkpoint_load_abs_delta"),
            }.items()
        }
        equivalence_pass = (
            behavior.get("mechanism_off_execution") == "PASS"
            and parent_pass
            and all(value is not None and value <= REALIZATION_EQUIVALENCE_TOLERANCE for value in deltas.values())
            and _protocol_equivalence_pass(behavior, realization)
        )
        if equivalence_pass:
            mechanism_allowed = True
            mechanism_state = "ACTIVE_SUPPORTED" if active else "INACTIVE"
            reason = "NESTED_MECHANISM_PARENT_AND_PROTOCOL_EQUIVALENCE_CONFIRMED"
        else:
            reason = "NESTED_MECHANISM_GATE_NOT_SATISFIED"
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q5-realization-typing-authority.v1",
            "realization_class": realization_class,
            "realization_typing": declared_typing,
            "mechanism_state": mechanism_state,
            "mechanism_information_authority": (
                "MECHANISM_INFORMATION" if mechanism_allowed else "NOT_ASSESSED"
            ),
            "effect_authority": "NOT_ASSESSED",
            "effect_input_allowed": False,
            "effect_eligibility": "ELIGIBLE_ONLY_AFTER_FULL_MATCHED_EPISODE",
            "mechanism_information_input_allowed": mechanism_allowed,
            "reason": reason,
            "resource_status_changes_mechanism": False,
            "no_off_switch_is_not_negative_mechanism_evidence": True,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "authority_digest": sha256_digest(payload)}


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProspectivePolicyComparisonError(f"JSON root is not an object: {path}")
    return value


def verify_semantic_digest(
    value: Mapping[str, Any], *, digest_field: str, expected: str
) -> None:
    payload = dict(value)
    observed_field = payload.pop(digest_field, None)
    observed = sha256_digest(payload)
    if observed_field != expected or observed != expected:
        raise ProspectivePolicyComparisonError(
            f"{digest_field} semantic identity drift: {observed_field} / {observed}"
        )


def verify_file(path: Path, expected: str, *, label: str) -> None:
    observed = bytes_sha256(path.read_bytes())
    if observed != expected:
        raise ProspectivePolicyComparisonError(
            f"{label} byte identity drift: {observed}"
        )


def flatten_frozen_pool(pool: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        pool.get("held_out_reads") != 0
        or pool.get("implementation_or_qualification_outcomes_present_when_written")
        != 0
        or pool.get("outcome_fields_consumed") != []
    ):
        raise ProspectivePolicyComparisonError("pool is not outcome-blind")
    rows: list[dict[str, Any]] = []
    for source_group in sorted(pool["candidate_pools"]):
        for value in pool["candidate_pools"][source_group]:
            row = dict(value)
            if row.get("stage") != "OPENSPEC_FROZEN":
                raise ProspectivePolicyComparisonError(
                    "the complete Provider denominator did not freeze"
                )
            if row.get("resolution", {}).get("resolution") != "INNOVATION_REQUIRED":
                raise ProspectivePolicyComparisonError(
                    "shared open-idea pool contains a non-innovation candidate"
                )
            candidate_id = row.get("preoutcome_score", {}).get("spec_digest")
            if not isinstance(candidate_id, str):
                raise ProspectivePolicyComparisonError("candidate identity is missing")
            row["candidate_id"] = candidate_id
            row["source_group"] = str(source_group)
            rows.append(row)
    if not rows or len({row["candidate_id"] for row in rows}) != len(rows):
        raise ProspectivePolicyComparisonError("pool is empty or has duplicate identities")
    return rows


def _common_manifest_fields(
    *, policy_name: str, pool_digest: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "recclaw.research-line.q4-prospective-selection.v1",
        "policy_name": policy_name,
        "pool_digest": pool_digest,
        "candidate_count": len(rows),
        "selection_budget": 1,
        "candidate_identity_feature": False,
        "origin_feature": False,
        "qualification_feature": False,
        "resource_feature": False,
        "implementation_feature": False,
        "development_outcome_feature": False,
        "held_out_reads": 0,
        "development_only": True,
        "scientific_effect_claim": False,
    }


def select_static(
    pool: Mapping[str, Any], *, pool_digest: str
) -> dict[str, Any]:
    """Consume the existing F0 static high-change identity-order policy."""

    rows = flatten_frozen_pool(pool)
    direction_digests = {
        str(row["producer_role"]): sha256_digest(
            {"producer_role": str(row["producer_role"])}
        )
        for row in rows
    }
    candidates = tuple(
        IdeaCandidateIdentityV1(
            research_spec_ref=f"recclaw-open-research-spec-v1:{row['candidate_id']}",
            research_spec_digest=str(row["candidate_id"]),
            direction_ref=f"producer-direction:{row['producer_role']}",
            direction_digest=direction_digests[str(row["producer_role"])],
            high_change=True,
            current_profile_expressible=False,
        )
        for row in rows
    )
    first_spec = rows[0]["research_spec"]
    policy_input = IdeaPolicyInputV1(
        research_context_ref="recclaw-q4-prospective-context-v1",
        research_context_digest=sha256_digest(
            {"scope": "Q4_PROSPECTIVE_OPEN_RECOMMENDATION_RESEARCH"}
        ),
        protocol_ref=str(first_spec["protocol_ref"]),
        protocol_digest=str(first_spec["protocol_digest"]),
        current_profile_ref=str(first_spec["current_profile_ref"]),
        current_profile_digest=str(first_spec["current_profile_digest"]),
        candidates=candidates,
        budget=IdeaBudgetV1(
            ideation_slots=1, implementation_slots=1, qualification_slots=1
        ),
    )
    decision = run_static_idea_policy(policy_input)
    selected = [
        value
        for value in decision.acquisition_decisions
        if value.disposition is AcquisitionDispositionV1.SELECT
    ]
    if len(selected) != 1:
        raise ProspectivePolicyComparisonError("STATIC did not select exactly one row")
    selected_id = selected[0].subject_digest
    records = []
    for candidate in sorted(rows, key=lambda value: value["candidate_id"]):
        chosen = candidate["candidate_id"] == selected_id
        records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "producer_role": candidate["producer_role"],
                "selection_score": None,
                "selection_score_semantics": "STATIC_CANONICAL_OPAQUE_IDENTITY_ORDER",
                "selection_probability": 1.0 if chosen else 0.0,
                "selected": chosen,
                "decision_reason": (
                    "STATIC_HIGH_CHANGE_IDENTITY_ORDER"
                    if chosen
                    else "STATIC_IDEA_BUDGET_EXHAUSTED"
                ),
            }
        )
    payload = canonical_value(
        {
            **_common_manifest_fields(
                policy_name="STATIC", pool_digest=pool_digest, rows=rows
            ),
            "policy_ref": STATIC_IDEA_POLICY_REF_V1,
            "policy_digest": STATIC_IDEA_POLICY_DIGEST_V1,
            "selection_rule": "EXISTING_F0_STATIC_HIGH_CHANGE_IDENTITY_ORDER",
            "selection_probability_semantics": "DETERMINISTIC_POINT_MASS",
            "selected_candidate_id": selected_id,
            "candidates": records,
        }
    )
    return {**payload, "selection_digest": sha256_digest(payload)}


def select_current_f1(
    pool: Mapping[str, Any], *, pool_digest: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the accepted F1 Idea direction order to an already-frozen pool.

    F1's physical campaign used this exact order to allocate Producer calls.
    The prospective consumer only moves that established ordering across the
    shared-pool boundary; it does not learn, revise, or use outcomes.
    """

    rows = flatten_frozen_pool(pool)
    direction_order = tuple(str(value) for value in policy["direction_order"])
    if len(direction_order) != len(set(direction_order)):
        raise ProspectivePolicyComparisonError("F1 direction order is invalid")
    rank = {direction: index for index, direction in enumerate(direction_order)}
    if any(str(row["producer_role"]) not in rank for row in rows):
        raise ProspectivePolicyComparisonError("pool direction is outside F1 authority")
    ranked = sorted(
        rows,
        key=lambda value: (
            rank[str(value["producer_role"])],
            str(value["candidate_id"]),
        ),
    )
    selected_id = str(ranked[0]["candidate_id"])
    records = []
    for row in sorted(rows, key=lambda value: str(value["candidate_id"])):
        direction_rank = rank[str(row["producer_role"])]
        chosen = str(row["candidate_id"]) == selected_id
        records.append(
            {
                "candidate_id": row["candidate_id"],
                "producer_role": row["producer_role"],
                "direction_rank": direction_rank,
                "selection_score": -direction_rank,
                "selection_score_semantics": "ACCEPTED_F1_LEARNED_DIRECTION_RANK",
                "selection_probability": 1.0 if chosen else 0.0,
                "selected": chosen,
                "decision_reason": (
                    "SELECTED_BY_ACCEPTED_F1_DIRECTION_ORDER"
                    if chosen
                    else "NOT_SELECTED_LOWER_ACCEPTED_F1_DIRECTION_ORDER"
                ),
            }
        )
    payload = canonical_value(
        {
            **_common_manifest_fields(
                policy_name="CURRENT_F1", pool_digest=pool_digest, rows=rows
            ),
            "policy_ref": policy["policy_ref"],
            "policy_version": policy["policy_version"],
            "policy_digest": policy["policy_digest"],
            "direction_order": direction_order,
            "selection_rule": "ACCEPTED_F1_IDEA_DIRECTION_ORDER_THEN_CANDIDATE_ID",
            "tie_break": "CANDIDATE_ID_ASCENDING",
            "selection_probability_semantics": "DETERMINISTIC_POINT_MASS",
            "selected_candidate_id": selected_id,
            "candidates": records,
        }
    )
    return {**payload, "selection_digest": sha256_digest(payload)}


def select_outcome_aware(
    pool: Mapping[str, Any],
    *,
    pool_digest: str,
    policy: Mapping[str, Any],
    activation: Mapping[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Invoke the accepted Q3 consumer with its real 15% exploration draw."""

    flatten_frozen_pool(pool)
    manifest = build_q3_acquisition_manifest(
        policy=policy,
        activation=activation,
        frozen_pool=pool,
        task_type="IDEA",
        random_seed=random_seed,
    )
    if manifest["pool_digest"] == pool_digest:
        raise ProspectivePolicyComparisonError(
            "raw pool digest must differ from the Q3 origin-blind projection digest"
        )
    return canonical_value(
        {
            **manifest,
            "policy_name": "OUTCOME_AWARE",
            "raw_full_pool_digest": pool_digest,
            "qualification_feature": False,
            "resource_feature": False,
            "implementation_feature": False,
            "development_outcome_feature": False,
        }
    )


def build_arm_manifest(
    *,
    campaign_id: str,
    arm_index: int,
    policy_name: str,
    selection: Mapping[str, Any],
    full_pool_file: Path,
    common_execution: Mapping[str, Any],
) -> dict[str, Any]:
    if policy_name not in POLICY_ORDER or POLICY_ORDER[arm_index - 1] != policy_name:
        raise ProspectivePolicyComparisonError("physical policy order drift")
    selected_id = str(selection["selected_candidate_id"])
    selected_row = next(
        row for row in selection["candidates"] if row.get("selected") is True
    )
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-arm-manifest.v1",
            "campaign_id": campaign_id,
            "round_index": arm_index,
            "policy_name": policy_name,
            "physical_order": POLICY_ORDER,
            "frozen_inputs": {
                "full_pool_file": str(full_pool_file),
                "full_pool_file_sha256": bytes_sha256(full_pool_file.read_bytes()),
                "selection_digest": selection.get(
                    "selection_digest", selection.get("acquisition_digest")
                ),
            },
            "frozen_execution": canonical_value(common_execution),
            "selected_candidate": {
                "candidate_id": selected_id,
                "selection_probability": selected_row["selection_probability"],
                "selection_score": selected_row["selection_score"],
                "exploration_probability": selection.get(
                    "exploration_probability", 0.0
                ),
                "exploration_draw": selection.get("random_draw"),
                "exploration_selected": selection.get(
                    "exploration_selected", False
                ),
            },
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "manifest_digest": sha256_digest(payload)}


def classify_mechanism_probe(
    *,
    qualification: Mapping[str, Any],
    admission: Mapping[str, Any],
    realization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify only real qualifier/resource probe evidence before full outcome."""

    if realization is not None:
        return classify_realization_authority(
            realization=realization,
            qualification=qualification,
            admission=admission,
        )

    if admission.get("status") != "RESOURCE_ADMITTED":
        state = "NOT_ASSESSED"
        reason = "RESOURCE_FAILURE_HAS_NO_MECHANISM_AUTHORITY"
        evidence = {}
    elif qualification.get("status") != "QUALIFICATION_PASS":
        state = "NOT_ASSESSED"
        reason = "QUALIFICATION_FAILURE_HAS_NO_MECHANISM_AUTHORITY"
        evidence = {}
    else:
        behavior = dict(qualification.get("behavioral_evidence") or {})
        loss_delta = float(behavior.get("behavioral_loss_max_abs_delta") or 0.0)
        score_delta = float(behavior.get("behavioral_score_max_abs_delta") or 0.0)
        active = (
            behavior.get("probe_status") == "PASS_STRUCTURAL_BEHAVIOR_ACTIVE"
            and bool(behavior.get("overridden_behavioral_methods"))
            and max(loss_delta, score_delta) > 0.0
        )
        mechanism_off = behavior.get("mechanism_off_execution")
        off_deltas = [
            float(behavior.get(name) or 0.0)
            for name in (
                "mechanism_off_full_sort_max_abs_delta",
                "mechanism_off_loss_abs_delta",
                "mechanism_off_predict_max_abs_delta",
            )
        ]
        if not active:
            state = "INACTIVE"
            reason = "REAL_BEHAVIOR_PROBE_FOUND_NO_ACTIVE_PARTICIPATION"
        elif mechanism_off != "PASS":
            state = "NON_IDENTIFIABLE"
            reason = "ACTIVE_PARTICIPATION_WITHOUT_EXECUTABLE_MECHANISM_OFF"
        elif max(off_deltas) > 1e-8:
            state = "ACTIVE_CONTRADICTED"
            reason = "MECHANISM_OFF_FAILED_PARENT_EQUIVALENCE"
        else:
            state = "ACTIVE_SUPPORTED"
            reason = "ACTIVE_PARTICIPATION_AND_PARENT_EQUIVALENT_MECHANISM_OFF"
        evidence = {
            "loss_participation_observed": loss_delta > 0.0,
            "loss_max_abs_delta": loss_delta,
            "score_or_propagation_participation_observed": score_delta > 0.0,
            "score_max_abs_delta": score_delta,
            "overridden_behavioral_methods": behavior.get(
                "overridden_behavioral_methods", []
            ),
            "extra_parameter_names": behavior.get("extra_parameter_names", []),
            "mechanism_off_execution": mechanism_off,
            "mechanism_off_parent_equivalence_deltas": off_deltas,
            "resource_probe_exit_status": admission.get("resource_probe", {}).get(
                "exit_status"
            ),
            "resource_probe_wall_time_ms": admission.get("resource_probe", {}).get(
                "wall_time_ms"
            ),
            "train_eval_resource_observed": admission.get("resource_probe") is not None,
        }
    if state not in MECHANISM_STATES:
        raise ProspectivePolicyComparisonError("unknown mechanism state")
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-mechanism-probe.v1",
            "status": "PROBE_CLASSIFIED",
            "mechanism_state": state,
            "reason": reason,
            "evidence": evidence,
            "effect_fields_consumed": [],
            "resource_failure_updates_mechanism": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "probe_digest": sha256_digest(payload)}


def _sum_stage_costs(arm_root: Path, stages: Sequence[str]) -> tuple[int, dict[str, int]]:
    components: dict[str, int] = {}
    for stage in stages:
        path = arm_root / "stage_costs" / f"{stage}.json"
        value = read_object(path)
        components[stage] = int(value["wall_time_ms"])
    return sum(components.values()), components


def compute_four_metrics(
    *,
    arm_root: Path,
    upstream_arm_root: Path | None = None,
    full_pool_count: int,
    shared_pool_provider_wall_time_ms: int,
) -> dict[str, Any]:
    """Compute exactly the four frozen main metrics for one policy arm."""

    episode_receipt = read_object(arm_root / "EPISODE_RECEIPT.json")
    mechanism = read_object(arm_root / "MECHANISM_PROBE_RECEIPT.json")
    complete = episode_receipt.get("status") == "EPISODE_CREATED"
    effect = None
    if complete:
        summary = episode_receipt["outcome_summary"]
        effect = round(
            float(summary["candidate_metrics"]["ndcg@10"])
            - float(summary["baseline_metrics"]["ndcg@10"]),
            12,
        )
    identifiable = (
        complete
        and mechanism.get("mechanism_state") in MECHANISM_IDENTIFIABLE_STATES
    )
    informative = complete and (effect is not None or identifiable)
    upstream_root = arm_root if upstream_arm_root is None else upstream_arm_root
    upstream_wall, upstream_components = _sum_stage_costs(
        upstream_root,
        ("implementer", "materialize-qualifier"),
    )
    physical_wall, physical_components = _sum_stage_costs(
        arm_root,
        (
            "resource-admission",
            "mechanism-probe",
            "matched-execution",
        ),
    )
    selected_wall = upstream_wall + physical_wall
    stage_components = {**upstream_components, **physical_components}
    total_wall = shared_pool_provider_wall_time_ms + selected_wall
    cost_per_informative = total_wall if informative else None
    payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-four-metrics.v1",
            "A_best_parent_relative_development_effect_ndcg_at_10": effect,
            "A_missing_not_zero_or_negative": effect is None,
            "B_mechanism_identifiable_episode_count": 1 if identifiable else 0,
            "C_cost_per_informative_episode_wall_time_ms": cost_per_informative,
            "C_zero_informative_semantics": (
                None if informative else "INF_UNDEFINED_NO_SMOOTHING"
            ),
            "C_cost_components": {
                "shared_pool_provider_full_cost_charged_to_each_policy_ms": (
                    shared_pool_provider_wall_time_ms
                ),
                "selected_chain_stage_wall_time_ms": stage_components,
                "total_wall_time_ms": total_wall,
                "informative_episode_count": 1 if informative else 0,
            },
            "D_full_episode_completion_rate_main_selected_denominator": (
                1.0 if complete else 0.0
            ),
            "D_selected_denominator": 1,
            "D_full_pool_denominator": full_pool_count,
            "D_full_pool_completion_coverage_diagnostic": (
                round(1.0 / full_pool_count, 12) if complete else 0.0
            ),
            "mechanism_state": mechanism.get("mechanism_state"),
            "episode_status": episode_receipt.get("status"),
            "missingness": episode_receipt.get("missingness"),
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "metrics_digest": sha256_digest(payload)}


__all__ = [
    "MECHANISM_IDENTIFIABLE_STATES",
    "MECHANISM_STATES",
    "POLICY_ORDER",
    "ProspectivePolicyComparisonError",
    "REALIZATION_CLASSES",
    "REALIZATION_CONTRACT_SCHEMA",
    "REALIZATION_TYPINGS",
    "SHARED_OUTCOME_LEDGER_SCHEMA",
    "SHARED_REALIZATION_POOL_SCHEMA",
    "bind_observation_to_realization",
    "build_arm_manifest",
    "build_shared_package_seed_outcome_ledger",
    "build_shared_realization_contract",
    "build_shared_realization_pool",
    "classify_realization_authority",
    "classify_mechanism_probe",
    "compute_four_metrics",
    "flatten_frozen_pool",
    "read_object",
    "select_current_f1",
    "select_outcome_aware",
    "select_static",
    "verify_file",
    "verify_semantic_digest",
]
