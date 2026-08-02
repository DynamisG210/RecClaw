"""Minimal Q2 mechanism-evidence consumer.

This module owns input binding and the outcome-independent state decision.  The
physical worker lives in ``scripts/q2_mechanism_probe_worker.py`` so the core
decision can be tested without importing RecBole or Torch locally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


class MechanismCharacterizationError(RuntimeError):
    """The frozen Q2 input or physical evidence is incomplete or inconsistent."""


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    """Create one durable JSON artifact without overwriting prior evidence."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(value))
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return bytes_sha256(data)


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "recclaw.research-line.q2-mechanism-probe-contract.v1"
        or value.get("status") != "FROZEN_BEFORE_Q2_OUTCOME"
        or value.get("development_only") is not True
    ):
        raise MechanismCharacterizationError("Q2 probe contract identity drift")
    return value


def validate_selected_package(
    repo_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the sole Q1 package and return its immutable candidate root."""

    repo_root = repo_root.resolve()
    expected = dict(contract["input"])
    q1_receipt_path = repo_root / str(expected["q1_physical_receipt_path"])
    q1_receipt_bytes = q1_receipt_path.read_bytes()
    if bytes_sha256(q1_receipt_bytes) != expected["q1_physical_receipt_sha256"]:
        raise MechanismCharacterizationError("Q1 physical receipt byte drift")
    q1_receipt = json.loads(q1_receipt_bytes)
    receipt_selected = q1_receipt.get("selected_results", {}).get("enriched", {})
    if receipt_selected.get("candidate_package_digest") != expected[
        "candidate_package_digest"
    ]:
        raise MechanismCharacterizationError("selected Q1 package digest drift")
    package_path = repo_root / str(expected["candidate_package_path"])
    package_bytes = package_path.read_bytes()
    if bytes_sha256(package_bytes) != expected["candidate_package_file_sha256"]:
        raise MechanismCharacterizationError("selected Q1 package byte drift")
    package = json.loads(package_bytes)
    package_identity = package.get("package", {})
    checks = {
        "candidate_source_tree_digest": package_identity.get("source_tree_digest"),
        "entrypoint": package_identity.get("executable_entrypoint"),
        "research_spec_digest": package_identity.get("research_spec_digest"),
    }
    observed = {
        "candidate_source_tree_digest": checks["candidate_source_tree_digest"],
        "entrypoint": checks["entrypoint"],
        "research_spec_digest": checks["research_spec_digest"],
    }
    wanted = {
        key: expected[key]
        for key in (
            "candidate_source_tree_digest",
            "entrypoint",
            "research_spec_digest",
        )
    }
    if observed != wanted:
        raise MechanismCharacterizationError("selected Q1 package identity drift")

    candidate_roots = []
    for source_path in package_path.parent.glob("candidates/*/*/recclaw_ext/candidate.py"):
        if bytes_sha256(source_path.read_bytes()) == expected["candidate_source_sha256"]:
            candidate_roots.append(source_path.parents[1])
    if len(candidate_roots) != 1:
        raise MechanismCharacterizationError(
            "selected Q1 candidate root is missing or ambiguous"
        )
    candidate_root = candidate_roots[0]
    for row in package["implementation_receipt"]["written_files"]:
        path = candidate_root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or bytes_sha256(path.read_bytes()) != row["sha256"]
        ):
            raise MechanismCharacterizationError(
                f"selected candidate file drift: {row['path']}"
            )
    return {
        "candidate_package": package,
        "candidate_package_path": package_path,
        "candidate_root": candidate_root,
        "candidate_source_path": candidate_root / "recclaw_ext/candidate.py",
    }


_PARTICIPATION_PROBES = (
    "loss_participation",
    "gate_activation",
    "routing_and_propagation",
)
_IDENTIFIABILITY_PROBES = (
    "target_conditioning",
    "mechanism_off_parent_equivalence",
)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_probe_statistics(
    statistics: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert physical observations into the preregistered probe decisions."""

    numeric = dict(contract["numeric_rules"])
    population = dict(contract["probe_population"])
    loss = dict(statistics.get("loss_participation", {}))
    gradient_norms = dict(loss.get("mechanism_gradient_l2", {}))
    loss_values = [
        loss.get("full_loss"),
        loss.get("mechanism_off_loss"),
        loss.get("stability_contribution"),
        loss.get("full_off_abs_delta"),
        *gradient_norms.values(),
    ]
    loss_pass = (
        bool(gradient_norms)
        and all(_finite_number(value) for value in loss_values)
        and float(loss["stability_contribution"])
        > float(numeric["finite_nonzero_floor"])
        and float(loss["full_off_abs_delta"])
        > float(numeric["routing_score_or_margin_delta_floor"])
        and all(
            float(value) > float(numeric["mechanism_gradient_l2_floor"])
            for value in gradient_norms.values()
        )
    )

    gate = dict(statistics.get("gate_activation", {}))
    gate_values = [
        gate.get("raw_logit_std"),
        gate.get("support_saturation_rate"),
    ]
    gate_pass = (
        all(_finite_number(value) for value in gate_values)
        and int(gate.get("distinct_user_count", 0))
        >= int(population["minimum_distinct_users"])
        and int(gate.get("eligible_user_count", 0))
        >= int(population["minimum_eligible_users_with_two_supports"])
        and float(gate["raw_logit_std"])
        > float(numeric["activation_std_floor"])
        and float(gate["support_saturation_rate"])
        <= float(numeric["maximum_support_saturation_rate"])
        and int(gate.get("distinct_centered_logit_signatures", 0)) >= 2
    )

    routing = dict(statistics.get("routing_and_propagation", {}))
    routing_values = [
        routing.get("positive_score_max_abs_delta"),
        routing.get("pairwise_margin_max_abs_delta"),
    ]
    routing_pass = (
        all(_finite_number(value) for value in routing_values)
        and int(routing.get("support_gate_call_count", 0)) > 0
        and max(float(value) for value in routing_values)
        > float(numeric["routing_score_or_margin_delta_floor"])
    )

    target = dict(statistics.get("target_conditioning", {}))
    target_values = [target.get("max_centered_common_logit_delta")]
    target_pass = (
        all(_finite_number(value) for value in target_values)
        and int(target.get("compared_user_count", 0))
        >= int(population["minimum_eligible_users_with_two_supports"])
        and float(target["max_centered_common_logit_delta"])
        > float(numeric["activation_std_floor"])
        and int(target.get("pairwise_order_change_count", 0)) > 0
    )

    parent = dict(statistics.get("mechanism_off_parent_equivalence", {}))
    deltas = dict(parent.get("max_abs_deltas", {}))
    reference_maxima = dict(parent.get("reference_max_abs_values", {}))
    required_surfaces = {
        "predict",
        "full_sort_predict",
        "calculate_loss",
        "shared_parameter_gradients",
    }
    parent_pass = set(deltas) == required_surfaces and all(
        _finite_number(deltas[name])
        and _finite_number(reference_maxima.get(name))
        and float(deltas[name])
        <= float(numeric["parent_equivalence_atol"])
        + float(numeric["parent_equivalence_rtol"])
        * float(reference_maxima[name])
        for name in required_surfaces
    )

    discriminative = dict(statistics.get("discriminative_prediction", {}))
    learned = discriminative.get("learned_removal_mean_abs_margin_damage")
    control = discriminative.get("control_removal_mean_abs_margin_damage")
    if (
        int(discriminative.get("eligible_example_count", 0)) < 1
        or not _finite_number(learned)
        or not _finite_number(control)
    ):
        discriminative_status = "NOT_RUN"
    else:
        margin = float(numeric["discriminative_relative_damage_margin"])
        floor = float(numeric["finite_nonzero_floor"])
        learned_value = float(learned)
        control_value = float(control)
        if learned_value >= control_value + margin * max(control_value, floor):
            discriminative_status = "SUPPORTED"
        elif control_value >= learned_value + margin * max(learned_value, floor):
            discriminative_status = "CONTRADICTED"
        else:
            discriminative_status = "INCONCLUSIVE"

    return {
        "loss_participation": {"status": "PASS" if loss_pass else "FAIL", **loss},
        "gate_activation": {"status": "PASS" if gate_pass else "FAIL", **gate},
        "routing_and_propagation": {
            "status": "PASS" if routing_pass else "FAIL",
            **routing,
        },
        "target_conditioning": {
            "status": "PASS" if target_pass else "FAIL",
            **target,
        },
        "mechanism_off_parent_equivalence": {
            "status": "PASS" if parent_pass else "FAIL",
            **parent,
        },
        "discriminative_prediction": {
            "status": discriminative_status,
            **discriminative,
        },
    }


def classify_mechanism_evidence(evidence: Mapping[str, Any]) -> str:
    """Apply the frozen Q2 precedence without consulting resource missingness."""

    required = (*_PARTICIPATION_PROBES, *_IDENTIFIABILITY_PROBES)
    statuses = {
        name: dict(evidence.get(name, {})).get("status") for name in required
    }
    discriminative = dict(evidence.get("discriminative_prediction", {})).get(
        "status"
    )

    def not_assessed(status: object) -> bool:
        return status is None or status == "NOT_RUN" or (
            isinstance(status, str) and status.startswith("NOT_ASSESSED")
        )

    # A frozen identifiability discriminator that physically fails is decisive:
    # later learned-effect evidence cannot make the declared mechanism separable.
    if any(statuses[name] == "FAIL" for name in _IDENTIFIABILITY_PROBES):
        return "NON_IDENTIFIABLE"
    if any(not_assessed(status) for status in statuses.values()):
        return "NOT_ASSESSED"
    if not_assessed(discriminative):
        return "NOT_ASSESSED"
    if any(statuses[name] != "PASS" for name in _PARTICIPATION_PROBES):
        return "INACTIVE"
    if any(statuses[name] != "PASS" for name in _IDENTIFIABILITY_PROBES):
        return "NON_IDENTIFIABLE"
    if discriminative == "SUPPORTED":
        return "ACTIVE_SUPPORTED"
    if discriminative == "CONTRADICTED":
        return "ACTIVE_CONTRADICTED"
    return "NON_IDENTIFIABLE"


def full_ablation_allowed(
    *, state: str, evidence: Mapping[str, Any], resource_status: str
) -> bool:
    if state not in {"ACTIVE_SUPPORTED", "ACTIVE_CONTRADICTED"}:
        return False
    if resource_status != "SUCCESS":
        return False
    return all(
        dict(evidence.get(name, {})).get("status") == "PASS"
        for name in _IDENTIFIABILITY_PROBES
    )


def q3_evidence_package(
    *,
    contract_sha256: str,
    physical_result_sha256: str,
    state: str,
    evidence: Mapping[str, Any],
    resource_status: str,
    full_ablation_executed: bool,
) -> dict[str, Any]:
    effect_update_allowed = state in {
        "ACTIVE_SUPPORTED",
        "ACTIVE_CONTRADICTED",
    }
    return {
        "schema": "recclaw.research-line.q3-mechanism-evidence-input.v1",
        "development_only": True,
        "scientific_effect_claim": False,
        "held_out_reads": 0,
        "q2_contract_sha256": contract_sha256,
        "q2_physical_result_sha256": physical_result_sha256,
        "mechanism_state": state,
        "mechanism_effect_update_allowed": effect_update_allowed,
        "resource_status": resource_status,
        "resource_updates_mechanism_effect": False,
        "full_ablation_executed": full_ablation_executed,
        "full_ablation_eligible": full_ablation_allowed(
            state=state,
            evidence=evidence,
            resource_status=resource_status,
        ),
        "next_consumer": "Q3_FEASIBILITY_MECHANISM_EFFECT_MODEL",
    }


__all__ = [
    "MechanismCharacterizationError",
    "bytes_sha256",
    "canonical_json_bytes",
    "classify_mechanism_evidence",
    "evaluate_probe_statistics",
    "full_ablation_allowed",
    "load_contract",
    "q3_evidence_package",
    "validate_selected_package",
    "write_new_json",
]
