from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT
    / "docs/research_line/vnext/"
    "F1_RESOURCE_COMPATIBLE_EXECUTION_CONTRACT.json"
)
PRIOR_CANONICAL_PATH = (
    ROOT
    / "docs/research_line/vnext/"
    "F1_RESOURCE_COMPATIBLE_REALIZATION_CANONICAL_RECEIPT.json"
)
RECOVERY_PATH = (
    ROOT
    / "docs/research_line/vnext/"
    "F1_OPEN_META_RUNTIME_RECOVERY_V2_CANONICAL_RECEIPT.json"
)
EXECUTION_CANONICAL_PATH = (
    ROOT
    / "docs/research_line/vnext/"
    "F1_RESOURCE_COMPATIBLE_EXECUTION_CANONICAL_RECEIPT.json"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner():
    path = ROOT / "scripts/run_f1_gpu35_closure.py"
    spec = importlib.util.spec_from_file_location(
        "f1_gpu35_closure_execution_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_budget_and_deadlines_are_mechanical_resource_only() -> None:
    contract = _read(CONTRACT_PATH)
    prior = _read(PRIOR_CANONICAL_PATH)
    recovery = _read(RECOVERY_PATH)
    prediction = prior["resource_decision"]["accepted_consumer_output"][
        "predictions"
    ]["resource_compatible_realization"]
    control = recovery["runtime_recovery"]["control"]

    candidate_seconds = prediction["estimated_total_wall_time_seconds"]
    control_seconds = control["wall_time_ms"] / 1000
    margin = contract["admission"]["resource_margin_multiplier"]
    expected_budget = (
        math.ceil(((candidate_seconds + control_seconds) * margin) / 3600)
        * 3600
    )
    expected_deadlines = {
        "matched_bpr_control": math.ceil(control_seconds * margin),
        "resource_compatible_realization": math.ceil(
            candidate_seconds * margin
        ),
    }

    assert expected_budget == 10800
    assert contract["admission"]["campaign_budget_seconds"] == expected_budget
    assert contract["admission"]["deadlines_seconds"] == expected_deadlines
    assert contract["admission"]["deadline_sum_seconds"] == sum(
        expected_deadlines.values()
    )
    assert max(expected_deadlines.values()) < contract["admission"][
        "watchdog_seconds"
    ]
    assert contract["outcome_fields_consumed"] == []
    assert contract["held_out_reads"] == contract["new_provider_calls"] == 0
    assert contract["claims"] == {
        "comparable_to_prior_7200_campaign": False,
        "original_sealed_candidate_pass": False,
        "policy_superiority": False,
        "scientific_effect": False,
    }


def test_execution_contract_binds_sealed_prefix_and_control_receipts() -> None:
    prior = _read(PRIOR_CANONICAL_PATH)
    prior_campaign = Path(prior["external_receipt_ref"]).parent
    validated = _runner()._validate_compatible_execution_contract(
        ROOT,
        prior_prefix_campaign=prior_campaign,
    )

    assert validated == _read(CONTRACT_PATH)
    assert validated["inputs"]["candidate_prefix"]["prefix_status"] == "SUCCESS"
    assert validated["inputs"]["candidate_prefix"][
        "peak_memory_observed_mib"
    ] == 6696.0
    assert validated["admission"]["arm_order"] == [
        "matched_bpr_control",
        "resource_compatible_realization",
    ]


def test_completed_execution_preserves_negative_outcome_without_overclaim() -> None:
    import hashlib

    receipt = _read(EXECUTION_CANONICAL_PATH)
    external = Path(receipt["external_receipt_ref"])
    assert hashlib.sha256(external.read_bytes()).hexdigest() == receipt[
        "external_receipt_sha256"
    ]
    assert receipt["status"] == "F1_RESOURCE_COMPATIBLE_REALIZATION_PASS"
    assert all(receipt["architecture_effect_gates"].values())
    assert receipt["experiment_executed"] is True
    assert receipt["resource_disposition"] == "COMPLETED_MATCHED_PAIR"
    assert receipt["new_physical_training_runs"] == 2
    assert receipt["held_out_reads"] == receipt["new_provider_calls"] == 0
    assert receipt["original_sealed_candidate_pass"] is False
    assert receipt["original_sealed_candidate_disposition"] == "RESOURCE_DEFERRED"
    assert receipt["old_7200_campaign_comparison_allowed"] is False
    assert receipt["policy_superiority_claim"] is False
    assert receipt["scientific_effect_claim"] is False

    runs = receipt["matched_full_runs"]
    assert [runs[arm]["exit_status"] for arm in (
        "matched_bpr_control",
        "resource_compatible_realization",
    )] == ["SUCCESS", "SUCCESS"]
    assert runs["matched_bpr_control"]["resource_deadline_seconds"] == 426
    assert runs["resource_compatible_realization"][
        "resource_deadline_seconds"
    ] == 9771
    assert runs["resource_compatible_realization"]["metrics"]["ndcg@10"] < runs[
        "matched_bpr_control"
    ]["metrics"]["ndcg@10"]

    episode = receipt["episode"]
    assert episode["experiment_executed"] is True
    assert episode["evidence_class"] == "INCONCLUSIVE_EXPERIMENT"
    assert episode["mechanism_interpretation"] == "NOT_ADJUDICATED"
    assert episode["failure_class"] == "INCONCLUSIVE"
    assert episode["mechanism_negative_evidence"] is False
