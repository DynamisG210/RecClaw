from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
Q4 = ROOT / "results/research_line/q4_prospective_policy_comparison_20260803_01"
DOCS = ROOT / "docs/research_line/vnext"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_q4_sealed_bytes_match_declared_independent_audit_bindings() -> None:
    expected = {
        DOCS / "Q4_PROSPECTIVE_POLICY_COMPARISON_CANONICAL_RECEIPT.json": (
            "304b0a0b1649983fefda9bbc3d50528eb75c40644c5e4d98d61661c851a30281"
        ),
        Q4 / "Q4_PROSPECTIVE_CANONICAL_RECEIPT.json": (
            "304b0a0b1649983fefda9bbc3d50528eb75c40644c5e4d98d61661c851a30281"
        ),
        Q4 / "Q4_PROSPECTIVE_PHYSICAL_RECEIPT.json": (
            "5b0e6914d17960ffec025a65df6aaeb445f53ab7e3878616a60f0080891d6100"
        ),
        Q4 / "Q4_PROSPECTIVE_POLICY_COMPARISON_PACKAGE.json": (
            "c7d246a88e3857fedbade67887fcdde431497c4dc960913fe2c686f240492f8b"
        ),
        Q4 / "fairness_v2/PREFREEZE_FAIRNESS_CONTRACT_V2.json": (
            "76e8ee755b457cc8b0d93af33e9b2d67fe2793100b22393378f665dd26dfce29"
        ),
        Q4 / "VERIFICATION_AUDIT_BEFORE_FINALIZE.json": (
            "3909b7df70c99536b914900eda551b59e208272ca6d5f27063bb71a7edeb31b3"
        ),
        Q4 / "SHA256SUMS": (
            "1d1f053b5561f2e9eab9f7997530d2e85f303a75d05129217205becbdce686bf"
        ),
    }
    assert {path: _sha256(path) for path in expected} == expected
    assert (
        DOCS / "Q4_PROSPECTIVE_POLICY_COMPARISON_CANONICAL_RECEIPT.json"
    ).read_bytes() == (Q4 / "Q4_PROSPECTIVE_CANONICAL_RECEIPT.json").read_bytes()

    entries = (Q4 / "SHA256SUMS").read_text().splitlines()
    assert len(entries) == 182
    for entry in entries:
        digest, relative_path = entry.split("  ", 1)
        assert _sha256(Q4 / relative_path) == digest


def test_q4_provider_and_protocol_ledgers_preserve_preoutcome_failures() -> None:
    broker_paths = sorted(Q4.glob("**/broker.sqlite3"))
    assert len(broker_paths) == 7
    broker_rows: list[tuple[object, ...]] = []
    for path in broker_paths:
        with sqlite3.connect(path) as connection:
            broker_rows.extend(
                connection.execute(
                    "SELECT status, returned_model, outcome_json FROM calls"
                ).fetchall()
            )
    assert broker_rows == [("SUCCESS", "gpt-5.4", None)] * 7

    candidate_ids = {
        "01_static": (
            "38a26af380e16a8ef7e2ada045f75be761e0ff38eda747f1c2e83a830db12b64"
        ),
        "02_current_f1": (
            "38a26af380e16a8ef7e2ada045f75be761e0ff38eda747f1c2e83a830db12b64"
        ),
    }
    for arm, candidate_id in candidate_ids.items():
        original = _load(Q4 / "arms" / arm / "ARM_MANIFEST.json")
        corrected = _load(Q4 / "fairness_v2/arms" / arm / "ARM_MANIFEST_V2.json")
        invalid = _load(
            Q4 / "fairness_v2/arms" / arm / "INVALID_PROTOCOL_BINDING_RECEIPT.json"
        )
        assert original["frozen_execution"]["resource_probe_seed"] == 54302
        assert corrected["frozen_execution"]["resource_probe_seed"] == 54102
        assert original["selected_candidate"]["candidate_id"] == candidate_id
        assert corrected["selected_candidate"]["candidate_id"] == candidate_id
        assert invalid["classification"] == "INVALID_PROTOCOL_BINDING"
        assert invalid["training_batch_count"] == 0
        assert invalid["effect_authority"] == 0
        assert invalid["mechanism_authority"] == 0
        assert invalid["candidate_retry_count"] == 0

    physical = _load(Q4 / "Q4_PROSPECTIVE_PHYSICAL_RECEIPT.json")
    assert physical["shared_pool_provider_usage"]["physical_calls"] == 4
    assert physical["shared_pool_provider_usage"]["retries"] == 0
    assert physical["totals"] == {
        "all_gpu_worker_launches_including_invalid": 10,
        "candidate_retries": 0,
        "complete_episodes": 2,
        "held_out_reads": 0,
        "implementation_provider_calls": 3,
        "invalid_protocol_pretraining_launches": 2,
        "matched_development_training_runs": 4,
        "missing_episodes": 1,
        "proposal_provider_calls": 4,
        "qualification_training_runs": 2,
        "stage_retries": 0,
        "valid_resource_probe_training_runs": 2,
    }


def test_q4_metrics_keep_missingness_and_claim_authority_separate() -> None:
    static = _load(Q4 / "fairness_v2/arms/01_static/FOUR_MAIN_METRICS.json")
    current = _load(Q4 / "fairness_v2/arms/02_current_f1/FOUR_MAIN_METRICS.json")
    outcome = _load(Q4 / "fairness_v2/arms/03_outcome_aware/FOUR_MAIN_METRICS.json")

    assert static["A_best_parent_relative_development_effect_ndcg_at_10"] == -0.0064
    assert current["A_best_parent_relative_development_effect_ndcg_at_10"] == -0.0072
    assert static["mechanism_state"] == current["mechanism_state"] == "NON_IDENTIFIABLE"
    assert outcome["A_best_parent_relative_development_effect_ndcg_at_10"] is None
    assert outcome["A_missing_not_zero_or_negative"] is True
    assert outcome["C_cost_per_informative_episode_wall_time_ms"] is None
    assert outcome["C_zero_informative_semantics"] == "INF_UNDEFINED_NO_SMOOTHING"
    assert outcome["mechanism_state"] == "NOT_ASSESSED"
    assert outcome["episode_status"] == "EPISODE_MISSING"
    assert all(row["held_out_reads"] == 0 for row in (static, current, outcome))
    assert all(row["scientific_effect_claim"] is False for row in (static, current, outcome))

    audit_receipt = _load(DOCS / "Q4_FINAL_INDEPENDENT_AUDIT_CANONICAL_RECEIPT.json")
    assert audit_receipt["audited_source"] == {
        "commit": "3615a8c9b2cc6d473df6fb1348fd09c47a883426",
        "parent": "63e44341bd1da982f00a9db7de2be7e779c7a63d",
        "tree": "4492bc796707f24080a9ae799023b9d73b15b3c4",
    }
    assert audit_receipt["verdict"] == "PASS_WITH_FINDINGS"
    assert audit_receipt["permitted_claim"]["authority"] == "DEVELOPMENT_ONLY"
