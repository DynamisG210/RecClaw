from __future__ import annotations

from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    ARM_ORDER,
    CAMPAIGN_TOTAL_BUDGET_SECONDS,
    ENGINEERING_WATCHDOG_SECONDS,
    PROBE_EPOCHS,
    finalize_hard_block_receipt,
    predict_resources,
    structural_features,
)


def _probe(*, epoch_ms: tuple[int, int, int], peak_mib: float) -> dict[str, object]:
    phases = []
    for epoch, wall_ms in enumerate(epoch_ms):
        phases.extend(
            [
                {
                    "epoch": epoch,
                    "phase": "TRAIN",
                    "wall_time_ms": wall_ms - 100,
                    "peak_allocated_mib": peak_mib,
                    "peak_reserved_mib": peak_mib,
                },
                {
                    "epoch": epoch,
                    "phase": "EVAL",
                    "wall_time_ms": 100,
                    "peak_allocated_mib": peak_mib / 2,
                    "peak_reserved_mib": peak_mib / 2,
                },
            ]
        )
    return {
        "exit_status": "SUCCESS",
        "resource_telemetry": {
            "parameter_count": 1000,
            "phase_records": phases,
            "trainable_parameter_count": 1000,
        },
        "wall_time_ms": sum(epoch_ms) + 500,
    }


def test_q0r_prediction_is_uniform_outcome_blind_and_budget_bounded() -> None:
    features = {
        arm: {
            "bottleneck_feature_count": index,
            "dense_compute": index > 1,
            "full_sort_path": True,
            "graph_propagation": index > 0,
            "routing_path": index > 2,
            "sparse_compute": index > 0,
        }
        for index, arm in enumerate(ARM_ORDER)
    }
    probes = {
        arm: _probe(
            epoch_ms=(1000 + index * 500,) * PROBE_EPOCHS,
            peak_mib=1000 + index * 100,
        )
        for index, arm in enumerate(ARM_ORDER)
    }

    decision = predict_resources(arm_features=features, probe_runs=probes)

    assert decision["outcome_fields_consumed"] == []
    assert decision["probe_contract"]["execution_purpose"] == "RESOURCE_PROBE_ONLY"
    assert decision["probe_contract"]["uniform_across_arms"] is True
    assert (
        sum(row["deadline_seconds"] for row in decision["schedule"])
        + decision["probe_cost_seconds"]
        <= CAMPAIGN_TOTAL_BUDGET_SECONDS
    )
    assert [row["arm"] for row in decision["schedule"]] == list(ARM_ORDER)
    assert all(
        row["features"]["parameter_count"] == 1000
        for row in decision["predictions"].values()
    )
    assert all(
        row["deadline_seconds"] < ENGINEERING_WATCHDOG_SECONDS
        for row in decision["schedule"]
    )
    assert decision["deadline_formula"].find("1500") == -1


def test_q0r_structural_features_are_visible_and_auditable(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text(
        """
import torch

class Candidate:
    def full_sort_predict(self, x):
        routed = torch.matmul(x, self.router.weight)
        return torch.sparse.mm(self.norm_adj, routed)
""",
        encoding="utf-8",
    )

    features = structural_features(source)

    assert features["dense_compute"] is True
    assert features["full_sort_path"] is True
    assert features["graph_propagation"] is True
    assert features["routing_path"] is True
    assert features["sparse_compute"] is True
    assert features["source_bytes"] > 0


def test_q0r_hard_block_receipt_separates_watchdog_and_effect(tmp_path: Path) -> None:
    external = tmp_path / "external.json"
    external.write_text(
        (
            '{"development_only":true,"held_out_reads":0,"probe_runs":'
            '{"frontier_candidate":{"censoring_semantics":'
            '"RESOURCE_OR_COMPLETION_ONLY; NEVER_MECHANISM_EFFECT",'
            '"censoring_trigger":"RESOURCE_BUDGET_DEADLINE",'
            '"exit_status":"RESOURCE_CENSORED",'
            '"resource_deadline_seconds":300,"wall_time_ms":300001,'
            '"watchdog_seconds":10800}},"q1_allowed":false,'
            '"schema":"recclaw.research-line.q0r-canonical-receipt.v1",'
            '"scientific_effect_claim":false,"status":"HARD_BLOCK"}\n'
        ),
        encoding="utf-8",
    )
    canonical = tmp_path / "canonical.json"

    receipt = finalize_hard_block_receipt(
        external,
        canonical_receipt_path=canonical,
    )

    separation = receipt["engineering_safety_and_research_budget_separation"]
    blocker = receipt["resource_blockers"]["frontier_candidate"]
    assert separation["legacy_1500_seconds_controls_q0r_full_runs"] is False
    assert separation["engineering_watchdog_seconds"] == 10800
    assert blocker["resource_deadline_seconds"] == 300
    assert blocker["watchdog_seconds"] == 10800
    assert blocker["resource_disposition"] == "RESOURCE_DEFERRED"
    assert blocker["mechanism_effect_update_allowed"] is False
    assert receipt["evaluation"]["q1_allowed"] is False
