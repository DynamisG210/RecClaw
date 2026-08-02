from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    ARM_ORDER,
    CAMPAIGN_TOTAL_BUDGET_SECONDS,
    ENGINEERING_WATCHDOG_SECONDS,
    FIXED_EVAL_BATCH_INDICES,
    FIXED_TRAIN_BATCH_INDICES,
    PROBE_EPOCHS,
    build_fixed_batch_prefix_contract,
    finalize_fixed_batch_hard_block_receipt,
    finalize_hard_block_receipt,
    predict_resources,
    structural_features,
)


def _probe(*, batch_ms: int, peak_mib: float) -> dict[str, object]:
    batches = []
    for epoch in range(PROBE_EPOCHS):
        for position in range(len(FIXED_TRAIN_BATCH_INDICES)):
            batches.append(
                {
                    "epoch": epoch,
                    "loss": 1.0 / (position + 1),
                    "phase": "TRAIN",
                    "status": "BATCH_COMPLETED",
                    "wall_time_ms": batch_ms,
                    "peak_allocated_mib": peak_mib,
                    "peak_reserved_mib": peak_mib,
                }
            )
        for _position in range(len(FIXED_EVAL_BATCH_INDICES)):
            batches.append(
                {
                    "epoch": epoch,
                    "loss": None,
                    "phase": "EVAL",
                    "status": "BATCH_COMPLETED",
                    "wall_time_ms": max(1, batch_ms // 2),
                    "peak_allocated_mib": peak_mib / 2,
                    "peak_reserved_mib": peak_mib / 2,
                }
            )
    return {
        "exit_status": "SUCCESS",
        "resource_telemetry": {
            "batch_records": batches,
            "full_train_batches_per_epoch": 100,
            "full_validation_batches_per_eval": 20,
            "initialization_wall_time_ms": 500,
            "parameter_count": 1000,
            "prefix_contract": {"contract_file_sha256": "a" * 64},
            "trainable_parameter_count": 1000,
        },
        "wall_time_ms": sum(int(row["wall_time_ms"]) for row in batches) + 500,
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
            batch_ms=10 + index * 5,
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


def test_q0r_fixed_batch_contract_is_frozen_origin_blind() -> None:
    contract = build_fixed_batch_prefix_contract()

    assert contract["epochs"] == 3
    assert contract["deadline_rule"]["resource_deadline_seconds"] == 300
    assert contract["deadline_rule"]["legacy_1500_seconds_controls_probe"] is False
    assert contract["train_batch_indices"] == list(range(32))
    assert contract["eval_batch_indices"] == list(range(64))
    assert contract["execution_purpose"] == "RESOURCE_PROBE_ONLY"
    encoded = json.dumps(contract, sort_keys=True)
    assert "candidate" not in encoded.lower()
    assert "ndcg" not in encoded.lower()
    assert "outcome" not in encoded.lower()


def test_censored_fixed_batch_writer_preserves_atomic_partial_telemetry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "resource_telemetry.json"
    worker = Path(__file__).resolve().parents[3] / "scripts/campaign_train_worker.py"
    code = r'''
import importlib.util
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("campaign_train_worker", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
path = Path(sys.argv[2])
batches = []
telemetry = {
    "active_progress": {},
    "batch_records": batches,
    "phase_records": [],
    "parameter_count": 1,
    "trainable_parameter_count": 1,
}
def started(position, source_index):
    telemetry["active_progress"] = {
        "batch_position": position,
        "phase": "TRAIN",
        "source_batch_index": source_index,
        "status": "BATCH_STARTED",
    }
    module._write_durable_json(path, module._finalize_resource_telemetry(telemetry))
def completed(position, source_index, started_ns):
    batches.append({
        "batch_position": position,
        "epoch": 0,
        "loss": 1.0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
        "phase": "TRAIN",
        "source_batch_index": source_index,
        "status": "BATCH_COMPLETED",
        "wall_time_ms": 1,
    })
    module._write_durable_json(path, module._finalize_resource_telemetry(telemetry))
view = module._FixedBatchDataLoaderView(
    object(), ("first", "second"), (0, 1),
    on_batch_started=started, on_batch_completed=completed,
)
for position, _batch in enumerate(view):
    if position == 1:
        time.sleep(30)
'''
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(worker), str(artifact)]
    )
    deadline = time.monotonic() + 10
    observed = None
    while time.monotonic() < deadline:
        if artifact.is_file():
            observed = json.loads(artifact.read_text(encoding="utf-8"))
            if observed.get("completed_batch_records") == 1:
                break
        time.sleep(0.02)
    process.kill()
    process.wait(timeout=5)

    assert observed is not None
    preserved = json.loads(artifact.read_text(encoding="utf-8"))
    assert preserved["completed_batch_records"] == 1
    assert preserved["batch_records"][0]["status"] == "BATCH_COMPLETED"
    assert preserved["batch_records"][0]["loss"] == 1.0
    assert preserved["peak_gpu_memory_mib"] == 1.0


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


def test_fixed_batch_hard_block_finalizer_records_progress_without_effect(
    tmp_path: Path,
) -> None:
    external = tmp_path / "physical.json"
    external.write_text(
        json.dumps(
            {
                "held_out_reads": 0,
                "probe_runs": {
                    "frontier_candidate": {
                        "exit_status": "RESOURCE_CENSORED",
                        "resource_telemetry": {
                            "active_progress": {"status": "BATCH_STARTED"},
                            "batch_records": [
                                {"phase": "TRAIN", "status": "BATCH_COMPLETED"}
                            ],
                        },
                        "wall_time_ms": 300001,
                    }
                },
                "q1_allowed": False,
                "schema": "recclaw.research-line.q0r-canonical-receipt.v2",
                "status": "HARD_BLOCK",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    canonical = tmp_path / "canonical.json"

    receipt = finalize_fixed_batch_hard_block_receipt(
        external,
        canonical_receipt_path=canonical,
        external_receipt_ref="results/physical.json",
    )

    blocker = receipt["resource_blockers"]["frontier_candidate"]
    assert blocker["completed_train_batches"] == 1
    assert blocker["completed_eval_batches"] == 0
    assert blocker["mechanism_effect_update_allowed"] is False
    assert blocker["resource_disposition"] == "RESOURCE_DEFERRED"
    assert receipt["full_outcomes_present"] == 0
    assert receipt["evaluation"]["h2_resource_modeling_and_scheduling"] == "NOT_CLOSED"
    assert receipt["evaluation"]["q1_allowed"] is False
