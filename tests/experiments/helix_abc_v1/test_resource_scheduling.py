from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
    ARM_ORDER,
    CAMPAIGN_TOTAL_BUDGET_SECONDS,
    ENGINEERING_WATCHDOG_SECONDS,
    FIXED_EVAL_BATCH_INDICES,
    FIXED_TRAIN_BATCH_INDICES,
    PROBE_EPOCHS,
    admit_resource_only_evidence,
    build_fixed_batch_prefix_contract,
    finalize_fixed_batch_hard_block_receipt,
    finalize_hard_block_receipt,
    predict_resources,
    project_resource_only_evidence,
    structural_features,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    RECBole_ROOT,
    SEARCH_DATA_ROOT,
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


def _accepted_resource_receipts() -> tuple[dict[str, object], dict[str, object]]:
    root = Path(__file__).resolve().parents[3]
    q0 = json.loads(
        (root / "docs/research_line/vnext/Q0_QUALITY_CALIBRATION_CANONICAL_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    q0r = json.loads(
        (root / "docs/research_line/vnext/Q0R_TYPE_PRESERVING_RESOURCE_SCHEDULING_CANONICAL_RECEIPT.json").read_text(
            encoding="utf-8"
        )
    )
    return q0, q0r


def test_q0r2_resource_projection_is_effect_blind() -> None:
    q0, q0r = _accepted_resource_receipts()
    changed = copy.deepcopy(q0)
    changed["evaluation"]["outcomes"]["parent_equivalent_null"].update(
        {
            "candidate_minus_bpr": 999.0,
            "candidate_ndcg_at_10": -999.0,
            "mechanism_effect_update_allowed": False,
        }
    )
    changed["evaluation"]["matched_bpr_control"]["metrics"] = {"ndcg@10": -42.0}
    changed["arm_records"]["parent_equivalent_null"]["training_run"]["metrics"] = {
        "loss": 123.0,
        "ndcg@10": -42.0,
    }

    projected = project_resource_only_evidence(q0_receipt=q0, q0r_receipt=q0r)
    changed_projected = project_resource_only_evidence(
        q0_receipt=changed, q0r_receipt=q0r
    )
    decision = admit_resource_only_evidence(projected)

    assert projected == changed_projected
    assert decision == admit_resource_only_evidence(changed_projected)
    assert projected["effect_fields_consumed"] == []
    assert all(
        row["mechanism_effect_update_allowed"] is False
        for row in projected["arms"].values()
    )
    encoded = json.dumps(projected, sort_keys=True).lower()
    assert "ndcg" not in encoded
    assert "candidate_minus" not in encoded
    assert '"loss"' not in encoded


def test_q0r2_first_principles_admission_is_non_empty() -> None:
    q0, q0r = _accepted_resource_receipts()
    evidence = project_resource_only_evidence(q0_receipt=q0, q0r_receipt=q0r)

    decision = admit_resource_only_evidence(evidence)

    scheduled = {row["arm"] for row in decision["schedule"]}
    deferred = {row["arm"]: row for row in decision["deferred_arms"]}
    assert scheduled == {"matched_bpr_control", "parent_equivalent_null"}
    assert decision["schedule"]
    assert sum(row["deadline_seconds"] for row in decision["schedule"]) <= 7200
    assert deferred["known_good_reference"]["resource_disposition"] == "RESOURCE_DEFERRED"
    assert deferred["frontier_candidate"]["future_eligible"] is True
    assert decision["engineering_watchdog_seconds"] == 10800
    assert all(row["deadline_seconds"] < 10800 for row in decision["schedule"])


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


def test_resource_prediction_accepts_frozen_two_arm_consumer() -> None:
    order = ("matched_bpr_control", "sealed_f1_candidate")
    features = {
        arm: {
            "ast_call_count": 1,
            "bottleneck_feature_count": 0,
            "dense_compute": False,
            "full_sort_path": True,
            "graph_propagation": False,
            "routing_path": False,
            "source_bytes": 10,
            "source_sha256": str(index + 1) * 64,
            "sparse_compute": False,
        }
        for index, arm in enumerate(order)
    }
    probes = {arm: _probe(batch_ms=2, peak_mib=100) for arm in order}

    decision = predict_resources(
        arm_features=features,
        probe_runs=probes,
        arm_order=order,
        probe_seed=53102,
    )

    assert set(decision["predictions"]) == set(order)
    assert decision["probe_contract"]["seed"] == 53102
    assert {row["arm"] for row in decision["schedule"]} == set(order)


def test_fixed_batch_prefix_contract_accepts_frozen_campaign_seed() -> None:
    contract = build_fixed_batch_prefix_contract(seed=53102)

    assert contract["seed"] == 53102
    assert contract["uniform_across_arms"] is True
    assert contract["deadline_rule"]["legacy_1500_seconds_controls_probe"] is False


def test_q0r_training_lower_bound_defers_arm_without_eval_signal() -> None:
    features = {
        arm: {
            "bottleneck_feature_count": 0,
            "dense_compute": False,
            "full_sort_path": True,
            "graph_propagation": False,
            "routing_path": False,
            "sparse_compute": False,
        }
        for arm in ARM_ORDER
    }
    probes = {arm: _probe(batch_ms=2, peak_mib=100) for arm in ARM_ORDER}
    frontier = probes["frontier_candidate"]
    telemetry = frontier["resource_telemetry"]
    telemetry["batch_records"] = [
        {**row, "wall_time_ms": 15_000}
        for row in telemetry["batch_records"]
        if row["phase"] == "TRAIN"
    ]
    telemetry["full_train_batches_per_epoch"] = 390
    frontier["exit_status"] = "RESOURCE_CENSORED"
    frontier["wall_time_ms"] = 300_000

    decision = predict_resources(arm_features=features, probe_runs=probes)

    deferred = {row["arm"]: row for row in decision["deferred_arms"]}
    assert deferred["frontier_candidate"]["reason"] == (
        "TRAINING_LOWER_BOUND_EXCEEDS_CAMPAIGN_BUDGET"
    )
    assert deferred["frontier_candidate"]["resource_disposition"] == (
        "RESOURCE_DEFERRED"
    )
    assert deferred["frontier_candidate"]["mechanism_effect_update_allowed"] is False
    assert "frontier_candidate" not in {row["arm"] for row in decision["schedule"]}
    assert decision["predictions"]["frontier_candidate"]["estimate_scope"] == (
        "TRAINING_ONLY_RESOURCE_LOWER_BOUND"
    )


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
class Loader:
    def __iter__(self):
        yield "unused"
loader = Loader()
module._install_fixed_batch_iteration(
    loader, ("first", "second"), (0, 1),
    on_batch_started=started, on_batch_completed=completed,
)
for position, _batch in enumerate(loader):
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


def test_fixed_batch_iteration_preserves_full_sort_evaluator_type() -> None:
    import importlib.util

    worker = Path(__file__).resolve().parents[3] / "scripts/campaign_train_worker.py"
    spec = importlib.util.spec_from_file_location("campaign_train_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FullSortEvalDataLoader:
        def __iter__(self):
            yield ({"user_id": [1]}, None, [0], [1])

    class GenericView:
        def __init__(self, source: object) -> None:
            self.source = source

        def __iter__(self):
            yield from self.source

    def evaluator(loader: object) -> str:
        interaction, _history, _positive_u, _positive_i = next(iter(loader))
        if isinstance(loader, FullSortEvalDataLoader):
            return "FULL_SORT"
        return str(interaction["item_id"])

    loader = FullSortEvalDataLoader()
    with pytest.raises(KeyError, match="item_id"):
        evaluator(GenericView(loader))

    restore = module._install_fixed_batch_iteration(
        loader,
        (({"user_id": [1]}, None, [0], [1]),),
        (0,),
        on_batch_started=lambda *_args: None,
        on_batch_completed=lambda *_args: None,
    )
    try:
        assert isinstance(loader, FullSortEvalDataLoader)
        assert evaluator(loader) == "FULL_SORT"
    finally:
        restore()


@pytest.mark.skipif(
    not (RECBole_ROOT / "recbole").is_dir() or not SEARCH_DATA_ROOT.is_dir(),
    reason="exact RecBole search runtime is unavailable",
)
def test_real_recbole_full_sort_fixed_batch_keeps_item_id_semantics(
    tmp_path: Path,
) -> None:
    import importlib.util

    import numpy as np
    import torch

    for name, value in {
        "float_": np.float64,
        "int_": np.int64,
        "complex_": np.complex128,
        "unicode_": np.str_,
        "string_": np.bytes_,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(RECBole_ROOT))
    import run_candidate

    run_candidate.install_optional_dependency_stubs()
    run_candidate.patch_recbole_runtime_compat()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
    from recbole.utils import get_model, get_trainer, init_seed

    worker = root / "scripts/campaign_train_worker.py"
    spec = importlib.util.spec_from_file_location("campaign_train_worker", worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = Config(
        model="BPR",
        dataset="ml-1m",
        config_file_list=[
            str(RECBole_ROOT / "recbole/properties/model/BPR.yaml"),
            str(root / "configs/task_ml1m.yaml"),
            str(root / "configs/lightgcn_metrics.yaml"),
        ],
        config_dict={
            "benchmark_filename": ["train", "dev", "dev"],
            "checkpoint_dir": str(tmp_path),
            "data_path": str(SEARCH_DATA_ROOT),
            "epochs": 1,
            "eval_step": 1,
            "reproducibility": True,
            "seed": 54102,
            "show_progress": False,
            "state": "ERROR",
            "use_gpu": True,
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, _unused_test_data = data_preparation(config, dataset)
    assert isinstance(valid_data, FullSortEvalDataLoader)
    fixed_batch = module._preallocate_batches(valid_data, (0,))
    model = get_model(config["model"])(config, train_data._dataset).to(
        config["device"]
    )
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.eval_collector.data_collect(train_data)
    completed: list[int] = []
    restore = module._install_fixed_batch_iteration(
        valid_data,
        fixed_batch,
        (0,),
        on_batch_started=lambda *_args: None,
        on_batch_completed=lambda position, *_args: completed.append(position),
    )
    try:
        assert isinstance(valid_data, FullSortEvalDataLoader)
        valid_score, valid_result = trainer._valid_epoch(valid_data)
    finally:
        restore()

    assert torch.isfinite(torch.tensor(valid_score))
    assert "ndcg@10" in valid_result
    assert completed == [0]


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
