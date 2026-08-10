from __future__ import annotations

import ast
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
    _round_test_feedback_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "scripts" / "campaign_train_worker.py"


def _worker_source() -> str:
    return WORKER_PATH.read_text(encoding="utf-8")


def _worker_main(source: str) -> ast.FunctionDef:
    tree = ast.parse(source, filename=str(WORKER_PATH))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _constant_bool(value: ast.AST, expected: bool) -> bool:
    return isinstance(value, ast.Constant) and value.value is expected


def test_worker_uses_proven_single_process_recbole_loader() -> None:
    source = _worker_source()
    assert "set_start_method" not in source
    assert '"worker": 0' in source


def test_worker_runs_saved_best_checkpoint_test_evaluation() -> None:
    source = _worker_source()
    main = _worker_main(source)

    fit_calls = tuple(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    )
    assert any(
        any(
            keyword.arg == "saved" and _constant_bool(keyword.value, True)
            for keyword in call.keywords
        )
        for call in fit_calls
    )

    evaluate_calls = tuple(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate"
    )
    assert any(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "test_data"
        and any(
            keyword.arg == "load_best_model"
            and _constant_bool(keyword.value, True)
            for keyword in call.keywords
        )
        for call in evaluate_calls
    )

    assert '"benchmark_filename": ["train", "dev", "heldout"]' in source
    assert '"eval_batch_size": 65536' in source
    assert '"worker": 0' in source
    assert '"metric_source": "BEST_CHECKPOINT_TEST_RESULT"' in source
    assert '"online_partition_role": "ROUND_TEST_FEEDBACK"' in source
    assert '"test_result": test_result' in source


def test_round_feedback_contract_is_consumed_by_leased_artifacts() -> None:
    profile = json.loads(
        (
            REPO_ROOT
            / "src/recclaw_core/experiments/helix_abc_v1/resources"
            / "campaign_training_profile_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["benchmark_filename"] == ["train", "dev", "heldout"]
    assert profile["partition_profile_id"] == (
        "RECCLAW_ML1M_ROUND_TEST_FEEDBACK_PARTITION_V1"
    )
    assert profile["online_metric_source"] == "BEST_CHECKPOINT_TEST_RESULT"
    assert profile["online_partition_role"] == "ROUND_TEST_FEEDBACK"
    assert profile["heldout_access"] == "AFTER_BEST_CHECKPOINT_SELECTION"

    config = (REPO_ROOT / "configs/task_ml1m.yaml").read_text(encoding="utf-8")
    assert "eval_batch_size: 65536" in config
    assert "worker: 0" in config

    single_round = (
        REPO_ROOT / "src/recclaw_core/research_line/single_round.py"
    ).read_text(encoding="utf-8")
    assert '"held_out_reads": 0' not in single_round
    assert "ML-1M train/dev offline full-sort" not in single_round
    assert '"online_partition_role": "ROUND_TEST_FEEDBACK"' in single_round
    assert 'worker.get("test_result", {})' in single_round

    pilot_training = (
        REPO_ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py"
    ).read_text(encoding="utf-8")
    assert 'worker.get("test_result", {})' in pilot_training

    metrics, matches = _round_test_feedback_metrics(
        {
            "best_valid_result": {"ndcg@10": 0.21},
            "test_result": {"ndcg@10": 0.29},
            "metric_source": "BEST_CHECKPOINT_TEST_RESULT",
            "online_partition_role": "ROUND_TEST_FEEDBACK",
        }
    )
    assert matches is True
    assert metrics == {"ndcg@10": 0.29}

    metrics, matches = _round_test_feedback_metrics(
        {
            "test_result": {"ndcg@10": 0.31},
            "metric_source": "BEST_VALID_RESULT",
            "online_partition_role": "DEVELOPMENT_VALIDATION",
        }
    )
    assert matches is False
    assert metrics == {}

    campaign_runtime = (
        REPO_ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/campaign_runtime.py"
    ).read_text(encoding="utf-8")
    assert '"BEST_CHECKPOINT_TEST_RESULT"' in campaign_runtime
    assert '"ROUND_TEST_FEEDBACK"' in campaign_runtime
    assert '"AFTER_BEST_CHECKPOINT_SELECTION"' in campaign_runtime

    training_release = (
        REPO_ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/training_runtime_release.py"
    ).read_text(encoding="utf-8")
    assert '"BEST_CHECKPOINT_TEST_RESULT"' in training_release
