"""Run and neutrally audit the fixed three-seed LightGCN comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from recclaw_phase1.ab002_launcher import CONTRACT, canonical_write
from recclaw_phase1.ab002_preflight import exact_file_identities, tree_identity
from recclaw_phase1.ab002_s0 import copy_s0_source
from recclaw_phase1.neutral_outcome_auditor import audit_raw_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json"
SEEDS = (2026, 2027, 2028)
BASELINE_REGISTRY = """candidates:
  - candidate_id: ab002_fixed_lightgcn_comparator
    category: frozen_comparator
    base_model: LightGCN
    implementation_type: config
    runner_type: config_only
    status: implemented
    wired: true
    entrypoint: recbole.model.general_recommender.lightgcn:LightGCN
    consumes: []
"""


def summarize_metric(values: list[float]) -> dict[str, Any]:
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("three finite comparator metric values are required")
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        "mean": statistics.fmean(values),
        "population_variance": statistics.pvariance(values),
        "seed_count": 3,
    }


def run_baseline(*, runtime_root: Path, python_executable: str) -> dict[str, Any]:
    if os.environ.get("RECCLAW_AB002_FULL_START_AUTHORIZED") != "YES":
        raise ValueError("three-seed comparator requires external Canary GO")
    root = runtime_root.expanduser().resolve()
    comparator_root = root / "three_seed_lightgcn_comparator"
    if comparator_root.exists():
        raise FileExistsError(f"comparator root must be absent: {comparator_root}")
    source_root = comparator_root / "source"
    source_root.mkdir(parents=True)
    source_files = copy_s0_source(source_root)
    registry = comparator_root / "fixed_lightgcn_registry.yaml"
    registry.write_text(BASELINE_REGISTRY, encoding="utf-8")
    result_dir = comparator_root / "raw_runs"
    results_csv = comparator_root / "results.csv"
    override_dir = comparator_root / "overrides"
    checkpoint_dir = comparator_root / "checkpoints"
    ledger = comparator_root / "candidate_execution_budget.jsonl"
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    recbole_root = Path(contract["execution_environment"]["recbole_root"])
    recbole_before = exact_file_identities(
        recbole_root, contract["execution_environment"]["recbole_runtime_files_sha256"]
    )
    dataset_root = recbole_root / "dataset" / contract["protocol"]["dataset"]
    dataset_before, _ = tree_identity(dataset_root)
    environment = dict(os.environ)
    environment.update(
        {
            "RECBOLE_ROOT": str(recbole_root),
            "PYTHONNOUSERSITE": "1",
            "RECCLAW_CANDIDATE_EXECUTION_BUDGET": "3",
            "RECCLAW_CANDIDATE_EXECUTION_LEDGER": str(ledger),
        }
    )
    runs = []
    ndcg_values = []
    for index, seed in enumerate(SEEDS, start=1):
        before_json = set(result_dir.glob("*.json")) if result_dir.exists() else set()
        command = [
            python_executable,
            str(source_root / "scripts/run_candidate.py"),
            "ab002_fixed_lightgcn_comparator",
            "--registry-path",
            str(registry),
            "--base-model",
            "LightGCN",
            "--recbole-root",
            str(recbole_root),
            "--result-dir",
            str(result_dir),
            "--results-csv",
            str(results_csv),
            "--override-dir",
            str(override_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--cleanup-checkpoints",
            "--execution-purpose",
            "baseline",
            "--set",
            f"seed={seed}",
        ]
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        execution_log = comparator_root / f"launcher_seed_{seed}.log"
        execution_log.write_text(completed.stdout, encoding="utf-8")
        after_json = set(result_dir.glob("*.json"))
        created = sorted(after_json - before_json)
        if completed.returncode != 0 or len(created) != 1:
            raise RuntimeError(
                f"LightGCN comparator seed {seed} failed: rc={completed.returncode}, artifacts={len(created)}"
            )
        result_path = created[0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        log_path = Path(result["log_path"])
        envelope = {
            "anonymous_run_id": f"AB002-BLIND-COMPARATOR-{index}",
            "log_path": str(log_path),
            "result_json_path": str(result_path),
            "planned_training_seed": seed,
            "dataset_snapshot_sha256": f"sha256:{dataset_before}",
            "shared_state_integrity": {
                "dataset_before_sha256": f"sha256:{dataset_before}",
                "dataset_after_sha256": f"sha256:{tree_identity(dataset_root)[0]}",
                "recbole_files_before": recbole_before,
                "recbole_files_after": exact_file_identities(
                    recbole_root,
                    contract["execution_environment"]["recbole_runtime_files_sha256"],
                ),
            },
            "candidate_artifacts": [
                {
                    "role": "candidate_config",
                    "path": str(source_root / "configs/lightgcn.yaml"),
                    "sha256": hashlib.sha256(
                        (source_root / "configs/lightgcn.yaml").read_bytes()
                    ).hexdigest(),
                },
                {
                    "role": "candidate_source",
                    "path": str(recbole_root / "recbole/model/general_recommender/lightgcn.py"),
                    "sha256": recbole_before[
                        "recbole/model/general_recommender/lightgcn.py"
                    ],
                },
            ],
        }
        audit = audit_raw_run(envelope, policy)
        if not audit["eligible_for_performance_analysis"]:
            raise RuntimeError(f"comparator seed {seed} failed neutral audit: {audit['diagnostics']}")
        value = float(result["ndcg@10"])
        ndcg_values.append(value)
        runs.append(
            {
                "training_seed": seed,
                "run_id": result["run_id"],
                "result_json_path": str(result_path),
                "log_path": str(log_path),
                "neutral_audit": audit,
                "launcher_log_path": str(execution_log),
            }
        )
    return {
        "record_type": "RECCLAW_PHASE1_AB002_THREE_SEED_BASELINE_REPORT",
        "schema_version": "recclaw.phase1.ab002.three_seed_baseline_report.v1",
        "status": "LOCAL_COMPLETE",
        "gate_status": "NOT_STARTED",
        "comparator": "LightGCN",
        "protocol_id": contract["protocol"]["protocol_id"],
        "seeds": list(SEEDS),
        "ndcg_at_10": summarize_metric(ndcg_values),
        "runs": runs,
        "source_files": source_files,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed AB-002 three-seed LightGCN comparator")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_baseline(runtime_root=args.runtime_root, python_executable=args.python_executable)
    canonical_write(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
