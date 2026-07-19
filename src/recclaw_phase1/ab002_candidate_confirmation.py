"""Arm-blind selection and three-seed confirmation for one AB-002 run packet."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from recclaw_phase1.ab002_launcher import CONTRACT, canonical_write, sha256_file
from recclaw_phase1.ab002_preflight import exact_file_identities, tree_identity
from recclaw_phase1.neutral_outcome_auditor import audit_raw_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json"
SEEDS = (2026, 2027, 2028)
CANDIDATE_NOTE = re.compile(r"(?:^|\s|\|)candidate_id=([^|\s]+)")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _registry_entry(path: Path, candidate_id: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for row in value.get("candidates", []):
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id:
            return row
    raise ValueError(f"selected candidate absent from registry: {candidate_id}")


def _artifact_paths(source: Path, recbole_root: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    candidate_id = str(entry["candidate_id"])
    base_model = str(entry["base_model"])
    candidate_config = source / "configs/candidates" / f"{candidate_id}.yaml"
    if not candidate_config.is_file():
        candidate_config = source / "configs" / ("lightgcn.yaml" if base_model == "LightGCN" else "bpr.yaml")
    runner_type = str(entry.get("runner_type") or "")
    if runner_type == "model":
        module = str(entry["entrypoint"]).split(":", 1)[0]
        candidate_source = source / (module.replace(".", "/") + ".py")
    else:
        candidate_source = recbole_root / (
            "recbole/model/general_recommender/lightgcn.py"
            if base_model == "LightGCN"
            else "recbole/model/general_recommender/bpr.py"
        )
    if not candidate_config.is_file() or not candidate_source.is_file():
        raise ValueError(f"candidate artifact closure incomplete: {candidate_id}")
    return candidate_config, candidate_source


def _envelope(
    *,
    blind_id: str,
    seed: int,
    result_path: Path,
    log_path: Path,
    candidate_config: Path,
    candidate_source: Path,
    dataset_snapshot: str,
    recbole_files: dict[str, str],
) -> dict[str, Any]:
    return {
        "anonymous_run_id": f"{blind_id}-SEED-{seed}",
        "log_path": str(log_path),
        "result_json_path": str(result_path),
        "planned_training_seed": seed,
        "dataset_snapshot_sha256": dataset_snapshot,
        "shared_state_integrity": {
            "dataset_before_sha256": dataset_snapshot,
            "dataset_after_sha256": dataset_snapshot,
            "recbole_files_before": recbole_files,
            "recbole_files_after": recbole_files,
        },
        "candidate_artifacts": [
            {"role": "candidate_config", "path": str(candidate_config), "sha256": sha256_file(candidate_config)},
            {"role": "candidate_source", "path": str(candidate_source), "sha256": sha256_file(candidate_source)},
        ],
    }


def select_seed_2026_candidate(
    *, packet_root: Path, blind_id: str, policy: dict[str, Any], recbole_files: dict[str, str]
) -> dict[str, Any]:
    pilot = packet_root / "pilot"
    source = packet_root / "source"
    proposed = {
        str(row.get("candidate_id"))
        for row in _jsonl(pilot / "candidate_proposals.jsonl")
        if row.get("candidate_id")
    }
    candidates = []
    with (pilot / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("status") or "").lower() != "success":
            continue
        matched = CANDIDATE_NOTE.search(str(row.get("notes") or ""))
        if matched is None or matched.group(1) not in proposed:
            continue
        candidate_id = matched.group(1)
        run_id = str(row.get("run_id") or "")
        result_path = pilot / "candidates" / f"{run_id}.json"
        log_path = pilot / "candidates" / f"{run_id}.log"
        entry = _registry_entry(source / "configs/candidate_registry.yaml", candidate_id)
        recbole_root = Path(json.loads(CONTRACT.read_text())["execution_environment"]["recbole_root"])
        config, implementation = _artifact_paths(source, recbole_root, entry)
        envelope = _envelope(
            blind_id=blind_id,
            seed=2026,
            result_path=result_path,
            log_path=log_path,
            candidate_config=config,
            candidate_source=implementation,
            dataset_snapshot=policy["dataset_snapshot_sha256"],
            recbole_files=recbole_files,
        )
        audit = audit_raw_run(envelope, policy)
        value = float(row["ndcg@10"])
        if audit["eligible_for_performance_analysis"] and math.isfinite(value):
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "base_model": entry["base_model"],
                    "seed_2026_ndcg_at_10": value,
                    "seed_2026_result_path": str(result_path),
                    "seed_2026_log_path": str(log_path),
                    "candidate_config_path": str(config),
                    "candidate_source_path": str(implementation),
                    "neutral_audit": audit,
                }
            )
    candidates.sort(key=lambda item: (-item["seed_2026_ndcg_at_10"], item["candidate_id"]))
    return {
        "selection_status": "SELECTED_NEW_CANDIDATE" if candidates else "BASELINE_FALLBACK",
        "eligible_candidate_count": len(candidates),
        "selected": candidates[0] if candidates else None,
        "selection_tie_break": "highest seed-2026 NDCG@10 then lexicographically smallest candidate_id",
    }


def confirm_selected(
    *, packet_root: Path, blind_id: str, python_executable: str
) -> dict[str, Any]:
    if os.environ.get("RECCLAW_AB002_FULL_START_AUTHORIZED") != "YES":
        raise ValueError("candidate confirmation requires external Canary GO")
    packet = packet_root.expanduser().resolve()
    if any(component.lower() in {"control", "treatment"} for component in packet.parts):
        raise ValueError("neutral confirmation packet path contains an arm-label component")
    manifest = json.loads((packet / "blind_packet_manifest.json").read_text(encoding="utf-8"))
    blind_id = str(manifest["blind_id"])
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    recbole_root = Path(contract["execution_environment"]["recbole_root"])
    recbole_files = exact_file_identities(
        recbole_root, contract["execution_environment"]["recbole_runtime_files_sha256"]
    )
    dataset_root = recbole_root / "dataset" / contract["protocol"]["dataset"]
    dataset_tree, _ = tree_identity(dataset_root)
    if f"sha256:{dataset_tree}" != policy["dataset_snapshot_sha256"]:
        raise ValueError("dataset identity drift before confirmation")
    selection = select_seed_2026_candidate(
        packet_root=packet, blind_id=blind_id, policy=policy, recbole_files=recbole_files
    )
    if selection["selected"] is None:
        return {
            "record_type": "RECCLAW_PHASE1_AB002_BLIND_CONFIRMATION",
            "schema_version": "recclaw.phase1.ab002.blind_confirmation.v1",
            "status": "LOCAL_COMPLETE",
            "gate_status": "NOT_STARTED",
            "blind_id": blind_id,
            **selection,
            "frontier_policy": "USE_FIXED_THREE_SEED_LIGHTGCN_COMPARATOR",
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
        }
    selected = selection["selected"]
    source = packet / "source"
    confirmation = packet / "confirmation"
    if confirmation.exists():
        raise FileExistsError("confirmation root must be absent")
    result_dir = confirmation / "raw_runs"
    results_csv = confirmation / "results.csv"
    registry = source / "configs/candidate_registry.yaml"
    ledger = confirmation / "candidate_execution_budget.jsonl"
    env = dict(os.environ)
    env.update(
        {
            "RECBOLE_ROOT": str(recbole_root),
            "PYTHONNOUSERSITE": "1",
            "RECCLAW_CANDIDATE_EXECUTION_BUDGET": "2",
            "RECCLAW_CANDIDATE_EXECUTION_LEDGER": str(ledger),
        }
    )
    seed_runs = [
        {
            "training_seed": 2026,
            "result_path": Path(selected["seed_2026_result_path"]),
            "log_path": Path(selected["seed_2026_log_path"]),
            "audit": selected["neutral_audit"],
            "ndcg@10": selected["seed_2026_ndcg_at_10"],
        }
    ]
    for seed in (2027, 2028):
        before = set(result_dir.glob("*.json")) if result_dir.exists() else set()
        command = [
            python_executable,
            str(source / "scripts/run_candidate.py"),
            selected["candidate_id"],
            "--registry-path",
            str(registry),
            "--base-model",
            selected["base_model"],
            "--recbole-root",
            str(recbole_root),
            "--result-dir",
            str(result_dir),
            "--results-csv",
            str(results_csv),
            "--override-dir",
            str(confirmation / "overrides"),
            "--checkpoint-dir",
            str(confirmation / "checkpoints"),
            "--cleanup-checkpoints",
            "--execution-purpose",
            "confirmation",
            "--set",
            f"seed={seed}",
        ]
        completed = subprocess.run(
            command,
            cwd=source,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (confirmation / f"launcher_seed_{seed}.log").parent.mkdir(parents=True, exist_ok=True)
        (confirmation / f"launcher_seed_{seed}.log").write_text(completed.stdout, encoding="utf-8")
        created = sorted(set(result_dir.glob("*.json")) - before)
        if completed.returncode != 0 or len(created) != 1:
            raise RuntimeError(f"confirmation seed {seed} failed")
        result_path = created[0]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        log_path = result_dir / f"{result['run_id']}.log"
        audit = audit_raw_run(
            _envelope(
                blind_id=blind_id,
                seed=seed,
                result_path=result_path,
                log_path=log_path,
                candidate_config=Path(selected["candidate_config_path"]),
                candidate_source=Path(selected["candidate_source_path"]),
                dataset_snapshot=policy["dataset_snapshot_sha256"],
                recbole_files=recbole_files,
            ),
            policy,
        )
        if not audit["eligible_for_performance_analysis"]:
            raise RuntimeError(f"confirmation seed {seed} failed neutral audit: {audit['diagnostics']}")
        seed_runs.append(
            {
                "training_seed": seed,
                "result_path": result_path,
                "log_path": log_path,
                "audit": audit,
                "ndcg@10": float(result["ndcg@10"]),
            }
        )
    values = [float(run["ndcg@10"]) for run in seed_runs]
    return {
        "record_type": "RECCLAW_PHASE1_AB002_BLIND_CONFIRMATION",
        "schema_version": "recclaw.phase1.ab002.blind_confirmation.v1",
        "status": "LOCAL_COMPLETE",
        "gate_status": "NOT_STARTED",
        "blind_id": blind_id,
        **selection,
        "confirmation_seeds": list(SEEDS),
        "three_seed_ndcg_at_10": {
            "values": values,
            "mean": statistics.fmean(values),
            "population_variance": statistics.pvariance(values),
        },
        "seed_runs": [
            {**run, "result_path": str(run["result_path"]), "log_path": str(run["log_path"])}
            for run in seed_runs
        ],
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and confirm one blinded AB-002 frontier")
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = confirm_selected(
        packet_root=args.packet_root,
        blind_id="RESOLVED_FROM_PACKET",
        python_executable=args.python_executable,
    )
    canonical_write(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
