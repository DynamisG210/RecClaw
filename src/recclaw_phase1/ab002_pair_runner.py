"""Materialize and run one contemporaneous AB-002 Control/Treatment pair."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urlrequest

from recclaw_phase1.ab002_launcher import (
    CANARY_SEED,
    FULL_SEARCH_SEEDS,
    _execute,
    build_launch_plan,
    canonical_write,
    expected_gpu_id,
    expected_pair_id,
    materialize_arm,
    validate_baseline_inputs,
    validate_runtime_root,
)


def broker_health(base_url: str) -> dict[str, object]:
    with urlrequest.urlopen(f"{base_url.rstrip('/')}/health", timeout=5.0) as response:
        body = response.read()
        value = json.loads(body)
        if response.status != 200 or not isinstance(value, dict) or value.get("status") != "ok":
            raise ValueError("paired broker health response is invalid")
        return {"status_code": int(response.status), "response": value}


def run_pair(
    *,
    runtime_root: Path,
    search_seed: int,
    broker_url: str,
    baseline_dir: Path,
    python_executable: str,
    client_token_env: str,
) -> dict[str, object]:
    root = validate_runtime_root(runtime_root)
    if os.environ.get("RECCLAW_AB002_START_AUTHORIZED") != "YES":
        raise ValueError("pair execution requires RECCLAW_AB002_START_AUTHORIZED=YES")
    if search_seed in FULL_SEARCH_SEEDS and os.environ.get("RECCLAW_AB002_FULL_START_AUTHORIZED") != "YES":
        raise ValueError("Full A/B pair execution requires external Canary GO and RECCLAW_AB002_FULL_START_AUTHORIZED=YES")
    if search_seed not in (CANARY_SEED, *FULL_SEARCH_SEEDS):
        raise ValueError(f"unsupported AB-002 search seed: {search_seed}")
    if not os.environ.get(client_token_env, ""):
        raise ValueError(f"missing broker client token env: {client_token_env}")
    baseline_rows = validate_baseline_inputs(baseline_dir)
    health = broker_health(broker_url)
    pair_id = expected_pair_id(search_seed)
    plans = {}
    materializations = {}
    for arm in ("control", "treatment"):
        gpu_id = expected_gpu_id(search_seed, arm)
        materializations[arm] = materialize_arm(
            runtime_root=root,
            arm=arm,
            search_seed=search_seed,
        )
        plans[arm] = build_launch_plan(
            runtime_root=root,
            arm=arm,
            search_seed=search_seed,
            gpu_id=gpu_id,
            pair_id=pair_id,
            broker_url=broker_url,
            baseline_dir=baseline_dir,
            python_executable=python_executable,
        )

    record_path = root / "execution_records" / f"pair_search_seed_{search_seed}.json"
    record: dict[str, object] = {
        "record_type": "RECCLAW_PHASE1_AB002_PAIR_EXECUTION",
        "schema_version": "recclaw.phase1.ab002.pair_execution.v1",
        "status": "IN_PROGRESS",
        "search_seed": search_seed,
        "pair_id": pair_id,
        "canary": search_seed == CANARY_SEED,
        "included_in_primary_outcomes": search_seed in FULL_SEARCH_SEEDS,
        "broker_health": health,
        "baseline_inputs": baseline_rows,
        "materialization_manifests": {
            arm: str(Path(value["source_root"]).parent / "materialization_manifest.json")
            for arm, value in materializations.items()
        },
        "plans": plans,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    canonical_write(record_path, record)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"ab002-{search_seed}") as pool:
        futures = {
            arm: pool.submit(_execute, plan, client_token_env=client_token_env)
            for arm, plan in plans.items()
        }
        exit_codes = {arm: future.result() for arm, future in futures.items()}
    record["exit_codes"] = exit_codes
    record["status"] = "LOCAL_COMPLETE" if all(code == 0 for code in exit_codes.values()) else "STOPPED"
    canonical_write(record_path, record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one contemporaneous AB-002 pair")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--search-seed", type=int, choices=(CANARY_SEED, *FULL_SEARCH_SEEDS), required=True)
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--client-token-env", default="RECCLAW_BROKER_CLIENT_TOKEN")
    args = parser.parse_args(argv)
    result = run_pair(
        runtime_root=args.runtime_root,
        search_seed=args.search_seed,
        broker_url=args.broker_url,
        baseline_dir=args.baseline_dir,
        python_executable=args.python_executable,
        client_token_env=args.client_token_env,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "LOCAL_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
