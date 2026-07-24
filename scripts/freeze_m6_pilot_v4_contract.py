#!/usr/bin/env python3
"""Freeze the one authorized post-M6R Pilot V4 contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    PilotStoreContractV2,
    pilot_budget,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    TrainingExecutionPurposeV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    TRAINING_RUNNER_ABI,
    training_runtime_release,
)


CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "DEVELOPMENT_PILOT_CONTRACT_V4.json"
)
PREDECESSOR_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "DEVELOPMENT_PILOT_CONTRACT_V3.json"
)
V3_FAILURE_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "M6_PILOT_V3_FAILURE_RECORD.json"
)
M6R_AUDIT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6_runtime_recovery"
    / "M6R_INDEPENDENT_AUDIT.md"
)
MODELS_CACHE = Path("/mnt/c/Users/gtrho/.codex/models_cache.json")


SOURCE_FILES = (
    "scripts/pilot_train_worker.py",
    "scripts/run_m6_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/canary_broker.py",
    "src/recclaw_core/experiments/helix_abc_v1/common_execution_guard.py",
    "src/recclaw_core/experiments/helix_abc_v1/materialization.py",
    "src/recclaw_core/experiments/helix_abc_v1/pilot_analysis.py",
    "src/recclaw_core/experiments/helix_abc_v1/pilot_training.py",
    "src/recclaw_core/experiments/helix_abc_v1/precanary_orchestration.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_canary.py",
    "src/recclaw_core/experiments/helix_abc_v1/real_pilot.py",
    "src/recclaw_core/experiments/helix_abc_v1/research_controller.py",
    "src/recclaw_core/experiments/helix_abc_v1/runtime_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/state_store.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_execution_guard.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_materialization.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_contracts.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_runtime_release.py",
    "src/recclaw_core/experiments/helix_abc_v1/training_state_store.py",
    "src/recclaw_core/helix/composition.py",
    "src/recclaw_core/helix/contracts.py",
    "src/recclaw_core/helix/fusion.py",
    "src/recclaw_core/helix/guard_adapter.py",
    "src/recclaw_core/helix/ledger.py",
    "src/recclaw_core/helix/ports.py",
    "tests/experiments/helix_abc_v1/test_m3_helix_composition.py",
    "tests/experiments/helix_abc_v1/test_m6_pilot.py",
    "tests/experiments/helix_abc_v1/test_m6r_training_runtime.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_contract() -> dict[str, Any]:
    predecessor = json.loads(PREDECESSOR_PATH.read_text(encoding="utf-8"))
    contract = {
        key: value
        for key, value in predecessor.items()
        if key not in {"content_digest", "repair_predecessor"}
    }
    store = PilotStoreContractV2.create()
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce="M6-PILOT-9204-OPAQUE-V4",
    )
    release = training_runtime_release()
    source_files = {
        relative: file_sha256(ROOT / relative)
        for relative in SOURCE_FILES
    }
    models_cache = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))
    contract.update(
        {
            "assignment": {
                "commitment": assignment.commitment,
                "opaque": True,
            },
            "budget_per_arm_round": pilot_budget().to_dict(),
            "historical_development_costs": {
                "V1": {
                    "broker_calls": 1,
                    "broker_input_tokens": 13361,
                    "broker_output_tokens": 828,
                    "broker_tokens": 14189,
                    "ordinary_execution_debit": 0,
                    "search_seed": 9201,
                    "training_backend_starts": 0,
                },
                "V2": {
                    "broker_calls": 4,
                    "broker_input_tokens": 53470,
                    "broker_output_tokens": 1155,
                    "broker_tokens": 54625,
                    "ordinary_execution_debit": 0,
                    "search_seed": 9202,
                    "training_backend_starts": 0,
                },
                "V3": {
                    "broker_calls": 1,
                    "broker_input_tokens": 13361,
                    "broker_output_tokens": 809,
                    "broker_tokens": 14170,
                    "ordinary_execution_debit": 0,
                    "search_seed": 9203,
                    "training_backend_starts": 0,
                },
            },
            "lineage_predecessors": {
                "m6r_audit_sha256": file_sha256(M6R_AUDIT_PATH),
                "m6r_checkpoint_commit": git_value("rev-parse", "HEAD"),
                "m6r_checkpoint_tree": git_value(
                    "rev-parse", "HEAD^{tree}"
                ),
                "v3_contract_content_digest": predecessor["content_digest"],
                "v3_contract_sha256": file_sha256(PREDECESSOR_PATH),
                "v3_failure_record_sha256": file_sha256(V3_FAILURE_PATH),
            },
            "pilot": {
                "experiment_id": store.experiment_id,
                "ordinary_execution_seed": store.ordinary_execution_seed,
                "rounds_per_arm": store.scheduled_slots_per_arm_seed,
                "search_seeds": list(store.search_seeds),
                "store_contract_identity_digest": store.identity_digest,
            },
            "record_schema": "recclaw.development-pilot-contract.v2",
            "source": {
                "base_checkpoint_commit": git_value("rev-parse", "HEAD"),
                "base_checkpoint_tree": git_value(
                    "rev-parse", "HEAD^{tree}"
                ),
                "files": source_files,
                "source_projection_digest": sha256_digest(source_files),
            },
            "status": "FROZEN_PRE_OUTCOME",
            "training": {
                "execution_purpose": TrainingExecutionPurposeV1.PILOT.value,
                "profile_path": predecessor["training"]["profile_path"],
                "profile_sha256": predecessor["training"]["profile_sha256"],
                "runner_abi": TRAINING_RUNNER_ABI,
                "runtime_release_digest": release.digest,
                "runtime_release_schema": release.record_type,
            },
        }
    )
    contract["analysis"]["code_sha256"] = source_files[
        "src/recclaw_core/experiments/helix_abc_v1/pilot_analysis.py"
    ]
    contract["broker"]["models_cache_etag"] = models_cache["etag"]
    contract["broker"]["models_cache_sha256"] = file_sha256(MODELS_CACHE)
    contract["runtime"]["rfc8785"] = "0.1.4"
    contract["content_digest"] = sha256_digest(contract)
    return contract


def main() -> int:
    if CONTRACT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite frozen contract: {CONTRACT_PATH}")
    contract = build_contract()
    CONTRACT_PATH.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "contract_path": str(CONTRACT_PATH),
                "contract_sha256": file_sha256(CONTRACT_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
