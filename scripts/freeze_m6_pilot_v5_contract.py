#!/usr/bin/env python3
"""Freeze the one authorized post-M6E Pilot V5 contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from freeze_m6_pilot_v4_contract import (  # noqa: E402
    SOURCE_FILES as V4_SOURCE_FILES,
    file_sha256,
    git_value,
)

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    FRESH_PILOT_V5_SEARCH_SEED,
    PilotStoreContractV3,
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
    / "DEVELOPMENT_PILOT_CONTRACT_V5.json"
)
M6E_CHECKPOINT_COMMIT = "57c9d520cd3788ab355acdef4f8a5ced81e8585d"
M6E_CHECKPOINT_TREE = "85cfcbea43400b7406be4cbdf2fd84dd1a3252b3"
V4_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "DEVELOPMENT_PILOT_CONTRACT_V4.json"
)
V4_FAILURE_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "M6_PILOT_V4_FAILURE_RECORD.json"
)
M6E_PACKET_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6e"
    / "M6E_TRAINING_RUNTIME_CONFORMANCE_PACKET.json"
)
M6E_AUDIT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6e"
    / "M6E_INDEPENDENT_AUDIT.md"
)
M6E_EXECUTION_RECORD_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6e"
    / "M6E_EXECUTION_RECORD.json"
)
MODELS_CACHE = (
    ROOT
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "pilot_v5_model_catalog_snapshot.json"
)

SOURCE_FILES = tuple(
    sorted(
        set(V4_SOURCE_FILES)
        | {
            "scripts/freeze_m6_pilot_v5_contract.py",
            "scripts/run_m6_pilot_v5.py",
            "src/recclaw_core/experiments/helix_abc_v1/m6e_conformance.py",
            "src/recclaw_core/experiments/helix_abc_v1/store_audit.py",
            "src/recclaw_core/experiments/helix_abc_v1/training_filesystem.py",
            "src/recclaw_core/experiments/helix_abc_v1/resources/training_runtime_release_v2.json",
            "src/recclaw_core/experiments/helix_abc_v1/resources/training_runtime_v2_lock.json",
            "src/recclaw_core/experiments/helix_abc_v1/resources/pilot_v5_model_catalog_snapshot.json",
            "tests/experiments/helix_abc_v1/test_m6e_environment_closure.py",
        }
    )
)


def _sealed_seed_registry() -> list[int]:
    seeds = []
    for version in range(1, 5):
        path = (
            ROOT
            / "docs"
            / "research_line"
            / "m6"
            / f"DEVELOPMENT_PILOT_CONTRACT_V{version}.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        seeds.extend(int(seed) for seed in document["pilot"]["search_seeds"])
    return sorted(set(seeds))


def build_contract() -> dict[str, Any]:
    predecessor = json.loads(V4_CONTRACT_PATH.read_text(encoding="utf-8"))
    failure = json.loads(V4_FAILURE_PATH.read_text(encoding="utf-8"))
    m6e_packet = json.loads(M6E_PACKET_PATH.read_text(encoding="utf-8"))
    if (
        m6e_packet["verdict"] != "PASS"
        or m6e_packet["P0"] != 0
        or m6e_packet["P1"] != 0
    ):
        raise RuntimeError("refusing to freeze Pilot V5 before M6E PASS")
    sealed_seeds = _sealed_seed_registry()
    if (
        sealed_seeds != [9201, 9202, 9203, 9204]
        or FRESH_PILOT_V5_SEARCH_SEED != max(sealed_seeds) + 1
    ):
        raise RuntimeError("Pilot V5 seed is not the smallest unused seed")

    contract = {
        key: value
        for key, value in predecessor.items()
        if key not in {"content_digest", "repair_predecessor"}
    }
    store = PilotStoreContractV3.create()
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id,
        nonce="M6-PILOT-9205-OPAQUE-V5",
    )
    release = training_runtime_release()
    source_files = {
        relative: file_sha256(ROOT / relative)
        for relative in SOURCE_FILES
    }
    models_cache = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))
    historical = dict(predecessor["historical_development_costs"])
    attempted = failure["attempted_execution"]
    historical["V4"] = {
        "broker_calls": int(attempted["broker_calls_total"]),
        "broker_input_tokens": int(attempted["broker_input_tokens"]),
        "broker_output_tokens": int(attempted["broker_output_tokens"]),
        "broker_tokens": int(attempted["broker_total_tokens"]),
        "gpu_cost_microunits": int(attempted["gpu_cost_microunits"]),
        "gpu_device_time_ms": int(attempted["gpu_device_time_ms"]),
        "ordinary_execution_debit": int(
            attempted["ordinary_execution_debit"]
        ),
        "search_seed": int(attempted["search_seed"]),
        "training_backend_starts": int(
            attempted["training_backend_starts"]
        ),
    }
    runtime = dict(predecessor["runtime"])
    runtime.update(
        {
            "conda_explicit_digest": release.python_environment_lock_digest,
            "python": (
                "/root/projects/RecClaw_m6_training_runtime_v2/bin/python"
            ),
            "python_environment_lock_digest": (
                release.python_environment_lock_digest
            ),
            "versions": {
                "cuda": True,
                "numpy": release.backend_identity[
                    "python_package_versions"
                ]["numpy"],
                "python": release.backend_identity["python_version"],
                "recbole": release.backend_identity["recbole_version"],
                "scipy": release.backend_identity[
                    "python_package_versions"
                ]["scipy"],
                "torch": release.backend_identity[
                    "torch_cuda_environment"
                ]["torch_version"],
            },
        }
    )
    contract.update(
        {
            "assignment": {
                "commitment": assignment.commitment,
                "opaque": True,
            },
            "budget_per_arm_round": pilot_budget().to_dict(),
            "historical_development_costs": historical,
            "lineage_predecessors": {
                "m6e_audit_sha256": file_sha256(M6E_AUDIT_PATH),
                "m6e_checkpoint_commit": M6E_CHECKPOINT_COMMIT,
                "m6e_checkpoint_tree": M6E_CHECKPOINT_TREE,
                "m6e_execution_record_sha256": file_sha256(
                    M6E_EXECUTION_RECORD_PATH
                ),
                "m6e_packet_content_digest": m6e_packet["content_digest"],
                "m6e_packet_sha256": file_sha256(M6E_PACKET_PATH),
                "v4_contract_content_digest": predecessor["content_digest"],
                "v4_contract_sha256": file_sha256(V4_CONTRACT_PATH),
                "v4_failure_record_sha256": file_sha256(V4_FAILURE_PATH),
            },
            "m6e": {
                "P0": 0,
                "P1": 0,
                "P2": int(m6e_packet["P2"]),
                "conformance_packet_digest": m6e_packet["content_digest"],
                "independent_audit_sha256": m6e_packet[
                    "independent_audit_sha256"
                ],
                "runtime_release_digest": (
                    m6e_packet["training_runtime_release_digest"]
                ),
            },
            "pilot": {
                "experiment_id": store.experiment_id,
                "ordinary_execution_seed": store.ordinary_execution_seed,
                "rounds_per_arm": store.scheduled_slots_per_arm_seed,
                "search_seeds": list(store.search_seeds),
                "store_contract_identity_digest": store.identity_digest,
            },
            "record_schema": "recclaw.development-pilot-contract.v3",
            "runtime": runtime,
            "sealed_pilot_seed_registry": sealed_seeds,
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
    contract["broker"]["models_cache_mode"] = (
        "FROZEN_SELECTED_MODEL_CATALOG_PROJECTION_V1"
    )
    contract["broker"]["models_cache_path"] = str(MODELS_CACHE)
    contract["broker"]["models_cache_sha256"] = file_sha256(MODELS_CACHE)
    contract["runtime"]["rfc8785"] = "0.1.4"
    contract["content_digest"] = sha256_digest(contract)
    return contract


def main() -> int:
    if CONTRACT_PATH.exists():
        raise RuntimeError(
            f"refusing to overwrite frozen contract: {CONTRACT_PATH}"
        )
    contract = build_contract()
    CONTRACT_PATH.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "contract_path": str(CONTRACT_PATH),
                "contract_sha256": file_sha256(CONTRACT_PATH),
                "search_seed": FRESH_PILOT_V5_SEARCH_SEED,
                "source_projection_digest": contract["source"][
                    "source_projection_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
