#!/usr/bin/env python3
"""Close the sealed F1 matched pair on one gpu35 environment.

The two physical phases are deliberately separate. ``prefix`` freezes the
resource-only decision before any full outcome exists; ``full`` consumes that
decision exactly once. ``bind`` adds an immutable physical receipt to the
repository without re-running either arm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (  # noqa: E402
    _write_new_json,
    run_development_training,
)
from recclaw_core.experiments.helix_abc_v1.innovation_recbole_adapter import (  # noqa: E402
    snapshot_candidate_tree,
)
from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (  # noqa: E402
    ENGINEERING_WATCHDOG_SECONDS,
    PROBE_EPOCHS,
    PROBE_TIMEOUT_SECONDS,
    build_fixed_batch_prefix_contract,
    predict_resources,
    structural_features,
)
ACCEPTED_Q0R2_COMMIT = "278391ab47c78508af211978045ad5573d6fc135"
ACCEPTED_Q0R2_PARENT = "5372da07829c92d73b530855fded95de4f57059b"
ACCEPTED_Q0R2_TREE = "4ef40aa98ddb3ddd1020c2a8ffc7b0dee0752edf"
ACCEPTED_F1_COMMIT = "0041d1cd4dafb3e1a1e2aced97c6a4988db72fc8"
ACCEPTED_F1_PARENT = "3505885c69737064ba5bd59a5aa8c96e5309d15d"
ACCEPTED_F1_TREE = "a9395474954743ab88e3604156cdc2a812f99258"
F1_V1_REPO_RECEIPT_SHA256 = (
    "f160436a9df1bb8d849849f6cb61e367dd54c62d4b5fcc68101a21c86a72b4b3"
)
F1_RECOVERY_REPO_RECEIPT_SHA256 = (
    "59d576b0bbc48fcef17d601acdd53c2205df8f775d5f58a01f5d7314ac8d27fe"
)
F1_V1_EXTERNAL_RECEIPT_SHA256 = (
    "5bb7361179f13bca085884d6926e6002c9f7dfc9632c8471de9fcd0bd631cebb"
)
F1_RECOVERY_EXTERNAL_RECEIPT_SHA256 = (
    "fc4baf9e92b0a95669099ecf480be57274a6ad80abda394e7674b64315458d2b"
)
SEALED_FILE_SHA256 = {
    "policy/versioned_policy.json": (
        "1bc607a8f0d4e2f266ae471b79c29c0251fdcf8ec37697c127875d6e38dfe88c"
    ),
    "activation/activation.json": (
        "7ad0d44da4d5357a36908a82b8edbe775ef347aedf0665e22d111be0da32969a"
    ),
    "selection/selection.json": (
        "c4fc92b5b53c1194cf9089ff09155ee578625324c9a625fbcfcc5cab750f44e6"
    ),
    "qualification/slot-01.json": (
        "1709c92bd91f6e332527aeb71a3ec7b501ed10390114aa5dbb3b87df2d04ee9c"
    ),
    "registry/active_search_profile.json": (
        "e5f3400050bbfba4147181de3d1ffadf57e5d4c36016fb2f6458c77b623592a7"
    ),
    "specs/slot-01.json": (
        "7d0bbbd4303ac74c6f9794c9795be70fc839f4097893a4dee770cd80de806d8c"
    ),
    "execution/experiments/learned-policy-selected-candidate/execution_recipe.json": (
        "715f451d9daa4707db255bf787e318e895023a03e65db642c49b430c64f65bef"
    ),
    "F1_CANONICAL_RECEIPT.json": F1_V1_EXTERNAL_RECEIPT_SHA256,
    "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json": (
        F1_RECOVERY_EXTERNAL_RECEIPT_SHA256
    ),
}
CANDIDATE_RELATIVE_ROOT = (
    "execution/candidates/slot-01/"
    "innovation-candidate-e2fdeaacdf44bbcd5f85f905"
)
CANDIDATE_SOURCE_SHA256 = (
    "ade9963f19c1d5e90e261a71b6cf21c28af92af86f9f511920a39e18b75c8df3"
)
CANDIDATE_SOURCE_TREE_DIGEST = (
    "3f5c6b73e9750ece273e49f7134d409827401b67034ed20624379a6f8623d0a5"
)
CANDIDATE_PACKAGE_DIGEST = (
    "6d21c0babd5b29c748423ffeba0cec51e73fe3fce0d724f50b5fe0a8a8d60fbf"
)
BPR_SOURCE_SHA256 = (
    "d7f859638fc1578ee1ab1b0c147b11bc3cda797f4aedb029100466eea4d6a151"
)
RECBole_COMMIT_IDENTITY = "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
TRAINING_SEED = 53102
FULL_EPOCHS = 100
ARM_ORDER = ("matched_bpr_control", "sealed_f1_candidate")
RUN_IDENTITY = "f1-gpu35-closure-v1"
CAMPAIGN_ID = "recclaw-f1-gpu35-closure-v1"


class F1Gpu35ClosureError(RuntimeError):
    """The sealed F1 closure cannot proceed without identity or fairness drift."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise F1Gpu35ClosureError(f"JSON object required: {path}")
    return value


def _assert_new(path: Path, *, label: str) -> None:
    if path.exists():
        raise F1Gpu35ClosureError(f"{label} already exists: {path}")


def _assert_gpu_idle() -> dict[str, Any]:
    worker_processes: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == current_pid:
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "campaign_train_worker.py" in command:
            worker_processes.append({"pid": int(item.name), "command": command})
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    compute_apps = tuple(
        line.strip() for line in query.stdout.splitlines() if line.strip()
    )
    if worker_processes or compute_apps:
        raise F1Gpu35ClosureError(
            "gpu35 is occupied: "
            + json.dumps(
                {"campaign_train_workers": worker_processes, "compute_apps": compute_apps},
                sort_keys=True,
            )
        )
    return canonical_value(
        {
            "campaign_train_workers": [],
            "compute_apps": [],
            "hostname": subprocess.run(
                ["hostname"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "nvidia_smi_query_return_code": query.returncode,
            "status": "IDLE",
        }
    )


def _validate_sealed_inputs(
    repo_root: Path,
    sealed_f1_root: Path,
) -> dict[str, Any]:
    observed = {
        relative: bytes_sha256((sealed_f1_root / relative).read_bytes())
        for relative in SEALED_FILE_SHA256
    }
    drift = {
        relative: {"expected": SEALED_FILE_SHA256[relative], "observed": digest}
        for relative, digest in observed.items()
        if digest != SEALED_FILE_SHA256[relative]
    }
    repo_receipts = {
        "v1": bytes_sha256(
            (
                repo_root
                / "docs/research_line/vnext/F1_OPEN_META_CANONICAL_RECEIPT.json"
            ).read_bytes()
        ),
        "recovery": bytes_sha256(
            (
                repo_root
                / "docs/research_line/vnext/"
                "F1_OPEN_META_RUNTIME_RECOVERY_V2_CANONICAL_RECEIPT.json"
            ).read_bytes()
        ),
    }
    expected_repo_receipts = {
        "v1": F1_V1_REPO_RECEIPT_SHA256,
        "recovery": F1_RECOVERY_REPO_RECEIPT_SHA256,
    }
    for name, digest in repo_receipts.items():
        if digest != expected_repo_receipts[name]:
            drift[f"repo_receipt:{name}"] = {
                "expected": expected_repo_receipts[name],
                "observed": digest,
            }
    if drift:
        raise F1Gpu35ClosureError(
            "sealed F1 byte identity drift: " + json.dumps(drift, sort_keys=True)
        )

    policy = _read_json(sealed_f1_root / "policy/versioned_policy.json")
    activation = _read_json(sealed_f1_root / "activation/activation.json")
    selection = _read_json(sealed_f1_root / "selection/selection.json")
    qualification = _read_json(sealed_f1_root / "qualification/slot-01.json")
    active_profile = _read_json(
        sealed_f1_root / "registry/active_search_profile.json"
    )
    selected_spec = _read_json(sealed_f1_root / "specs/slot-01.json")
    v1_receipt = _read_json(sealed_f1_root / "F1_CANONICAL_RECEIPT.json")
    recovery_receipt = _read_json(
        sealed_f1_root / "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json"
    )
    recipe = _read_json(
        sealed_f1_root
        / "execution/experiments/learned-policy-selected-candidate/"
        "execution_recipe.json"
    )["execution_recipe"]
    candidate_root = sealed_f1_root / CANDIDATE_RELATIVE_ROOT
    source_tree_digest = sha256_digest(
        {"files": snapshot_candidate_tree(candidate_root)}
    )
    candidate_source_digest = bytes_sha256(
        (candidate_root / "recclaw_ext/candidate.py").read_bytes()
    )
    qualification_without_behavior = {
        key: value
        for key, value in qualification.items()
        if key != "behavioral_mechanism_evidence"
    }
    selected_slot_record = next(
        row
        for row in v1_receipt["proposal_records"]
        if row.get("logical_slot_id") == selection.get("selected_slot")
    )
    policy_digest = policy.get("policy_digest")
    if (
        policy_digest != activation.get("policy_digest")
        or policy_digest != selection.get("activated_policy_digest")
        or selection.get("activation_digest") != activation.get("activation_digest")
        or selection.get("selected_slot") != "slot-01"
        or v1_receipt.get("selection") != selection
        or v1_receipt.get("activation") != activation
        or v1_receipt.get("qualification") != qualification_without_behavior
        or qualification["receipt"].get("status") != "PASS"
        or qualification["receipt"].get("candidate_package_digest")
        != CANDIDATE_PACKAGE_DIGEST
        or qualification["receipt"].get("source_tree_digest")
        != CANDIDATE_SOURCE_TREE_DIGEST
        or source_tree_digest != CANDIDATE_SOURCE_TREE_DIGEST
        or candidate_source_digest != CANDIDATE_SOURCE_SHA256
        or recipe.get("entrypoint_source_sha256") != CANDIDATE_SOURCE_SHA256
        or sha256_digest(selected_spec["research_spec"])
        != selected_slot_record.get("spec_digest")
        or active_profile.get("profile_digest")
        != v1_receipt["profile"]["active_profile_digest"]
        or recovery_receipt.get("status") != "F1_RESOURCE_HARD_BLOCKED"
        or recovery_receipt.get("episode") is not None
        or recovery_receipt.get("held_out_reads") != 0
    ):
        raise F1Gpu35ClosureError("sealed F1 semantic binding drift")
    return canonical_value(
        {
            "active_profile_file_sha256": observed[
                "registry/active_search_profile.json"
            ],
            "activation_digest": activation["activation_digest"],
            "candidate_entrypoint": recipe["entrypoint"],
            "candidate_package_digest": CANDIDATE_PACKAGE_DIGEST,
            "candidate_root": str(candidate_root),
            "candidate_source_sha256": candidate_source_digest,
            "candidate_source_tree_digest": source_tree_digest,
            "policy_digest": policy["policy_digest"],
            "qualification_file_sha256": observed["qualification/slot-01.json"],
            "sealed_file_sha256": observed,
            "selected_capability_digest": selection["selected_capability_digest"],
            "selected_capability_ref": selection["selected_capability_ref"],
            "selected_spec_file_sha256": observed["specs/slot-01.json"],
            "selection_file_sha256": observed["selection/selection.json"],
        }
    )


def _run_arm(
    *,
    repo_root: Path,
    side_root: Path,
    run_id: str,
    candidate_root: Path | None,
    entrypoint: str,
    source_sha256: str,
    epochs: int,
    purpose: str,
    timeout_seconds: int,
    prefix_contract_path: Path | None = None,
) -> dict[str, Any]:
    return run_development_training(
        repo_root=repo_root,
        side_root=side_root,
        run_id=run_id,
        seed=TRAINING_SEED,
        candidate_root=candidate_root,
        entrypoint=entrypoint,
        source_sha256=source_sha256,
        run_identity=RUN_IDENTITY,
        authority="user-delegated-f1-gpu35-closure",
        timeout_seconds=timeout_seconds,
        recbole_commit_identity=RECBole_COMMIT_IDENTITY,
        epochs=epochs,
        execution_purpose=purpose,
        resource_telemetry=prefix_contract_path is not None,
        watchdog_seconds=ENGINEERING_WATCHDOG_SECONDS,
        prefix_contract_path=prefix_contract_path,
    )


def run_prefix(
    *,
    repo_root: Path,
    campaign_root: Path,
    sealed_f1_root: Path,
) -> dict[str, Any]:
    _assert_new(campaign_root, label="unique campaign root")
    binding = _validate_sealed_inputs(repo_root, sealed_f1_root)
    idle = _assert_gpu_idle()
    campaign_root.mkdir(parents=True)
    identity = canonical_value(
        {
            "accepted_f1": {
                "commit": ACCEPTED_F1_COMMIT,
                "parent": ACCEPTED_F1_PARENT,
                "tree": ACCEPTED_F1_TREE,
            },
            "accepted_q0r2": {
                "commit": ACCEPTED_Q0R2_COMMIT,
                "parent": ACCEPTED_Q0R2_PARENT,
                "tree": ACCEPTED_Q0R2_TREE,
            },
            "arm_order": ARM_ORDER,
            "campaign_id": CAMPAIGN_ID,
            "epochs_per_full_arm": FULL_EPOCHS,
            "held_out_reads": 0,
            "matched_seed": TRAINING_SEED,
            "new_provider_calls": 0,
            "remote_root": str(campaign_root.parent),
            "run_identity": RUN_IDENTITY,
            "schema": "recclaw.research-line.f1-gpu35-closure-identity.v1",
            "sealed_binding": binding,
        }
    )
    _write_new_json(campaign_root / "RUN_IDENTITY.json", identity)
    _write_new_json(campaign_root / "GPU_IDLE_BEFORE_PREFIX.json", idle)
    contract_path = campaign_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    contract_digest = _write_new_json(
        contract_path, build_fixed_batch_prefix_contract(seed=TRAINING_SEED)
    )
    candidate_root = Path(binding["candidate_root"])
    recbole_root = Path(os.environ["RECCLAW_RECBOLE_ROOT"])
    baseline_source = recbole_root / "recbole/model/general_recommender/bpr.py"
    if bytes_sha256(baseline_source.read_bytes()) != BPR_SOURCE_SHA256:
        raise F1Gpu35ClosureError("gpu35 BPR comparator source identity drift")
    arm_inputs = {
        "matched_bpr_control": {
            "candidate_root": None,
            "entrypoint": "recbole.model.general_recommender.bpr:BPR",
            "source_path": baseline_source,
            "source_sha256": BPR_SOURCE_SHA256,
        },
        "sealed_f1_candidate": {
            "candidate_root": candidate_root,
            "entrypoint": binding["candidate_entrypoint"],
            "source_path": candidate_root / "recclaw_ext/candidate.py",
            "source_sha256": CANDIDATE_SOURCE_SHA256,
        },
    }
    probe_runs: dict[str, Any] = {}
    for arm in ARM_ORDER:
        row = arm_inputs[arm]
        probe_runs[arm] = _run_arm(
            repo_root=repo_root,
            side_root=campaign_root / "prefix_runs",
            run_id=f"{arm}-prefix",
            candidate_root=row["candidate_root"],
            entrypoint=str(row["entrypoint"]),
            source_sha256=str(row["source_sha256"]),
            epochs=PROBE_EPOCHS,
            purpose="RESOURCE_PROBE_ONLY",
            timeout_seconds=PROBE_TIMEOUT_SECONDS,
            prefix_contract_path=contract_path,
        )
        _write_new_json(
            campaign_root / "prefix_results" / f"{arm}.json", probe_runs[arm]
        )
    features = {
        arm: structural_features(Path(row["source_path"]))
        for arm, row in arm_inputs.items()
    }
    decision = predict_resources(
        arm_features=features,
        probe_runs=probe_runs,
        arm_order=ARM_ORDER,
        probe_seed=TRAINING_SEED,
    )
    decision_path = campaign_root / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json"
    decision_digest = _write_new_json(decision_path, decision)
    scheduled = {row["arm"] for row in decision["schedule"]}
    disposition = (
        "FULL_MATCHED_PAIR_ADMITTED"
        if scheduled == set(ARM_ORDER)
        else "RESOURCE_DEFERRED"
    )
    receipt = canonical_value(
        {
            "decision_artifact_sha256": decision_digest,
            "effect_fields_consumed": [],
            "full_outcomes_present_when_written": 0,
            "held_out_reads": 0,
            "identity_sha256": bytes_sha256(
                (campaign_root / "RUN_IDENTITY.json").read_bytes()
            ),
            "legacy_1500_seconds_controls_full": False,
            "mechanism_effect_update_allowed": False,
            "prefix_contract_sha256": contract_digest,
            "prefix_results": probe_runs,
            "qualification_preserved": True,
            "resource_disposition": disposition,
            "schema": "recclaw.research-line.f1-gpu35-prefix-receipt.v1",
            "watchdog_purpose": "HANG_OR_UNBOUNDED_EXECUTION_ONLY",
        }
    )
    _write_new_json(campaign_root / "PREFIX_PHYSICAL_RECEIPT.json", receipt)
    if disposition == "RESOURCE_DEFERRED":
        physical = _resource_deferred_receipt(
            campaign_root=campaign_root,
            identity=identity,
            decision=decision,
            prefix_receipt=receipt,
        )
        _write_new_json(
            campaign_root / "F1_GPU35_CLOSURE_PHYSICAL_RECEIPT.json", physical
        )
    return receipt


def _resource_deferred_receipt(
    *,
    campaign_root: Path,
    identity: Mapping[str, Any],
    decision: Mapping[str, Any],
    prefix_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return canonical_value(
        {
            "architecture_effect_gates": {
                "end_to_end_result_chain_real_and_valid": False,
                "function_real_and_runnable": False,
                "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": True,
                "serves_open_algorithm_research_target": True,
            },
            "attempt_identity": identity,
            "development_only": True,
            "effect_update": "COMPLETION_RESOURCE_ONLY",
            "episode": None,
            "experiment_executed": False,
            "held_out_reads": 0,
            "mechanism_negative_evidence": False,
            "new_provider_calls": 0,
            "policy_superiority_claim": False,
            "prefix_receipt_sha256": bytes_sha256(
                (campaign_root / "PREFIX_PHYSICAL_RECEIPT.json").read_bytes()
            ),
            "qualification_preserved": True,
            "resource_decision": decision,
            "resource_disposition": prefix_receipt["resource_disposition"],
            "schema": "recclaw.research-line.f1-gpu35-closure-physical-receipt.v1",
            "scientific_effect_claim": False,
            "scientific_interpretation": "INCONCLUSIVE_NOT_ADJUDICATED",
            "status": "RESOURCE_DEFERRED",
        }
    )


def finalize_prefix_failure(
    *,
    repo_root: Path,
    campaign_root: Path,
    sealed_f1_root: Path,
) -> dict[str, Any]:
    """Freeze a no-full decision from the one existing matched prefix pair."""

    decision_path = campaign_root / "PREDICTION_AND_SCHEDULE_BEFORE_OUTCOME.json"
    prefix_receipt_path = campaign_root / "PREFIX_PHYSICAL_RECEIPT.json"
    physical_path = campaign_root / "F1_GPU35_CLOSURE_PHYSICAL_RECEIPT.json"
    _assert_new(decision_path, label="resource decision")
    _assert_new(prefix_receipt_path, label="prefix physical receipt")
    _assert_new(physical_path, label="physical closure receipt")
    if (campaign_root / "full_runs").exists():
        raise F1Gpu35ClosureError("full outcome root exists before resource decision")
    identity = _read_json(campaign_root / "RUN_IDENTITY.json")
    binding = _validate_sealed_inputs(repo_root, sealed_f1_root)
    if identity.get("sealed_binding") != binding:
        raise F1Gpu35ClosureError("sealed binding changed after prefix execution")
    contract_path = campaign_root / "FIXED_BATCH_PREFIX_CONTRACT.json"
    contract_digest = bytes_sha256(contract_path.read_bytes())
    runs = {
        arm: _read_json(campaign_root / "prefix_results" / f"{arm}.json")
        for arm in ARM_ORDER
    }
    control = runs["matched_bpr_control"]
    candidate = runs["sealed_f1_candidate"]
    candidate_telemetry = candidate.get("resource_telemetry")
    if (
        control.get("exit_status") != "SUCCESS"
        or candidate.get("exit_status") != "RUNTIME_FAILURE"
        or candidate.get("worker_error_type") != "OutOfMemoryError"
        or not isinstance(candidate_telemetry, Mapping)
        or candidate.get("metrics") != {}
        or any(run.get("seed") != TRAINING_SEED for run in runs.values())
        or any(run.get("epochs_requested") != PROBE_EPOCHS for run in runs.values())
        or any(
            run.get("resource_deadline_seconds") != PROBE_TIMEOUT_SECONDS
            for run in runs.values()
        )
        or any(
            run.get("watchdog_seconds") != ENGINEERING_WATCHDOG_SECONDS
            for run in runs.values()
        )
        or any(
            run.get("resource_telemetry", {})
            .get("prefix_contract", {})
            .get("contract_file_sha256")
            != contract_digest
            for run in runs.values()
        )
    ):
        raise F1Gpu35ClosureError("prefix failure is not the exact uniform resource block")
    decision = canonical_value(
        {
            "campaign_total_budget_seconds": 7200,
            "consumer": "accepted_q0r2_resource_only_admission_consumer",
            "consumer_input_status": "REJECTED_REAL_PREFIX_TELEMETRY_UNAVAILABLE",
            "deferred_arms": (
                {
                    "arm": "sealed_f1_candidate",
                    "future_eligible": True,
                    "mechanism_effect_update_allowed": False,
                    "observed_peak_gpu_memory_mib": candidate_telemetry.get(
                        "peak_gpu_memory_mib"
                    ),
                    "reason": "OBSERVED_GPU_MEMORY_CAPACITY_OOM",
                    "resource_disposition": "RESOURCE_INFEASIBLE",
                },
                {
                    "arm": "matched_bpr_control",
                    "future_eligible": True,
                    "mechanism_effect_update_allowed": False,
                    "reason": "MATCHED_PAIR_CANDIDATE_RESOURCE_INFEASIBLE",
                    "resource_disposition": "RESOURCE_DEFERRED",
                },
            ),
            "effect_fields_consumed": [],
            "engineering_watchdog_seconds": ENGINEERING_WATCHDOG_SECONDS,
            "full_outcomes_present_when_written": 0,
            "legacy_1500_seconds_controls_full": False,
            "prefix_contract_sha256": contract_digest,
            "prefix_failure": {
                "error_type": candidate["worker_error_type"],
                "exit_status": candidate["exit_status"],
                "launcher_return_code": candidate["launcher_return_code"],
                "mechanism_negative_evidence": False,
                "peak_gpu_memory_mib": candidate_telemetry.get(
                    "peak_gpu_memory_mib"
                ),
                "result_sha256": candidate.get("result_sha256"),
                "resource_telemetry_sha256": candidate.get(
                    "resource_telemetry_sha256"
                ),
            },
            "schedule": (),
            "schema": "recclaw.q0r2-f1-prefix-resource-admission.v1",
            "watchdog_purpose": "HANG_OR_UNBOUNDED_EXECUTION_ONLY",
        }
    )
    decision_digest = _write_new_json(decision_path, decision)
    prefix_receipt = canonical_value(
        {
            "decision_artifact_sha256": decision_digest,
            "effect_fields_consumed": [],
            "full_outcomes_present_when_written": 0,
            "held_out_reads": 0,
            "identity_sha256": bytes_sha256(
                (campaign_root / "RUN_IDENTITY.json").read_bytes()
            ),
            "legacy_1500_seconds_controls_full": False,
            "mechanism_effect_update_allowed": False,
            "prefix_contract_sha256": contract_digest,
            "prefix_results": runs,
            "qualification_preserved": True,
            "resource_disposition": "RESOURCE_DEFERRED",
            "schema": "recclaw.research-line.f1-gpu35-prefix-receipt.v1",
            "watchdog_purpose": "HANG_OR_UNBOUNDED_EXECUTION_ONLY",
        }
    )
    _write_new_json(prefix_receipt_path, prefix_receipt)
    physical = _resource_deferred_receipt(
        campaign_root=campaign_root,
        identity=identity,
        decision=decision,
        prefix_receipt=prefix_receipt,
    )
    _write_new_json(physical_path, physical)
    return physical


def bind_receipt(
    *,
    physical_receipt_path: Path,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    _assert_new(canonical_receipt_path, label="repository canonical receipt")
    physical = _read_json(physical_receipt_path)
    if (
        physical.get("schema")
        != "recclaw.research-line.f1-gpu35-closure-physical-receipt.v1"
        or physical.get("development_only") is not True
        or physical.get("held_out_reads") != 0
        or physical.get("new_provider_calls") != 0
        or physical.get("scientific_effect_claim") is not False
        or physical.get("policy_superiority_claim") is not False
    ):
        raise F1Gpu35ClosureError("invalid physical F1 gpu35 closure receipt")
    repository_receipt = canonical_value(
        {
            **physical,
            "external_receipt_ref": str(physical_receipt_path),
            "external_receipt_sha256": bytes_sha256(
                physical_receipt_path.read_bytes()
            ),
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("prefix", "finalize-prefix-failure", "bind"),
        required=True,
    )
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--sealed-f1-root", type=Path)
    parser.add_argument("--physical-receipt", type=Path)
    parser.add_argument("--canonical-receipt", type=Path)
    args = parser.parse_args()
    if args.phase in {"prefix", "finalize-prefix-failure"}:
        if args.campaign_root is None or args.sealed_f1_root is None:
            parser.error(
                "physical phases require --campaign-root and --sealed-f1-root"
            )
        functions = {
            "prefix": run_prefix,
            "finalize-prefix-failure": finalize_prefix_failure,
        }
        function = functions[args.phase]
        result = function(
            repo_root=ROOT,
            campaign_root=args.campaign_root.resolve(),
            sealed_f1_root=args.sealed_f1_root.resolve(),
        )
    else:
        if args.physical_receipt is None or args.canonical_receipt is None:
            parser.error("bind requires --physical-receipt and --canonical-receipt")
        result = bind_receipt(
            physical_receipt_path=args.physical_receipt.resolve(),
            canonical_receipt_path=args.canonical_receipt.resolve(),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
