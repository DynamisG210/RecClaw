#!/usr/bin/env python3
"""Freeze or verify the fresh V16 resource-closure recovery Pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from freeze_v13_pilot_contract import (
    DEFAULT_LLM_CONFIG,
    PROPOSAL_SCHEMA,
    RESOURCE_ROOT,
    ROOT,
    build_contract,
    file_sha256,
)
from freeze_v15_pilot_contract import (
    BACKEND_ROOT,
    ORIGINAL_GIT_CONFIG,
    ORIGINAL_GIT_CONFIG_HOME,
    ORIGINAL_GIT_EXECUTABLE,
    ORIGINAL_GIT_EXEC_PATH,
    _dataset_projection,
    _v15_source_files,
    activate_original_git_tool,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
    V16_PILOT_ASSIGNMENT_NONCE,
    V16_PILOT_ROUNDS_PER_ARM,
    V16_PILOT_SEARCH_SEED,
    V16PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    CHECKPOINT_SHA256_V19,
    POLICY_BUNDLE_DIGEST_V19,
    PROMOTION_DECISION_DIGEST_V19,
    MetaV19CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_COMMIT,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    campaign_training_runtime_release,
    validate_campaign_training_runtime_release,
)


DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V16_FROZEN_CHAIN_PILOT_CONTRACT.json"
)
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "pilot_9218_v16"
DEFAULT_TRAINING_PYTHON = BACKEND_ROOT / "runtime_exact_v2/bin/python"
DEFAULT_RECBOLE_ROOT = BACKEND_ROOT / "recbole"
DEFAULT_DATA_PATH = BACKEND_ROOT / "search_dataset"
META_CHECKPOINT = RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v19.json"
META_PROMOTION = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "META_V19_TRANSPORT_REQUALIFICATION.json"
)
BROKER_RELEASE = (
    RESOURCE_ROOT / "lab_api_broker_release_v1_v14_schema_v5.json"
)
TRAINING_RELEASE = RESOURCE_ROOT / "training_runtime_release_v9.json"
SCIENTIFIC_GATE = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V16_SCIENTIFIC_ATTRIBUTION_GATE.json"
)
RECOVERY_AUDIT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V16_RESOURCE_CLOSURE_REQUALIFICATION.json"
)
RECORD_SCHEMA = "recclaw.v16-pilot-contract.v1"
SOURCE_BUNDLE = BACKEND_ROOT / "recclaw_v16_source.bundle"


def _v16_source_files() -> tuple[str, ...]:
    replacements = {
        "scripts/freeze_campaign_training_runtime_release_v8.py": (
            "scripts/freeze_campaign_training_runtime_release_v9.py"
        ),
        "scripts/freeze_v15_pilot_contract.py": (
            "scripts/freeze_v16_pilot_contract.py"
        ),
        "scripts/run_v15_pilot.py": "scripts/run_v16_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v15.py": (
            "src/recclaw_core/experiments/helix_abc_v1/"
            "campaign_pilot_v16.py"
        ),
        "src/recclaw_core/experiments/helix_abc_v1/resources/"
        "training_runtime_release_v8.json": (
            "src/recclaw_core/experiments/helix_abc_v1/resources/"
            "training_runtime_release_v9.json"
        ),
    }
    rows = {replacements.get(item, item) for item in _v15_source_files()}
    rows.add("scripts/validate_campaign_training_runtime_v9.py")
    return tuple(sorted(rows))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def build_v16_contract(
    *,
    llm_api_config: Path,
    output_root: Path,
    training_python: Path,
    recbole_root: Path,
    data_path: Path,
) -> dict[str, object]:
    activate_original_git_tool()
    contract = build_contract(
        llm_api_config=llm_api_config,
        output_root=output_root,
        store_contract_factory=V16PilotStoreContractV1.create,
        assignment_nonce=V16_PILOT_ASSIGNMENT_NONCE,
        search_seed=V16_PILOT_SEARCH_SEED,
        rounds_per_arm=V16_PILOT_ROUNDS_PER_ARM,
        meta_runtime_class=MetaV19CampaignRuntimeV1,
        meta_checkpoint=META_CHECKPOINT,
        meta_promotion=META_PROMOTION,
        meta_checkpoint_sha256=CHECKPOINT_SHA256_V19,
        meta_policy_bundle_digest=POLICY_BUNDLE_DIGEST_V19,
        meta_promotion_digest=PROMOTION_DECISION_DIGEST_V19,
        broker_release_path=BROKER_RELEASE,
        training_release_path=TRAINING_RELEASE,
        scientific_gate_path=SCIENTIFIC_GATE,
        record_schema=RECORD_SCHEMA,
        source_files=_v16_source_files(),
        training_python=training_python,
        training_recbole_root=recbole_root,
        training_data_path=data_path,
        dataset_projection=_dataset_projection(data_path),
        git_head_at_freeze=_git_head(),
    )
    payload = dict(contract)
    payload.pop("content_digest")
    if Path(shutil.which("git") or "").resolve() != (
        ORIGINAL_GIT_EXECUTABLE.resolve()
    ):
        raise RuntimeError("V16 pinned Original Git tool is not active")
    subprocess.run(
        ["git", "cat-file", "-e", f"{ORIGINAL_MAIN_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    payload["original"] = {
        **payload["original"],
        "git_config_path": ORIGINAL_GIT_CONFIG.as_posix(),
        "git_config_sha256": file_sha256(ORIGINAL_GIT_CONFIG),
        "git_executable": ORIGINAL_GIT_EXECUTABLE.as_posix(),
        "git_executable_sha256": file_sha256(ORIGINAL_GIT_EXECUTABLE),
        "git_exec_path": ORIGINAL_GIT_EXEC_PATH.as_posix(),
        "source_bundle_path": SOURCE_BUNDLE.as_posix(),
        "source_bundle_sha256": file_sha256(SOURCE_BUNDLE),
        "source_repository_head": _git_head(),
    }
    recovery = json.loads(RECOVERY_AUDIT.read_text(encoding="utf-8"))
    payload["backend_qualification"] = {
        "audit_path": RECOVERY_AUDIT.as_posix(),
        "audit_sha256": file_sha256(RECOVERY_AUDIT),
        "audit_verdict": recovery["verdict"],
        "p0": recovery["p0"],
        "p1": recovery["p1"],
        "release_id": recovery["training_runtime_release_id"],
        "runtime_release_digest": recovery[
            "training_runtime_release_digest"
        ],
    }
    return {**payload, "content_digest": sha256_digest(payload)}


def verify_v16_pilot_contract(path: Path) -> dict[str, object]:
    activate_original_git_tool()
    contract = json.loads(path.read_text(encoding="utf-8"))
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError("V16 contract content digest mismatch")
    gate = json.loads(SCIENTIFIC_GATE.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_AUDIT.read_text(encoding="utf-8"))
    if (
        contract["record_schema"] != RECORD_SCHEMA
        or contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["pilot_started"] is not False
        or contract["broker"]["model"] != "gpt-5.4"
        or contract["common_substrate"]["executable_mechanism_count"] != 66
        or contract["meta"]["policy_bundle_digest"]
        != POLICY_BUNDLE_DIGEST_V19
        or contract["guard_and_fusion"]["gate_result_digest"]
        != gate["gate_result_digest"]
        or contract["backend_qualification"]["audit_verdict"] != "PASS"
        or contract["backend_qualification"]["p0"] != 0
        or contract["backend_qualification"]["p1"] != 0
        or recovery["verdict"] != "PASS"
    ):
        raise RuntimeError("V16 contract invariant mismatch")
    expected_store = V16PilotStoreContractV1.create()
    if contract["pilot"] != {
        "experiment_id": expected_store.experiment_id,
        "ordinary_execution_seed": expected_store.ordinary_execution_seed,
        "rounds_per_arm": expected_store.scheduled_slots_per_arm_seed,
        "search_seeds": list(expected_store.search_seeds),
        "store_contract_identity_digest": expected_store.identity_digest,
    }:
        raise RuntimeError("V16 store contract identity mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V16 source identity mismatch: {relative}")
    exact = {
        Path(contract["backend_qualification"]["audit_path"]): contract[
            "backend_qualification"
        ]["audit_sha256"],
        Path(contract["broker"]["release_manifest_path"]): contract[
            "broker"
        ]["release_manifest_sha256"],
        Path(contract["broker"]["response_schema_path"]): contract[
            "broker"
        ]["response_schema_sha256"],
        Path(contract["meta"]["checkpoint_path"]): contract["meta"][
            "checkpoint_sha256"
        ],
        Path(contract["training"]["release_manifest_path"]): contract[
            "training"
        ]["release_manifest_sha256"],
        Path(contract["original"]["git_config_path"]): contract[
            "original"
        ]["git_config_sha256"],
        Path(contract["original"]["git_executable"]): contract[
            "original"
        ]["git_executable_sha256"],
        Path(contract["original"]["source_bundle_path"]): contract[
            "original"
        ]["source_bundle_sha256"],
    }
    for artifact, digest in exact.items():
        if file_sha256(artifact) != digest:
            raise RuntimeError(f"V16 release identity mismatch: {artifact}")
    if (
        campaign_training_runtime_release().digest
        != contract["training"]["release_digest"]
    ):
        raise RuntimeError("V16 active training release changed")
    if Path(shutil.which("git") or "").resolve() != Path(
        contract["original"]["git_executable"]
    ).resolve():
        raise RuntimeError("V16 pinned Original Git tool is not active")
    runtime_failures = validate_campaign_training_runtime_release(
        data_path=Path(contract["dataset"]["search_parent"]),
        python_executable=Path(contract["training"]["python"]),
        recbole_root=Path(contract["training"]["recbole_root"]),
    )
    if runtime_failures:
        raise RuntimeError(
            "V16 training backend identity mismatch: "
            + ",".join(runtime_failures)
        )
    subprocess.run(
        ["git", "cat-file", "-e", f"{ORIGINAL_MAIN_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    if Path(contract["output_root"]).exists():
        raise RuntimeError("V16 Pilot output root is no longer fresh")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--llm-api-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
    )
    parser.add_argument(
        "--python", type=Path, default=DEFAULT_TRAINING_PYTHON
    )
    parser.add_argument(
        "--recbole-root", type=Path, default=DEFAULT_RECBOLE_ROOT
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    if args.verify is not None:
        contract = verify_v16_pilot_contract(args.verify.resolve())
        print(
            json.dumps(
                {
                    "content_digest": contract["content_digest"],
                    "pilot_started": contract["pilot_started"],
                    "verdict": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    contract = build_v16_contract(
        llm_api_config=args.llm_api_config.resolve(),
        output_root=args.output_root.resolve(),
        training_python=args.python.absolute(),
        recbole_root=args.recbole_root.resolve(),
        data_path=args.data_path.resolve(),
    )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("V16 frozen contract already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "path": output.as_posix(),
                "sha256": file_sha256(output),
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
