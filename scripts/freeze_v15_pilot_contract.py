#!/usr/bin/env python3
"""Freeze or verify the fresh V15 native-backend chain Pilot contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

try:
    from scripts.freeze_v13_pilot_contract import (
        DEFAULT_LLM_CONFIG,
        PROPOSAL_SCHEMA,
        RESOURCE_ROOT,
        ROOT,
        build_contract,
        file_sha256,
    )
    from scripts.freeze_v14_pilot_contract import _v14_source_files
except ModuleNotFoundError:
    from freeze_v13_pilot_contract import (
        DEFAULT_LLM_CONFIG,
        PROPOSAL_SCHEMA,
        RESOURCE_ROOT,
        ROOT,
        build_contract,
        file_sha256,
    )
    from freeze_v14_pilot_contract import _v14_source_files

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v15 import (
    V15_PILOT_ASSIGNMENT_NONCE,
    V15_PILOT_ROUNDS_PER_ARM,
    V15_PILOT_SEARCH_SEED,
    V15PilotStoreContractV1,
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
    / "V15_FROZEN_CHAIN_PILOT_CONTRACT.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/pilot_9217_v15"
)
DEFAULT_TRAINING_PYTHON = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/runtime_exact_v2/bin/python"
)
DEFAULT_RECBOLE_ROOT = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/recbole"
)
DEFAULT_DATA_PATH = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/"
    "recclaw_v15_backend_v1/search_dataset"
)
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
TRAINING_RELEASE = RESOURCE_ROOT / "training_runtime_release_v8.json"
SCIENTIFIC_GATE = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V15_SCIENTIFIC_ATTRIBUTION_GATE.json"
)
BACKEND_AUDIT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V15_BACKEND_CONFORMANCE_AUDIT.json"
)
PARENT_CONTRACT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V14_FROZEN_CHAIN_PILOT_CONTRACT.json"
)
RECORD_SCHEMA = "recclaw.v15-pilot-contract.v1"
BACKEND_ROOT = Path(
    "/NAS2020/Workspaces/DMGroup/tingrangan/recclaw_v15_backend_v1"
)
ORIGINAL_GIT_EXECUTABLE = (
    BACKEND_ROOT / "tools/git-focal/usr/bin/git"
)
ORIGINAL_GIT_EXEC_PATH = (
    BACKEND_ROOT / "tools/git-focal/usr/lib/git-core"
)
ORIGINAL_GIT_CONFIG_HOME = BACKEND_ROOT / "tools/git-config"
ORIGINAL_GIT_CONFIG = ORIGINAL_GIT_CONFIG_HOME / "git/config"
SOURCE_BUNDLE = BACKEND_ROOT / "recclaw_v15_source.bundle"


def _v15_source_files() -> tuple[str, ...]:
    replacements = {
        "scripts/freeze_campaign_training_runtime_release_v6.py": (
            "scripts/freeze_campaign_training_runtime_release_v8.py"
        ),
        "scripts/freeze_v14_pilot_contract.py": (
            "scripts/freeze_v15_pilot_contract.py"
        ),
        "scripts/run_v14_pilot.py": "scripts/run_v15_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v14.py": (
            "src/recclaw_core/experiments/helix_abc_v1/"
            "campaign_pilot_v15.py"
        ),
    }
    return tuple(
        sorted({replacements.get(item, item) for item in _v14_source_files()})
    )


def _dataset_projection(data_path: Path) -> dict[str, object]:
    parent = json.loads(PARENT_CONTRACT.read_text(encoding="utf-8"))[
        "dataset"
    ]
    projection = copy.deepcopy(parent)
    search = data_path / "ml-1m"
    for name, expected in projection["search_files"].items():
        observed = file_sha256(search / name)
        if observed != expected:
            raise RuntimeError(f"V15 dataset identity mismatch: {name}")
    projection["search_parent"] = data_path.as_posix()
    projection["search_dataset"] = search.as_posix()
    projection["online_backend_heldout_mount"] = "ABSENT"
    return projection


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def activate_original_git_tool() -> None:
    os.environ["PATH"] = (
        ORIGINAL_GIT_EXECUTABLE.parent.as_posix()
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    os.environ["GIT_EXEC_PATH"] = ORIGINAL_GIT_EXEC_PATH.as_posix()
    os.environ["XDG_CONFIG_HOME"] = ORIGINAL_GIT_CONFIG_HOME.as_posix()


def build_v15_contract(
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
        store_contract_factory=V15PilotStoreContractV1.create,
        assignment_nonce=V15_PILOT_ASSIGNMENT_NONCE,
        search_seed=V15_PILOT_SEARCH_SEED,
        rounds_per_arm=V15_PILOT_ROUNDS_PER_ARM,
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
        source_files=_v15_source_files(),
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
        raise RuntimeError("V15 pinned Original Git tool is not active")
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
    backend = json.loads(BACKEND_AUDIT.read_text(encoding="utf-8"))
    payload["backend_qualification"] = {
        "audit_path": BACKEND_AUDIT.as_posix(),
        "audit_sha256": file_sha256(BACKEND_AUDIT),
        "audit_verdict": backend["audit"]["verdict"],
        "p0": backend["audit"]["P0"],
        "p1": backend["audit"]["P1"],
        "release_id": backend["backend_release"]["release_id"],
        "runtime_release_digest": backend["backend_release"][
            "campaign_runtime_release_digest"
        ],
    }
    return {**payload, "content_digest": sha256_digest(payload)}


def verify_v15_pilot_contract(path: Path) -> dict[str, object]:
    activate_original_git_tool()
    contract = json.loads(path.read_text(encoding="utf-8"))
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError("V15 contract content digest mismatch")
    gate = json.loads(SCIENTIFIC_GATE.read_text(encoding="utf-8"))
    backend = json.loads(BACKEND_AUDIT.read_text(encoding="utf-8"))
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
        or backend["audit"]["verdict"] != "PASS"
    ):
        raise RuntimeError("V15 contract invariant mismatch")
    expected_store = V15PilotStoreContractV1.create()
    if contract["pilot"] != {
        "experiment_id": expected_store.experiment_id,
        "ordinary_execution_seed": expected_store.ordinary_execution_seed,
        "rounds_per_arm": expected_store.scheduled_slots_per_arm_seed,
        "search_seeds": list(expected_store.search_seeds),
        "store_contract_identity_digest": expected_store.identity_digest,
    }:
        raise RuntimeError("V15 store contract identity mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V15 source identity mismatch: {relative}")
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
            raise RuntimeError(f"V15 release identity mismatch: {artifact}")
    if (
        campaign_training_runtime_release().digest
        != contract["training"]["release_digest"]
    ):
        raise RuntimeError("V15 active training release changed")
    if Path(shutil.which("git") or "").resolve() != Path(
        contract["original"]["git_executable"]
    ).resolve():
        raise RuntimeError("V15 pinned Original Git tool is not active")
    runtime_failures = validate_campaign_training_runtime_release(
        data_path=Path(contract["dataset"]["search_parent"]),
        python_executable=Path(contract["training"]["python"]),
        recbole_root=Path(contract["training"]["recbole_root"]),
    )
    if runtime_failures:
        raise RuntimeError(
            "V15 training backend identity mismatch: "
            + ",".join(runtime_failures)
        )
    subprocess.run(
        ["git", "cat-file", "-e", f"{ORIGINAL_MAIN_COMMIT}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )
    if Path(contract["output_root"]).exists():
        raise RuntimeError("V15 Pilot output root is no longer fresh")
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
        contract = verify_v15_pilot_contract(args.verify.resolve())
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
    contract = build_v15_contract(
        llm_api_config=args.llm_api_config.resolve(),
        output_root=args.output_root.resolve(),
        training_python=args.python.resolve(),
        recbole_root=args.recbole_root.resolve(),
        data_path=args.data_path.resolve(),
    )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("V15 frozen contract already exists")
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
