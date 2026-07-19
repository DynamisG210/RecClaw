"""Direct, development-only preflight for the frozen AB-002 Canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

from recclaw_phase1.ab002_launcher import (
    ARM_MANIFEST,
    CONTRACT,
    FROZEN_GUARD_FILES,
    FROZEN_TREATMENT_AGENT_SHA256,
    canonical_write,
    materialize_arm,
    sha256_file,
    validate_baseline_inputs,
    validate_runtime_root,
)
from recclaw_phase1.ab002_s0 import LEAKAGE_AUDIT, validate_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUMS = PROJECT_ROOT / "SOURCE_SHA256SUMS"
CLEAN_MANIFEST = PROJECT_ROOT / "clean_tree_manifest.json"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_GIT_ID = re.compile(r"^[0-9a-f]{40}$")


def tree_identity(root: Path) -> tuple[str, dict[str, str]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    rows: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        rows[relative] = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest(), rows


def exact_file_identities(root: Path, expected: dict[str, str]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required file missing: {path}")
        observed = sha256_file(path)
        if observed != wanted:
            raise ValueError(f"file identity drift: {relative}: {observed} != {wanted}")
        actual[relative] = observed
    return actual


def verify_source_sums(root: Path, sums_path: Path) -> int:
    checked = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wanted, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != wanted:
            raise ValueError(f"source checksum mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise ValueError("source checksum inventory is empty")
    return checked


def verify_clean_tree_manifest(root: Path, manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files")
    if not isinstance(rows, list) or int(payload.get("file_count", -1)) != len(rows):
        raise ValueError("clean-tree manifest file count is invalid")
    expected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("clean-tree manifest row is invalid")
        relative = str(row.get("path") or "")
        wanted = str(row.get("sha256") or "")
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in expected
            or not HEX_SHA256.fullmatch(wanted)
        ):
            raise ValueError("clean-tree manifest path or digest is invalid")
        path = root / candidate
        if not path.is_file() or sha256_file(path) != wanted:
            raise ValueError(f"clean-tree file identity drift: {relative}")
        expected[relative] = wanted
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }
    allowed = set(expected) | {manifest_path.relative_to(root).as_posix()}
    if actual != allowed:
        extras = sorted(actual - allowed)
        missing = sorted(allowed - actual)
        raise ValueError(f"clean-tree inventory drift: extras={extras}, missing={missing}")
    return len(expected) + 1


def resolve_release_identity(
    *,
    root: Path,
    expected_tag: str,
    release_identity_path: Path | None,
    source_sums_path: Path,
    clean_manifest_path: Path,
) -> dict[str, Any]:
    git = shutil.which("git")
    if git and (root / ".git").exists():
        if git_value("status", "--porcelain"):
            raise ValueError("preflight requires a clean delivery checkout")
        head = git_value("rev-parse", "HEAD")
        tree = git_value("rev-parse", "HEAD^{tree}")
        tag_commit = git_value("rev-list", "-n", "1", expected_tag)
        if tag_commit != head:
            raise ValueError(f"release tag does not resolve to HEAD: {tag_commit} != {head}")
        return {
            "commit": head,
            "tree": tree,
            "tag": expected_tag,
            "verification_mode": "LIVE_GIT_AND_EXACT_FILE_MANIFEST",
        }
    if release_identity_path is None:
        raise ValueError("gitless preflight requires an external release identity record")
    identity_path = release_identity_path.expanduser().resolve()
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    required = {
        "record_type": "RECCLAW_PHASE1_AB002_EXTERNAL_RELEASE_IDENTITY",
        "schema_version": "recclaw.phase1.ab002.external_release_identity.v1",
        "status": "LOCAL_COMPLETE",
        "tag": expected_tag,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    for key, wanted in required.items():
        if payload.get(key) != wanted:
            raise ValueError(f"external release identity field mismatch: {key}")
    commit = str(payload.get("commit") or "")
    tree = str(payload.get("tree") or "")
    if not HEX_GIT_ID.fullmatch(commit) or not HEX_GIT_ID.fullmatch(tree):
        raise ValueError("external release identity has invalid Git identities")
    archive = Path(str(payload.get("source_archive_path") or "")).expanduser().resolve()
    archive_sha256 = str(payload.get("source_archive_sha256") or "")
    if not archive.is_file() or sha256_file(archive) != archive_sha256:
        raise ValueError("external release archive identity drift")
    if payload.get("source_sha256s_sha256") != sha256_file(source_sums_path):
        raise ValueError("external release SOURCE_SHA256SUMS identity drift")
    if payload.get("clean_tree_manifest_sha256") != sha256_file(clean_manifest_path):
        raise ValueError("external release clean-tree manifest identity drift")
    return {
        "commit": commit,
        "tree": tree,
        "tag": expected_tag,
        "verification_mode": "EXTERNAL_RELEASE_ARCHIVE_AND_EXACT_FILE_MANIFEST",
        "release_identity_record": str(identity_path),
        "release_identity_record_sha256": sha256_file(identity_path),
        "source_archive": str(archive),
        "source_archive_sha256": archive_sha256,
    }


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def gpu_inventory() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"nvidia-smi failed: {completed.stderr.strip()}")
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError("unexpected nvidia-smi CSV shape")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "driver": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    by_index = {row["index"]: row for row in rows}
    for index in (0, 1):
        if index not in by_index or by_index[index]["memory_free_mib"] < 20_000:
            raise ValueError(f"required Canary GPU {index} is not sufficiently free")
    return rows


def broker_health(base_url: str) -> dict[str, Any]:
    with urlrequest.urlopen(f"{base_url.rstrip('/')}/health", timeout=5.0) as response:
        body = json.loads(response.read())
        if response.status != 200 or body.get("status") != "ok":
            raise ValueError("paired broker health check failed")
        return {"status_code": response.status, "body": body}


def build_preflight(
    *,
    runtime_root: Path,
    baseline_dir: Path,
    broker_url: str,
    expected_tag: str,
    release_identity_path: Path | None,
    upstream_key_env: str,
    client_token_env: str,
) -> dict[str, Any]:
    root = validate_runtime_root(runtime_root)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    clean_tree_count = verify_clean_tree_manifest(PROJECT_ROOT, CLEAN_MANIFEST)
    release = resolve_release_identity(
        root=PROJECT_ROOT,
        expected_tag=expected_tag,
        release_identity_path=release_identity_path,
        source_sums_path=SOURCE_SUMS,
        clean_manifest_path=CLEAN_MANIFEST,
    )
    source_count = verify_source_sums(PROJECT_ROOT, SOURCE_SUMS)
    s0 = validate_records()
    leakage = json.loads(LEAKAGE_AUDIT.read_text(encoding="utf-8"))
    if leakage["audit_verdict"] != "NO_KNOWN_HISTORICAL_SEARCH_LEAKAGE_DETECTED":
        raise ValueError("S0 historical leakage audit did not pass")

    recbole_root = Path(contract["execution_environment"]["recbole_root"]).resolve()
    expected_recbole = contract["execution_environment"]["recbole_runtime_files_sha256"]
    recbole = exact_file_identities(recbole_root, expected_recbole)
    dataset_root = recbole_root / "dataset" / contract["protocol"]["dataset"]
    dataset_tree, dataset_files = tree_identity(dataset_root)
    expected_dataset = str(contract["protocol"]["dataset_snapshot"]).removeprefix("sha256:")
    if dataset_tree != expected_dataset:
        raise ValueError(f"dataset tree identity drift: {dataset_tree} != {expected_dataset}")
    if dataset_files != contract["execution_environment"]["dataset_files_sha256"]:
        raise ValueError("dataset file inventory or identity drift")
    baseline_rows = validate_baseline_inputs(baseline_dir)

    for secret_name in (upstream_key_env, client_token_env):
        if not os.environ.get(secret_name, ""):
            raise ValueError(f"required non-empty secret variable missing: {secret_name}")
    if (root / "runs").exists():
        raise ValueError("fresh Canary runtime must not already contain the primary runs root")
    broker_db = root / "broker" / "paired_llm.sqlite3"
    if not broker_db.is_file():
        raise ValueError("paired broker database is missing after broker health start")
    with sqlite3.connect(broker_db) as connection:
        request_count = int(connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
    if request_count != 0:
        raise ValueError("fresh Canary broker database already contains requests")
    capacity = shutil.disk_usage(root.parent if not root.exists() else root)
    if capacity.free < 100 * 1024**3:
        raise ValueError("external runtime filesystem has less than 100 GiB free")

    preflight_materialization = root / "preflight" / "materialization"
    if preflight_materialization.exists():
        raise ValueError("preflight materialization root must be absent")
    control = materialize_arm(
        runtime_root=preflight_materialization, arm="control", search_seed=9001
    )
    treatment = materialize_arm(
        runtime_root=preflight_materialization, arm="treatment", search_seed=9001
    )
    if control["control_guard_material_count"] != 0 or control["integration_files"]:
        raise ValueError("Control materialization contains Guard material")
    treatment_agent = next(
        row for row in treatment["source_files"] if row["path"] == "scripts/agent.py"
    )
    if treatment_agent["sha256"] != FROZEN_TREATMENT_AGENT_SHA256:
        raise ValueError("Treatment materialized agent identity drift")

    try:
        import torch
        import recbole
    except ImportError as exc:
        raise ValueError(f"required runtime import failed: {exc}") from exc
    if recbole.__version__ != contract["execution_environment"]["recbole"]:
        raise ValueError("imported RecBole version drift")
    if torch.__version__ != contract["execution_environment"]["torch"]:
        raise ValueError("imported PyTorch version drift")

    return {
        "record_type": "RECCLAW_PHASE1_AB002_GPU5_PRE_CANARY_PREFLIGHT",
        "schema_version": "recclaw.phase1.ab002.gpu5_pre_canary_preflight.v1",
        "status": "LOCAL_COMPLETE",
        "gate_status": "NOT_STARTED",
        "canary_started": False,
        "full_ab_started": False,
        "host": platform.node(),
        "release": {
            **release,
            "source_sha256s_digest": sha256_file(SOURCE_SUMS),
            "source_sha256_checked_count": source_count,
            "clean_tree_manifest_digest": sha256_file(CLEAN_MANIFEST),
            "clean_tree_checked_count": clean_tree_count,
        },
        "s0": {
            "s0_id": s0["s0_id"],
            "source_tree_sha256": s0["source_tree_sha256"],
            "manifest_sha256": sha256_file(PROJECT_ROOT / "phase1/s0/ab002/initial_state_manifest.json"),
            "leakage_audit_sha256": sha256_file(LEAKAGE_AUDIT),
            "leakage_verdict": leakage["audit_verdict"],
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "recclaw_distribution": importlib.metadata.version("recclaw-phase1"),
            "recbole": recbole.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "gpu_inspection_not_reservation": gpu_inventory(),
        "dataset": {
            "path": str(dataset_root),
            "tree_sha256": dataset_tree,
            "files_sha256": dataset_files,
        },
        "recbole_root": str(recbole_root),
        "recbole_runtime_files_sha256": recbole,
        "baseline_inputs": baseline_rows,
        "shared_state_before": {
            "dataset_tree_sha256": dataset_tree,
            "recbole_runtime_files_sha256": recbole,
        },
        "materialization": {
            "root": str(preflight_materialization),
            "control_manifest": str(
                Path(control["source_root"]).parent / "materialization_manifest.json"
            ),
            "treatment_manifest": str(
                Path(treatment["source_root"]).parent / "materialization_manifest.json"
            ),
            "control_guard_material_count": 0,
            "treatment_agent_sha256": treatment_agent["sha256"],
            "frozen_guard_files_sha256": FROZEN_GUARD_FILES,
        },
        "runtime_isolation": {
            "external_root": str(root),
            "primary_runs_root_absent": True,
            "fresh_broker_db_path": str(broker_db),
            "fresh_broker_request_count": request_count,
            "free_bytes": capacity.free,
        },
        "broker": broker_health(broker_url),
        "secrets": {
            upstream_key_env: "NON_EMPTY_NOT_RECORDED",
            client_token_env: "NON_EMPTY_NOT_RECORDED",
        },
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the direct AB-002 gpu5 pre-Canary preflight")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--release-identity", type=Path)
    parser.add_argument("--upstream-api-key-env", default="RECCLAW_LAB_LLM_API_KEY")
    parser.add_argument("--client-token-env", default="RECCLAW_BROKER_CLIENT_TOKEN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_preflight(
        runtime_root=args.runtime_root,
        baseline_dir=args.baseline_dir,
        broker_url=args.broker_url,
        expected_tag=args.expected_tag,
        release_identity_path=args.release_identity,
        upstream_key_env=args.upstream_api_key_env,
        client_token_env=args.client_token_env,
    )
    canonical_write(args.output.expanduser().resolve(), record)
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
