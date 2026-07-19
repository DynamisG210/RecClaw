#!/usr/bin/env python3
"""Build deterministic Phase-1 delivery and provenance index manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


BASE_COMMIT = "0b44db72f2e44bfbf8139b43c9624e1e89f52b35"
BASE_TREE = "3f9049509e5e09ae59a0d6aba79a5c2094dd3c2c"
GENERATED_ROOTS = {"SOURCE_SHA256SUMS", "clean_tree_manifest.json"}
REMOVED_BASE_PATHS = (
    "notes/candidate_library.md",
    "notes/candidate_proposal_workflow.md",
    "notes/experiment_log.md",
    "notes/method_change_space.md",
    "recclaw_program.md",
    "results/baseline/.gitkeep",
    "results/candidates/.gitkeep",
    "results/results.csv",
    "scripts/analysis/summarize_ablation_results.py",
    "scripts/run_baseline.sh",
    "scripts/sync_to_server.sh",
)
MIGRATED_PATHS = (
    ".gitattributes",
    "configs/phase1/ab002/arm_manifest.json",
    "configs/phase1/ab002/common_llm_broker_spec.json",
    "configs/phase1/ab002/experiment_contract.json",
    "configs/phase1/ab002/experiment_contract.md",
    "configs/phase1/ab002/neutral_run_audit_policy.json",
    "configs/phase1/ab002/outcome_audit_policy.json",
    "phase1/overlays/treatment_agent.patch",
    "pyproject.toml",
    "schemas/evidence_guard_envelope.schema.json",
    "schemas/evidence_guard_result.schema.json",
    "schemas/phase1_ab002_contract.schema.json",
    "schemas/phase1_milestone_record.schema.json",
    "scripts/run_reflection_pilot.py",
    "src/recclaw_core/__init__.py",
    "src/recclaw_core/exploration/__init__.py",
    "src/recclaw_core/exploration/evidence_guard.py",
    "src/recclaw_core/exploration/original_recclaw_adapter.py",
    "src/recclaw_core/exploration/original_recclaw_guard_hook.py",
    "src/recclaw_phase1/__init__.py",
    "src/recclaw_phase1/ab002_analysis.py",
    "src/recclaw_phase1/ab002_blind_packager.py",
    "src/recclaw_phase1/ab002_canary_audit.py",
    "src/recclaw_phase1/ab002_canary_orchestrator.py",
    "src/recclaw_phase1/ab002_canary_probes.py",
    "src/recclaw_phase1/ab002_candidate_confirmation.py",
    "src/recclaw_phase1/ab002_final_analysis.py",
    "src/recclaw_phase1/ab002_launcher.py",
    "src/recclaw_phase1/ab002_pair_runner.py",
    "src/recclaw_phase1/ab002_preflight.py",
    "src/recclaw_phase1/ab002_s0.py",
    "src/recclaw_phase1/ab002_three_seed_baseline.py",
    "src/recclaw_phase1/neutral_outcome_auditor.py",
    "src/recclaw_phase1/paired_llm_broker.py",
    "tests/exploration/__init__.py",
    "tests/exploration/test_evidence_guard.py",
    "tests/exploration/test_original_recclaw_adapter.py",
    "tests/exploration/test_original_recclaw_treatment_overlay.py",
    "tests/phase1/__init__.py",
    "tests/phase1/test_ab002_analysis.py",
    "tests/phase1/test_ab002_blind_packager.py",
    "tests/phase1/test_ab002_canary_probes.py",
    "tests/phase1/test_ab002_canary_orchestrator.py",
    "tests/phase1/test_ab002_candidate_confirmation.py",
    "tests/phase1/test_ab002_final_analysis.py",
    "tests/phase1/test_ab002_launcher.py",
    "tests/phase1/test_ab002_pair_runner.py",
    "tests/phase1/test_ab002_preflight.py",
    "tests/phase1/test_ab002_s0_runner_budget.py",
    "tests/phase1/test_ab002_three_seed_baseline.py",
    "tests/phase1/test_neutral_run_auditor.py",
    "tests/phase1/test_paired_llm_broker.py",
    "tests/phase1/test_phase1_schemas.py",
)


def migrated_paths(root: Path) -> tuple[str, ...]:
    s0_paths = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "phase1/s0/ab002").rglob("*")
        if path.is_file()
    )
    return tuple(sorted(set((*MIGRATED_PATHS, *s0_paths))))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inventory_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return sorted(
        path
        for path in completed.stdout.splitlines()
        if path and path not in GENERATED_ROOTS and (root / path).is_file()
    )


def file_rows(root: Path, paths: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def build(root: Path, archive_dir: Path) -> None:
    archive_root = archive_dir / "RecClaw_development_provenance_v2"
    archive_tar = archive_dir / f"{archive_dir.name}.tar.gz"
    archive_manifest = archive_root / "provenance_archive_manifest.json"
    archive_sums = archive_root / "SHA256SUMS"
    excluded_inventory = archive_root / "inventory/archive_excluded_file_inventory.json"
    status_inventory = archive_root / "inventory/git_status_porcelain.txt"
    required_archive_files = (
        archive_tar,
        archive_manifest,
        archive_sums,
        excluded_inventory,
        status_inventory,
    )
    missing = [str(path) for path in required_archive_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("archive inputs missing: " + ", ".join(missing))

    archived = json.loads(archive_manifest.read_text(encoding="utf-8"))
    excluded = json.loads(excluded_inventory.read_text(encoding="utf-8"))
    reason_counts = Counter(str(row["reason"]) for row in excluded["files"])

    provenance_index = {
        "record_type": "RECCLAW_PHASE1_PROVENANCE_ARCHIVE_INDEX",
        "schema_version": "recclaw.phase1.provenance_archive_index.v1",
        "archive_basename": archive_tar.name,
        "archive_external_path": str(archive_tar),
        "archive_sha256": sha256_file(archive_tar),
        "archive_size_bytes": archive_tar.stat().st_size,
        "archive_manifest_sha256": sha256_file(archive_manifest),
        "archive_sha256sums_sha256": sha256_file(archive_sums),
        "archived_workspace_head": archived["source_identity"]["head"],
        "archived_workspace_tree": archived["source_identity"]["tree"],
        "archived_workspace_branch": archived["source_identity"]["branch"],
        "included_file_count": archived["included_file_count"],
        "excluded_candidate_count": archived["excluded_candidate_count"],
        "source_workspace_files_modified": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    write_json(root / "provenance_archive_manifest.json", provenance_index)

    exclusion_index = {
        "record_type": "RECCLAW_PHASE1_EXCLUDED_FILE_INVENTORY_INDEX",
        "schema_version": "recclaw.phase1.excluded_file_inventory.v1",
        "path_level_inventory_location": str(excluded_inventory),
        "path_level_inventory_sha256": sha256_file(excluded_inventory),
        "path_level_inventory_count": excluded["count"],
        "reason_counts": dict(sorted(reason_counts.items())),
        "delivery_migration_exclusions": [
            "superseded drafts",
            "intermediate Codex work packages",
            "stopped package candidates and duplicate handoffs",
            "local wheel, tar, and unpacked source copies",
            "raw review transcripts and temporary source snapshots",
            "pycache, pyc, cache, environment-dump, and temporary log files",
            "large raw artifacts and historical experiment outputs",
            "historical execution records other than five Phase-1 milestones",
            "failed permission-expanded research-loop implementation",
        ],
        "removed_from_base_tree": list(REMOVED_BASE_PATHS),
        "rationale": (
            "The exact path-level exclusion inventory remains in the external provenance "
            "archive so historical names and bulk inventory do not expand the minimal delivery tree."
        ),
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    write_json(root / "excluded_file_inventory.json", exclusion_index)

    migrated_rows = file_rows(root, migrated_paths(root))
    allowlist = {
        "record_type": "RECCLAW_PHASE1_MIGRATED_FILE_ALLOWLIST",
        "schema_version": "recclaw.phase1.migrated_file_allowlist.v1",
        "base_commit": BASE_COMMIT,
        "base_tree": BASE_TREE,
        "policy": "Only paths listed here were migrated or authored beyond retained base code.",
        "migrated_functional_file_count": len(migrated_rows),
        "migrated_functional_files": migrated_rows,
        "common_base_modifications": [
            "the closed S0 common source carries identical search-seed, budget, and leakage-sanitization changes for both arms"
        ],
        "control_guard_source_count": 0,
        "treatment_guard_binding": "src/recclaw_phase1/ab002_launcher.py:FROZEN_GUARD_FILES",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    write_json(root / "migrated_file_allowlist.json", allowlist)

    source_paths = inventory_paths(root)
    source_rows = file_rows(root, source_paths)
    (root / "SOURCE_SHA256SUMS").write_text(
        "".join(f'{row["sha256"]}  {row["path"]}\n' for row in source_rows),
        encoding="utf-8",
    )

    delivery_paths = inventory_paths(root) + ["SOURCE_SHA256SUMS"]
    delivery_rows = file_rows(root, sorted(set(delivery_paths)))
    clean_manifest = {
        "record_type": "RECCLAW_PHASE1_CLEAN_TREE_MANIFEST",
        "schema_version": "recclaw.phase1.clean_tree_manifest.v1",
        "base_commit": BASE_COMMIT,
        "base_tree": BASE_TREE,
        "branch": "phase1/evidence-guard-ab002",
        "manifest_self_excluded": True,
        "file_count": len(delivery_rows),
        "files": delivery_rows,
        "source_sha256sums_sha256": sha256_file(root / "SOURCE_SHA256SUMS"),
        "canary_started": False,
        "training_started": False,
        "llm_api_used": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    write_json(root / "clean_tree_manifest.json", clean_manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(), args.archive_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
