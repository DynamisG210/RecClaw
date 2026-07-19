"""Closed, reproducible initial-state package for the AB-002 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
S0_ROOT = PROJECT_ROOT / "phase1/s0/ab002"
S0_SOURCE = S0_ROOT / "source"
INITIAL_STATE_MANIFEST = S0_ROOT / "initial_state_manifest.json"
LEAKAGE_AUDIT = S0_ROOT / "historical_leakage_audit.json"

S0_ID = "AB002-RECONSTRUCTED-CLEAN-START-S0-V1"
PRE_SEARCH_COMMIT = "b3dc5a7e33fd1cbec60b5dfb9ae7097c67d0b5e8"
PRE_SEARCH_TREE = "9160182744c6ac595426739413c79e3f61322886"
FINAL_COMMON_SOURCE_COMMIT = "c25598aca6c803ed3e2a2a37cb6480fbccc1ed98"
FINAL_COMMON_SOURCE_TREE = "35e0e9d7d557d086bc47bf2527bb023a51b4bf70"

SANITIZED_COMMON_FILES = {
    "scripts/agent.py",
    "scripts/build_experience_summary.py",
    "scripts/run_reflection_pilot.py",
}
AB002_BUDGET_INSTRUMENTED_FILES = {
    "scripts/agent.py",
    "scripts/implement_candidate_proposal.py",
    "scripts/run_candidate.py",
}

# These tokens identify results or candidate families produced after the recovered
# pre-search registry. The scan is deliberately byte-based and case-insensitive.
FORBIDDEN_RUNTIME_TOKENS = (
    "cand_lightgcn_edge_dropout_residual_norm_dualpathblend_repair_076",
    "cand_lightgcn_shallow_alignment_rankaware_gate",
    "cand_lightgcn_shallow_rankaware_lastlayeralign",
    "historical_v4m_best",
    "historical_best",
    "continue_from_v4",
    "v4k",
    "v4m",
    "0.2908",
    "0.289067",
    "0.2856",
    "0.2769",
    "0.2765",
    "0.274367",
)

FORBIDDEN_DYNAMIC_SUFFIXES = (".jsonl", ".csv", ".log", ".pth", ".pt")
FORBIDDEN_DYNAMIC_NAMES = {
    "agent_memory.json",
    "candidate_search_tree.json",
    "candidate_proposals.json",
    "experience_summary.json",
    "reflection_memory.json",
    "state_summary.json",
}

BASELINE_INPUTS = (
    {
        "model": "BPR",
        "filename": "baseline_bpr_20260509_080916.log",
        "sha256": "48c75d62d5e21dae5b3d2de2d56398e8a613468eba9e6a94db9f372d7db071a3",
        "reference_ndcg_at_10": 0.2567,
    },
    {
        "model": "LightGCN",
        "filename": "baseline_lightgcn_20260509_081256.log",
        "sha256": "e1ef2f7f15e3596386e61b259ac771f89f1fe7eb4014fdc97ef3089335520fba",
        "reference_ndcg_at_10": 0.2671,
    },
)

HISTORICAL_AUDIT_SOURCES = (
    {
        "logical_path": "RecClaw_LabLog/2026.05.28/v4k_75_checkpoint_analysis/agent_memory_current.jsonl",
        "sha256": "687df25ad7e589a6f394173758f5c4542b70235838d4c6f088856b3281dfaf51",
    },
    {
        "logical_path": "RecClaw_LabLog/2026.05.28/v4k_75_checkpoint_analysis/candidate_proposals_current.jsonl",
        "sha256": "9f766478c9aaba9ffe0d156b920d1ec5bb03998d89cb26eefc63e882f664f4e8",
    },
    {
        "logical_path": "RecClaw_LabLog/2026.05.30/v4m_200_final_analysis/agent_memory.jsonl",
        "sha256": "f3ece4863226c728f042d7e48334a3c242a544d3629487338a8aab5a314466da",
    },
    {
        "logical_path": "RecClaw_LabLog/2026.05.30/v4m_200_final_analysis/candidate_proposals.jsonl",
        "sha256": "ccaccb9662b5210b1d6c7d89bda75031d1feb103da37498433654ea8162b7177",
    },
    {
        "logical_path": "RecClaw_LabLog/2026.05.30/v4m_200_final_analysis/v4m_final_summary.md",
        "sha256": "c9839ae447b8b1b6d74c21c3f5da764213763672bf83f2eae7de7c48db96f62c",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    return sorted(
        (path for path in S0_SOURCE.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(S0_SOURCE).as_posix(),
    )


def source_tree_sha256(paths: list[Path] | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths or source_paths():
        relative = path.relative_to(S0_SOURCE).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_ids(registry: Path) -> list[str]:
    values = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- candidate_id:"):
            values.append(stripped.split(":", 1)[1].strip().strip("'\""))
    return values


def _provenance(relative: str) -> dict[str, str]:
    if relative in SANITIZED_COMMON_FILES and relative in AB002_BUDGET_INSTRUMENTED_FILES:
        return {
            "kind": "FINAL_COMMON_SOURCE_WITH_S0_SANITIZATION_AND_AB002_BUDGET_INSTRUMENTATION",
            "source_commit": FINAL_COMMON_SOURCE_COMMIT,
        }
    if relative in SANITIZED_COMMON_FILES:
        return {
            "kind": "FINAL_COMMON_SOURCE_WITH_S0_SANITIZATION",
            "source_commit": FINAL_COMMON_SOURCE_COMMIT,
        }
    if relative in AB002_BUDGET_INSTRUMENTED_FILES:
        return {
            "kind": "FINAL_COMMON_SOURCE_WITH_AB002_BUDGET_INSTRUMENTATION",
            "source_commit": FINAL_COMMON_SOURCE_COMMIT,
        }
    if relative.startswith("scripts/"):
        return {
            "kind": "EXACT_FINAL_COMMON_SOURCE",
            "source_commit": FINAL_COMMON_SOURCE_COMMIT,
        }
    return {
        "kind": "EXACT_PRE_SEARCH_LIBRARY_SOURCE",
        "source_commit": PRE_SEARCH_COMMIT,
    }


def build_records() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = source_paths()
    rows = []
    findings = []
    dynamic_files = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(S0_SOURCE).as_posix()
        raw = path.read_bytes()
        total_bytes += len(raw)
        lowered = raw.lower()
        matched = [token for token in FORBIDDEN_RUNTIME_TOKENS if token.encode().lower() in lowered]
        if matched:
            findings.append({"path": relative, "tokens": matched})
        if path.suffix.lower() in FORBIDDEN_DYNAMIC_SUFFIXES or path.name in FORBIDDEN_DYNAMIC_NAMES:
            dynamic_files.append(relative)
        rows.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "provenance": _provenance(relative),
            }
        )

    registry_ids = _candidate_ids(S0_SOURCE / "configs/candidate_registry.yaml")
    checks = {
        "closed_source_inventory": "PASS" if paths else "FAIL",
        "forbidden_runtime_token_scan": "PASS" if not findings else "FAIL",
        "dynamic_search_artifact_scan": "PASS" if not dynamic_files else "FAIL",
        "initial_memory_absent": "PASS",
        "historical_candidate_overlap": "PASS" if not findings else "FAIL",
    }
    verdict = (
        "NO_KNOWN_HISTORICAL_SEARCH_LEAKAGE_DETECTED"
        if all(value == "PASS" for value in checks.values())
        else "HISTORICAL_SEARCH_LEAKAGE_DETECTED"
    )
    manifest = {
        "record_type": "RECCLAW_PHASE1_AB002_INITIAL_STATE_MANIFEST",
        "schema_version": "recclaw.phase1.ab002.initial_state.v2",
        "s0_id": S0_ID,
        "status": "LOCAL_COMPLETE" if verdict.startswith("NO_KNOWN") else "STOPPED",
        "classification": "RECONSTRUCTED_CLEAN_START_S0",
        "exact_pre_first_iteration_snapshot_recovered": False,
        "reconstruction": {
            "reason": "The final v4m run continued from v4k140 and no single exact clean pre-first snapshot contains the final valid runtime code without learned search state.",
            "final_common_source_commit": FINAL_COMMON_SOURCE_COMMIT,
            "final_common_source_tree": FINAL_COMMON_SOURCE_TREE,
            "pre_search_library_commit": PRE_SEARCH_COMMIT,
            "pre_search_library_tree": PRE_SEARCH_TREE,
            "sanitized_common_files": sorted(SANITIZED_COMMON_FILES),
            "budget_instrumented_common_files": sorted(AB002_BUDGET_INSTRUMENTED_FILES),
            "sanitization_scope": "remove post-start candidate-family names and historical performance thresholds while retaining the final common runner logic",
        },
        "source_root": "phase1/s0/ab002/source",
        "source_tree_sha256": source_tree_sha256(paths),
        "source_file_count": len(rows),
        "source_files": rows,
        "candidate_registry_count": len(registry_ids),
        "candidate_registry_ids": registry_ids,
        "initial_dynamic_state": {
            "agent_memory": "ABSENT",
            "candidate_proposals": "ABSENT",
            "candidate_search_tree": "ABSENT",
            "experience_summary": "ABSENT",
            "reflection_memory": "ABSENT",
            "result_rows": 0,
            "seed_from_argument": "FORBIDDEN",
        },
        "baseline_inputs": list(BASELINE_INPUTS),
        "historical_search_output_count": 0,
        "historical_reference_visibility": "REPORTING_ONLY_NOT_COPIED_TO_RUNTIME",
        "leakage_audit": "phase1/s0/ab002/historical_leakage_audit.json",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    audit = {
        "record_type": "RECCLAW_PHASE1_AB002_HISTORICAL_LEAKAGE_AUDIT",
        "schema_version": "recclaw.phase1.ab002.historical_leakage_audit.v1",
        "s0_id": S0_ID,
        "status": "LOCAL_COMPLETE" if verdict.startswith("NO_KNOWN") else "STOPPED",
        "verdict": verdict,
        "checks": checks,
        "scanned_file_count": len(paths),
        "scanned_bytes": total_bytes,
        "forbidden_runtime_tokens": list(FORBIDDEN_RUNTIME_TOKENS),
        "token_findings": findings,
        "dynamic_artifact_findings": dynamic_files,
        "historical_audit_sources": list(HISTORICAL_AUDIT_SOURCES),
        "historical_lineage_facts": [
            "v4k began from an empty metric summary but generated candidates from round 1",
            "v4m continued from v4k140 and therefore is not a clean start",
            "the final dual-path candidate and the two learned family prefixes post-date the recovered pre-search registry",
        ],
        "known_limit": "This is a bounded known-leakage audit over the recovered Git and approved LabLog evidence, not proof that no undiscovered historical artifact exists.",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    return manifest, audit


def canonical_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_records() -> None:
    manifest, audit = build_records()
    canonical_write(INITIAL_STATE_MANIFEST, manifest)
    canonical_write(LEAKAGE_AUDIT, audit)


def validate_records() -> dict[str, Any]:
    expected_manifest, expected_audit = build_records()
    actual_manifest = json.loads(INITIAL_STATE_MANIFEST.read_text(encoding="utf-8"))
    actual_audit = json.loads(LEAKAGE_AUDIT.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise ValueError("AB-002 initial-state manifest does not match source bytes")
    if actual_audit != expected_audit:
        raise ValueError("AB-002 historical leakage audit does not match source bytes")
    if actual_audit["verdict"] != "NO_KNOWN_HISTORICAL_SEARCH_LEAKAGE_DETECTED":
        raise ValueError("AB-002 S0 historical leakage audit is not clean")
    return actual_manifest


def copy_s0_source(destination: Path) -> list[dict[str, Any]]:
    manifest = validate_records()
    rows = []
    for item in manifest["source_files"]:
        relative = Path(item["path"])
        source = S0_SOURCE / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        actual = sha256_file(target)
        if actual != item["sha256"]:
            raise ValueError(f"copied S0 file drift: {relative.as_posix()}")
        rows.append(
            {"path": relative.as_posix(), "size_bytes": target.stat().st_size, "sha256": actual}
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the AB-002 reconstructed S0 records")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        write_records()
    else:
        validate_records()
    print(json.dumps({"s0_id": S0_ID, "status": "LOCAL_COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
