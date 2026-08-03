#!/usr/bin/env python3
"""Package sealed Q4 multi-round soak evidence for the next research-line gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.multiround_soak import (  # noqa: E402
    ROUND_STAGE_ORDER,
    resume_round,
)


EXPECTED_GPU = {
    "hostname": "gpu35-tingrangan",
    "user": "tingrangan",
    "name": "NVIDIA GeForce RTX 3080",
    "memory_mib": 10240,
    "driver": "550.76",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def _file(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": bytes_sha256(resolved.read_bytes()),
        "size_bytes": resolved.stat().st_size,
    }


def _receipt(round_root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    path = round_root / "remote_evidence" / name
    return path, _read(path)


def _round_summary(round_root: Path, *, campaign_root: Path) -> dict[str, Any]:
    manifest_path = round_root / "ROUND_MANIFEST.json"
    ledger_path = round_root / "STAGE_LEDGER.json"
    manifest = _read(manifest_path)
    ledger = _read(ledger_path)
    resume = resume_round(manifest_path=manifest_path, ledger_path=ledger_path)
    if not resume["complete"] or resume["held_out_reads"] != 0 or resume["retries"] != 0:
        raise RuntimeError(f"round is not cleanly sealed: {round_root}")
    stages = ledger["stages"]
    if [stage["stage"] for stage in stages] != list(ROUND_STAGE_ORDER):
        raise RuntimeError(f"round stage chain drift: {round_root}")

    provider_path, provider = _receipt(
        round_root, "next_pool/PROVIDER_RESOLVER_RECEIPT.json"
    )
    implementer_path, implementer = _receipt(round_root, "IMPLEMENTER_RECEIPT.json")
    qualifier_path, qualifier = _receipt(round_root, "MATERIALIZE_QUALIFIER_RECEIPT.json")
    resource_path, resource = _receipt(round_root, "RESOURCE_ADMISSION_RECEIPT.json")
    matched_path, matched = _receipt(round_root, "MATCHED_EXECUTION_RECEIPT.json")
    episode_path, episode = _receipt(round_root, "EPISODE_RECEIPT.json")
    authority_path, authority = _receipt(round_root, "AUTHORITY_UPDATE_RECEIPT.json")
    activation_path, activation = _receipt(round_root, "POLICY_ACTIVATION_RECEIPT.json")
    preflight_path, preflight = _receipt(round_root, "REMOTE_PREFLIGHT_RECEIPT.json")

    if (
        preflight.get("status") != "PASS"
        or preflight.get("cuda_available") is not True
        or preflight.get("cuda_device_count") != 1
        or preflight.get("cuda_device_name") != EXPECTED_GPU["name"]
    ):
        raise RuntimeError(f"gpu35 physical identity drift: {round_root}")
    for receipt in (
        provider,
        implementer,
        qualifier,
        resource,
        matched,
        episode,
        authority,
        activation,
    ):
        if receipt.get("held_out_reads") != 0:
            raise RuntimeError(f"held-out read in round: {round_root}")
        if receipt.get("scientific_effect_claim") is not False:
            raise RuntimeError(f"scientific claim escaped development lane: {round_root}")

    physical_ids = resume["reuse_forbidden_physical_call_ids"]
    if len(physical_ids) != len(set(physical_ids)):
        raise RuntimeError(f"physical identity reuse: {round_root}")
    task_acquisitions = manifest["task_acquisitions"]
    if sorted(task_acquisitions) != ["EXPERIMENT", "IDEA", "REPLICATION"]:
        raise RuntimeError(f"acquisition head separation drift: {round_root}")
    if any(
        value.get("candidate_count") != 4
        or len(value.get("candidates", [])) != 4
        or value.get("selection_probabilities_sum") != 1.0
        for value in task_acquisitions.values()
    ):
        raise RuntimeError(f"full-pool denominator drift: {round_root}")

    qualification_receipt = qualifier.get("qualification", {}).get("receipt", {})
    qualification_failure = qualifier.get("qualification", {}).get("failure_detail") or {}

    attempts = implementer.get("attempts", [])
    provider_calls = int(provider["proposal_provider_usage"]["physical_calls"]) + len(attempts)
    training_runs = int(matched.get("physical_training_runs", 0)) + (
        1 if resource.get("resource_probe") else 0
    )
    matched_summary: dict[str, Any] = {
        "status": matched["status"],
        "physical_training_runs": matched.get("physical_training_runs", 0),
        "serial_execution": matched.get("serial_execution", True),
        "retries": matched.get("retries", 0),
    }
    if matched.get("baseline") and matched.get("candidate"):
        matched_summary.update(
            {
                "seed": matched["matched_seed"],
                "baseline": {
                    "exit_status": matched["baseline"]["exit_status"],
                    "epochs": matched["baseline"]["epochs_requested"],
                    "wall_time_ms": matched["baseline"]["wall_time_ms"],
                    "metrics": matched["baseline"]["metrics"],
                },
                "candidate": {
                    "exit_status": matched["candidate"]["exit_status"],
                    "epochs": matched["candidate"]["epochs_requested"],
                    "wall_time_ms": matched["candidate"]["wall_time_ms"],
                    "metrics": matched["candidate"]["metrics"],
                },
            }
        )

    return canonical_value(
        {
            "round_index": manifest["round_index"],
            "manifest_digest": manifest["manifest_digest"],
            "manifest_sha256": bytes_sha256(manifest_path.read_bytes()),
            "input_policy_digest": manifest["frozen_inputs"]["policy_digest"],
            "input_activation_digest": manifest["frozen_inputs"]["activation_digest"],
            "expected_previous_round_activation_digest": manifest[
                "expected_previous_round_activation_digest"
            ],
            "input_pool_digest": manifest["frozen_inputs"]["full_pool_digest"],
            "selection": manifest["selected_candidate"],
            "task_acquisitions": task_acquisitions,
            "stages": [
                {
                    "stage": stage["stage"],
                    "status": stage["status"],
                    "stage_digest": stage["stage_digest"],
                    "physical_call_ids": stage["process"]["physical_call_ids"],
                    "artifact_count": len(stage["artifacts"]),
                }
                for stage in stages
            ],
            "provider_resolver": {
                "status": provider["status"],
                "candidate_count": provider["candidate_count"],
                "physical_calls": provider["proposal_provider_usage"]["physical_calls"],
                "successful_calls": provider["proposal_provider_usage"]["successful_calls"],
                "retries": provider["retries"],
                "next_pool_sha256": provider["frozen_pool_sha256"],
            },
            "implementer": {
                "status": implementer["status"],
                "physical_calls": len(attempts),
                "maximum_physical_attempts": implementer["maximum_physical_attempts"],
                "retries": implementer["retries"],
                "manual_candidate_patches": implementer["manual_candidate_patches"],
            },
            "qualifier": {
                "status": qualifier["status"],
                "qualification_status": qualifier.get("qualification_status"),
                "static_result": qualification_receipt.get("static_result"),
                "construction_result": qualification_receipt.get("construction_result"),
                "api_contract_result": qualification_receipt.get("api_contract_result"),
                "smoke_result": qualification_receipt.get("smoke_result"),
                "failure_class": qualification_failure.get("failure_class"),
                "reason_code": qualification_failure.get("reason_code"),
                "manual_candidate_patches": qualifier["manual_candidate_patches"],
            },
            "resource_admission": {
                "status": resource["status"],
                "admitted": resource["admitted"],
                "future_eligible": resource["future_eligible"],
                "effect_update_allowed": resource["effect_update_allowed"],
            },
            "matched_development": matched_summary,
            "episode": {
                "status": episode["status"],
                "missingness": episode["missingness"],
                "effect_update_allowed": episode["effect_update_allowed"],
                "evidence_class": (
                    episode.get("typed_episode") or {}
                ).get("evidence_class"),
                "failure_class": (
                    episode.get("typed_episode") or {}
                ).get("failure_class"),
                "outcome_summary": episode.get("outcome_summary"),
            },
            "authority_update": {
                "status": authority["status"],
                "round_denominator_count": authority["round_denominator_count"],
                "round_mechanism_information_update_count": authority[
                    "round_mechanism_information_update_count"
                ],
                "round_effect_update_count": authority["round_effect_update_count"],
                "head_update_counts": authority["head_update_counts"],
                "projection_digest": authority["projection_digest"],
                "policy_digest": authority["policy_digest"],
            },
            "activation": {
                "status": activation["status"],
                "activation_digest": activation["activation_digest"],
                "policy_digest": activation["policy_digest"],
                "parent_policy_digest": activation["parent_policy_digest"],
                "reversible": activation["reversible"],
            },
            "counts": {
                "physical_provider_calls": provider_calls,
                "physical_training_runs": training_runs,
                "retries": 0,
                "held_out_reads": 0,
            },
            "artifacts": [
                _file(path, root=campaign_root)
                for path in (
                    manifest_path,
                    ledger_path,
                    preflight_path,
                    provider_path,
                    implementer_path,
                    qualifier_path,
                    resource_path,
                    matched_path,
                    episode_path,
                    authority_path,
                    activation_path,
                )
            ],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--canonical-receipt", type=Path, required=True)
    args = parser.parse_args()
    campaign_root = args.campaign_root.resolve()
    plan_path = campaign_root / "CAMPAIGN_PLAN.json"
    plan = _read(plan_path)
    rounds = [
        _round_summary(campaign_root / f"round_{index:02d}", campaign_root=campaign_root)
        for index in (1, 2, 3)
    ]
    for previous, current in zip(rounds, rounds[1:]):
        if current["input_activation_digest"] != previous["activation"]["activation_digest"]:
            raise RuntimeError("round did not consume the previous activation")
        if current["expected_previous_round_activation_digest"] != previous["activation"][
            "activation_digest"
        ]:
            raise RuntimeError("round activation expectation drift")

    totals = {
        "rounds": len(rounds),
        "physical_provider_calls": sum(
            item["counts"]["physical_provider_calls"] for item in rounds
        ),
        "physical_training_runs": sum(
            item["counts"]["physical_training_runs"] for item in rounds
        ),
        "complete_matched_episodes": sum(
            item["episode"]["effect_update_allowed"] is True for item in rounds
        ),
        "missing_episodes": sum(item["episode"]["missingness"] is not None for item in rounds),
        "retries": 0,
        "held_out_reads": 0,
        "maximum_training_concurrency": 1,
    }
    package_payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-multiround-soak-package.v1",
            "campaign_id": plan["campaign_id"],
            "status": "DEVELOPMENT_ONLY_PASS",
            "development_only": True,
            "scientific_effect_claim": False,
            "policy_superiority_claim": False,
            "held_out_reads": 0,
            "base": {
                "commit": "fe1678d286fc710202265d145ed58940b1a373f7",
                "parent": "49f9225517d76582500bcf80bf7fc6a5d112f137",
                "tree": "11658cd121bcc074f4f1436efe25e32ab3eb239e",
            },
            "accepted_q3_inputs": {
                "physical_receipt_sha256": "dfdd6d4dbb5f62c43f59cf48f4080816485e589df8da7af31a804ea1ca9731d2",
                "active_policy_file_sha256": "f5359e5430be13c4d596b99f194f4a263d8e0ae222777cb1292ae7b87443eeb1",
                "soak_package_file_sha256": "085071c9f1fad03a436a44fe81cf13710458e2ffcf1f1c09e10db4fdf7dd59e0",
                "soak_package_semantic_digest": "b91c18d404505b596f3aa6eecb318a2e73dcaa3cb17782455e06cbff706e604c",
            },
            "gpu35_identity": EXPECTED_GPU,
            "gpu_identity_correction": {
                "incorrect_text_only": "A prior commentary called gpu35 an A100; that wording was incorrect.",
                "actual_and_only_execution_target": EXPECTED_GPU,
                "wrong_target_execution_detected": False,
                "wsl_gpu_alias_used": False,
            },
            "campaign_plan": _file(plan_path, root=campaign_root),
            "rounds": rounds,
            "totals": totals,
            "failure_and_recovery_evidence": {
                "natural_candidate_failure": {
                    "round_index": 1,
                    "stage": "MATERIALIZE_QUALIFIER",
                    "status": rounds[0]["qualifier"]["status"],
                    "failure_class": rounds[0]["qualifier"]["failure_class"],
                    "reason_code": rounds[0]["qualifier"]["reason_code"],
                    "round_continued": True,
                    "effect_update_allowed": False,
                },
                "controlled_sealed_stage_resume": {
                    **_read(campaign_root / "round_01" / "CONTROLLED_RESUME_EVIDENCE.json"),
                    "artifact": _file(
                        campaign_root / "round_01" / "CONTROLLED_RESUME_EVIDENCE.json",
                        root=campaign_root,
                    ),
                },
                "preoutcome_environment_corrections": [
                    {
                        "round_index": 1,
                        "root_cause": "accepted R1 external receipt was absent from the relocated remote dependency root",
                        "correction": "copied the accepted read-only R1 artifact and verified its receipt hash under the unchanged manifest",
                        "accepted_receipt_sha256": "3c952ac60ee58d4c469e82fc14f2cff8a2f67a290524146b1da9f2bbcf0a6427",
                        "provider_calls_before_correction": 0,
                        "training_runs_before_correction": 0,
                        "outcomes_before_correction": 0,
                    },
                    {
                        "round_index": 1,
                        "root_cause": "RecBole was absent from the Resolver initialization import path",
                        "correction": "added the frozen RecBole root to bootstrap and passed comprehensive remote preflight",
                        "provider_calls_before_correction": 0,
                        "training_runs_before_correction": 0,
                        "outcomes_before_correction": 0,
                    },
                    {
                        "round_index": 2,
                        "root_cause": "the remote runtime had no /usr/bin/git for RecBole identity discovery",
                        "correction": "verified detached .git/HEAD and passed its exact commit identity to the existing training consumer",
                        "verified_recbole_commit": "7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
                        "provider_calls_before_correction": 0,
                        "training_runs_before_correction": 0,
                        "outcomes_before_correction": 0,
                        "preserved_artifacts": [
                            _file(path, root=campaign_root)
                            for path in sorted(
                                (
                                    campaign_root
                                    / "round_02"
                                    / "remote_evidence"
                                    / "preoutcome_missing_git_binary"
                                ).glob("*.json")
                            )
                        ],
                    },
                ],
            },
            "authority_boundary": {
                "feasibility_consumed_full_denominator_and_resource_protocol_censoring_only": True,
                "mechanism_information_consumed_real_probe_ablation_status_only": True,
                "effect_consumed_complete_fresh_matched_development_episodes_only": True,
                "resource_and_interface_failures_updated_effect": False,
                "formal_scientific_effect_claim": False,
            },
            "remaining_limitations": [
                "All outcomes are DEVELOPMENT_ONLY.",
                "The two complete matched episodes each use one development seed and are inconclusive.",
                "No mechanism-information probe or ablation completed in Q4, so that head remains unchanged.",
                "No held-out data was read; formal effect or policy-superiority claims are not supported.",
            ],
        }
    )
    package = {**package_payload, "package_digest": sha256_digest(package_payload)}
    package_path = campaign_root / "Q4_MULTIROUND_SOAK_PACKAGE.json"
    _write_new(package_path, package)

    physical_payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-multiround-physical-receipt.v1",
            "campaign_id": plan["campaign_id"],
            "status": "DEVELOPMENT_ONLY_PASS",
            "development_only": True,
            "scientific_effect_claim": False,
            "package": _file(package_path, root=campaign_root),
            "gpu35_identity": EXPECTED_GPU,
            "remote_root": "/NAS2020/Workspaces/DMGroup/tingrangan/q4_multiround_soak_20260803_01",
            "round_manifest_digests": [item["manifest_digest"] for item in rounds],
            "round_activation_digests": [item["activation"]["activation_digest"] for item in rounds],
            "final_head_update_counts": rounds[-1]["authority_update"]["head_update_counts"],
            "totals": totals,
            "retries": 0,
            "held_out_reads": 0,
            "gpu_identity_correction_recorded": True,
        }
    )
    physical = {**physical_payload, "receipt_digest": sha256_digest(physical_payload)}
    physical_path = campaign_root / "Q4_PHYSICAL_RECEIPT.json"
    _write_new(physical_path, physical)

    canonical_payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-multiround-canonical-receipt.v1",
            "campaign_id": plan["campaign_id"],
            "status": "DEVELOPMENT_ONLY_PASS",
            "development_only": True,
            "scientific_effect_claim": False,
            "package_sha256": bytes_sha256(package_path.read_bytes()),
            "package_digest": package["package_digest"],
            "physical_receipt_sha256": bytes_sha256(physical_path.read_bytes()),
            "physical_receipt_digest": physical["receipt_digest"],
            "final_policy_digest": rounds[-1]["activation"]["policy_digest"],
            "final_activation_digest": rounds[-1]["activation"]["activation_digest"],
            "held_out_reads": 0,
        }
    )
    canonical = {**canonical_payload, "receipt_digest": sha256_digest(canonical_payload)}
    canonical_path = args.canonical_receipt.resolve()
    _write_new(canonical_path, canonical)

    hash_paths = [package_path, physical_path, canonical_path]
    for round_root in (campaign_root / f"round_{index:02d}" for index in (1, 2, 3)):
        hash_paths.extend(path for path in round_root.rglob("*") if path.is_file())
    lines = []
    for path in sorted(set(hash_paths), key=lambda item: str(item)):
        try:
            label = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            label = str(path.resolve())
        lines.append(f"{bytes_sha256(path.read_bytes())}  {label}")
    sums_path = campaign_root / "SHA256SUMS"
    with sums_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "package": str(package_path),
                "package_sha256": bytes_sha256(package_path.read_bytes()),
                "physical_receipt": str(physical_path),
                "physical_receipt_sha256": bytes_sha256(physical_path.read_bytes()),
                "canonical_receipt": str(canonical_path),
                "canonical_receipt_sha256": bytes_sha256(canonical_path.read_bytes()),
                "sha256sums": str(sums_path),
                "sha256sums_sha256": bytes_sha256(sums_path.read_bytes()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
