"""Fail-closed V31 matched formal experiment planning and orchestration.

The two arms share one GPU and are advanced in balanced, paired rounds.  A
process failure may leave the first arm of a pair one round ahead; a clean
resume always catches up the lagging arm before opening another pair.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.research_line.standalone import load_research_profile_source

from .matched_budget import (
    PAIRED_ROUND_SCHEDULE_V31,
    audit_matched_budget_contract_v31,
)
from .outer_panel import validate_outer_panel_manifest_v31


FORMAL_LAUNCH_PLAN_SCHEMA_V31 = "recclaw.helix.formal-launch-plan.v31"
FORMAL_LAUNCH_EVENT_SCHEMA_V31 = "recclaw.helix.formal-launch-event.v31"
FORMAL_COMPLETION_SCHEMA_V31 = "recclaw.helix.formal-completion.v31"


class FormalExperimentError(RuntimeError):
    """The frozen V31 experiment cannot be prepared or advanced safely."""


def _object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalExperimentError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise FormalExperimentError(f"{label} must be a JSON object: {path}")
    return value


def _resolve(repo_root: Path, ref: object, *, label: str) -> Path:
    if not isinstance(ref, str) or not ref.strip():
        raise FormalExperimentError(f"{label} ref is required")
    path = Path(ref)
    resolved = path if path.is_absolute() else repo_root / path
    return resolved.resolve()


def _verify_sha(path: Path, expected: object, *, label: str) -> str:
    if not path.is_file():
        raise FormalExperimentError(f"{label} is missing: {path}")
    actual = bytes_sha256(path.read_bytes())
    if actual != expected:
        raise FormalExperimentError(
            f"{label} SHA256 drift: expected {expected}, observed {actual}"
        )
    return actual


def blocked_formal_preflight_receipt_v31(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    audit = audit_matched_budget_contract_v31(manifest)
    return canonical_value(
        {
            "schema": "recclaw.helix.formal-preflight.v31",
            "ready": audit.ready,
            "errors": audit.errors,
            "manifest_digest": audit.manifest_digest,
            "shared_contract_digest": audit.shared_contract_digest,
            "worker_launch_authorized": audit.ready,
            "heldout_accessed": False,
        }
    )


def bind_outer_panel_to_manifest_v31(
    manifest: Mapping[str, Any],
    *,
    outer_manifest_path: Path,
) -> dict[str, Any]:
    """Bind a custodian-sealed outer manifest without reading panel outcomes."""

    value = canonical_value(dict(manifest))
    if value.get("status") != "PREFLIGHT_BLOCKED":
        raise FormalExperimentError("outer binding requires PREFLIGHT_BLOCKED input")
    if value.get("unresolved_preflight_requirements") != [
        "NEW_PRECOMMITTED_UNREAD_OUTER_PANEL"
    ]:
        raise FormalExperimentError("outer binding is not the sole remaining gate")
    outer_manifest = _object(outer_manifest_path, label="outer manifest")
    validate_outer_panel_manifest_v31(outer_manifest)
    shared = value["shared_contract"]
    shared["outer_heldout_contract"] = canonical_value(
        {
            "access_policy": "POST_SELECTION_ONLY",
            "online_feedback_allowed": False,
            "status": "PRECOMMITTED_UNREAD",
            "manifest_ref": str(outer_manifest_path.resolve()),
            "manifest_sha256": bytes_sha256(outer_manifest_path.read_bytes()),
            "manifest_digest": outer_manifest["manifest_digest"],
        }
    )
    shared_digest = sha256_digest(shared)
    value["status"] = "PREFLIGHT_READY"
    value["shared_contract_digest"] = shared_digest
    value["unresolved_preflight_requirements"] = []
    for arm in value["arms"]:
        arm["shared_contract_digest"] = shared_digest
    audit = audit_matched_budget_contract_v31(value)
    if not audit.ready:
        raise FormalExperimentError("; ".join(audit.errors))
    return value


def _arm_command(
    *,
    python_executable: Path,
    repo_root: Path,
    run_root: Path,
    arm: Mapping[str, Any],
    shared: Mapping[str, Any],
    baseline_plan: Mapping[str, Any],
    baseline_receipt_path: Path,
    profile_source_path: Path,
    seed_schedule_path: Path,
) -> list[str]:
    execution = shared["execution_budget"]
    runtime = shared["runtime_release"]
    provider = shared["provider_budget"]
    baseline = shared["development_baseline"]
    comparator = baseline_plan["comparator"]
    recipe = comparator["execution_recipe"]
    command = [
        str(python_executable),
        str((repo_root / "scripts" / "run_research_line_standalone.py").resolve()),
        "--run-root",
        str(run_root),
        "--api-config",
        str(Path(provider["config_ref"])),
        "--campaign-id",
        str(arm["arm_id"]),
        "--round-count",
        str(execution["metric_round_count"]),
        "--max-rounds-this-invocation",
        "1",
        "--seed",
        str(execution["ordered_seed_schedule"][0]),
        "--epochs",
        str(execution["epochs_requested"]),
        "--timeout-seconds",
        str(execution["worker_ceiling_seconds"]),
        "--watchdog-seconds",
        str(execution["worker_ceiling_seconds"]),
        "--final-worker-ceiling-seconds",
        str(execution["worker_ceiling_seconds"]),
        "--gpu-id",
        str(runtime["native_gpu_id"]),
        "--observation-seed-schedule-json",
        str(seed_schedule_path),
        "--baseline-ref",
        str(recipe["capability_ref"]),
        "--baseline-digest",
        str(recipe["capability_digest"]),
        "--baseline-ndcg-at-10",
        str(baseline["ndcg_at_10"]),
        "--baseline-protocol-digest",
        str(shared["online_metric_contract"]["protocol_digest"]),
        "--source-ref",
        f"receipt:{baseline_receipt_path}",
        "--source-digest",
        str(baseline["receipt_sha256"]),
        "--development-validation-v31",
        "--attempt-scheduler",
        "--max-attempts-per-round",
        str(execution["max_attempts_per_round"]),
        "--profile-source-json",
        str(profile_source_path),
    ]
    if arm["arm_kind"] == "FULL_HELIX":
        treatment = arm["treatment"]
        command.extend(
            [
                "--evidence-guard-expected-dataset-manifest-digest",
                str(shared["dataset_contract"]["search_partition_manifest_digest"]),
                "--evidence-guard-validation-seed-schedule-json",
                str(seed_schedule_path),
                "--evidence-guard-required-seed-count",
                str(treatment["required_seed_count"]),
                "--evidence-guard-comparator",
                str(recipe["capability_ref"]),
                "--evidence-guard-minimum-effect-delta",
                str(treatment["minimum_support_effect_delta"]),
                "--evidence-guard-replication-trigger-delta",
                str(treatment["replication_trigger_delta"]),
                "--evidence-guard-max-allocation-actions",
                str(treatment["max_allocation_actions"]),
                "--evidence-guard-max-open-allocation-actions",
                str(treatment["max_open_allocation_actions"]),
            ]
        )
    return command


def build_formal_launch_plan_v31(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    python_executable: Path,
) -> dict[str, Any]:
    """Validate all frozen bytes and render secret-free arm commands."""

    value = canonical_value(dict(manifest))
    audit = audit_matched_budget_contract_v31(value)
    if not audit.ready:
        raise FormalExperimentError("; ".join(audit.errors))
    root = repo_root.resolve()
    if not python_executable.is_file():
        raise FormalExperimentError(f"Python executable is missing: {python_executable}")
    entrypoint = root / "scripts" / "run_research_line_standalone.py"
    if not entrypoint.is_file():
        raise FormalExperimentError(f"standalone entrypoint is missing: {entrypoint}")

    shared = value["shared_contract"]
    source = shared["campaign_source_release"]
    source_path = _resolve(root, source["archive_ref"], label="source archive")
    _verify_sha(source_path, source["archive_sha256"], label="source archive")
    provider = shared["provider_budget"]
    api_path = _resolve(root, provider["config_ref"], label="Provider config")
    _verify_sha(api_path, provider["config_digest"], label="Provider config")

    baseline = shared["development_baseline"]
    baseline_plan_path = _resolve(root, baseline["plan_ref"], label="baseline plan")
    baseline_receipt_path = _resolve(
        root, baseline["receipt_ref"], label="baseline receipt"
    )
    baseline_plan = _object(baseline_plan_path, label="baseline plan")
    baseline_receipt = _object(baseline_receipt_path, label="baseline receipt")
    if baseline_plan.get("plan_digest") != baseline["plan_digest"]:
        raise FormalExperimentError("baseline plan digest drift")
    _verify_sha(
        baseline_receipt_path,
        baseline["receipt_sha256"],
        label="baseline receipt",
    )
    if baseline_receipt.get("receipt_digest") != baseline["receipt_digest"]:
        raise FormalExperimentError("baseline receipt self-digest drift")
    if baseline_receipt.get("ndcg_at_10") != baseline["ndcg_at_10"]:
        raise FormalExperimentError("baseline score drift")

    formal = shared["formal_execution"]
    seed_source = formal["observation_seed_source"]
    seed_path = _resolve(root, seed_source["ref"], label="seed schedule")
    _verify_sha(seed_path, seed_source["sha256"], label="seed schedule")
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    if seeds != shared["execution_budget"]["ordered_seed_schedule"]:
        raise FormalExperimentError("seed source differs from shared schedule")

    profile_binding = formal["research_profile_source"]
    profile_path = _resolve(root, profile_binding["ref"], label="profile source")
    _verify_sha(profile_path, profile_binding["sha256"], label="profile source")
    profile_source = load_research_profile_source(profile_path)
    if profile_source.identity != {
        "schema": profile_binding["schema"],
        "schema_version": "v1",
        "source_ref": profile_binding["source_ref"],
        "source_digest": profile_binding["source_digest"],
    }:
        raise FormalExperimentError("Research profile source identity drift")

    outer = shared["outer_heldout_contract"]
    outer_path = _resolve(root, outer["manifest_ref"], label="outer manifest")
    _verify_sha(outer_path, outer["manifest_sha256"], label="outer manifest")
    outer_manifest = _object(outer_path, label="outer manifest")
    validate_outer_panel_manifest_v31(outer_manifest)
    if outer_manifest.get("manifest_digest") != outer["manifest_digest"]:
        raise FormalExperimentError("outer manifest semantic digest drift")

    formal_root = Path(formal["formal_root"]).resolve()
    arm_rows: list[dict[str, Any]] = []
    for arm in value["arms"]:
        run_root = formal_root / "arms" / str(arm["arm_id"])
        command = _arm_command(
            python_executable=python_executable.resolve(),
            repo_root=root,
            run_root=run_root,
            arm=arm,
            shared=shared,
            baseline_plan=baseline_plan,
            baseline_receipt_path=baseline_receipt_path,
            profile_source_path=profile_path,
            seed_schedule_path=seed_path,
        )
        arm_rows.append(
            canonical_value(
                {
                    "arm_id": arm["arm_id"],
                    "arm_kind": arm["arm_kind"],
                    "run_root": str(run_root),
                    "command": command,
                    "command_digest": sha256_digest(command),
                }
            )
        )
    payload = canonical_value(
        {
            "schema": FORMAL_LAUNCH_PLAN_SCHEMA_V31,
            "status": "READY_NOT_STARTED",
            "manifest_digest": audit.manifest_digest,
            "shared_contract_digest": audit.shared_contract_digest,
            "source_archive_ref": str(source_path),
            "source_archive_sha256": source["archive_sha256"],
            "repo_root": str(root),
            "formal_root": str(formal_root),
            "python_executable": str(python_executable.resolve()),
            "strategy": PAIRED_ROUND_SCHEDULE_V31,
            "metric_round_count": shared["execution_budget"]["metric_round_count"],
            "max_progress_skew": 1,
            "heldout_access_during_search": "FORBIDDEN",
            "arms": arm_rows,
        }
    )
    return {**payload, "plan_digest": sha256_digest(payload)}


def _campaign_state_projection(run_root: Path) -> tuple[dict[str, Any], str]:
    state_path = run_root / "CAMPAIGN_STATE.json"
    if not state_path.is_file():
        raise FormalExperimentError(
            f"existing arm root lacks CAMPAIGN_STATE.json: {run_root}"
        )
    projection = _object(state_path, label="campaign state projection")
    state = projection.get("state")
    state_digest = projection.get("state_digest")
    if (
        projection.get("schema")
        != "recclaw.research-line.campaign-state-projection.v1"
        or not isinstance(state, dict)
        or state_digest != sha256_digest(state)
    ):
        raise FormalExperimentError(
            f"campaign state projection identity drift: {state_path}"
        )
    return state, str(state_digest)


def _progress(run_root: Path, *, target: int) -> int:
    if not run_root.exists():
        return 0
    state, _ = _campaign_state_projection(run_root)
    next_round = state.get("next_round_index")
    if (
        isinstance(next_round, bool)
        or not isinstance(next_round, int)
        or not 1 <= next_round <= target + 1
    ):
        raise FormalExperimentError(f"invalid next_round_index in {state_path}")
    return next_round - 1


def _summary_from_stdout(stdout: str) -> Mapping[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and value.get("schema") == (
            "recclaw.research-line.standalone-run-summary.v1"
        ):
            return value
    return None


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FormalExperimentError(f"write-once artifact drift: {path}")
        return
    path.write_bytes(payload)


def _run_arm_once(
    *,
    plan: Mapping[str, Any],
    arm: Mapping[str, Any],
    sequence: int,
) -> int:
    target = int(plan["metric_round_count"])
    run_root = Path(str(arm["run_root"]))
    before = _progress(run_root, target=target)
    command = list(arm["command"])
    if run_root.exists():
        command.append("--resume")
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        command,
        cwd=str(plan["repo_root"]),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    launcher_root = Path(str(plan["formal_root"])) / "launcher"
    stem = f"{sequence:04d}-{arm['arm_id']}"
    stdout_path = launcher_root / f"{stem}.stdout.log"
    stderr_path = launcher_root / f"{stem}.stderr.log"
    _write_once(stdout_path, result.stdout.encode("utf-8"))
    _write_once(stderr_path, result.stderr.encode("utf-8"))
    progress_error = None
    try:
        after: int | None = _progress(run_root, target=target)
    except FormalExperimentError as error:
        after = None
        progress_error = str(error)
    summary = _summary_from_stdout(result.stdout)
    event = canonical_value(
        {
            "schema": FORMAL_LAUNCH_EVENT_SCHEMA_V31,
            "sequence": sequence,
            "arm_id": arm["arm_id"],
            "before_metric_rounds": before,
            "after_metric_rounds": after,
            "return_code": result.returncode,
            "progress_error": progress_error,
            "summary": summary,
            "stdout_ref": str(stdout_path),
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_ref": str(stderr_path),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "cuda_visible_devices": "UNSET_BY_LAUNCHER",
        }
    )
    launcher_root.mkdir(parents=True, exist_ok=True)
    with (launcher_root / "EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_bytes(event).decode("utf-8") + "\n")
    if progress_error is not None:
        raise FormalExperimentError(progress_error)
    assert after is not None
    if after != before + 1:
        raise FormalExperimentError(
            f"{arm['arm_id']} did not advance exactly one metric round"
        )
    if summary is None or summary.get("metric_round_count") != after:
        raise FormalExperimentError(f"{arm['arm_id']} summary is missing or drifted")
    expected_code = 0 if after == target else 3
    if result.returncode != expected_code:
        raise FormalExperimentError(
            f"{arm['arm_id']} returned {result.returncode}, expected {expected_code}"
        )
    return after


def execute_formal_launch_plan_v31(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Advance both arms to 100 paired metric rounds, with safe resume."""

    value = canonical_value(dict(plan))
    declared = value.pop("plan_digest", None)
    if value.get("schema") != FORMAL_LAUNCH_PLAN_SCHEMA_V31:
        raise FormalExperimentError("formal launch plan schema drift")
    if declared != sha256_digest(value):
        raise FormalExperimentError("formal launch plan digest drift")
    if value.get("strategy") != PAIRED_ROUND_SCHEDULE_V31:
        raise FormalExperimentError("formal launch schedule drift")
    arms = {str(arm["arm_kind"]): arm for arm in value["arms"]}
    if set(arms) != {"RESEARCH_LINE", "FULL_HELIX"}:
        raise FormalExperimentError("formal plan must contain exactly two arms")
    target = int(value["metric_round_count"])
    events_path = Path(str(value["formal_root"])) / "launcher" / "EVENTS.jsonl"
    sequence = (
        sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line)
        if events_path.is_file()
        else 0
    )
    while True:
        research_progress = _progress(
            Path(str(arms["RESEARCH_LINE"]["run_root"])), target=target
        )
        helix_progress = _progress(
            Path(str(arms["FULL_HELIX"]["run_root"])), target=target
        )
        if abs(research_progress - helix_progress) > 1:
            raise FormalExperimentError("paired arm progress skew exceeds one round")
        if research_progress == helix_progress == target:
            break
        if research_progress > helix_progress:
            next_arm = arms["FULL_HELIX"]
        elif helix_progress > research_progress:
            next_arm = arms["RESEARCH_LINE"]
        else:
            paired_round = research_progress + 1
            next_arm = (
                arms["RESEARCH_LINE"]
                if paired_round % 2 == 1
                else arms["FULL_HELIX"]
            )
        sequence += 1
        _run_arm_once(plan=value, arm=next_arm, sequence=sequence)

    arm_receipts: list[dict[str, Any]] = []
    for arm in value["arms"]:
        state_path = Path(str(arm["run_root"])) / "CAMPAIGN_STATE.json"
        state, state_digest = _campaign_state_projection(
            Path(str(arm["run_root"]))
        )
        arm_receipts.append(
            {
                "arm_id": arm["arm_id"],
                "run_root": arm["run_root"],
                "next_round_index": state["next_round_index"],
                "state_digest": state_digest,
                "state_file_sha256": bytes_sha256(state_path.read_bytes()),
            }
        )
    completion_payload = canonical_value(
        {
            "schema": FORMAL_COMPLETION_SCHEMA_V31,
            "status": "COMPLETE_AWAITING_POST_SELECTION_OUTER_EVALUATION",
            "plan_digest": declared,
            "manifest_digest": value["manifest_digest"],
            "metric_round_count_per_arm": target,
            "arm_receipts": arm_receipts,
            "outer_heldout_accessed": False,
        }
    )
    completion = {
        **completion_payload,
        "completion_digest": sha256_digest(completion_payload),
    }
    completion_path = Path(str(value["formal_root"])) / "FORMAL_COMPLETION.json"
    _write_once(completion_path, canonical_json_bytes(completion) + b"\n")
    return completion


__all__ = [
    "FORMAL_COMPLETION_SCHEMA_V31",
    "FORMAL_LAUNCH_EVENT_SCHEMA_V31",
    "FORMAL_LAUNCH_PLAN_SCHEMA_V31",
    "FormalExperimentError",
    "bind_outer_panel_to_manifest_v31",
    "blocked_formal_preflight_receipt_v31",
    "build_formal_launch_plan_v31",
    "execute_formal_launch_plan_v31",
]
