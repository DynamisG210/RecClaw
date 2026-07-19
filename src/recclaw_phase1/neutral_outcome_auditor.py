"""Pre-registered neutral outcome classification for AB-002 development results."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json"
DEFAULT_RUN_POLICY = PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json"
CONFIG_LINE = re.compile(r"^(?:.*?INFO\s+)?([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
ARM_VALUES = {"control", "treatment"}
ARM_KEYS = {"arm", "arm_label", "experimental_arm", "group_label"}
GUARD_KEYS = {"guard_enabled", "guard_classification", "evidence_guard_result"}


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    if text == "None":
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _log_facts(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    config: dict[str, Any] = {}
    conflicts = []
    for line in text.splitlines():
        matched = CONFIG_LINE.match(line)
        if matched is None:
            continue
        key = matched.group(1)
        value = _parse_value(matched.group(2))
        if key in config and config[key] != value:
            conflicts.append(key)
        else:
            config[key] = value
    command_model = None
    command_dataset = None
    for matched in re.finditer(r"--(model|dataset)=([^'\",\]\s]+)", text):
        if matched.group(1) == "model" and command_model is None:
            command_model = matched.group(2)
        if matched.group(1) == "dataset" and command_dataset is None:
            command_dataset = matched.group(2)
    return {
        "config": config,
        "conflicting_keys": sorted(set(conflicts)),
        "command_model": command_model,
        "command_dataset": command_dataset,
        "has_test_result": "test result:" in text.lower(),
        "has_traceback": "traceback (most recent call last)" in text.lower(),
        "has_crash_marker": "\ncrash:" in text.lower(),
    }


def _contains_arm_label(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in ARM_KEYS or lowered in GUARD_KEYS:
                return True
            if _contains_arm_label(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_arm_label(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ARM_VALUES:
            return True
        normalized = lowered.replace("\\", "/")
        return any(component in ARM_VALUES for component in normalized.split("/"))
    return False


def _compare(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def audit_raw_run(envelope: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    diagnostics: list[str] = []
    if _contains_arm_label(envelope):
        diagnostics.append("AUDITOR_INPUT_NOT_ARM_BLIND")
    anonymous_id = str(envelope.get("anonymous_run_id") or "")
    if not anonymous_id:
        diagnostics.append("ANONYMOUS_RUN_ID_MISSING")

    log_path = Path(str(envelope.get("log_path") or "")).expanduser().resolve()
    result_path = Path(str(envelope.get("result_json_path") or "")).expanduser().resolve()
    if not log_path.is_file():
        diagnostics.append("RUN_LOG_MISSING")
    if not result_path.is_file():
        diagnostics.append("RESULT_JSON_MISSING")

    facts: dict[str, Any] = {}
    result: dict[str, Any] = {}
    if "RUN_LOG_MISSING" not in diagnostics:
        facts = _log_facts(log_path)
        if facts["conflicting_keys"]:
            diagnostics.append("RECBole_CONFIG_CONFLICT")
        if facts["has_traceback"] or facts["has_crash_marker"]:
            diagnostics.append("RUN_CRASHED")
        if not facts["has_test_result"]:
            diagnostics.append("TEST_RESULT_MISSING")
    if "RESULT_JSON_MISSING" not in diagnostics:
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            result = loaded if isinstance(loaded, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            diagnostics.append("RESULT_JSON_INVALID")
        if result and str(result.get("status") or "").lower() != "success":
            diagnostics.append("RESULT_STATUS_NOT_SUCCESS")

    expected = policy["expected_log_facts"]
    config = facts.get("config") if isinstance(facts.get("config"), dict) else {}
    for key, expected_value in expected.items():
        if not _compare(config.get(key), expected_value):
            diagnostics.append(f"PROTOCOL_MISMATCH:{key}")
    if facts and facts.get("command_dataset") != policy["dataset"]:
        diagnostics.append("PROTOCOL_MISMATCH:dataset")
    if facts and not facts.get("command_model"):
        diagnostics.append("MODEL_IDENTITY_MISSING")

    observed_seed = config.get("seed")
    if observed_seed not in policy["allowed_training_seeds"]:
        diagnostics.append("PROTOCOL_MISMATCH:seed")
    if envelope.get("planned_training_seed") != observed_seed:
        diagnostics.append("PLANNED_OBSERVED_SEED_MISMATCH")

    expected_dataset = policy["dataset_snapshot_sha256"]
    if envelope.get("dataset_snapshot_sha256") != expected_dataset:
        diagnostics.append("DATASET_SNAPSHOT_MISMATCH")
    shared = envelope.get("shared_state_integrity")
    if not isinstance(shared, dict):
        diagnostics.append("SHARED_STATE_INTEGRITY_MISSING")
    else:
        if shared.get("dataset_before_sha256") != expected_dataset:
            diagnostics.append("DATASET_BEFORE_MISMATCH")
        if shared.get("dataset_after_sha256") != expected_dataset:
            diagnostics.append("DATASET_AFTER_MISMATCH")
        expected_recbole = policy["recbole_runtime_files_sha256"]
        recbole_before = shared.get("recbole_files_before")
        recbole_after = shared.get("recbole_files_after")
        if recbole_before != expected_recbole:
            diagnostics.append("RECBOLE_BEFORE_IDENTITY_MISMATCH")
        if recbole_after != expected_recbole:
            diagnostics.append("RECBOLE_AFTER_IDENTITY_MISMATCH")
        if recbole_before != recbole_after:
            diagnostics.append("RECBOLE_SHARED_STATE_CHANGED")

    artifacts = envelope.get("candidate_artifacts")
    roles = set()
    if not isinstance(artifacts, list):
        diagnostics.append("CANDIDATE_ARTIFACTS_MISSING")
        artifacts = []
    for item in artifacts:
        if not isinstance(item, dict):
            diagnostics.append("CANDIDATE_ARTIFACT_ENTRY_INVALID")
            continue
        role = str(item.get("role") or "")
        roles.add(role)
        path = Path(str(item.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            diagnostics.append(f"ARTIFACT_MISSING:{role or 'unknown'}")
            continue
        if _sha256(path) != item.get("sha256"):
            diagnostics.append(f"ARTIFACT_DIGEST_MISMATCH:{role or 'unknown'}")
    for role in policy["required_candidate_artifact_roles"]:
        if role not in roles:
            diagnostics.append(f"ARTIFACT_ROLE_MISSING:{role}")

    if result:
        if result.get("run_id") != log_path.stem:
            diagnostics.append("RUN_ID_LOG_STEM_MISMATCH")
        for metric in policy["required_metrics"]:
            value = result.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                diagnostics.append(f"METRIC_MISSING_OR_NONFINITE:{metric}")

    diagnostics = sorted(set(diagnostics))
    eligible = not diagnostics
    return {
        "record_type": "RECCLAW_PHASE1_AB002_NEUTRAL_RAW_RUN_AUDIT",
        "schema_version": "recclaw.phase1.ab002.neutral_raw_run_audit.v1",
        "status": "LOCAL_COMPLETE",
        "anonymous_run_id": anonymous_id,
        "eligible_for_performance_analysis": eligible,
        "diagnostics": diagnostics,
        "observed_seed": observed_seed,
        "observed_model": facts.get("command_model"),
        "run_log_sha256": _sha256(log_path) if log_path.is_file() else None,
        "result_json_sha256": _sha256(result_path) if result_path.is_file() else None,
        "metrics": {
            metric: result.get(metric) for metric in policy["required_metrics"] if metric in result
        },
        "policy_id": policy["policy_id"],
        "arm_label_read": False,
        "guard_classification_read": False,
        "decision_effect": "DEVELOPMENT_PERFORMANCE_ANALYSIS_ELIGIBILITY_ONLY",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def audit_outcome(analysis: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    aggregate = analysis.get("aggregate") if isinstance(analysis.get("aggregate"), dict) else {}
    complete_pairs = int(analysis.get("complete_pair_count") or 0)
    assigned_pairs = int(analysis.get("assigned_pair_count") or 0)
    paired_mean = _number(aggregate.get("paired_mean_primary_difference"))
    paired_median = _number(aggregate.get("paired_median_primary_difference"))
    treatment_wins = aggregate.get("treatment_primary_win_count")
    treatment_wins = int(treatment_wins) if isinstance(treatment_wins, int) else None
    false_block = _number(aggregate.get("valid_action_false_block_rate"))
    invalid_reduction = _number(aggregate.get("invalid_feedback_relative_reduction"))
    signal_delta = _number(aggregate.get("valid_informative_signal_rate_delta"))
    critical_false_blocks = aggregate.get("critical_valid_action_false_block_count")
    critical_false_blocks = (
        int(critical_false_blocks) if isinstance(critical_false_blocks, int) else None
    )
    simple_rule_same = aggregate.get("simple_rule_same_benefit")
    reasons = []
    outcome = "INCONCLUSIVE"
    minimum_pairs = int(policy["assigned_pair_count"])
    if assigned_pairs != minimum_pairs:
        reasons.append("ASSIGNED_PAIR_COUNT_MISMATCH")
    if complete_pairs < minimum_pairs:
        reasons.append("INSUFFICIENT_COMPLETE_PAIRS")
    required = {
        "PAIRED_MEAN_MISSING": paired_mean,
        "PAIRED_MEDIAN_MISSING": paired_median,
        "TREATMENT_WIN_COUNT_MISSING": treatment_wins,
        "INDEPENDENT_FALSE_BLOCK_ADJUDICATION_MISSING": false_block,
        "INVALID_FEEDBACK_REDUCTION_MISSING": invalid_reduction,
        "SIGNAL_RATE_DELTA_MISSING": signal_delta,
        "CRITICAL_FALSE_BLOCK_COUNT_MISSING": critical_false_blocks,
    }
    reasons.extend(code for code, value in required.items() if value is None)
    if not reasons:
        feedback_improved = invalid_reduction > 0 and signal_delta > 0
        strong = (
            treatment_wins >= int(policy["strong_positive_minimum_treatment_wins"])
            and paired_median > 0
            and critical_false_blocks == 0
            and feedback_improved
        )
        reliability = (
            paired_mean >= -float(policy["reliability_noninferiority_margin"])
            and invalid_reduction >= float(policy["reliability_minimum_invalid_feedback_reduction"])
            and signal_delta > 0
            and false_block <= float(policy["reliability_maximum_false_block_rate"])
        )
        negative = (
            paired_median < -float(policy["negative_paired_median_margin"])
            or false_block > float(policy["negative_false_block_rate"])
            or (paired_mean <= 0 and not feedback_improved)
            or simple_rule_same is True
        )
        if negative:
            outcome = "NEGATIVE"
        elif strong:
            outcome = "STRONG_POSITIVE"
        elif reliability:
            outcome = "RELIABILITY_POSITIVE"
        else:
            outcome = "INCONCLUSIVE"
            reasons.append("PRE_REGISTERED_OUTCOME_CONJUNCTION_NOT_MET")
    return {
        "record_type": "RECCLAW_PHASE1_AB002_NEUTRAL_OUTCOME_AUDIT",
        "schema_version": "recclaw.phase1.ab002.neutral_outcome_audit.v1",
        "status": "LOCAL_COMPLETE",
        "experiment_outcome": outcome,
        "reason_codes": reasons,
        "assigned_pair_count": assigned_pairs,
        "complete_pair_count": complete_pairs,
        "paired_mean_primary_difference": paired_mean,
        "paired_median_primary_difference": paired_median,
        "treatment_primary_win_count": treatment_wins,
        "valid_action_false_block_rate": false_block,
        "invalid_feedback_relative_reduction": invalid_reduction,
        "valid_informative_signal_rate_delta": signal_delta,
        "simple_rule_same_benefit": simple_rule_same,
        "simple_rule_comparison_status": (
            "LOCAL_COMPLETE" if isinstance(simple_rule_same, bool) else "NOT_STARTED"
        ),
        "policy_id": policy["policy_id"],
        "decision_effect": "DEVELOPMENT_REPORT_ONLY",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a frozen neutral AB-002 audit")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--analysis", type=Path)
    sources.add_argument("--raw-run-envelope", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.raw_run_envelope is not None:
        envelope = json.loads(args.raw_run_envelope.read_text(encoding="utf-8"))
        policy_path = args.policy or DEFAULT_RUN_POLICY
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        result = audit_raw_run(envelope, policy)
    else:
        analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
        policy_path = args.policy or DEFAULT_POLICY
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        result = audit_outcome(analysis, policy)
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise SystemExit("outcome audit output must be outside the source checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
