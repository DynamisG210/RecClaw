"""Optional treatment hook for the final action-space self-reflection RecClaw.

The control arm does not import or enable this hook.  The treatment overlay enables
it through explicit environment variables and keeps the original producer, runner,
and reflection result intact alongside bounded Guard feedback.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .original_recclaw_adapter import (
    build_next_round_feedback_event,
    evaluate_postcheck_fail_open,
    evaluate_precheck_fail_open,
)


MATERIAL_PROTOCOL_KEYS = {
    "candidate_policy",
    "data_path",
    "dataset",
    "dataset_snapshot",
    "eval_args",
    "evaluation_candidate_universe",
    "metrics",
    "protocol_id",
    "repeatable",
    "seed",
    "split",
    "test_neg_sample_args",
    "topk",
    "train_neg_sample_args",
    "training_sampling",
    "valid_metric",
    "valid_neg_sample_args",
}

REQUIRED_RECBOLE_LOG_KEYS = {
    "candidate_id",
    "data_path",
    "epochs",
    "eval_args",
    "eval_batch_size",
    "eval_step",
    "learner",
    "learning_rate",
    "metrics",
    "repeatable",
    "reproducibility",
    "seed",
    "stopping_step",
    "test_neg_sample_args",
    "topk",
    "train_batch_size",
    "train_neg_sample_args",
    "valid_metric",
    "valid_neg_sample_args",
    "worker",
}

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONFIG_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_COMMAND_LINE = re.compile(r"\bINFO\s+(\[.*\])\s*$")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): deepcopy(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {
            str(key): deepcopy(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _parse_config_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip()


def _as_lower_text(value: object) -> str:
    return str(value).strip().lower()


def _normalized_metric_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({_as_lower_text(item) for item in value if str(item).strip()})


def _material_difference(
    differences: list[dict[str, Any]],
    *,
    code: str,
    field: str,
    expected: object,
    actual: object,
) -> None:
    differences.append(
        {
            "code": code,
            "field": field,
            "expected": deepcopy(expected),
            "actual": deepcopy(actual),
        }
    )


def _extract_recbole_log(path: Path) -> dict[str, Any]:
    text = _ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    command: list[str] | None = None
    config: dict[str, Any] = {}
    conflicting_keys: list[str] = []
    for raw_line in text.splitlines():
        if command is None:
            match = _COMMAND_LINE.search(raw_line)
            if match:
                parsed = _parse_config_value(match.group(1))
                if isinstance(parsed, list) and all(
                    isinstance(item, str) for item in parsed
                ):
                    command = parsed
        config_match = _CONFIG_LINE.match(raw_line)
        if not config_match:
            continue
        key = config_match.group(1)
        value = _parse_config_value(config_match.group(2))
        if key in config and config[key] != value:
            conflicting_keys.append(key)
        else:
            config[key] = value
    return {
        "command": command,
        "config": config,
        "conflicting_keys": sorted(set(conflicting_keys)),
    }


class OriginalRecClawGuardHook:
    """Treatment-only pre/post hook with an explicit fail-open boundary."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        feedback_path: Path,
        mode: str,
    ) -> None:
        if mode not in {"active", "shadow"}:
            raise ValueError("Evidence Guard hook mode must be active or shadow")
        self.contract = _as_mapping(contract)
        self.feedback_path = feedback_path
        self.mode = mode
        self._pending_prechecks: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_environment(cls) -> "OriginalRecClawGuardHook | None":
        contract_text = os.environ.get("RECCLAW_EVIDENCE_GUARD_CONTRACT", "").strip()
        if not contract_text:
            return None
        contract_path = Path(contract_text)
        value = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Evidence Guard contract must be a JSON object")
        feedback_text = os.environ.get("RECCLAW_EVIDENCE_GUARD_FEEDBACK", "").strip()
        if not feedback_text:
            raise ValueError("RECCLAW_EVIDENCE_GUARD_FEEDBACK is required")
        mode = os.environ.get("RECCLAW_EVIDENCE_GUARD_MODE", "shadow").strip()
        return cls(contract=value, feedback_path=Path(feedback_text), mode=mode)

    def _candidate_packet(
        self,
        *,
        candidate: Mapping[str, Any],
        params: Mapping[str, Any],
        round_id: int,
    ) -> dict[str, Any]:
        candidate_value = _as_mapping(candidate)
        params_value = _as_mapping(params)
        candidate_id = str(candidate_value.get("candidate_id") or "").strip()
        parent_id = str(
            candidate_value.get("parent_candidate_id")
            or candidate_value.get("ablation_parent")
            or ""
        ).strip()
        scaffold_id = str(
            candidate_value.get("entrypoint")
            or candidate_value.get("model")
            or ""
        ).strip()
        expected_recbole_model = (
            scaffold_id.rsplit(":", 1)[1].strip()
            if ":" in scaffold_id
            else str(candidate_value.get("model") or "").strip()
        )
        purpose = str(
            candidate_value.get("minimal_experiment")
            or candidate_value.get("hypothesis")
            or candidate_value.get("mechanism")
            or candidate_value.get("description")
            or "run the selected Original RecClaw candidate under the fixed protocol"
        ).strip()
        touched_keys = set(candidate_value) | set(params_value)
        material_keys = sorted(touched_keys & MATERIAL_PROTOCOL_KEYS)
        source = str(candidate_value.get("source") or "")
        recbole_core_change = (
            candidate_value.get("recbole_core_change_required") is True
            or scaffold_id.startswith("recbole.")
            and source not in {"", "registered_builtin"}
        )
        default_seed = str(
            _as_mapping(self.contract.get("seed_policy")).get(
                "default_training_seed", ""
            )
        )
        return {
            "candidate_id": candidate_id,
            "round_id": round_id,
            "original_decision": "PENDING_ORIGINAL_EXECUTION",
            "planned_seed_ids": [default_seed],
            "expected_recbole_model": expected_recbole_model,
            "proposal": {
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_id,
                "scaffold_id": scaffold_id,
                "minimal_experiment": purpose,
            },
            "protocol_projection_evidence": {
                "explicit_fixed_dataset": "dataset" not in material_keys
                and "dataset_snapshot" not in material_keys
                and "data_path" not in material_keys,
                "explicit_full_sort_evaluation": not bool(
                    {"eval_args", "evaluation_candidate_universe"} & set(material_keys)
                ),
                "explicit_unchanged_split": "split" not in material_keys,
                "explicit_unchanged_training_sampling": not bool(
                    {"train_neg_sample_args", "training_sampling"}
                    & set(material_keys)
                ),
                "recbole_core_change_required_false": not recbole_core_change,
                "parent_bound": bool(parent_id),
                "scaffold_bound": bool(scaffold_id),
                "training_seed_bound": bool(default_seed) and "seed" not in material_keys,
            },
            "live_projection_diagnostics": {
                "material_protocol_keys_in_candidate_or_params": material_keys,
                "projection_basis": "fixed experiment contract plus original action-space candidate fields",
            },
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
        }

    def precheck(
        self,
        *,
        candidate: Mapping[str, Any],
        params: Mapping[str, Any],
        round_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        packet = self._candidate_packet(
            candidate=candidate, params=params, round_id=round_id
        )
        run = evaluate_precheck_fail_open(
            contract=self.contract, candidate_packet=packet
        )
        candidate_id = str(packet["candidate_id"])
        self._pending_prechecks[candidate_id] = {"run": run, "packet": packet}
        event = build_next_round_feedback_event(
            precheck_run=run,
            postcheck_run=None,
            original_decision="PENDING_ORIGINAL_EXECUTION",
        )
        event["phase"] = "PRECHECK"
        event["hook_mode"] = self.mode
        event["guard_latency_ms"] = run["guard_latency_ms"]
        return run, event

    def should_defer(self, precheck_run: Mapping[str, Any]) -> bool:
        if self.mode != "active" or precheck_run.get("guard_succeeded") is not True:
            return False
        feedback = _as_mapping(precheck_run.get("feedback"))
        return feedback.get("recommendation") in {
            "REVISE_BEFORE_RUN",
            "ROUTE_TO_NEW_PROTOCOL_BRANCH",
        }

    def _inspect_live_log(
        self,
        *,
        log_path: Path,
        expected_run_id: str,
        expected_candidate_id: str,
        expected_model_name: str,
        expected_seed: str,
        artifact_kind: str,
    ) -> dict[str, Any]:
        """Bind an exact RecBole log to one expected run and derive its protocol."""

        protocol = deepcopy(self.contract["protocol"])
        fallback = {
            "log_path": str(log_path),
            "expected_run_id": expected_run_id,
            "expected_candidate_id": expected_candidate_id,
            "expected_model_name": expected_model_name,
            "expected_seed": expected_seed,
        }
        if not log_path.is_file():
            return {
                "artifact": {
                    "sha256": hashlib.sha256(_canonical_bytes(fallback)).hexdigest(),
                    "source_path": "",
                    "artifact_kind": f"derived_{artifact_kind}_identity_no_log",
                    "protocol_binding_status": "INCOMPLETE",
                    "protocol_binding_diagnostics": ["LIVE_LOG_NOT_FOUND"],
                },
                "artifact_identity_status": "PARTIAL",
                "observed_protocol": protocol,
                "protocol_binding_diagnostics": ["LIVE_LOG_NOT_FOUND"],
                "material_differences": [],
            }

        artifact_sha = _sha256_file(log_path)
        identity_diagnostics: list[str] = []
        if log_path.suffix.lower() != ".log":
            identity_diagnostics.append("LIVE_ARTIFACT_NOT_LOG_SUFFIX")
        if not expected_run_id:
            identity_diagnostics.append("EXPECTED_RUN_ID_MISSING")
        elif log_path.stem != expected_run_id:
            identity_diagnostics.append("LIVE_LOG_RUN_ID_PATH_MISMATCH")
        if not expected_candidate_id:
            identity_diagnostics.append("EXPECTED_LOG_CANDIDATE_ID_MISSING")
        if not expected_model_name:
            identity_diagnostics.append("EXPECTED_RECBOLE_MODEL_MISSING")
        if not expected_seed:
            identity_diagnostics.append("EXPECTED_TRAINING_SEED_MISSING")

        try:
            extracted = _extract_recbole_log(log_path)
        except OSError:
            extracted = {"command": None, "config": {}, "conflicting_keys": []}
            identity_diagnostics.append("LIVE_LOG_READ_FAILED")
        command = extracted["command"]
        config = extracted["config"]
        if command is None:
            identity_diagnostics.append("RECBOLE_COMMAND_HEADER_MISSING")
            command = []
        conflicting_keys = extracted["conflicting_keys"]
        if conflicting_keys:
            identity_diagnostics.append("RECBOLE_CONFIG_KEYS_CONFLICT")
        missing_keys = sorted(REQUIRED_RECBOLE_LOG_KEYS - set(config))
        if missing_keys:
            identity_diagnostics.append("RECBOLE_MATERIAL_CONFIG_INCOMPLETE")

        command_dataset = ""
        command_model = ""
        for index, item in enumerate(command):
            if item.startswith("--dataset="):
                command_dataset = item.split("=", 1)[1].strip()
            elif item == "--dataset" and index + 1 < len(command):
                command_dataset = command[index + 1].strip()
            elif item.startswith("--model="):
                command_model = item.split("=", 1)[1].strip()
            elif item == "--model" and index + 1 < len(command):
                command_model = command[index + 1].strip()
        if not command_dataset:
            identity_diagnostics.append("RECBOLE_COMMAND_DATASET_MISSING")
        if not command_model:
            identity_diagnostics.append("RECBOLE_COMMAND_MODEL_MISSING")
        elif expected_model_name and command_model != expected_model_name:
            identity_diagnostics.append("RECBOLE_COMMAND_MODEL_MISMATCH")
        if "candidate_id" in config and str(config["candidate_id"]) != expected_candidate_id:
            identity_diagnostics.append("RECBOLE_CANDIDATE_ID_MISMATCH")
        if "seed" in config and str(config["seed"]) != expected_seed:
            identity_diagnostics.append("RECBOLE_TRAINING_SEED_MISMATCH")

        differences: list[dict[str, Any]] = []
        expected_dataset = str(protocol.get("dataset") or "")
        if command_dataset and command_dataset != expected_dataset:
            _material_difference(
                differences,
                code="RECBOLE_DATASET_MISMATCH",
                field="dataset",
                expected=expected_dataset,
                actual=command_dataset,
            )
            protocol["dataset"] = command_dataset
            protocol["dataset_snapshot"] = (
                "unverified-live-dataset:"
                + hashlib.sha256(command_dataset.encode("utf-8")).hexdigest()
            )

        data_path = str(config.get("data_path") or "").rstrip("/")
        execution_environment = _as_mapping(
            self.contract.get("execution_environment")
        )
        recbole_root = str(execution_environment.get("recbole_root") or "").rstrip("/")
        expected_data_path = (
            f"{recbole_root}/dataset/{expected_dataset}" if recbole_root else ""
        )
        if data_path and Path(data_path).name != expected_dataset:
            _material_difference(
                differences,
                code="RECBOLE_DATA_PATH_DATASET_MISMATCH",
                field="dataset_snapshot",
                expected=expected_dataset,
                actual=data_path,
            )
            protocol["dataset_snapshot"] = (
                "unverified-live-data-path:"
                + hashlib.sha256(data_path.encode("utf-8")).hexdigest()
            )
        elif expected_data_path and data_path and data_path != expected_data_path:
            _material_difference(
                differences,
                code="RECBOLE_DATA_PATH_BINDING_MISMATCH",
                field="dataset_snapshot",
                expected=expected_data_path,
                actual=data_path,
            )
            protocol["dataset_snapshot"] = (
                "unverified-live-data-path:"
                + hashlib.sha256(data_path.encode("utf-8")).hexdigest()
            )

        training = _as_mapping(protocol.get("training_procedure"))
        training_checks = (
            ("train_batch_size", "train_batch_size", "RECBOLE_TRAIN_BATCH_SIZE_MISMATCH"),
            ("optimizer", "learner", "RECBOLE_OPTIMIZER_MISMATCH"),
            ("learning_rate", "learning_rate", "RECBOLE_LEARNING_RATE_MISMATCH"),
            ("max_epochs", "epochs", "RECBOLE_MAX_EPOCHS_MISMATCH"),
            (
                "evaluation_step",
                "eval_step",
                "RECBOLE_EVALUATION_STEP_MISMATCH",
            ),
            (
                "early_stopping_patience",
                "stopping_step",
                "RECBOLE_EARLY_STOPPING_MISMATCH",
            ),
            (
                "reproducibility",
                "reproducibility",
                "RECBOLE_REPRODUCIBILITY_MISMATCH",
            ),
        )
        for protocol_key, config_key, code in training_checks:
            if protocol_key not in training or config_key not in config:
                continue
            expected = training[protocol_key]
            actual = config[config_key]
            if protocol_key == "optimizer":
                matches = _as_lower_text(expected) == _as_lower_text(actual)
            else:
                matches = expected == actual
            if not matches:
                _material_difference(
                    differences,
                    code=code,
                    field=f"training_procedure.{protocol_key}",
                    expected=expected,
                    actual=actual,
                )
                training[protocol_key] = deepcopy(actual)
        protocol["training_procedure"] = training

        expected_sampling = _as_mapping(protocol.get("training_sampling"))
        actual_sampling = config.get("train_neg_sample_args")
        if isinstance(actual_sampling, Mapping):
            actual_distribution = _as_lower_text(actual_sampling.get("distribution"))
            actual_sample_num = actual_sampling.get("sample_num")
            actual_dynamic = actual_sampling.get("dynamic")
            sampling_checks = (
                ("distribution", actual_distribution, "RECBOLE_TRAIN_DISTRIBUTION_MISMATCH"),
                ("sample_num", actual_sample_num, "RECBOLE_TRAIN_SAMPLE_NUM_MISMATCH"),
                ("dynamic", actual_dynamic, "RECBOLE_TRAIN_DYNAMIC_MISMATCH"),
            )
            for key, actual, code in sampling_checks:
                if key not in expected_sampling:
                    continue
                expected = expected_sampling.get(key)
                comparable_expected = (
                    _as_lower_text(expected) if key == "distribution" else expected
                )
                if comparable_expected != actual:
                    _material_difference(
                        differences,
                        code=code,
                        field=f"training_sampling.{key}",
                        expected=expected,
                        actual=actual,
                    )
                    expected_sampling[key] = deepcopy(actual)
            expected_sampling["mode"] = (
                "uniform_negative"
                if actual_distribution == "uniform" and actual_sample_num == 1
                else "observed_recbole_negative_sampling"
            )
        elif "train_neg_sample_args" in config:
            _material_difference(
                differences,
                code="RECBOLE_TRAIN_SAMPLING_UNPARSEABLE",
                field="training_sampling",
                expected=expected_sampling,
                actual=actual_sampling,
            )
            expected_sampling = {"mode": "unparseable_recbole_training_sampling"}
        protocol["training_sampling"] = expected_sampling

        expected_split = _as_mapping(protocol.get("split"))
        expected_universe = _as_mapping(protocol.get("evaluation_candidate_universe"))
        eval_args = config.get("eval_args")
        if isinstance(eval_args, Mapping):
            raw_split = eval_args.get("split")
            if isinstance(raw_split, Mapping) and len(raw_split) == 1:
                split_token, split_ratio = next(iter(raw_split.items()))
                split_token = str(split_token)
            else:
                split_token, split_ratio = "UNPARSEABLE", raw_split
            order_token = str(eval_args.get("order") or "")
            group_by = str(eval_args.get("group_by") or "")
            eval_mode = eval_args.get("mode")
            expected_ratio = expected_split.get("ratio")
            expected_order_token = str(
                expected_split.get("recbole_order_token") or "RO"
            )
            expected_split_token = str(expected_split.get("recbole_token") or "RS")
            split_checks = (
                ("recbole_token", expected_split_token, split_token, "RECBOLE_SPLIT_TOKEN_MISMATCH"),
                ("ratio", expected_ratio, split_ratio, "RECBOLE_SPLIT_RATIO_MISMATCH"),
                ("recbole_order_token", expected_order_token, order_token, "RECBOLE_ORDER_MISMATCH"),
                ("group_by", expected_split.get("group_by"), group_by, "RECBOLE_GROUP_BY_MISMATCH"),
            )
            for key, expected, actual, code in split_checks:
                if expected != actual:
                    _material_difference(
                        differences,
                        code=code,
                        field=f"split.{key}",
                        expected=expected,
                        actual=actual,
                    )
                    expected_split[key] = deepcopy(actual)
            expected_split["strategy"] = (
                "random_user_holdout"
                if split_token == "RS" and order_token == "RO"
                else "observed_recbole_split"
            )
            expected_split["order"] = (
                "random" if order_token == "RO" else "observed_recbole_order"
            )
            if isinstance(eval_mode, Mapping):
                valid_mode = _as_lower_text(eval_mode.get("valid"))
                test_mode = _as_lower_text(eval_mode.get("test"))
            else:
                valid_mode = test_mode = _as_lower_text(eval_mode)
            if valid_mode != "full" or test_mode != "full":
                _material_difference(
                    differences,
                    code="RECBOLE_EVALUATION_MODE_MISMATCH",
                    field="evaluation_candidate_universe.mode",
                    expected={"valid": "full", "test": "full"},
                    actual={"valid": valid_mode, "test": test_mode},
                )
                expected_universe["mode"] = "sampled_or_other"
                expected_universe["recbole_modes"] = {
                    "valid": valid_mode,
                    "test": test_mode,
                }
        elif "eval_args" in config:
            _material_difference(
                differences,
                code="RECBOLE_EVAL_ARGS_UNPARSEABLE",
                field="split",
                expected=expected_split,
                actual=eval_args,
            )
            expected_split["strategy"] = "unparseable_recbole_split"

        for config_key in ("valid_neg_sample_args", "test_neg_sample_args"):
            sampling = config.get(config_key)
            sample_num = sampling.get("sample_num") if isinstance(sampling, Mapping) else None
            if config_key in config and _as_lower_text(sample_num) != "none":
                _material_difference(
                    differences,
                    code="RECBOLE_EVALUATION_SAMPLING_MISMATCH",
                    field=f"evaluation_candidate_universe.{config_key}",
                    expected="none",
                    actual=sample_num,
                )
                expected_universe["mode"] = "sampled_or_other"
                expected_universe[config_key] = deepcopy(sampling)
        protocol["split"] = expected_split
        protocol["evaluation_candidate_universe"] = expected_universe

        metric = _as_mapping(protocol.get("metric"))
        cutoff = metric.get("cutoff")
        metrics = _normalized_metric_names(config.get("metrics"))
        topk = config.get("topk")
        valid_metric = _as_lower_text(config.get("valid_metric"))
        implementation = _as_mapping(self.contract.get("protocol_implementation"))
        expected_metrics = implementation.get("reported_metrics")
        if isinstance(expected_metrics, list):
            normalized_expected_metrics = sorted(
                {
                    _as_lower_text(str(item).split("@", 1)[0])
                    for item in expected_metrics
                }
            )
        else:
            normalized_expected_metrics = [_as_lower_text(metric.get("name"))]
        metric_mismatch = False
        if "metrics" in config and metrics != normalized_expected_metrics:
            metric_mismatch = True
            _material_difference(
                differences,
                code="RECBOLE_REPORTED_METRICS_MISMATCH",
                field="metric.reported_metrics",
                expected=normalized_expected_metrics,
                actual=metrics,
            )
            metric["reported_metrics"] = metrics
        if "topk" in config and topk != [cutoff]:
            metric_mismatch = True
            _material_difference(
                differences,
                code="RECBOLE_TOPK_MISMATCH",
                field="metric.cutoff",
                expected=[cutoff],
                actual=topk,
            )
            if isinstance(topk, list) and topk and isinstance(topk[0], int) and topk[0] > 0:
                metric["cutoff"] = topk[0]
        expected_valid_metric = f"{_as_lower_text(metric.get('name'))}@{cutoff}"
        if "valid_metric" in config and valid_metric != expected_valid_metric:
            metric_mismatch = True
            _material_difference(
                differences,
                code="RECBOLE_VALID_METRIC_MISMATCH",
                field="metric.valid_metric",
                expected=expected_valid_metric,
                actual=valid_metric,
            )
            metric["valid_metric"] = valid_metric
        if metric_mismatch and not metric.get("name"):
            metric["name"] = "unknown"
        protocol["metric"] = metric

        implementation_checks = (
            ("repeatable", "repeatable", "RECBOLE_REPEATABLE_MISMATCH"),
            ("evaluation_batch_size", "eval_batch_size", "RECBOLE_EVAL_BATCH_SIZE_MISMATCH"),
            ("worker_count", "worker", "RECBOLE_WORKER_COUNT_MISMATCH"),
        )
        for expected_key, config_key, code in implementation_checks:
            if expected_key not in implementation or config_key not in config:
                continue
            expected = implementation[expected_key]
            actual = config[config_key]
            if expected != actual:
                _material_difference(
                    differences,
                    code=code,
                    field=f"protocol_implementation.{expected_key}",
                    expected=expected,
                    actual=actual,
                )
                protocol["training_procedure"][expected_key] = deepcopy(actual)

        if differences:
            protocol["protocol_id"] = f"OBSERVED-MATERIAL-FLIP-{artifact_sha[:16]}"
        artifact_identity = "PARTIAL" if identity_diagnostics else "EXACT"
        binding_status = (
            "INCOMPLETE"
            if identity_diagnostics
            else "MATERIAL_PROTOCOL_MISMATCH"
            if differences
            else "EXACT_PROTOCOL_MATCH"
        )
        diagnostics = [
            *sorted(set(identity_diagnostics)),
            *[str(item["code"]) for item in differences],
        ]
        return {
            "artifact": {
                "sha256": artifact_sha,
                "source_path": str(log_path),
                "artifact_kind": artifact_kind,
                "protocol_binding_status": binding_status,
                "protocol_binding_diagnostics": diagnostics,
                "parsed_material_config_sha256": hashlib.sha256(
                    _canonical_bytes(
                        {
                            key: config[key]
                            for key in sorted(REQUIRED_RECBOLE_LOG_KEYS & set(config))
                        }
                    )
                ).hexdigest(),
                "dataset_snapshot_binding": "FROZEN_EXECUTION_PREFLIGHT_REQUIRED",
            },
            "artifact_identity_status": artifact_identity,
            "observed_protocol": protocol,
            "protocol_binding_diagnostics": diagnostics,
            "material_differences": differences,
        }

    @staticmethod
    def original_trial_memory_disposition(
        postcheck_run: Mapping[str, Any]
    ) -> str:
        """Decide whether the original raw trial may enter primary search memory."""

        if postcheck_run.get("guard_succeeded") is not True:
            return "FAIL_OPEN_ORIGINAL"
        feedback = _as_mapping(postcheck_run.get("feedback"))
        if feedback.get("outcome_classification") in {
            "VALID_IMPROVEMENT",
            "INFORMATIVE_NEGATIVE",
        } and feedback.get("may_update_primary_search_memory") is True:
            return "ADMIT_ORIGINAL_TRIAL"
        return "QUARANTINE_ORIGINAL_TRIAL"

    @staticmethod
    def prepare_feedback_for_original_memory(
        event: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Keep runtime blockers useful without presenting them as research evidence."""

        payload = _as_mapping(event)
        original_status_value = payload.get("original_run_status")
        original_status = (
            _as_lower_text(original_status_value)
            if original_status_value is not None
            else ""
        )
        if (
            payload.get("postcheck_outcome_classification") == "RUNTIME_BLOCKER"
            or original_status not in {"", "success"}
        ):
            payload["decision"] = "crash"
            payload["diagnostic_blocker_signal"] = True
            payload["expose_to_next_iteration"] = True
            payload["may_update_primary_search_memory"] = False
            payload["may_update_diagnostic_memory"] = True
            payload["memory_channel"] = "DIAGNOSTIC_FEEDBACK"
            payload["next_iteration_effect"] = "INGEST_DIAGNOSTIC_FEEDBACK_ONLY"
        return payload

    def postcheck(
        self,
        *,
        candidate: Mapping[str, Any],
        candidate_result: Mapping[str, Any],
        original_record: object,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_value = _as_mapping(candidate)
        candidate_id = str(candidate_value.get("candidate_id") or "")
        pending = self._pending_prechecks.pop(candidate_id)
        packet = _as_mapping(pending["packet"])
        record = _as_mapping(original_record)
        result = _as_mapping(candidate_result)
        log_path = Path(str(result.get("log_path") or ""))
        record_run_id = str(record.get("run_id") or "").strip()
        result_run_id = str(result.get("run_id") or "").strip()
        run_id = record_run_id or result_run_id
        expected_log_candidate_id = str(
            candidate_value.get("run_candidate_id") or candidate_id
        ).strip()
        expected_seed = str(packet["planned_seed_ids"][0])
        inspection = self._inspect_live_log(
            log_path=log_path,
            expected_run_id=run_id,
            expected_candidate_id=expected_log_candidate_id,
            expected_model_name=str(packet.get("expected_recbole_model") or ""),
            expected_seed=expected_seed,
            artifact_kind="live_runner_log",
        )
        if record_run_id and result_run_id and record_run_id != result_run_id:
            inspection["artifact_identity_status"] = "PARTIAL"
            inspection["artifact"]["protocol_binding_status"] = "INCOMPLETE"
            inspection["artifact"]["protocol_binding_diagnostics"].append(
                "ORIGINAL_RECORD_RESULT_RUN_ID_MISMATCH"
            )
            inspection["protocol_binding_diagnostics"].append(
                "ORIGINAL_RECORD_RESULT_RUN_ID_MISMATCH"
            )
        status = str(result.get("status") or record.get("status") or "").lower()
        run_status = "SUCCESS" if status == "success" else "RUNTIME_BLOCKED"
        packet["original_decision"] = record.get("decision")
        packet["original_interpretation"] = {
            "reason": record.get("reason"),
            "next_action": record.get("next_action"),
            "compare_baseline": record.get("compare_baseline"),
            "compare_history_best": record.get("compare_history_best"),
        }
        packet["runner_observation"] = {
            "observation_id": f"LIVE-{record.get('run_id') or candidate_id}",
            "observation_kind": "METRIC_EVALUATION",
            "run_status": run_status,
            "artifact_identity_status": inspection["artifact_identity_status"],
            "observed_protocol": inspection["observed_protocol"],
            "metrics": {
                str(name): metric
                for name, metric in result.items()
                if isinstance(metric, (int, float)) and not isinstance(metric, bool)
            },
            "seed_runs": [
                {
                    "seed_id": packet["planned_seed_ids"][0],
                    "run_id": run_id,
                    "primary_artifact": inspection["artifact"],
                }
            ],
            "blocker": (
                None
                if run_status == "SUCCESS"
                else {
                    "kind": "ORIGINAL_RECCLAW_RUNTIME_BLOCKER",
                    "reason": record.get("reason"),
                }
            ),
        }
        post = evaluate_postcheck_fail_open(
            contract=self.contract,
            candidate_packet=packet,
            precheck_envelope=pending["run"]["envelope"],
        )
        event = build_next_round_feedback_event(
            precheck_run=pending["run"],
            postcheck_run=post,
            original_decision=record.get("decision"),
        )
        event["phase"] = "POSTCHECK"
        event["hook_mode"] = self.mode
        event["guard_latency_ms"] = post["guard_latency_ms"]
        event["original_run_status"] = status
        event["original_record_snapshot"] = record
        event["original_result_snapshot"] = result
        event["live_protocol_binding"] = {
            "artifact_identity_status": inspection["artifact_identity_status"],
            "diagnostics": inspection["protocol_binding_diagnostics"],
            "material_differences": inspection["material_differences"],
        }
        event["original_trial_memory_disposition"] = (
            self.original_trial_memory_disposition(post)
        )
        return post, event

    def _validation_artifact(
        self,
        *,
        run_id: str,
        expected_candidate_id: str,
        expected_model_name: str,
        expected_seed: str,
    ) -> dict[str, Any]:
        results_text = os.environ.get("RECCLAW_RESULTS_CSV", "").strip()
        results_path = Path(results_text) if results_text else None
        matches: list[dict[str, str]] = []
        if results_path is not None and results_path.is_file():
            with results_path.open("r", encoding="utf-8", newline="") as handle:
                matches = [
                    dict(row)
                    for row in csv.DictReader(handle)
                    if str(row.get("run_id") or "") == run_id
                ]
        if len(matches) == 1:
            log_path = Path(str(matches[0].get("log_path") or ""))
            return self._inspect_live_log(
                log_path=log_path,
                expected_run_id=run_id,
                expected_candidate_id=expected_candidate_id,
                expected_model_name=expected_model_name,
                expected_seed=expected_seed,
                artifact_kind="live_seed_validation_log",
            )
        fallback = {
            "run_id": run_id,
            "results_csv": "" if results_path is None else str(results_path),
            "matching_row_count": len(matches),
        }
        protocol = deepcopy(self.contract["protocol"])
        return {
            "artifact": {
                "sha256": hashlib.sha256(_canonical_bytes(fallback)).hexdigest(),
                "source_path": "",
                "artifact_kind": "derived_seed_validation_identity_no_exact_log",
                "protocol_binding_status": "INCOMPLETE",
                "protocol_binding_diagnostics": [
                    "RESULTS_CSV_RUN_ID_BINDING_NOT_UNIQUE"
                ],
            },
            "artifact_identity_status": "PARTIAL",
            "observed_protocol": protocol,
            "protocol_binding_diagnostics": [
                "RESULTS_CSV_RUN_ID_BINDING_NOT_UNIQUE"
            ],
            "material_differences": [],
        }

    def postcheck_seed_validation(
        self,
        *,
        candidate: Mapping[str, Any],
        params: Mapping[str, Any],
        round_id: int,
        seed_validation: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Classify the original planner's completed multi-seed validation."""

        validation = _as_mapping(seed_validation)
        raw_runs = validation.get("runs")
        if not isinstance(raw_runs, list) or not raw_runs:
            raise ValueError("seed validation must contain non-empty runs")
        runs = [_as_mapping(row) for row in raw_runs]
        planned_seed_ids = [str(row.get("seed") or "").strip() for row in runs]
        if any(not seed_id for seed_id in planned_seed_ids):
            raise ValueError("seed validation run is missing seed")
        packet = self._candidate_packet(
            candidate=candidate,
            params=params,
            round_id=round_id,
        )
        packet["planned_seed_ids"] = planned_seed_ids
        packet["original_decision"] = "multi_seed_verify"
        packet["original_interpretation"] = {
            "seed_validation": deepcopy(validation),
        }
        pre = evaluate_precheck_fail_open(
            contract=self.contract,
            candidate_packet=packet,
        )
        seed_runs: list[dict[str, Any]] = []
        exact_artifacts = True
        successful_runs = True
        inspections: list[dict[str, Any]] = []
        expected_log_candidate_id = str(
            _as_mapping(candidate).get("run_candidate_id")
            or packet["candidate_id"]
        ).strip()
        for row, seed_id in zip(runs, planned_seed_ids):
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                raise ValueError("seed validation run is missing run_id")
            inspection = self._validation_artifact(
                run_id=run_id,
                expected_candidate_id=expected_log_candidate_id,
                expected_model_name=str(packet.get("expected_recbole_model") or ""),
                expected_seed=seed_id,
            )
            inspections.append(inspection)
            exact_artifacts = (
                exact_artifacts
                and inspection["artifact_identity_status"] == "EXACT"
            )
            successful_runs = successful_runs and str(
                row.get("status") or ""
            ).lower() == "success"
            seed_runs.append(
                {
                    "seed_id": seed_id,
                    "run_id": run_id,
                    "primary_artifact": inspection["artifact"],
                }
            )
        validation_status = str(validation.get("status") or "").lower()
        run_status = (
            "SUCCESS"
            if successful_runs and validation_status in {"passed", "failed"}
            else "RUNTIME_BLOCKED"
        )
        metric_name = str(validation.get("metric") or "ndcg@10")
        mean_value = validation.get("mean")
        metrics = (
            {metric_name: mean_value}
            if isinstance(mean_value, (int, float)) and not isinstance(mean_value, bool)
            else {}
        )
        candidate_id = str(packet["candidate_id"])
        observed_protocols = [item["observed_protocol"] for item in inspections]
        observed_protocol_digests = sorted(
            {
                hashlib.sha256(_canonical_bytes(item)).hexdigest()
                for item in observed_protocols
            }
        )
        if len(observed_protocol_digests) == 1:
            observed_protocol = observed_protocols[0]
        else:
            observed_protocol = deepcopy(self.contract["protocol"])
            observed_protocol["protocol_id"] = (
                "OBSERVED-MIXED-SEED-PROTOCOLS-"
                + hashlib.sha256(
                    _canonical_bytes(observed_protocol_digests)
                ).hexdigest()[:16]
            )
            observed_protocol["training_procedure"][
                "mixed_seed_protocol_digests"
            ] = observed_protocol_digests
        packet["runner_observation"] = {
            "observation_id": f"LIVE-SEED-VALIDATION-{candidate_id}-{round_id}",
            "observation_kind": "METRIC_EVALUATION",
            "run_status": run_status,
            "artifact_identity_status": "EXACT" if exact_artifacts else "PARTIAL",
            "observed_protocol": observed_protocol,
            "metrics": metrics,
            "seed_runs": seed_runs,
            "blocker": (
                None
                if run_status == "SUCCESS"
                else {
                    "kind": "ORIGINAL_RECCLAW_SEED_VALIDATION_BLOCKER",
                    "validation_status": validation_status,
                }
            ),
        }
        post = evaluate_postcheck_fail_open(
            contract=self.contract,
            candidate_packet=packet,
            precheck_envelope=pre["envelope"],
        )
        event = build_next_round_feedback_event(
            precheck_run=pre,
            postcheck_run=post,
            original_decision="multi_seed_verify",
        )
        event["phase"] = "SEED_VALIDATION_POSTCHECK"
        event["hook_mode"] = self.mode
        event["precheck_latency_ms"] = pre["guard_latency_ms"]
        event["postcheck_latency_ms"] = post["guard_latency_ms"]
        event["guard_latency_ms"] = (
            pre["guard_latency_ms"] + post["guard_latency_ms"]
        )
        event["original_run_status"] = (
            "success" if run_status == "SUCCESS" else "runtime_blocked"
        )
        event["original_seed_validation_snapshot"] = validation
        event["live_protocol_binding"] = {
            "artifact_identity_status": "EXACT" if exact_artifacts else "PARTIAL",
            "seed_run_diagnostics": [
                {
                    "run_id": seed_run["run_id"],
                    "diagnostics": inspection["protocol_binding_diagnostics"],
                    "material_differences": inspection["material_differences"],
                }
                for seed_run, inspection in zip(seed_runs, inspections)
            ],
            "observed_protocol_digests": observed_protocol_digests,
        }
        event["original_trial_memory_disposition"] = (
            self.original_trial_memory_disposition(post)
        )
        return post, event

    def persist_feedback(self, event: Mapping[str, Any], memory_path: Path) -> None:
        payload = _as_mapping(event)
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
        if (
            payload.get("expose_to_next_iteration") is True
            and self.feedback_path.resolve() != memory_path.resolve()
        ):
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            with memory_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
