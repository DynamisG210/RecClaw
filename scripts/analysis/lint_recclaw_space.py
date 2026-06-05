#!/usr/bin/env python3
"""Lint the RecClaw runtime search-space contracts.

This is a preflight check for action-space/reflection/search-policy edits. It
does not change agent behavior; it only verifies that the runtime contracts stay
self-consistent before a branch is used for automated proposal generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
SEARCH_POLICY_PATH = PROJECT_ROOT / "configs" / "search_policy.yaml"
CANDIDATE_CONFIG_DIR = PROJECT_ROOT / "configs" / "candidates"

ALLOWED_PROTOCOL_CONSUMES = {"train_neg_sample_args"}
CANDIDATE_CONFIG_META_KEYS = {
    "base_model",
    "candidate_id",
    "entrypoint",
    "model",
    "posthoc",
    "runner_type",
}
CONDITIONAL_CONTEXT_KEYS = {
    "action_type",
    "base_model",
    "proposal_type",
    "runnable_level",
    "runner_type",
}
PLANNED_STATUSES = {"planned", "future", "not_implemented", "unimplemented"}


@dataclass(frozen=True)
class LintIssue:
    severity: str
    check: str
    message: str
    object_id: str = ""
    path: str = ""

    def as_dict(self) -> dict[str, str]:
        payload = {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
        }
        if self.object_id:
            payload["object_id"] = self.object_id
        if self.path:
            payload["path"] = self.path
        return payload


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def named_set(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value}
    return set(as_str_list(value))


def contains_lablog_path(path: str | Path) -> bool:
    text = str(path).replace("\\", "/")
    return "RecClaw_LabLog" in text.split("/")


def reject_lablog_path(path: str | Path) -> None:
    if contains_lablog_path(path):
        raise ValueError(f"runtime lint refuses LabLog path: {path}")


def candidate_base_models(raw: Any, allowed_models: set[str]) -> set[str]:
    if isinstance(raw, list):
        return {str(item) for item in raw if str(item)}
    text = str(raw or "").strip()
    if not text:
        return set()
    if text in allowed_models:
        return {text}
    parts = {
        part.strip()
        for part in re.split(r"\s*(?:/|,|\||\+|\band\b)\s*", text)
        if part.strip()
    }
    resolved = {part for part in parts if part in allowed_models}
    if resolved:
        return resolved
    return {model for model in allowed_models if re.search(rf"\b{re.escape(model)}\b", text)}


def is_runnable_candidate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("wired")) or str(candidate.get("status") or "") == "implemented"


def is_planned_parameter(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    status = str(spec.get("status") or spec.get("lifecycle") or "").strip().lower()
    if status in PLANNED_STATUSES:
        return True
    if spec.get("implemented") is False or spec.get("runnable") is False:
        return True
    return False


def parameter_values(spec: Any) -> list[Any]:
    if isinstance(spec, dict):
        values = spec.get("values")
    else:
        values = spec
    return values if isinstance(values, list) else []


def declared_candidate_parameter_specs(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    raw = candidate.get("new_parameters") or []
    if not isinstance(raw, list):
        return specs
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            specs[name] = item
    return specs


def scalar_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def numeric_cast(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return value
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer() and not re.search(r"[.eE]", text):
        return int(number)
    return number


def value_allowed(value: Any, allowed_values: list[Any]) -> bool:
    normalized = numeric_cast(value)
    return normalized in [numeric_cast(item) for item in allowed_values]


def compatibility_models(spec: Any) -> set[str]:
    if not isinstance(spec, dict):
        return set()
    raw = spec.get("compatible_models", spec.get("model_compatibility"))
    return set(as_str_list(raw))


def parameter_action_types(spec: Any) -> set[str]:
    if not isinstance(spec, dict):
        return set()
    return set(as_str_list(spec.get("action_types")))


def iter_requires(rule: Any) -> list[tuple[str, Any]]:
    if not isinstance(rule, dict):
        return []
    requires = rule.get("requires")
    if isinstance(requires, dict):
        return [(str(key), value) for key, value in requires.items()]
    return []


def candidate_configs_from_dir(path: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return configs
    for config_path in sorted(path.glob("*.yaml")):
        configs[str(config_path)] = load_yaml(config_path)
    return configs


def add_issue(
    issues: list[LintIssue],
    severity: str,
    check: str,
    message: str,
    *,
    object_id: str = "",
    path: str = "",
) -> None:
    issues.append(LintIssue(severity, check, message, object_id=object_id, path=path))


def lint_conditional_rule(
    issues: list[LintIssue],
    *,
    rule: dict[str, Any],
    check: str,
    parameter_space: dict[str, Any],
    base_models: set[str],
    object_id: str,
) -> None:
    parameter = str(rule.get("parameter") or object_id).strip()
    if parameter and parameter not in parameter_space:
        add_issue(
            issues,
            "error",
            check,
            f"conditional rule references unknown parameter: {parameter}",
            object_id=object_id,
        )
    valid_require_keys = set(parameter_space) | CONDITIONAL_CONTEXT_KEYS
    for name, _ in iter_requires(rule):
        if name not in valid_require_keys:
            add_issue(
                issues,
                "error",
                check,
                f"conditional rule requires unknown parameter/context: {name}",
                object_id=object_id,
            )
    incompatible = set(as_str_list(rule.get("incompatible_with_models")))
    unknown_models = sorted(incompatible - base_models)
    if unknown_models:
        add_issue(
            issues,
            "error",
            check,
            f"conditional rule references unknown incompatible models: {unknown_models}",
            object_id=object_id,
        )


def lint_payloads(
    *,
    action_space: dict[str, Any],
    schema: dict[str, Any],
    registry: dict[str, Any],
    search_policy: dict[str, Any],
    candidate_configs: dict[str, dict[str, Any]] | None = None,
) -> list[LintIssue]:
    issues: list[LintIssue] = []
    candidate_configs = candidate_configs or {}

    base_models = set(as_str_list(action_space.get("base_models")))
    runner_types = set(as_str_list(action_space.get("runner_types")))
    action_types = named_set(action_space.get("action_types"))
    parameter_space = action_space.get("parameter_space")
    parameter_space = parameter_space if isinstance(parameter_space, dict) else {}
    planned_parameters = {str(name) for name, spec in parameter_space.items() if is_planned_parameter(spec)}

    if not base_models:
        add_issue(issues, "error", "action_space.base_models", "base_models must not be empty")
    if not runner_types:
        add_issue(issues, "error", "action_space.runner_types", "runner_types must not be empty")
    if not action_types:
        add_issue(issues, "error", "action_space.action_types", "action_types must not be empty")

    schema_base_models = set(as_str_list(schema.get("allowed_base_models")))
    schema_runner_types = set(as_str_list(schema.get("allowed_runner_types")))
    schema_action_types = set(as_str_list(schema.get("allowed_action_types")))
    if base_models - schema_base_models:
        add_issue(
            issues,
            "error",
            "schema.base_models",
            f"schema is missing action-space base models: {sorted(base_models - schema_base_models)}",
        )
    if schema_base_models - base_models:
        add_issue(
            issues,
            "error",
            "schema.base_models",
            f"schema allows base models outside action_space: {sorted(schema_base_models - base_models)}",
        )
    if runner_types - schema_runner_types:
        add_issue(
            issues,
            "error",
            "schema.runner_types",
            f"schema is missing action-space runner types: {sorted(runner_types - schema_runner_types)}",
        )
    if schema_runner_types - runner_types:
        add_issue(
            issues,
            "error",
            "schema.runner_types",
            f"schema allows runner types outside action_space: {sorted(schema_runner_types - runner_types)}",
        )
    if action_types - schema_action_types:
        add_issue(
            issues,
            "error",
            "schema.action_types",
            f"schema is missing action-space action types: {sorted(action_types - schema_action_types)}",
        )
    if schema_action_types - action_types:
        add_issue(
            issues,
            "error",
            "schema.action_types",
            f"schema allows action types outside action_space: {sorted(schema_action_types - action_types)}",
        )

    projection = action_space.get("method_space_projection")
    if isinstance(projection, dict):
        for category, spec in projection.items():
            if not isinstance(spec, dict):
                continue
            for action in as_str_list(spec.get("executable_actions")):
                if action not in action_types:
                    add_issue(
                        issues,
                        "error",
                        "method_space_projection.executable_actions",
                        f"projection references undefined action_type: {action}",
                        object_id=str(category),
                    )
            for parameter in as_str_list(spec.get("example_parameters")):
                if parameter not in parameter_space:
                    add_issue(
                        issues,
                        "error",
                        "method_space_projection.example_parameters",
                        f"projection references undefined parameter: {parameter}",
                        object_id=str(category),
                    )

    for name, spec in parameter_space.items():
        parameter = str(name)
        values = parameter_values(spec)
        if not values:
            add_issue(
                issues,
                "error",
                "parameter_space.values",
                "parameter must declare a non-empty values list",
                object_id=parameter,
            )
        elif any(not scalar_value(value) for value in values):
            add_issue(
                issues,
                "error",
                "parameter_space.values",
                "parameter values must be scalar JSON/YAML values",
                object_id=parameter,
            )
        for action in parameter_action_types(spec):
            if action not in action_types:
                add_issue(
                    issues,
                    "error",
                    "parameter_space.action_types",
                    f"parameter references undefined action_type: {action}",
                    object_id=parameter,
                )
        unknown_models = sorted(compatibility_models(spec) - base_models)
        if unknown_models:
            add_issue(
                issues,
                "error",
                "parameter_space.compatible_models",
                f"parameter references unknown compatible models: {unknown_models}",
                object_id=parameter,
            )
        if isinstance(spec, dict) and isinstance(spec.get("conditional_validity"), dict):
            lint_conditional_rule(
                issues,
                rule={"parameter": parameter, **spec["conditional_validity"]},
                check="parameter_space.conditional_validity",
                parameter_space=parameter_space,
                base_models=base_models,
                object_id=parameter,
            )

    raw_groups = action_space.get("parameter_groups")
    if isinstance(raw_groups, list):
        for group_index, group in enumerate(raw_groups, start=1):
            for parameter in as_str_list(group):
                if parameter not in parameter_space:
                    add_issue(
                        issues,
                        "error",
                        "parameter_groups",
                        f"parameter group references undefined parameter: {parameter}",
                        object_id=f"group_{group_index}",
                    )
                if parameter in planned_parameters:
                    add_issue(
                        issues,
                        "error",
                        "parameter_groups",
                        f"planned parameter appears in runnable parameter_groups: {parameter}",
                        object_id=f"group_{group_index}",
                    )
    else:
        add_issue(issues, "error", "parameter_groups", "parameter_groups must be a list")

    raw_conditional_rules = action_space.get("conditional_validity_rules")
    if raw_conditional_rules is not None:
        if not isinstance(raw_conditional_rules, list):
            add_issue(
                issues,
                "error",
                "conditional_validity_rules",
                "conditional_validity_rules must be a list when present",
            )
        else:
            for index, rule in enumerate(raw_conditional_rules, start=1):
                if not isinstance(rule, dict):
                    add_issue(
                        issues,
                        "error",
                        "conditional_validity_rules",
                        "conditional validity rule must be an object",
                        object_id=f"rule_{index}",
                    )
                    continue
                lint_conditional_rule(
                    issues,
                    rule=rule,
                    check="conditional_validity_rules",
                    parameter_space=parameter_space,
                    base_models=base_models,
                    object_id=str(rule.get("parameter") or f"rule_{index}"),
                )

    for root in as_str_list(action_space.get("allowed_implementation_roots")):
        clean = root.replace("\\", "/")
        if clean.startswith("/") or ".." in Path(clean).parts:
            add_issue(
                issues,
                "error",
                "allowed_implementation_roots",
                f"implementation root must be a relative in-repo path: {root}",
            )

    registry_rows = registry.get("candidates", [])
    if not isinstance(registry_rows, list):
        add_issue(issues, "error", "candidate_registry", "registry.candidates must be a list")
        registry_rows = []
    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(registry_rows, start=1):
        if not isinstance(candidate, dict):
            add_issue(issues, "error", "candidate_registry", "candidate row must be an object")
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            add_issue(issues, "error", "candidate_registry.candidate_id", "candidate_id is required")
            continue
        if candidate_id in registry_by_id:
            add_issue(
                issues,
                "error",
                "candidate_registry.candidate_id",
                "duplicate candidate_id in registry",
                object_id=candidate_id,
            )
        registry_by_id[candidate_id] = candidate
        models = candidate_base_models(candidate.get("base_model"), base_models)
        if not models:
            add_issue(
                issues,
                "error",
                "candidate_registry.base_model",
                f"candidate base_model is outside action_space: {candidate.get('base_model')}",
                object_id=candidate_id,
            )
        runner_type = str(candidate.get("runner_type") or "").strip()
        if runner_type and runner_type not in runner_types:
            add_issue(
                issues,
                "error",
                "candidate_registry.runner_type",
                f"candidate runner_type is outside action_space: {runner_type}",
                object_id=candidate_id,
            )
        local_parameter_specs = declared_candidate_parameter_specs(candidate)
        raw_new_parameters = candidate.get("new_parameters")
        if raw_new_parameters is not None and not isinstance(raw_new_parameters, list):
            add_issue(
                issues,
                "error",
                "candidate_registry.new_parameters",
                "new_parameters must be a list",
                object_id=candidate_id,
            )
        elif isinstance(raw_new_parameters, list):
            seen_local_names: set[str] = set()
            for item in raw_new_parameters:
                if not isinstance(item, dict):
                    add_issue(
                        issues,
                        "error",
                        "candidate_registry.new_parameters",
                        "each new parameter declaration must be an object",
                        object_id=candidate_id,
                    )
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    add_issue(
                        issues,
                        "error",
                        "candidate_registry.new_parameters",
                        "new parameter declaration requires a name",
                        object_id=candidate_id,
                    )
                    continue
                if name in seen_local_names:
                    add_issue(
                        issues,
                        "error",
                        "candidate_registry.new_parameters",
                        f"duplicate local parameter declaration: {name}",
                        object_id=candidate_id,
                    )
                seen_local_names.add(name)
                search_space = item.get("search_space")
                if not isinstance(search_space, list) or not search_space:
                    add_issue(
                        issues,
                        "error",
                        "candidate_registry.new_parameters",
                        f"local parameter requires a non-empty search_space: {name}",
                        object_id=candidate_id,
                    )
                elif "default" not in item or not value_allowed(item.get("default"), search_space):
                    add_issue(
                        issues,
                        "error",
                        "candidate_registry.new_parameters",
                        f"local parameter default must belong to search_space: {name}",
                        object_id=candidate_id,
                    )
        for consume in as_str_list(candidate.get("consumes")):
            if consume in ALLOWED_PROTOCOL_CONSUMES:
                continue
            if consume not in parameter_space and consume not in local_parameter_specs:
                add_issue(
                    issues,
                    "error",
                    "candidate_registry.consumes",
                    f"candidate consumes undeclared parameter: {consume}",
                    object_id=candidate_id,
                )
                continue
            if consume in local_parameter_specs:
                continue
            if consume in planned_parameters and is_runnable_candidate(candidate):
                add_issue(
                    issues,
                    "error",
                    "candidate_registry.planned_parameter",
                    f"runnable candidate consumes planned parameter: {consume}",
                    object_id=candidate_id,
                )
            compatible = compatibility_models(parameter_space[consume])
            incompatible = sorted(models - compatible) if compatible and models else []
            if incompatible:
                add_issue(
                    issues,
                    "error",
                    "candidate_registry.compatible_models",
                    f"candidate consumes {consume}, incompatible with base_model(s): {incompatible}",
                    object_id=candidate_id,
                )

    for config_path, payload in candidate_configs.items():
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not candidate_id:
            add_issue(
                issues,
                "error",
                "candidate_config.candidate_id",
                "candidate config must declare candidate_id",
                path=config_path,
            )
            continue
        if Path(config_path).stem != candidate_id:
            add_issue(
                issues,
                "error",
                "candidate_config.filename",
                f"candidate config filename does not match candidate_id: {candidate_id}",
                object_id=candidate_id,
                path=config_path,
            )
        registry_candidate = registry_by_id.get(candidate_id)
        if registry_candidate is None:
            add_issue(
                issues,
                "error",
                "candidate_config.registry",
                "candidate config has no matching registry entry",
                object_id=candidate_id,
                path=config_path,
            )
            models = set()
            config_is_runnable = False
        else:
            models = candidate_base_models(registry_candidate.get("base_model"), base_models)
            config_is_runnable = is_runnable_candidate(registry_candidate)
        local_parameter_specs = (
            declared_candidate_parameter_specs(registry_candidate)
            if registry_candidate is not None
            else {}
        )
        for key, value in payload.items():
            parameter = str(key)
            if parameter in CANDIDATE_CONFIG_META_KEYS:
                continue
            if parameter not in parameter_space and parameter not in local_parameter_specs:
                add_issue(
                    issues,
                    "error",
                    "candidate_config.parameters",
                    f"candidate config contains undeclared parameter: {parameter}",
                    object_id=candidate_id,
                    path=config_path,
                )
                continue
            if parameter in local_parameter_specs:
                allowed_values = local_parameter_specs[parameter].get("search_space")
                if isinstance(allowed_values, list) and allowed_values and not value_allowed(value, allowed_values):
                    add_issue(
                        issues,
                        "error" if config_is_runnable else "warning",
                        "candidate_config.values",
                        f"candidate config value is outside local search_space for {parameter}: {value}",
                        object_id=candidate_id,
                        path=config_path,
                    )
                continue
            if parameter in planned_parameters:
                add_issue(
                    issues,
                    "error",
                    "candidate_config.planned_parameter",
                    f"candidate config contains planned parameter: {parameter}",
                    object_id=candidate_id,
                    path=config_path,
                )
            allowed_values = parameter_values(parameter_space[parameter])
            if allowed_values and not value_allowed(value, allowed_values):
                add_issue(
                    issues,
                    "error" if config_is_runnable else "warning",
                    "candidate_config.values",
                    f"candidate config value is outside action_space for {parameter}: {value}",
                    object_id=candidate_id,
                    path=config_path,
                )
            compatible = compatibility_models(parameter_space[parameter])
            incompatible = sorted(models - compatible) if compatible and models else []
            if incompatible:
                add_issue(
                    issues,
                    "error",
                    "candidate_config.compatible_models",
                    f"candidate config uses {parameter}, incompatible with base_model(s): {incompatible}",
                    object_id=candidate_id,
                    path=config_path,
                )

    for stage_name, stage in (search_policy.get("search_stages") or {}).items():
        if not isinstance(stage, dict):
            continue
        for candidate_id in as_str_list(stage.get("priority_families")):
            if candidate_id not in registry_by_id:
                add_issue(
                    issues,
                    "error",
                    "search_policy.priority_families",
                    f"priority family is not in candidate registry: {candidate_id}",
                    object_id=str(stage_name),
                )
        for action in as_str_list(stage.get("preferred_actions")):
            if action not in action_types:
                add_issue(
                    issues,
                    "error",
                    "search_policy.preferred_actions",
                    f"preferred action is not defined in action_space: {action}",
                    object_id=str(stage_name),
                )

    return issues


def summarize_issues(issues: list[LintIssue]) -> dict[str, int]:
    return {
        "errors": sum(1 for issue in issues if issue.severity == "error"),
        "warnings": sum(1 for issue in issues if issue.severity == "warning"),
        "total": len(issues),
    }


def run_lint(
    *,
    action_space_path: Path = ACTION_SPACE_PATH,
    schema_path: Path = SCHEMA_PATH,
    registry_path: Path = REGISTRY_PATH,
    search_policy_path: Path = SEARCH_POLICY_PATH,
    candidate_config_dir: Path = CANDIDATE_CONFIG_DIR,
) -> dict[str, Any]:
    paths = [action_space_path, schema_path, registry_path, search_policy_path, candidate_config_dir]
    for path in paths:
        reject_lablog_path(path)
    issues = lint_payloads(
        action_space=load_yaml(action_space_path),
        schema=load_yaml(schema_path),
        registry=load_yaml(registry_path),
        search_policy=load_yaml(search_policy_path),
        candidate_configs=candidate_configs_from_dir(candidate_config_dir),
    )
    return {
        "summary": summarize_issues(issues),
        "issues": [issue.as_dict() for issue in issues],
    }


def format_text_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "RecClaw space lint",
        f"errors={summary['errors']} warnings={summary['warnings']} total={summary['total']}",
    ]
    for issue in report["issues"]:
        location = issue.get("object_id") or issue.get("path") or ""
        prefix = f"[{issue['severity']}] {issue['check']}"
        if location:
            prefix += f" ({location})"
        lines.append(f"- {prefix}: {issue['message']}")
    if summary["total"] == 0:
        lines.append("- ok: action space, schema, registry, search policy, and candidate configs are consistent")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint RecClaw action-space and search contracts.")
    parser.add_argument("--action-space", default=str(ACTION_SPACE_PATH), help="Path to action_space.yaml")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Path to candidate_proposal_schema.yaml")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Path to candidate_registry.yaml")
    parser.add_argument("--search-policy", default=str(SEARCH_POLICY_PATH), help="Path to search_policy.yaml")
    parser.add_argument(
        "--candidate-config-dir",
        default=str(CANDIDATE_CONFIG_DIR),
        help="Directory containing configs/candidates/*.yaml",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        report = run_lint(
            action_space_path=Path(args.action_space),
            schema_path=Path(args.schema),
            registry_path=Path(args.registry),
            search_policy_path=Path(args.search_policy),
            candidate_config_dir=Path(args.candidate_config_dir),
        )
    except ValueError as exc:
        report = {
            "summary": {"errors": 1, "warnings": 0, "total": 1},
            "issues": [
                {
                    "severity": "error",
                    "check": "runtime_path_guard",
                    "message": str(exc),
                }
            ],
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(format_text_report(report))
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
