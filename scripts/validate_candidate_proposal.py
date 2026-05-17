#!/usr/bin/env python3
"""Validate RecClaw candidate proposal JSONL files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
SCHEMA_PATH = PROJECT_ROOT / "configs" / "candidate_proposal_schema.yaml"
PROPOSAL_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
MEMORY_PATH = PROJECT_ROOT / "results" / "agent_memory.jsonl"

ACCEPTED = "accepted"
REJECTED = "rejected"
NEEDS_REVIEW = "needs_review"

NEXT_ACTIONS = {
    ACCEPTED: "run_candidate_with_overrides",
    NEEDS_REVIEW: "promote_to_implementation_queue",
    REJECTED: "revise_or_drop",
}

SIGNATURE_EXCLUDED_KEYS = {"seed", "reproducibility", "checkpoint_dir"}


def normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_signature_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_signature_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_signature_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    if not path.exists():
        return []
    rows: list[tuple[int, dict[str, Any] | None, str | None]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append((line_no, None, f"invalid JSON: {exc}"))
            continue
        if not isinstance(parsed, dict):
            rows.append((line_no, None, "proposal row must be a JSON object"))
            continue
        rows.append((line_no, parsed, None))
    return rows


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_consume_names(value: Any) -> list[str]:
    names: list[str] = []
    for item in as_list(value):
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]))
        else:
            names.append(str(item))
    return names


def parent_base_models(raw: Any, allowed_models: set[str]) -> set[str]:
    text = str(raw or "")
    return {model for model in allowed_models if model in text}


def new_parameter_names(proposal: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in as_list(proposal.get("new_parameters")):
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict) and item.get("name"):
            names.add(str(item["name"]))
    return names


def implementation_files(proposal: dict[str, Any]) -> list[str]:
    files: list[str] = []
    plan = proposal.get("implementation_plan")
    if isinstance(plan, dict):
        files.extend(str(item) for item in as_list(plan.get("files")))
    files.extend(str(item) for item in as_list(proposal.get("allowed_files")))
    return [item for item in files if item and item != "None"]


def params_signature(params: dict[str, Any]) -> str:
    normalized = {
        str(key): normalize_signature_value(value)
        for key, value in params.items()
        if str(key) not in SIGNATURE_EXCLUDED_KEYS
    }
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def parent_param_signature(parent_id: str, params: dict[str, Any]) -> str:
    return f"{parent_id}::{params_signature(params)}"


def canonical_parameter_signature_text(raw: Any) -> str:
    text = str(raw or "").strip()
    if "::" not in text:
        return text
    parent_id, raw_params = text.split("::", 1)
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        return text
    if not isinstance(params, dict):
        return text
    return parent_param_signature(parent_id, params)


def proposal_parameter_signature(proposal: dict[str, Any]) -> str:
    parent_id = str(proposal.get("parent_candidate_id") or "").strip()
    overrides = proposal.get("parameter_overrides") or {}
    if not parent_id or not isinstance(overrides, dict) or not overrides:
        return ""
    return parent_param_signature(parent_id, overrides)


def load_memory_param_signatures(path: Path) -> set[str]:
    signatures: set[str] = set()
    for _, row, parse_error in load_jsonl(path):
        if parse_error is not None or row is None:
            continue
        if not row.get("event") and row.get("parameter_signature"):
            signatures.add(canonical_parameter_signature_text(row.get("parameter_signature")))
            continue
        parent_id = str(row.get("parent_candidate_id") or row.get("candidate_id") or "").strip()
        params = row.get("parameter_overrides")
        if not isinstance(params, dict):
            params = row.get("params")
        if parent_id and isinstance(params, dict) and params:
            signatures.add(parent_param_signature(parent_id, params))
    return signatures


def path_is_allowed(path: str, allowed_roots: set[str]) -> bool:
    clean = path.strip().replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    if not clean or clean.startswith("/") or re.match(r"^[A-Za-z]:", clean):
        return False
    parts = PurePosixPath(clean).parts
    if any(part == ".." for part in parts):
        return False
    if PurePosixPath(clean).name == "__init__.py" and parts and parts[0] == "recclaw_ext":
        return False
    return any(clean == root.rstrip("/") or clean.startswith(root) for root in allowed_roots)


def mentions_recbole_core_change(proposal: dict[str, Any]) -> bool:
    text = json.dumps(proposal, ensure_ascii=False).lower()
    risky_patterns = (
        "modify recbole",
        "modifies recbole",
        "change recbole",
        "changes recbole",
        "patch recbole",
        "patches recbole",
        "requires recbole core",
        "requires changing recbole",
        "recbole core change required",
        "recbole/",
        "recbole\\",
    )
    return any(pattern in text for pattern in risky_patterns)


def next_action_for(status: str, runnable_level: str, runner_type: str) -> str:
    if status == ACCEPTED:
        if runnable_level == "parameter_only":
            return "run_candidate_with_overrides"
        return "promote_to_registry_then_run"
    if status == NEEDS_REVIEW:
        if runner_type == "posthoc":
            return "review_posthoc_runner_support"
        if runnable_level == "code_required":
            return "promote_to_implementation_queue"
        return "review_research_spec"
    return NEXT_ACTIONS[REJECTED]


def multiseed_warnings(proposal: dict[str, Any], runnable_level: str) -> list[str]:
    if str(proposal.get("proposal_type") or "") != "tuning":
        return []
    if runnable_level not in {"parameter_only", "config_only"}:
        return []
    plan = proposal.get("evaluation_plan")
    if not isinstance(plan, dict):
        return ["evaluation_plan is missing; use multi-seed validation before claiming improvement"]
    seeds = plan.get("validation_seeds")
    if not isinstance(seeds, list):
        return ["evaluation_plan.validation_seeds must be a list for multi-seed validation"]
    unique_seeds = {str(seed) for seed in seeds}
    if len(unique_seeds) < 3:
        return ["evaluation_plan.validation_seeds should contain at least 3 unique seeds"]
    aggregation = str(plan.get("aggregation") or "").lower()
    if "mean" not in aggregation or "std" not in aggregation:
        return ["evaluation_plan.aggregation should report mean and std across seeds"]
    return []


def validate_one(
    proposal: dict[str, Any],
    *,
    line_no: int,
    schema: dict[str, Any],
    registry_by_id: dict[str, dict[str, Any]],
    registry_ids: set[str],
    seen_ids: set[str],
    seen_param_signatures: set[str],
    memory_param_signatures: set[str],
    require_multiseed: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    review_reasons: list[str] = []
    warnings: list[str] = []

    required_fields = [str(item) for item in schema.get("required_fields", [])]
    allowed_levels = {str(item) for item in schema.get("allowed_runnable_levels", [])}
    allowed_models = {str(item) for item in schema.get("allowed_base_models", ["BPR", "LightGCN"])}
    allowed_runner_types = {str(item) for item in schema.get("allowed_runner_types", ["config_only", "model", "posthoc"])}
    allowed_proposal_types = {
        str(item) for item in schema.get("allowed_proposal_types", ["tuning", "algorithmic_variant", "research_spec"])
    }
    allowed_roots = {
        str(item).replace("\\", "/") for item in schema.get("allowed_implementation_roots", [])
    }

    for field in required_fields:
        if field not in proposal:
            errors.append(f"missing required field: {field}")

    proposal_type = str(proposal.get("proposal_type") or "").strip()
    candidate_id = str(proposal.get("candidate_id") or "").strip()
    parent_id = str(proposal.get("parent_candidate_id") or "").strip()
    base_model = str(proposal.get("base_model") or "").strip()
    runnable_level = str(proposal.get("runnable_level") or "").strip()
    runner_type = str(proposal.get("runner_type") or "").strip()

    if proposal_type not in allowed_proposal_types:
        errors.append(f"proposal_type must be one of {sorted(allowed_proposal_types)}: {proposal_type or '<missing>'}")

    if not candidate_id:
        errors.append("candidate_id is empty")
    elif not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", candidate_id):
        errors.append("candidate_id contains unsupported characters")
    elif candidate_id in registry_ids:
        errors.append(f"candidate_id already exists in registry: {candidate_id}")
    elif candidate_id in seen_ids:
        errors.append(f"candidate_id is duplicated in proposal file: {candidate_id}")

    parent = registry_by_id.get(parent_id)
    if not parent_id:
        errors.append("parent_candidate_id is empty")
    elif parent is None:
        errors.append(f"parent_candidate_id does not exist in registry: {parent_id}")

    if base_model not in allowed_models:
        errors.append(f"base_model must be one of {sorted(allowed_models)}: {base_model or '<missing>'}")
    elif parent is not None:
        parent_models = parent_base_models(parent.get("base_model"), allowed_models)
        if parent_models and base_model not in parent_models:
            errors.append(
                f"base_model does not match parent {parent_id}: "
                f"parent={sorted(parent_models)}, proposal={base_model}"
            )

    if runnable_level not in allowed_levels:
        errors.append(f"runnable_level must be one of {sorted(allowed_levels)}: {runnable_level or '<missing>'}")

    if runner_type not in allowed_runner_types:
        errors.append(f"runner_type must be one of {sorted(allowed_runner_types)}: {runner_type or '<missing>'}")

    consumes_value = proposal.get("consumes")
    consumes = extract_consume_names(consumes_value)
    if not isinstance(consumes_value, list):
        errors.append("consumes must be a list")
    if runnable_level == "parameter_only" and not consumes:
        errors.append("parameter_only proposal must consume at least one parent parameter")

    if parent is not None:
        parent_consumes = {str(item) for item in as_list(parent.get("consumes"))}
        declared_new_params = new_parameter_names(proposal)
        unsupported = sorted(set(consumes) - parent_consumes)

        if runnable_level in {"parameter_only", "config_only"} and unsupported:
            errors.append(f"consumes includes parameters not supported by parent {parent_id}: {unsupported}")
        elif runnable_level in {"code_required", "spec_only"}:
            undeclared = sorted(set(unsupported) - declared_new_params)
            if undeclared:
                errors.append(
                    f"new consumes must be declared in new_parameters for {runnable_level}: {undeclared}"
                )

        overrides = proposal.get("parameter_overrides") or {}
        if overrides and not isinstance(overrides, dict):
            errors.append("parameter_overrides must be an object when provided")
        elif isinstance(overrides, dict):
            unsupported_overrides = sorted(set(map(str, overrides.keys())) - parent_consumes)
            if runnable_level in {"parameter_only", "config_only"} and unsupported_overrides:
                errors.append(
                    f"parameter_overrides includes parameters not supported by parent {parent_id}: {unsupported_overrides}"
                )

        if runnable_level == "parameter_only" and not bool(parent.get("wired")):
            errors.append(f"parameter_only parent must be wired=true: {parent_id}")
        elif runnable_level == "config_only" and not bool(parent.get("wired")):
            review_reasons.append(f"parent is not wired yet: {parent_id}")

    parameter_signature = proposal_parameter_signature(proposal)
    if proposal_type == "tuning" and runnable_level in {"parameter_only", "config_only"}:
        if not parameter_signature:
            errors.append("tuning proposal must include non-empty parameter_overrides")
        elif (
            proposal.get("parameter_signature")
            and canonical_parameter_signature_text(proposal.get("parameter_signature")) != parameter_signature
        ):
            errors.append(
                "parameter_signature does not match normalized parent_candidate_id + parameter_overrides: "
                f"expected {parameter_signature}"
            )
        elif parameter_signature in memory_param_signatures:
            errors.append(f"parameter signature was already run in agent memory: {parameter_signature}")
        elif parameter_signature in seen_param_signatures:
            errors.append(f"parameter signature is duplicated in proposal file: {parameter_signature}")
        seen_param_signatures.add(parameter_signature)

    warnings.extend(multiseed_warnings(proposal, runnable_level))
    if require_multiseed:
        review_reasons.extend(f"multi-seed validation plan: {warning}" for warning in warnings)

    if mentions_recbole_core_change(proposal):
        errors.append("proposal appears to require modifying RecBole core")

    risk = proposal.get("risk")
    if isinstance(risk, dict) and risk.get("recbole_core_change_required") is True:
        errors.append("risk.recbole_core_change_required must not be true")

    if runnable_level == "code_required":
        if proposal_type not in {"algorithmic_variant", "research_spec"}:
            errors.append("code_required proposal must use algorithmic_variant or research_spec")
        if not isinstance(proposal.get("implementation_plan"), dict):
            review_reasons.append("code_required proposal should include implementation_plan")
        if not implementation_files(proposal):
            review_reasons.append("code_required proposal should include allowed_files or implementation_plan.files")
        review_reasons.append("code_required proposal needs local implementation before it can run")

    if runnable_level == "spec_only":
        review_reasons.append("spec_only proposal needs design review before implementation")

    if runner_type == "posthoc":
        review_reasons.append("posthoc is valid but current run_candidate.py has no posthoc execution flow")

    if allowed_roots:
        disallowed_files = sorted(
            path for path in implementation_files(proposal) if not path_is_allowed(path, allowed_roots)
        )
        if disallowed_files:
            errors.append(f"implementation files are outside allowed roots: {disallowed_files}")

    if errors:
        status = REJECTED
    elif review_reasons:
        status = NEEDS_REVIEW
    else:
        status = ACCEPTED

    if candidate_id:
        seen_ids.add(candidate_id)

    return {
        "line": line_no,
        "candidate_id": candidate_id,
        "parameter_signature": parameter_signature,
        "proposal_type": proposal_type,
        "runnable_level": runnable_level,
        "runner_type": runner_type,
        "status": status,
        "next_action": next_action_for(status, runnable_level, runner_type),
        "errors": errors,
        "review_reasons": review_reasons,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RecClaw candidate proposal JSONL.")
    parser.add_argument("--proposals", default=str(PROPOSAL_PATH), help="Path to candidate proposal JSONL")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Path to candidate_registry.yaml")
    parser.add_argument("--schema", default=str(SCHEMA_PATH), help="Path to candidate_proposal_schema.yaml")
    parser.add_argument(
        "--memory",
        default=str(MEMORY_PATH),
        help="Optional agent memory JSONL used to reject already-run parameter signatures",
    )
    parser.add_argument(
        "--require-multiseed",
        action="store_true",
        help="Route runnable tuning proposals without a 3-seed evaluation plan to needs_review",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    schema = load_yaml(Path(args.schema))
    registry = load_yaml(Path(args.registry)).get("candidates", [])
    if not isinstance(registry, list):
        raise ValueError(f"registry candidates must be a list: {args.registry}")
    registry_by_id = {str(item.get("candidate_id")): item for item in registry if item.get("candidate_id")}
    registry_ids = set(registry_by_id)
    memory_param_signatures = load_memory_param_signatures(Path(args.memory))

    proposal_path = Path(args.proposals)
    if not proposal_path.exists():
        raise FileNotFoundError(f"proposal file does not exist: {proposal_path}")

    seen_ids: set[str] = set()
    seen_param_signatures: set[str] = set()
    results: list[dict[str, Any]] = []
    for line_no, proposal, parse_error in load_jsonl(proposal_path):
        if parse_error is not None:
            results.append(
                {
                    "line": line_no,
                    "candidate_id": "",
                    "parameter_signature": "",
                    "proposal_type": "",
                    "runnable_level": "",
                    "runner_type": "",
                    "status": REJECTED,
                    "next_action": NEXT_ACTIONS[REJECTED],
                    "errors": [parse_error],
                    "review_reasons": [],
                    "warnings": [],
                }
            )
            continue
        assert proposal is not None
        results.append(
            validate_one(
                proposal,
                line_no=line_no,
                schema=schema,
                registry_by_id=registry_by_id,
                registry_ids=registry_ids,
                seen_ids=seen_ids,
                seen_param_signatures=seen_param_signatures,
                memory_param_signatures=memory_param_signatures,
                require_multiseed=bool(args.require_multiseed),
            )
        )

    summary = {
        ACCEPTED: sum(1 for item in results if item["status"] == ACCEPTED),
        REJECTED: sum(1 for item in results if item["status"] == REJECTED),
        NEEDS_REVIEW: sum(1 for item in results if item["status"] == NEEDS_REVIEW),
        "warnings": sum(len(item.get("warnings") or []) for item in results),
        "total": len(results),
    }
    next_actions: dict[str, int] = {}
    for item in results:
        action = str(item.get("next_action") or "")
        next_actions[action] = next_actions.get(action, 0) + 1
    report = {"summary": summary, "next_actions": next_actions, "results": results}

    text = json.dumps(report, ensure_ascii=True, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    return 1 if summary[REJECTED] else 0


if __name__ == "__main__":
    raise SystemExit(main())
