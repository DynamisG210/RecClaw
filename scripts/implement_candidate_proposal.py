#!/usr/bin/env python3
"""Implement a code_required RecClaw candidate proposal under strict allowlists."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml

try:
    from action_space import (
        allowed_implementation_roots,
        load_action_space,
        parameter_space_from_action_space,
    )
    from validate_candidate_proposal import load_jsonl, load_yaml
except ImportError:
    from .action_space import (
        allowed_implementation_roots,
        load_action_space,
        parameter_space_from_action_space,
    )
    from .validate_candidate_proposal import load_jsonl, load_yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH = PROJECT_ROOT / "configs" / "candidate_registry.yaml"
PROPOSAL_PATH = PROJECT_ROOT / "results" / "candidate_proposals.jsonl"
ACTION_SPACE_PATH = PROJECT_ROOT / "configs" / "action_space.yaml"
ACTION_SPACE = load_action_space(ACTION_SPACE_PATH)

ALLOWED_WRITE_ROOTS = tuple(sorted(allowed_implementation_roots(ACTION_SPACE)))
FORBIDDEN_PATHS = {
    "configs/task_ml1m.yaml",
    "configs/lightgcn.yaml",
    "configs/bpr.yaml",
    "configs/candidate_registry.yaml",
    "recclaw_ext/models/__init__.py",
    "recclaw_ext/posthoc/__init__.py",
}
IMPLEMENTATION_PARAMETER_KEYS = tuple(parameter_space_from_action_space(ACTION_SPACE))
IMPLEMENTATION_PARAMETER_SCHEMA = {
    key: {"type": ["number", "null"]} for key in IMPLEMENTATION_PARAMETER_KEYS
}

IMPLEMENTATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "candidate_config": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string"},
                "model": {"type": "string"},
                **IMPLEMENTATION_PARAMETER_SCHEMA,
            },
            "required": [
                "candidate_id",
                "model",
                *IMPLEMENTATION_PARAMETER_KEYS,
            ],
        },
        "registry_entry": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string"},
                "category": {"type": "string"},
                "base_model": {"type": "string"},
                "rs_problem": {"type": "string"},
                "hypothesis": {"type": "string"},
                "implementation_type": {"type": "string"},
                "minimal_change": {"type": "string"},
                "priority": {"type": "string"},
                "status": {"type": "string"},
                "wired": {"type": "boolean"},
                "runner_type": {"type": "string"},
                "entrypoint": {"type": "string"},
                "consumes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "candidate_id",
                "category",
                "base_model",
                "rs_problem",
                "hypothesis",
                "implementation_type",
                "minimal_change",
                "priority",
                "status",
                "wired",
                "runner_type",
                "entrypoint",
                "consumes",
            ],
        },
    },
    "required": ["files", "candidate_config", "registry_entry"],
}

SELF_REG_LOSS_PATTERN = re.compile(r"\bself\.reg_loss\s*\(")
SOFT_L2_POSITIONAL_MAX_NORM_PATTERN = re.compile(
    r"soft_l2_norm_penalty\s*\([^)\n]+,\s*(?:self\.)?(?:max_norm|lambda_norm)\b"
)
LIGHTGCN_BASE_CLASS_PATTERN = re.compile(r"class\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*LightGCN[^)]*\)")
VALID_SCORE_PATTERN = re.compile(r"valid_score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    parsed = json.loads(strip_json_fence(text))
    if not isinstance(parsed, dict):
        raise ValueError("LLM implementation response must be a JSON object")
    return parsed


def chat_completion(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key_env: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    messages: list[dict[str, str]],
    retries: int = 2,
) -> str:
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"missing LLM API key env var: {api_key_env}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "openai":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "recclaw_candidate_implementation",
                "strict": True,
                "schema": IMPLEMENTATION_RESPONSE_SCHEMA,
            },
        }
    req = urlrequest.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    context = ssl.create_default_context()
    ignore_unexpected_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
    if ignore_unexpected_eof:
        context.options |= ignore_unexpected_eof
    raw = ""
    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            with urlrequest.urlopen(req, timeout=timeout, context=context) as response:
                raw = response.read().decode("utf-8", errors="replace")
            break
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
        except (urlerror.URLError, TimeoutError, ssl.SSLError) as exc:
            if attempt >= max_attempts:
                raise RuntimeError(f"LLM API request failed after {attempt} attempts: {exc}") from exc
            time.sleep(min(2.0 * attempt, 5.0))

    parsed = json.loads(raw)
    choices = parsed.get("choices") or []
    if not choices:
        raise ValueError("LLM API response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM API response has empty content")
    return content


def load_proposal(path: Path, proposal_id: str) -> dict[str, Any]:
    for _, proposal, parse_error in load_jsonl(path):
        if parse_error or proposal is None:
            continue
        if str(proposal.get("candidate_id") or "") == proposal_id:
            return proposal
    raise KeyError(f"proposal_id not found: {proposal_id}")


def normalize_repo_path(path: str) -> str:
    clean = path.strip().replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    return clean


def path_is_allowed(path: str) -> bool:
    clean = normalize_repo_path(path)
    if not clean or clean.startswith("/") or re.match(r"^[A-Za-z]:", clean):
        return False
    parts = PurePosixPath(clean).parts
    if any(part == ".." for part in parts):
        return False
    if PurePosixPath(clean).name == "__init__.py" and parts and parts[0] == "recclaw_ext":
        return False
    if clean in FORBIDDEN_PATHS:
        return False
    return any(clean.startswith(root) for root in ALLOWED_WRITE_ROOTS)


@dataclass
class PreparedImplementation:
    candidate_id: str
    files: list[dict[str, str]]
    candidate_config: dict[str, Any]
    registry_entry: dict[str, Any]
    config_path: Path
    entrypoint: str
    source: str = "llm"


TEMPLATE_IMPLEMENTATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "bpr_hard_negative_margin",
        "base_model": "BPR",
        "parents": {"cand_bpr_hard_negative_mix", "cand_bpr_margin_loss"},
        "required_consumes": {"hard_negative_ratio", "margin"},
        "model": "BPRHardNegativeMargin",
        "entrypoint": "recclaw_ext.models.bpr_composed:BPRHardNegativeMargin",
        "defaults": {"hard_negative_ratio": 0.5, "margin": 0.2},
        "category": "Bias & Sample Construction",
        "priority": "high",
        "rs_problem": "negative sampling is weak and the pairwise boundary is soft",
        "minimal_change": "reuse existing hard-negative sampler and margin loss helper",
    },
    {
        "name": "bpr_popularity_aware_margin",
        "base_model": "BPR",
        "parents": {"cand_bpr_popularity_aware_negative", "cand_bpr_margin_loss"},
        "required_consumes": {"popularity_alpha", "margin"},
        "model": "BPRPopularityAwareMargin",
        "entrypoint": "recclaw_ext.models.bpr_composed:BPRPopularityAwareMargin",
        "defaults": {"popularity_alpha": 0.5, "margin": 0.2},
        "category": "Bias & Sample Construction",
        "priority": "medium",
        "rs_problem": "negative sampling and pairwise separation are both under-constrained",
        "minimal_change": "reuse existing popularity-aware sampler and margin loss helper",
    },
    {
        "name": "lightgcn_edge_dropout_residual_norm",
        "base_model": "LightGCN",
        "parents": {"cand_lightgcn_edge_dropout_residual_mix", "cand_lightgcn_residual_layer_mix"},
        "required_consumes": {"residual_weight", "edge_dropout", "lambda_norm", "max_norm"},
        "model": "LightGCNEdgeDropoutResidualNorm",
        "entrypoint": "recclaw_ext.models.lightgcn_residual_norm:LightGCNEdgeDropoutResidualNorm",
        "defaults": {
            "embedding_size": 128,
            "n_layers": 3,
            "residual_weight": 0.2,
            "edge_dropout": 0.1,
            "lambda_norm": 1e-4,
            "max_norm": 1.0,
        },
        "category": "Representation & Interaction",
        "priority": "high",
        "rs_problem": "residual graph propagation needs structural and scale regularization",
        "minimal_change": "reuse existing edge-dropout residual propagation and norm-control loss mixin",
    },
    {
        "name": "lightgcn_residual_norm",
        "base_model": "LightGCN",
        "parents": {"cand_lightgcn_residual_layer_mix"},
        "required_consumes": {"residual_weight", "lambda_norm", "max_norm"},
        "forbidden_consumes": {"edge_dropout"},
        "model": "LightGCNResidualNormConstrained",
        "entrypoint": "recclaw_ext.models.lightgcn_residual_norm:LightGCNResidualNormConstrained",
        "defaults": {
            "embedding_size": 128,
            "n_layers": 3,
            "residual_weight": 0.2,
            "lambda_norm": 1e-4,
            "max_norm": 1.0,
        },
        "category": "Representation & Interaction",
        "priority": "high",
        "rs_problem": "residual propagation can drift in embedding scale",
        "minimal_change": "reuse existing residual propagation and add soft norm control",
    },
)


def template_implementation_for_proposal(proposal: dict[str, Any]) -> PreparedImplementation | None:
    """Map known safe code_required proposals to reusable local classes.

    This lets the agent turn compatible, unseen composition proposals into
    runnable candidate configs without asking an LLM to rewrite code that already
    exists in the repository.
    """

    candidate_id = str(proposal.get("candidate_id") or "")
    parent_id = str(proposal.get("parent_candidate_id") or "")
    base_model = str(proposal.get("base_model") or "")
    consumes = {str(item) for item in (proposal.get("consumes") or [])}
    if not candidate_id or not consumes:
        return None

    for spec in TEMPLATE_IMPLEMENTATIONS:
        required = set(spec["required_consumes"])
        forbidden = set(spec.get("forbidden_consumes") or set())
        if base_model != spec["base_model"]:
            continue
        if parent_id not in spec["parents"]:
            continue
        if not required.issubset(consumes):
            continue
        if forbidden & consumes:
            continue

        candidate_config = {"candidate_id": candidate_id, "model": spec["model"]}
        defaults = dict(spec["defaults"])
        for key, value in defaults.items():
            if key in consumes or key in required or key in {"embedding_size", "n_layers"}:
                candidate_config[key] = value
        candidate_config = enforce_proposal_parameter_defaults(candidate_config, proposal)
        candidate_config = {key: value for key, value in candidate_config.items() if value is not None}
        validate_candidate_config_matches_proposal(candidate_config, proposal)

        registry_entry = {
            "candidate_id": candidate_id,
            "category": proposal.get("category") or spec["category"],
            "base_model": base_model,
            "rs_problem": spec["rs_problem"],
            "hypothesis": proposal.get("hypothesis") or "Known safe template implementation.",
            "implementation_type": "model",
            "minimal_change": spec["minimal_change"],
            "priority": spec["priority"],
            "status": "implemented",
            "wired": True,
            "runner_type": "model",
            "entrypoint": spec["entrypoint"],
            "consumes": [key for key in defaults if key in candidate_config and key not in {"candidate_id", "model"}],
        }
        config_path = PROJECT_ROOT / "configs" / "candidates" / f"{candidate_id}.yaml"
        if config_path.exists():
            raise ValueError(f"refusing to overwrite existing candidate config: {config_path.relative_to(PROJECT_ROOT)}")
        return PreparedImplementation(
            candidate_id=candidate_id,
            files=[],
            candidate_config=candidate_config,
            registry_entry=registry_entry,
            config_path=config_path,
            entrypoint=str(spec["entrypoint"]),
            source=f"template:{spec['name']}",
        )
    return None


def validate_files(files: Any) -> list[dict[str, str]]:
    if not isinstance(files, list):
        raise ValueError("implementation JSON must include files as a list")
    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for idx, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"files[{idx}] must be an object")
        path = normalize_repo_path(str(item.get("path") or ""))
        content = item.get("content")
        if not path:
            raise ValueError(f"files[{idx}].path is empty")
        if not path_is_allowed(path):
            raise ValueError(f"file path is outside implementation allowlist: {path}")
        if not path.endswith(".py"):
            raise ValueError(f"implementation files must be Python modules; configs are generated separately: {path}")
        if path in seen_paths:
            raise ValueError(f"duplicate implementation file path: {path}")
        seen_paths.add(path)
        target = (PROJECT_ROOT / path).resolve()
        try:
            target.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"file path escapes project root: {path}") from exc
        if target.exists():
            raise ValueError(f"refusing to overwrite existing implementation file: {path}")
        if not isinstance(content, str):
            raise ValueError(f"files[{idx}].content must be a string: {path}")
        validate_static_model_code(path, content)
        normalized.append({"path": path, "content": content})
    return normalized


def validate_static_model_code(path: str, content: str) -> None:
    try:
        compile(content, path, "exec")
    except SyntaxError as exc:
        raise ValueError(f"generated Python has syntax error in {path}: {exc}") from exc
    if "config.get" in content:
        raise ValueError(
            f"generated model uses config.get in {path}; RecBole Config does not support it. "
            "Use recclaw_ext.models._utils.config_get/config_float or config[key] with a fallback."
        )
    if (
        SELF_REG_LOSS_PATTERN.search(content)
        and "def reg_loss" not in content
        and "self.reg_loss =" not in content
        and not LIGHTGCN_BASE_CLASS_PATTERN.search(content)
    ):
        raise ValueError(
            f"generated model calls self.reg_loss in {path}, but RecBole BPR does not provide it; "
            "use recclaw_ext.models._utils.soft_l2_norm_penalty or an existing loss helper"
        )
    if SOFT_L2_POSITIONAL_MAX_NORM_PATTERN.search(content):
        raise ValueError(
            f"generated model passes max_norm/lambda_norm positionally to soft_l2_norm_penalty in {path}; "
            "call soft_l2_norm_penalty(emb1, emb2, max_norm=..., weight=...) so scalar values are not treated as tensors"
        )


def enforce_proposal_parameter_defaults(candidate_config: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    config = dict(candidate_config)
    overrides = proposal.get("parameter_overrides") or {}
    if not isinstance(overrides, dict):
        return config
    for key, value in overrides.items():
        if key in IMPLEMENTATION_PARAMETER_KEYS and value is not None:
            config[key] = value
    return config


def validate_candidate_config_matches_proposal(candidate_config: dict[str, Any], proposal: dict[str, Any]) -> None:
    overrides = proposal.get("parameter_overrides") or {}
    if not isinstance(overrides, dict):
        return
    mismatches: list[str] = []
    for key, value in overrides.items():
        if key not in IMPLEMENTATION_PARAMETER_KEYS or value is None:
            continue
        if candidate_config.get(key) != value:
            mismatches.append(f"{key} expected {value!r}, got {candidate_config.get(key)!r}")
    if mismatches:
        raise ValueError("candidate_config must preserve proposal parameter_overrides: " + "; ".join(mismatches))


def prepare_implementation(implementation: dict[str, Any], proposal: dict[str, Any]) -> PreparedImplementation:
    files = validate_files(implementation.get("files"))
    registry_entry = implementation.get("registry_entry")
    candidate_config = implementation.get("candidate_config")
    if not isinstance(registry_entry, dict):
        raise ValueError("implementation JSON must include registry_entry object")
    if not isinstance(candidate_config, dict):
        raise ValueError("implementation JSON must include candidate_config object")
    registry_entry = dict(registry_entry)
    candidate_config = dict(candidate_config)
    candidate_id = str(registry_entry.get("candidate_id") or proposal.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("candidate_id is required")
    registry_entry["candidate_id"] = candidate_id
    candidate_config["candidate_id"] = candidate_id
    candidate_config = enforce_proposal_parameter_defaults(candidate_config, proposal)
    candidate_config = {key: value for key, value in candidate_config.items() if value is not None}
    validate_candidate_config_matches_proposal(candidate_config, proposal)
    config_path = PROJECT_ROOT / "configs" / "candidates" / f"{candidate_id}.yaml"
    if config_path.exists():
        raise ValueError(f"refusing to overwrite existing candidate config: {config_path.relative_to(PROJECT_ROOT)}")
    if not candidate_config.get("model"):
        raise ValueError("candidate_config.model is required")
    entrypoint = str(registry_entry.get("entrypoint") or "")
    if not entrypoint:
        raise ValueError("registry_entry.entrypoint is required")
    entrypoint = normalize_entrypoint(entrypoint, files)
    registry_entry["entrypoint"] = entrypoint
    return PreparedImplementation(
        candidate_id=candidate_id,
        files=files,
        candidate_config=candidate_config,
        registry_entry=registry_entry,
        config_path=config_path,
        entrypoint=entrypoint,
    )


def camel_to_snake(name: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def normalize_entrypoint(entrypoint: str, files: list[dict[str, str]]) -> str:
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint must use module:attribute format: {entrypoint}")
    module_name, attr = entrypoint.split(":", 1)
    module_name = module_name.strip()
    attr = attr.strip()
    if not module_name or not attr:
        raise ValueError(f"entrypoint must use module:attribute format: {entrypoint}")
    file_by_path = {item["path"]: item for item in files}
    definition_pattern = re.compile(rf"^\s*(?:class|def)\s+{re.escape(attr)}\b", flags=re.MULTILINE)
    if module_name not in {"recclaw_ext.models", "recclaw_ext.posthoc"}:
        allowed_prefixes = ("recclaw_ext.models.", "recclaw_ext.posthoc.")
        if not module_name.startswith(allowed_prefixes):
            raise ValueError(f"entrypoint must stay inside recclaw_ext local packages: {entrypoint}")
        module_path = module_name.replace(".", "/") + ".py"
        if module_path not in file_by_path:
            raise ValueError(f"entrypoint module must be one of the generated files: {module_path}")
        content = file_by_path[module_path].get("content") or ""
        if not definition_pattern.search(content):
            raise ValueError(f"entrypoint target {attr} is not defined in generated file: {module_path}")
        return f"{module_name}:{attr}"

    root = module_name.replace(".", "/") + "/"
    candidate_files = [
        item["path"]
        for item in files
        if item["path"].startswith(root)
        and item["path"].endswith(".py")
        and not item["path"].endswith("/__init__.py")
    ]
    if not candidate_files:
        raise ValueError(
            f"package-level entrypoint {entrypoint} needs a concrete module file under {root}"
        )
    expected_stem = camel_to_snake(attr)
    class_matches = [
        item["path"]
        for item in files
        if item["path"] in candidate_files and definition_pattern.search(item.get("content") or "")
    ]
    if len(class_matches) > 1:
        raise ValueError(f"entrypoint class {attr} appears in multiple files: {class_matches}")
    exact_matches = [path for path in candidate_files if Path(path).stem == expected_stem]
    if len(class_matches) == 1:
        selected = class_matches[0]
    elif len(exact_matches) == 1:
        selected = exact_matches[0]
        content = file_by_path[selected].get("content") or ""
        if content.strip() and not definition_pattern.search(content):
            raise ValueError(f"entrypoint target {attr} is not defined in generated file: {selected}")
    else:
        raise ValueError(
            f"could not identify concrete module file for package-level entrypoint {entrypoint}; "
            f"use recclaw_ext.models.some_module:{attr}"
        )
    module = selected[:-3].replace("/", ".")
    return f"{module}:{attr}"


def import_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"entrypoint must use module:attribute format: {spec}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def registry_payload_with_entry(
    registry_path: Path,
    registry_entry: dict[str, Any],
    *,
    wired: bool,
    status: str,
) -> dict[str, Any]:
    payload = load_yaml(registry_path)
    candidates = payload.setdefault("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"registry candidates must be a list: {registry_path}")
    candidate_id = str(registry_entry.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("registry_entry.candidate_id is required")
    entry = dict(registry_entry)
    entry["wired"] = wired
    entry["status"] = status
    for idx, candidate in enumerate(candidates):
        if str(candidate.get("candidate_id") or "") == candidate_id:
            candidates[idx] = {**candidate, **entry}
            break
    else:
        candidates.append(entry)
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_dry_run(candidate_id: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_candidate.py"), candidate_id, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_training_smoke(candidate_id: str, smoke_sets: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_candidate.py"),
        candidate_id,
        "--cleanup-checkpoints",
    ]
    for item in smoke_sets:
        cmd.extend(["--set", item])
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_valid_scores(text: str) -> list[float]:
    values: list[float] = []
    for match in VALID_SCORE_PATTERN.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def compact_smoke_summary(smoke_run: subprocess.CompletedProcess[str], smoke_sets: list[str]) -> dict[str, Any]:
    text = f"{smoke_run.stdout}\n{smoke_run.stderr}"
    valid_scores = parse_valid_scores(text)
    rounded_unique = {round(value, 8) for value in valid_scores}
    run_ids = re.findall(r'"run_id"\s*:\s*"([^"]+)"', text)
    log_paths = re.findall(r'"log_path"\s*:\s*"([^"]+)"', text)
    summary: dict[str, Any] = {
        "exit_code": smoke_run.returncode,
        "overrides": smoke_sets,
        "valid_scores": [round(value, 8) for value in valid_scores],
        "valid_score_unique_count": len(rounded_unique),
        "metric_stagnation": len(valid_scores) >= 3 and len(rounded_unique) <= 1,
    }
    if run_ids:
        summary["run_id"] = run_ids[-1]
    if log_paths:
        summary["log_path"] = log_paths[-1]
    if smoke_run.returncode != 0 or smoke_run.stderr.strip():
        summary["stderr_tail"] = smoke_run.stderr[-800:]
    return summary


def cleanup_artifacts(prepared: PreparedImplementation, registry_path: Path, previous_registry_text: str | None) -> None:
    for item in prepared.files:
        path = PROJECT_ROOT / item["path"]
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        prepared.config_path.unlink()
    except FileNotFoundError:
        pass
    if previous_registry_text is not None:
        registry_path.write_text(previous_registry_text, encoding="utf-8")


def build_implementation_context(
    proposal: dict[str, Any],
    registry: dict[str, Any],
    experiment_directive: str = "",
) -> dict[str, Any]:
    directive = str(experiment_directive or "").strip()
    context = {
        "task": "Implement one RecClaw code_required candidate proposal under strict local allowlists.",
        "proposal": proposal,
        "registry": registry,
        "steering_priority": [
            "forbidden and allowlists",
            "experiment_directive when enabled",
            "proposal specification",
        ],
        "experiment_directive": {
            "enabled": bool(directive),
            "priority": "highest user steering instruction below forbidden and allowlists",
            "text": directive,
        },
        "required_json_shape": {
            "files": [{"path": "recclaw_ext/models/example.py", "content": "complete file content"}],
            "candidate_config": {"candidate_id": proposal.get("candidate_id"), "model": "LocalModelName"},
            "registry_entry": {
                "candidate_id": proposal.get("candidate_id"),
                "category": proposal.get("category"),
                "base_model": proposal.get("base_model"),
                "runner_type": proposal.get("runner_type") or "model",
                "entrypoint": "recclaw_ext.models.example:LocalModelName",
                "consumes": proposal.get("consumes") or [],
            },
        },
        "action_space": ACTION_SPACE,
        "allowed_write_roots": list(ALLOWED_WRITE_ROOTS),
        "config_update": "Use candidate_config only; do not include configs/candidates/*.yaml in files.",
        "registry_update": "Use registry_entry only; do not include configs/candidate_registry.yaml in files.",
        "forbidden": [
            "Do not modify RecBole core.",
            "Do not modify data split, evaluation protocol, baseline config, or task config.",
            "Do not write any __init__.py file; use concrete module entrypoints instead.",
            "Do not overwrite existing local helper/model files; return a new module file for this proposal.",
            "Do not include config files in files; candidate_config is written by this script.",
            "Stay inside action_space; candidate_config knobs should use declared action_space parameters.",
            "Use entrypoints like recclaw_ext.models.my_model:MyModel, not recclaw_ext.models:MyModel.",
            "If experiment_directive.enabled is true, follow experiment_directive.text unless it conflicts with forbidden rules or the proposal specification.",
            "RecBole Config is not a dict; never call config.get(...).",
            "Prefer helpers from recclaw_ext.models._utils: config_get, config_float, margin_bpr_loss, soft_l2_norm_penalty.",
            "Do not call self.reg_loss unless the generated class defines reg_loss itself.",
            "Call soft_l2_norm_penalty with max_norm=... and weight=... keyword arguments; never pass scalar knobs positionally.",
            "Preserve proposal.parameter_overrides exactly in candidate_config defaults.",
            "Do not override full_sort_predict unless the proposal explicitly requires it.",
            "Return strict JSON only.",
            "Provide complete file contents, not patches.",
        ],
    }
    return context


def status_report(status: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"status": status}
    payload.update(kwargs)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Implement one code_required candidate proposal via LLM.")
    parser.add_argument("--proposal-id", required=True, help="candidate_id of the proposal to implement")
    parser.add_argument("--proposals", default=str(PROPOSAL_PATH), help="candidate proposal JSONL path")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="candidate_registry.yaml path")
    parser.add_argument(
        "--llm-provider",
        choices=("deepseek", "openai", "compatible"),
        default=os.environ.get("RECCLAW_LLM_PROVIDER", "deepseek"),
        help="LLM provider. openai enables strict JSON schema responses.",
    )
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default="", help="LLM model name")
    parser.add_argument("--llm-api-key-env", default="", help="Environment variable containing API key")
    parser.add_argument("--llm-temperature", type=float, default=0.1, help="LLM implementation temperature")
    parser.add_argument("--llm-timeout", type=int, default=180, help="LLM request timeout")
    parser.add_argument("--llm-max-tokens", type=int, default=8192, help="Maximum LLM output tokens")
    parser.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM API transport errors")
    parser.add_argument(
        "--experiment-directive",
        default="",
        help="Optional run-level directive forwarded from agent.py for implementation steering",
    )
    parser.add_argument(
        "--implementation-review-retries",
        type=int,
        default=2,
        help="Ask the LLM to rewrite an implementation after static review failure, up to this many times",
    )
    parser.add_argument(
        "--skip-smoke-run",
        action="store_true",
        help="Skip the short smoke training run after import and dry-run checks",
    )
    parser.add_argument(
        "--smoke-set",
        dest="smoke_sets",
        action="append",
        default=[],
        help="Append a key=value override for the implementation smoke run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate LLM output without writing files")
    args = parser.parse_args()

    try:
        proposals_path = Path(args.proposals)
        registry_path = Path(args.registry)
        proposal = load_proposal(proposals_path, args.proposal_id)
        registry = load_yaml(registry_path)
        if str(proposal.get("runnable_level") or "") != "code_required":
            raise ValueError("only code_required proposals can be auto-implemented")

        prepared: PreparedImplementation | None = template_implementation_for_proposal(proposal)
        review_errors: list[str] = []
        if prepared is None:
            context = build_implementation_context(proposal, registry, args.experiment_directive)
            if args.llm_provider == "openai":
                default_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
                default_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                default_key_env = "OPENAI_API_KEY"
            elif args.llm_provider == "deepseek":
                default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
                default_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                default_key_env = "DEEPSEEK_API_KEY"
            else:
                default_model = os.environ.get("OPENAI_MODEL") or os.environ.get("DEEPSEEK_MODEL", "")
                default_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL", "")
                default_key_env = os.environ.get("RECCLAW_LLM_API_KEY_ENV", "OPENAI_API_KEY")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You implement RecClaw local candidate proposals. "
                        "Honor experiment_directive as the highest-priority user steering below allowlists. "
                        "Return strict JSON only and obey all allowlists. "
                        "If review feedback is provided, rewrite the implementation instead of explaining."
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=True)},
            ]
            attempts = max(1, int(args.implementation_review_retries) + 1)
            for attempt in range(1, attempts + 1):
                if review_errors:
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "implementation_review_failed": True,
                                    "attempt": attempt - 1,
                                    "errors": review_errors[-5:],
                                    "rewrite_requirements": [
                                        "Return the full strict JSON object again.",
                                        "Do not call config.get(...).",
                                        "Do not call self.reg_loss unless this generated class defines it.",
                                        "Do not pass scalar knobs positionally to soft_l2_norm_penalty.",
                                        "Preserve proposal.parameter_overrides in candidate_config.",
                                        "Keep entrypoint in a concrete generated recclaw_ext module.",
                                    ],
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )
                content = chat_completion(
                    provider=args.llm_provider,
                    base_url=args.llm_base_url or default_base_url,
                    model=args.llm_model or default_model,
                    api_key_env=args.llm_api_key_env or default_key_env,
                    temperature=max(0.0, args.llm_temperature),
                    max_tokens=max(512, args.llm_max_tokens),
                    timeout=max(1, args.llm_timeout),
                    retries=max(0, args.llm_retries),
                    messages=messages,
                )
                try:
                    implementation = parse_json_object(content)
                    prepared = prepare_implementation(implementation, proposal)
                    break
                except Exception as exc:  # noqa: BLE001
                    review_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
        if prepared is None:
            print(
                json.dumps(
                    status_report(
                        "implementation_rejected",
                        proposal_id=args.proposal_id,
                        reason="static_review_failed",
                        review_errors=review_errors,
                    ),
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 1

        candidate_id = prepared.candidate_id
        files = prepared.files
        candidate_config = prepared.candidate_config
        registry_entry = prepared.registry_entry
        config_path = prepared.config_path
        entrypoint = prepared.entrypoint

        if args.dry_run:
            print(
                json.dumps(
                    status_report(
                        "dry_run_ok",
                        proposal_id=args.proposal_id,
                        candidate_id=candidate_id,
                        files=[item["path"] for item in files],
                        config_path=str(config_path.relative_to(PROJECT_ROOT)),
                        entrypoint=entrypoint,
                        source=prepared.source,
                    ),
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 0

        previous_registry_text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else None
        for item in files:
            path = PROJECT_ROOT / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item["content"], encoding="utf-8")
        write_yaml(config_path, candidate_config)
        write_yaml(registry_path, registry_payload_with_entry(registry_path, registry_entry, wired=False, status="implement-ready"))

        try:
            import_object(entrypoint)
        except Exception as exc:  # noqa: BLE001
            cleanup_artifacts(prepared, registry_path, previous_registry_text)
            print(
                json.dumps(
                    status_report(
                        "implementation_failed",
                        proposal_id=args.proposal_id,
                        candidate_id=candidate_id,
                        reason=f"entrypoint import failed: {type(exc).__name__}: {exc}",
                    ),
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 1

        # run_candidate.py validates wired=true even for --dry-run, so briefly
        # expose the entry and immediately close it again if smoke planning fails.
        write_yaml(registry_path, registry_payload_with_entry(registry_path, registry_entry, wired=True, status="implemented"))
        dry_run = run_dry_run(candidate_id)
        if dry_run.returncode != 0:
            cleanup_artifacts(prepared, registry_path, previous_registry_text)
            print(
                json.dumps(
                    status_report(
                        "implemented_but_smoke_failed",
                        proposal_id=args.proposal_id,
                        candidate_id=candidate_id,
                        reason="run_candidate.py --dry-run failed",
                        stdout=dry_run.stdout[-2000:],
                        stderr=dry_run.stderr[-2000:],
                    ),
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 1

        smoke_status = "implemented_and_importable"
        smoke_summary: dict[str, Any] = {}
        if not args.skip_smoke_run:
            smoke_sets = list(args.smoke_sets or [])
            if not any(item.split("=", 1)[0] == "epochs" for item in smoke_sets if "=" in item):
                smoke_sets.append("epochs=3")
            if not any(item.split("=", 1)[0] == "stopping_step" for item in smoke_sets if "=" in item):
                smoke_sets.append("stopping_step=2")
            smoke_run = run_training_smoke(candidate_id, smoke_sets)
            smoke_summary = compact_smoke_summary(smoke_run, smoke_sets)
            if smoke_run.returncode != 0 or smoke_summary.get("metric_stagnation"):
                cleanup_artifacts(prepared, registry_path, previous_registry_text)
                fail_reason = (
                    "metric_stagnation_smoke"
                    if smoke_summary.get("metric_stagnation")
                    else "short smoke run failed"
                )
                print(
                    json.dumps(
                        status_report(
                            "implemented_but_smoke_failed",
                            proposal_id=args.proposal_id,
                            candidate_id=candidate_id,
                            reason=fail_reason,
                            smoke=smoke_summary,
                        ),
                        ensure_ascii=True,
                        indent=2,
                    )
                )
                return 1
            smoke_status = "implemented_and_smoke_passed"

        write_yaml(registry_path, registry_payload_with_entry(registry_path, registry_entry, wired=True, status="implemented"))
        print(
            json.dumps(
                status_report(
                    smoke_status,
                    proposal_id=args.proposal_id,
                    candidate_id=candidate_id,
                    files=[item["path"] for item in files],
                    config_path=str(config_path.relative_to(PROJECT_ROOT)),
                    entrypoint=entrypoint,
                    source=prepared.source,
                    smoke=smoke_summary,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                status_report("implementation_failed", proposal_id=args.proposal_id, reason=f"{type(exc).__name__}: {exc}"),
                ensure_ascii=True,
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
