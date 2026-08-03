"""Explicit Q5-A deployment inventory and no-call preflight.

The pilot runner is intentionally physical-call agnostic here.  This module
only inventories the source/import/runtime boundary and validates that the
existing consumers can be imported and rendered without invoking Provider,
Implementer, Qualifier, or RecBole training.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import bytes_sha256, canonical_value, sha256_digest


Q5A_DEPLOYMENT_SCHEMA = "recclaw.research-line.q5a-deployment-manifest.v1"
Q5A_PREFLIGHT_SCHEMA = "recclaw.research-line.q5a-comprehensive-preflight.v1"
EXPECTED_RECBOLE_COMMIT = "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
EXPECTED_PROVIDER_ENDPOINT_DIGEST = "810326f35f8f2efce5f4ca6f73d8df3300f60b4a7f3367fdab3ad73ec3363fcc"
EXPECTED_SEARCH_MANIFEST_DIGEST = "99213591aa3344b023e2d07f99f9f970fcdf49e80ef0122db29e89660d8fdf98"


class Q5ADeploymentError(RuntimeError):
    """A declared deployment dependency is absent or cannot be resolved."""


ENTRYPOINTS = (
    "scripts/run_q5a_idea_feasibility.py",
    "scripts/run_multiround_soak_stage.py",
    "scripts/run_prospective_policy_comparison.py",
)
REQUIRED_STAGE_SYMBOLS = (
    "run_preflight",
    "run_selected_resolver",
    "run_implementer",
    "run_materialize_qualifier",
    "run_resource_admission",
    "run_mechanism_probe",
    "run_matched_execution",
    "run_full_execution",
    "run_episode",
)
REQUIRED_RUNTIME_SYMBOLS = (
    ("recclaw_core.experiments.helix_abc_v1.fresh_r1", "bounded_provider_call"),
    ("recclaw_core.experiments.helix_abc_v1.fresh_r1", "run_development_training"),
    ("recclaw_core.experiments.helix_abc_v1.fresh_r1", "_materialize_and_qualify"),
    ("recclaw_core.experiments.helix_abc_v1.idea_quality", "_q1_unit_check"),
    ("recclaw_core.experiments.helix_abc_v1.idea_quality", "build_research_context"),
    ("recclaw_core.experiments.helix_abc_v1.fresh_r2", "load_registered_r1_artifacts"),
    ("recclaw_core.experiments.helix_abc_v1.fresh_r2", "build_active_r2_profile"),
)


def _file_record(path: Path, *, role: str, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise Q5ADeploymentError(f"missing file for {role}: {path}")
    relative = str(path.relative_to(relative_to)) if relative_to is not None else None
    return {"role": role, "path": str(path.resolve()), "relative_path": relative, "sha256": bytes_sha256(path.read_bytes())}


def _directory_files(path: Path, *, role: str, relative_to: Path | None = None) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise Q5ADeploymentError(f"missing directory for {role}: {path}")
    return [
        _file_record(item, role=role, relative_to=relative_to)
        for item in sorted(path.rglob("*"))
        if item.is_file() and "__pycache__" not in item.parts
    ]


def _module_path(module_name: str, *, src_root: Path, scripts_root: Path) -> Path | None:
    if module_name.startswith("recclaw_core"):
        base = src_root / Path(*module_name.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
    elif module_name == "recclaw_runtime_binding":
        candidates = (src_root / "recclaw_runtime_binding.py",)
    elif module_name.startswith("run_"):
        candidates = (scripts_root / f"{module_name}.py",)
    else:
        return None
    return next((path for path in candidates if path.is_file()), None)


def _relative_module_path(path: Path, *, src_root: Path, scripts_root: Path, node: ast.ImportFrom) -> Path | None:
    if node.level:
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        module = node.module or ""
        candidate = base / Path(*module.split(".")) if module else base
        paths = (candidate.with_suffix(".py"), candidate / "__init__.py")
        return next((value for value in paths if value.is_file()), None)
    return _module_path(node.module or "", src_root=src_root, scripts_root=scripts_root)


def _static_import_graph(repo_root: Path) -> dict[str, Any]:
    src_root = repo_root / "src"
    scripts_root = repo_root / "scripts"
    queue = [repo_root / entrypoint for entrypoint in ENTRYPOINTS]
    visited: set[Path] = set()
    edges: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if not path.is_file():
            unresolved.append({"from": str(path), "import": "entrypoint"})
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            raise Q5ADeploymentError(f"syntax error in static import graph: {path}: {error}") from error
        for node in ast.walk(tree):
            target = None
            import_name = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    target = _module_path(import_name, src_root=src_root, scripts_root=scripts_root)
                    if target is not None:
                        break
            elif isinstance(node, ast.ImportFrom):
                import_name = ("." * node.level) + (node.module or "")
                target = _relative_module_path(path, src_root=src_root, scripts_root=scripts_root, node=node)
            if target is not None:
                edges.append({"from": str(path.relative_to(repo_root)), "to": str(target.relative_to(repo_root)), "import": import_name})
                queue.append(target)
            elif isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("recclaw_core")):
                unresolved.append({"from": str(path.relative_to(repo_root)), "import": import_name})
    return {
        "entrypoints": ENTRYPOINTS,
        "files": tuple(sorted(str(path.relative_to(repo_root)) for path in visited)),
        "edges": tuple(sorted(edges, key=lambda value: (value["from"], value["to"], value["import"]))),
        "unresolved_internal_imports": tuple(sorted(unresolved, key=lambda value: (value["from"], value["import"]))),
    }


def _git_head(path: Path) -> str | None:
    head = path / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = path / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else value
    return value


def build_q5a_deployment_manifest(
    *,
    repo_root: Path,
    projects_root: Path,
    search_data_root: Path,
    recbole_root: Path,
    python_executable: Path,
    api_config: Path,
    prefreeze_manifest: Path | None = None,
    preflight_root: Path | None = None,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    """Inventory every known Q5-A source and runtime dependency without calls."""

    repo_root = repo_root.resolve()
    if prefreeze_manifest is None:
        prefreeze_manifest = repo_root / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
    prefreeze_manifest = prefreeze_manifest.resolve()
    if not prefreeze_manifest.is_file():
        raise Q5ADeploymentError(f"missing prefreeze manifest: {prefreeze_manifest}")
    prefreeze = _load(prefreeze_manifest)
    observed_prefreeze_digest = prefreeze.get("prefreeze_digest")
    prefreeze_payload = dict(prefreeze)
    prefreeze_payload.pop("prefreeze_digest", None)
    if not isinstance(observed_prefreeze_digest, str) or observed_prefreeze_digest != sha256_digest(prefreeze_payload):
        raise Q5ADeploymentError("prefreeze manifest self-digest mismatch")
    for field in ("source_tree_digest", "foundation_package_digest"):
        if not isinstance(prefreeze.get(field), str) or len(prefreeze[field]) != 64:
            raise Q5ADeploymentError(f"prefreeze manifest lacks {field}")
    foundation_commit = prefreeze.get("foundation_commit")
    if not isinstance(foundation_commit, str) or len(foundation_commit) != 40:
        raise Q5ADeploymentError("prefreeze foundation commit is invalid")
    preflight_root = (preflight_root or repo_root / "results/research_line/q5a_runtime_binding_preflight").resolve()
    campaign_root = (campaign_root or repo_root / "results/research_line/q5a_runtime_binding_campaign").resolve()
    source_graph = _static_import_graph(repo_root)
    q4_fixture = repo_root / "results/research_line/q4_prospective_policy_comparison_20260803_01/shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
    policy_root = repo_root / "results/research_line/q4_prospective_policy_comparison_20260803_01/policy_bindings"
    accepted_f1_policy = policy_root / "current_f1/policy/versioned_policy.json"
    accepted_outcome_policy = policy_root / "outcome_aware/versioned_policy.json"
    accepted_outcome_activation = policy_root / "outcome_aware/active_policy.json"
    accepted_r1_receipt = repo_root / "docs/research_line/vnext/R1_FRESH_TRAINING_FILESYSTEM_FIX_V3_CANONICAL_RECEIPT.json"
    accepted_profile = repo_root / "docs/research_line/vnext/WAVE2_INTEGRATED_GATE_RECEIPT.json"
    search_manifest = search_data_root / "search_partition_manifest.json"
    required_files = [
        _file_record(repo_root / entrypoint, role="runner_entrypoint", relative_to=repo_root)
        for entrypoint in ENTRYPOINTS
    ]
    required_files.extend(
        [
            _file_record(repo_root / "src/recclaw_runtime_binding.py", role="runtime_binding_bootstrap", relative_to=repo_root),
            _file_record(q4_fixture, role="accepted_outcome_blind_fixture", relative_to=repo_root),
            _file_record(accepted_f1_policy, role="accepted_current_f1_policy", relative_to=repo_root),
            _file_record(accepted_outcome_policy, role="accepted_outcome_aware_policy", relative_to=repo_root),
            _file_record(accepted_outcome_activation, role="accepted_outcome_aware_activation", relative_to=repo_root),
            _file_record(accepted_r1_receipt, role="accepted_r1_repository_receipt", relative_to=repo_root),
            _file_record(accepted_profile, role="accepted_active_profile_source", relative_to=repo_root),
            _file_record(search_manifest, role="search_partition_manifest"),
            _file_record(api_config, role="provider_config_reference"),
            _file_record(recbole_root / "recbole/model/general_recommender/bpr.py", role="recbole_baseline_entrypoint"),
            _file_record(prefreeze_manifest, role="prefreeze_manifest", relative_to=repo_root if repo_root in prefreeze_manifest.parents else None),
        ]
    )
    accepted_external_root = projects_root / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
    required_files.append(_file_record(accepted_external_root / "R1_CANONICAL_RECEIPT.json", role="accepted_r1_external_receipt"))
    test_files = _directory_files(repo_root / "tests/experiments/helix_abc_v1", role="q5a_contract_tests", relative_to=repo_root)
    provider_config = next(record for record in required_files if record["role"] == "provider_config_reference")
    search_record = next(record for record in required_files if record["role"] == "search_partition_manifest")
    external_receipt = required_files[-1]
    payload = canonical_value(
        {
            "schema": Q5A_DEPLOYMENT_SCHEMA,
            "entrypoints": ENTRYPOINTS,
            "static_import_graph": source_graph,
            "source_package": {
                "src_root": str((repo_root / "src").resolve()),
                "recclaw_core_files": _directory_files(repo_root / "src/recclaw_core", role="recclaw_core_source", relative_to=repo_root),
                "root_recclaw_ext_files": _directory_files(repo_root / "recclaw_ext", role="root_recclaw_ext", relative_to=repo_root),
                "configs_files": _directory_files(repo_root / "configs", role="configs", relative_to=repo_root),
            },
            "tests_and_fixtures": {"contract_test_files": test_files, "accepted_fixture": str(q4_fixture.resolve())},
            "accepted_external_roots": {
                "projects_root": str(projects_root.resolve()),
                "r1_external_root": str(accepted_external_root.resolve()),
                "r1_external_receipt_sha256": external_receipt["sha256"],
            },
            "provider": {
                "model": "gpt-5.4",
                "endpoint_digest": EXPECTED_PROVIDER_ENDPOINT_DIGEST,
                "config_reference": provider_config,
                "temperature": 0,
                "proposal_token_ceiling": 16000,
                "implementation_token_ceiling": 20000,
                "maximum_physical_attempts_per_call": 1,
                "retries": 0,
            },
            "runtime": {
                "python_executable": _file_record(python_executable, role="python_runtime"),
                "recbole_root": str(recbole_root.resolve()),
                "recbole_head": _git_head(recbole_root),
                "expected_recbole_commit": EXPECTED_RECBOLE_COMMIT,
                "search_data_root": str(search_data_root.resolve()),
                "search_manifest_digest": search_record["sha256"],
                "expected_search_manifest_digest": EXPECTED_SEARCH_MANIFEST_DIGEST,
            },
            "frozen_inputs": {
                "prefreeze_digest": observed_prefreeze_digest,
                "source_tree_digest": prefreeze["source_tree_digest"],
                "foundation_commit": foundation_commit,
                "foundation_package_digest": prefreeze["foundation_package_digest"],
            },
            "execution_binding": {
                "prefreeze_manifest": next(record for record in required_files if record["role"] == "prefreeze_manifest"),
                "preflight_root": str(preflight_root),
                "campaign_root": str(campaign_root),
                "execution_owner_contract": "ONE_PID_ONE_UNIQUE_CAMPAIGN_ROOT_EXCLUSIVE_OWNER",
            },
            "stage_consumers": {
                "stage_script": str((repo_root / "scripts/run_multiround_soak_stage.py").resolve()),
                "stage_symbols": REQUIRED_STAGE_SYMBOLS,
                "provider_consumer": "recclaw_core.experiments.helix_abc_v1.fresh_r1.bounded_provider_call",
                "implementer_consumer": "recclaw_core.experiments.helix_abc_v1.fresh_r1._materialize_and_qualify",
                "qualifier_consumer": "recclaw_core.experiments.helix_abc_v1.idea_quality._q1_unit_check",
                "resource_consumer": "recclaw_core.experiments.helix_abc_v1.fresh_r1.run_development_training",
                "dynamic_candidate_entrypoint": "candidate_package_relative:recclaw_ext/candidate.py",
                "recbole_baseline_entrypoint": "recbole.model.general_recommender.bpr:BPR",
            },
            "required_files": required_files,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    return {**payload, "deployment_digest": sha256_digest(payload)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Q5ADeploymentError(f"JSON root is not an object: {path}")
    return value


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def run_q5a_comprehensive_preflight(*, manifest_path: Path, repo_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Read-only import/render/path preflight; this function performs no physical call."""

    manifest = _load(manifest_path)
    if manifest.get("schema") != Q5A_DEPLOYMENT_SCHEMA:
        raise Q5ADeploymentError("deployment manifest schema drift")
    repo_root = repo_root.resolve()
    from recclaw_runtime_binding import RuntimeBindingV1

    binding = RuntimeBindingV1.from_manifest(manifest_path, repo_root=repo_root).activate()
    checks: list[dict[str, Any]] = []
    _check(checks, "runtime_binding_activation", all(
        os.environ.get(name) == str(value)
        for name, value in (
            ("RECCLAW_PROJECTS_ROOT", binding.projects_root),
            ("RECCLAW_SEARCH_DATA_ROOT", binding.search_data_root),
            ("RECCLAW_RECBOLE_ROOT", binding.recbole_root),
            ("RECCLAW_API_CONFIG", binding.api_config),
        )
    ), str(binding.binding_digest))
    pre_sensitive_imports = sorted(
        name for name in sys.modules
        if name in {
            "recclaw_core.experiments.helix_abc_v1.fresh_r1",
            "recclaw_core.experiments.helix_abc_v1.fresh_r2",
        }
    )
    _check(checks, "path_sensitive_import_order", not pre_sensitive_imports, str(pre_sensitive_imports))
    for record in manifest["required_files"]:
        path = Path(str(record["path"]))
        if not path.is_file() and record.get("relative_path"):
            path = repo_root / str(record["relative_path"])
        passed = path.is_file()
        if passed and record.get("sha256"):
            passed = bytes_sha256(path.read_bytes()) == record["sha256"]
        _check(checks, f"file:{record['role']}", passed, str(path))
    graph = manifest["static_import_graph"]
    _check(checks, "static_import_graph_unresolved", not graph["unresolved_internal_imports"], str(graph["unresolved_internal_imports"]))
    src_root = repo_root / "src"
    scripts_root = repo_root / "scripts"
    for path in (repo_root, src_root, scripts_root, repo_root / "recclaw_ext", repo_root / "configs"):
        _check(checks, f"directory:{path.name}", path.is_dir(), str(path))
    for module_name in (
        "recclaw_core.experiments.helix_abc_v1.canonical",
        "recclaw_core.experiments.helix_abc_v1.fresh_r1",
        "recclaw_core.experiments.helix_abc_v1.fresh_r2",
        "recclaw_core.experiments.helix_abc_v1.idea_quality",
        "recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison",
        "recclaw_core.experiments.helix_abc_v1.q5a_idea_feasibility",
        "recclaw_core.experiments.helix_abc_v1.q5a_deployment",
    ):
        try:
            importlib.import_module(module_name)
            _check(checks, f"import:{module_name}", True, "imported")
        except Exception as error:
            _check(checks, f"import:{module_name}", False, repr(error))
    try:
        from run_q5a_idea_feasibility import (
            _q5a_outcome_pool_view,
            _validate_outcome_policy_activation_pair,
        )

        policy_paths = {
            str(record["role"]): (
                Path(str(record["path"]))
                if Path(str(record["path"])).is_file()
                else repo_root / str(record["relative_path"])
            )
            for record in manifest["required_files"]
            if record.get("role")
            in {
                "accepted_outcome_aware_policy",
                "accepted_outcome_aware_activation",
            }
        }
        outcome_policy = _load(policy_paths["accepted_outcome_aware_policy"])
        outcome_activation = _load(policy_paths["accepted_outcome_aware_activation"])
        _validate_outcome_policy_activation_pair(outcome_policy, outcome_activation)
        projection_view = _q5a_outcome_pool_view(
            {
                "schema": "recclaw.research-line.q5a-raw-pool.v1",
                "candidate_pools": {"shared": []},
            }
        )
        _check(
            checks,
            "accepted_outcome_policy_activation_pair",
            outcome_activation.get("status") == "ACTIVE_DEVELOPMENT_ONLY"
            and outcome_activation.get("policy_digest")
            == outcome_policy.get("policy_digest"),
            str(outcome_activation.get("policy_digest")),
        )
        _check(
            checks,
            "q5a_q3_selection_rule_adapter",
            projection_view.get("selection_rule")
            == "NONE_POOL_ONLY_POLICIES_SELECT_AFTER_BYTE_FREEZE",
            str(projection_view.get("selection_rule")),
        )
    except Exception as error:
        _check(checks, "accepted_outcome_policy_activation_pair", False, repr(error))
        _check(checks, "q5a_q3_selection_rule_adapter", False, repr(error))
    try:
        stage_path = scripts_root / "run_multiround_soak_stage.py"
        spec = importlib.util.spec_from_file_location("q5a_stage_preflight", stage_path)
        if spec is None or spec.loader is None:
            raise ImportError("stage module spec unavailable")
        stage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage_module)
        missing = [name for name in REQUIRED_STAGE_SYMBOLS if not hasattr(stage_module, name)]
        _check(checks, "stage_consumer_symbols", not missing, str(missing))
    except Exception as error:
        _check(checks, "stage_consumer_symbols", False, repr(error))
    for module_name, symbol in REQUIRED_RUNTIME_SYMBOLS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol)
            _check(checks, f"symbol:{module_name}:{symbol}", True, "resolved")
        except Exception as error:
            _check(checks, f"symbol:{module_name}:{symbol}", False, repr(error))
    try:
        from .fresh_r2 import build_active_r2_profile, build_r1_registry, load_registered_r1_artifacts, public_active_profile_catalog
        from .idea_quality import build_research_context
        context = build_research_context(repo_root)
        artifacts, _receipt = load_registered_r1_artifacts(repo_root)
        registry = build_r1_registry(artifacts)
        _current, _build, _next, _build_receipt, active = build_active_r2_profile(registry)
        catalog = public_active_profile_catalog(active, artifacts, seed=57000)
        _check(checks, "accepted_profile_registry_context", bool(context and catalog and active.profile_digest), "resolved")
    except Exception as error:
        _check(checks, "accepted_profile_registry_context", False, repr(error))
    try:
        from . import fresh_r1, fresh_r2

        exact = (
            fresh_r1.PROJECTS_ROOT == binding.projects_root
            and fresh_r1.SEARCH_DATA_ROOT == binding.search_data_root
            and fresh_r1.RECBole_ROOT == binding.recbole_root
            and fresh_r2.PROJECTS_ROOT == binding.projects_root
            and fresh_r2.R1_EXTERNAL_ROOT == binding.projects_root / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
        )
        _check(checks, "import_time_runtime_paths_exact", exact, f"fresh_r1.PROJECTS_ROOT={fresh_r1.PROJECTS_ROOT}; fresh_r2.R1_EXTERNAL_ROOT={fresh_r2.R1_EXTERNAL_ROOT}")
    except Exception as error:
        _check(checks, "import_time_runtime_paths_exact", False, repr(error))
    try:
        from .idea_quality import build_research_context, derive_enriched_proposal_schema
        from .fresh_r2 import build_active_r2_profile, build_r1_registry, load_registered_r1_artifacts, public_active_profile_catalog, _r2_bindings, _r2_environment
        from .q5a_idea_feasibility import Q5A_POLICIES
        from run_q5a_idea_feasibility import _render_q5_prompt
        context = build_research_context(repo_root)
        artifacts, _receipt = load_registered_r1_artifacts(repo_root)
        registry = build_r1_registry(artifacts)
        _current, _build, _next, _build_receipt, active = build_active_r2_profile(registry)
        prompt_template = repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources/idea_quality_producer_prompt_v1.txt"
        prompt = _render_q5_prompt(prompt_template.read_text(encoding="utf-8"), slot="preflight-slot", role="mechanism_composer", mode="FRONTIER_HYPOTHESIS", seed=7100101, context=context, catalog=public_active_profile_catalog(active, artifacts, seed=57000), active=active, context_digest=sha256_digest(context))
        _check(checks, "provider_request_rendering_no_call", bool(prompt and Q5A_POLICIES), "rendered without provider call")
        schema = derive_enriched_proposal_schema()
        _check(checks, "provider_schema_read", isinstance(schema, dict), "schema loaded")
    except Exception as error:
        _check(checks, "provider_request_rendering_no_call", False, repr(error))
    try:
        fixture = _load(repo_root / "results/research_line/q4_prospective_policy_comparison_20260803_01/shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json")
        row = fixture["candidate_pools"]["shared"][0]
        stage_path = scripts_root / "run_multiround_soak_stage.py"
        spec = importlib.util.spec_from_file_location("q5a_stage_prompt_preflight", stage_path)
        assert spec and spec.loader
        stage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage_module)
        research_spec = stage_module._rehydrate_spec(row)
        from recclaw_core.experiments.helix_abc_v1.fresh_r1 import _shared_policy, render_implementation_prompt
        from recclaw_core.experiments.helix_abc_v1.innovation_spine import build_shared_implementer_request
        template_path = repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources/idea_quality_implementer_prompt_v1.txt"
        tool_policy_path = repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources/fresh_open_spec_tool_policy_v1.json"
        request = build_shared_implementer_request(research_spec, policy=_shared_policy(bytes_sha256(template_path.read_bytes()), bytes_sha256(tool_policy_path.read_bytes())))
        rendered = render_implementation_prompt(template_path.read_text(encoding="utf-8"), request)
        _check(checks, "implementer_prompt_rendering_no_call", bool(rendered), "rendered accepted fixture")
        entrypoint = importlib.util.find_spec("recbole.model.general_recommender.bpr")
        _check(checks, "resource_recbole_entrypoint_resolution", entrypoint is not None, str(entrypoint))
    except Exception as error:
        _check(checks, "implementer_prompt_rendering_no_call", False, repr(error))
    passed = all(check["status"] == "PASS" for check in checks)
    result = canonical_value(
        {
            "schema": Q5A_PREFLIGHT_SCHEMA,
            "deployment_manifest_digest": manifest.get("deployment_digest"),
            "deployment_digest": binding.deployment_digest,
            "binding_digest": binding.binding_digest,
            "prefreeze_digest": binding.prefreeze_digest,
            "source_tree_digest": binding.source_tree_digest,
            "foundation_commit": binding.foundation_commit,
            "foundation_package_digest": binding.foundation_package_digest,
            "preflight_root": str(binding.preflight_root),
            "campaign_root": str(binding.campaign_root),
            "status": "PASS" if passed else "HARD_BLOCK",
            "checks": checks,
            "provider_calls": 0,
            "implementer_calls": 0,
            "qualification_calls": 0,
            "training_runs": 0,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(json.dumps({**result, "preflight_digest": sha256_digest(result)}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {**result, "preflight_digest": sha256_digest(result)}


__all__ = [
    "EXPECTED_PROVIDER_ENDPOINT_DIGEST",
    "EXPECTED_RECBOLE_COMMIT",
    "EXPECTED_SEARCH_MANIFEST_DIGEST",
    "Q5A_DEPLOYMENT_SCHEMA",
    "Q5A_PREFLIGHT_SCHEMA",
    "Q5ADeploymentError",
    "build_q5a_deployment_manifest",
    "run_q5a_comprehensive_preflight",
]
