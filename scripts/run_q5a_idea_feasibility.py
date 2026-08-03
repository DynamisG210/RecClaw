#!/usr/bin/env python3
"""Run the real Q5-A idea/feasibility pilot in frozen serial stages."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recclaw_runtime_binding import (  # noqa: E402
    RuntimeBindingError,
    RuntimeBindingV1,
    claim_execution_owner,
    read_signed_execution_gate,
    read_signed_stage_release,
    release_execution_owner,
    read_verified_execution_gate,
    validate_realize_prerequisites,
    verify_execution_gate,
    verify_execution_owner,
    write_execution_gate,
)


_RUNTIME_CHILD = os.environ.get("RECCLAW_RUNTIME_CHILD") == "1"
_RUNTIME_BINDING: RuntimeBindingV1 | None = None
_EXECUTION_OWNER: dict[str, Any] | None = None
_CHILD_STATUS = "FAILED"


def _argv_value(name: str) -> str | None:
    for index, value in enumerate(sys.argv[:-1]):
        if value == name:
            return sys.argv[index + 1]
    return None


def _bootstrap_execution_binding() -> None:
    """Gate and bind before the first path-sensitive import, then exec once."""

    global _RUNTIME_BINDING, _EXECUTION_OWNER
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command not in {"pool", "realize"}:
        return
    manifest_value = _argv_value("--deployment-manifest")
    preflight_value = _argv_value("--preflight-receipt")
    prefreeze_value = _argv_value("--prefreeze-manifest")
    output_value = _argv_value("--output-root")
    if not all((manifest_value, preflight_value, prefreeze_value, output_value)):
        raise RuntimeBindingError("physical campaign requires manifest, PASS receipt, prefreeze, and output root")
    repo_value = _argv_value("--repo-root")
    repo_root = Path(repo_value).resolve() if repo_value else ROOT
    manifest_path = Path(manifest_value).resolve()
    preflight_path = Path(preflight_value).resolve()
    prefreeze_path = Path(prefreeze_value).resolve()
    output_root = Path(output_value).resolve()
    binding = RuntimeBindingV1.from_manifest(manifest_path, repo_root=repo_root).activate()
    if output_root != binding.campaign_root:
        raise RuntimeBindingError("campaign output root does not match manifest campaign_root")
    gate_path = output_root / "EXECUTION_GATE.json"
    if gate_path.exists():
        gate = read_verified_execution_gate(gate_path, binding)
    else:
        gate = verify_execution_gate(
            binding=binding,
            preflight_receipt_path=preflight_path,
            prefreeze_manifest_path=prefreeze_path,
        )
        write_execution_gate(gate_path, gate)
    owner = claim_execution_owner(binding, gate_receipt_path=gate_path, stage=command.upper())
    try:
        if command == "realize":
            validate_realize_prerequisites(binding)
    except Exception:
        release_execution_owner(binding, owner, status="FAILED")
        raise
    os.environ.update(
        {
            "RECCLAW_RUNTIME_CHILD": "1",
            "RECCLAW_BINDING_MANIFEST": str(manifest_path),
            "RECCLAW_PREFLIGHT_RECEIPT": str(preflight_path),
            "RECCLAW_PREFREEZE_MANIFEST": str(prefreeze_path),
            "RECCLAW_EXECUTION_GATE": str(gate_path),
            "RECCLAW_EXECUTION_OWNER_TOKEN": str(owner["owner_token"]),
            "RECCLAW_EXECUTION_GATE_DIGEST": str(gate["gate_digest"]),
        }
    )
    _RUNTIME_BINDING = binding
    _EXECUTION_OWNER = owner
    executable = str(binding.python_executable)
    try:
        os.execve(executable, [executable, str(Path(__file__).resolve()), *sys.argv[1:]], os.environ.copy())
    except BaseException:
        release_execution_owner(binding, owner, status="EXEC_FAILED")
        raise


def _activate_child_binding() -> None:
    global _RUNTIME_BINDING, _EXECUTION_OWNER
    if not _RUNTIME_CHILD:
        return
    manifest_value = os.environ.get("RECCLAW_BINDING_MANIFEST")
    if not manifest_value:
        raise RuntimeBindingError("runtime child is missing binding manifest")
    try:
        _RUNTIME_BINDING = RuntimeBindingV1.from_manifest(Path(manifest_value), repo_root=ROOT).activate()
        gate_path = Path(os.environ["RECCLAW_EXECUTION_GATE"])
        gate = read_verified_execution_gate(
            Path(os.environ["RECCLAW_EXECUTION_GATE"]), _RUNTIME_BINDING
        )
        if gate["gate_digest"] != os.environ.get("RECCLAW_EXECUTION_GATE_DIGEST"):
            raise RuntimeBindingError("execution gate digest drift in child")
        _EXECUTION_OWNER = verify_execution_owner(
            _RUNTIME_BINDING,
            owner_token=os.environ.get("RECCLAW_EXECUTION_OWNER_TOKEN"),
        )
        if len(sys.argv) > 1 and sys.argv[1] == "realize":
            validate_realize_prerequisites(_RUNTIME_BINDING)
        if gate_path.resolve() != _RUNTIME_BINDING.campaign_root / "EXECUTION_GATE.json":
            raise RuntimeBindingError("execution gate root drift in child")
    except BaseException:
        if _RUNTIME_BINDING is not None and _RUNTIME_BINDING.campaign_root.joinpath("EXECUTION_OWNER.json").is_file():
            try:
                observed = json.loads(_RUNTIME_BINDING.campaign_root.joinpath("EXECUTION_OWNER.json").read_text(encoding="utf-8"))
                if observed.get("owner_pid") == os.getpid() and observed.get("owner_token") == os.environ.get("RECCLAW_EXECUTION_OWNER_TOKEN"):
                    release_execution_owner(_RUNTIME_BINDING, observed, status="CHILD_BOOTSTRAP_FAILED")
            except Exception:
                pass
        raise


def _release_owned_at_exit() -> None:
    if _RUNTIME_CHILD and _RUNTIME_BINDING is not None and _EXECUTION_OWNER is not None:
        owner_path = _RUNTIME_BINDING.campaign_root / "EXECUTION_OWNER.json"
        if owner_path.is_file():
            try:
                release_execution_owner(_RUNTIME_BINDING, _EXECUTION_OWNER, status="ABNORMAL_EXIT")
            except Exception:
                pass


if _RUNTIME_CHILD:
    _activate_child_binding()
    atexit.register(_release_owned_at_exit)
else:
    _bootstrap_execution_binding()

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (  # noqa: E402
    AVAILABLE_DEPENDENCIES,
    BUDGET_LIMITS,
    MODEL,
    PROTOCOL_REQUIREMENTS,
    ROLE_INSTRUCTIONS,
    bounded_provider_call,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    _r2_bindings,
    _r2_environment,
    build_active_r2_profile,
    build_r1_registry,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from recclaw_core.experiments.helix_abc_v1.idea_quality import (  # noqa: E402
    Q1_CONTEXT_REF,
    Q1_PROPOSAL_TOKEN_CEILING,
    build_research_context,
    derive_enriched_proposal_schema,
    score_preoutcome_testability,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (  # noqa: E402
    FULL_DEVELOPMENT_SEEDS,
    FULL_EPOCHS,
    MAX_REPAIR_TURNS,
    RECBole_INTERFACE_CONTRACT,
    SCREEN_EPOCHS,
    build_conversion_execution_plan,
    choose_stable_promotions,
    finalize_conversion_execution_plan,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_f1 import (  # noqa: E402
    ALLOWED_DIRECTIONS,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (  # noqa: E402
    project_open_producer_draft,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison import (  # noqa: E402
    ProspectivePolicyComparisonError,
    build_shared_realization_contract,
    REALIZATION_EQUIVALENCE_TOLERANCE,
    classify_realization_authority,
    select_outcome_aware,
)
from recclaw_core.experiments.helix_abc_v1.q5a_idea_feasibility import (  # noqa: E402
    Q5A_POOL_COUNT,
    Q5A_POOL_SIZE,
    Q5A_POLICIES,
    Q5A_PREFREEZE_SCHEMA,
    Q5A_POLICY_SELECTION_BUDGET,
    Q5A_STAGES,
    Q5AIdeaFeasibilityError,
    build_q5a_pool_manifest,
    build_q5a_prefreeze_manifest,
    build_q5a_realization_union,
    build_q5a_selection_manifest,
    build_q5a_shared_exploration,
    build_q5a_stage_denominator,
)
from recclaw_core.experiments.helix_abc_v1.v4_response_contract import (  # noqa: E402
    validate_v4_response_contract,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    CapabilityResolutionResultV1,
)

from run_prospective_policy_comparison import (  # noqa: E402
    COMMON_EXECUTION,
    F1_POLICY_DIGEST,
    OUTCOME_POLICY_DIGEST,
    _shared_contract_instruction,
    _render_shared_prompt,
    _provider_usage,
    read_object,
)


CAMPAIGN_ID = os.environ.get("RECCLAW_Q5A_CAMPAIGN_ID", "q5a-idea-feasibility-20260803-01")
FOUNDATION_COMMIT = "885b81ed0b67ec0cc551a589d6d1e3a852b73c43"
FOUNDATION_PACKAGE_DIGEST = (
    "014bb27987086401de92092164fe715a5dba96f345fc80dc3fb51751dcc8de14"
)
POOL_SEEDS = (71001, 71002, 71003)
EXPLORATION_SEEDS = (72001, 72002, 72003)
CONVERSION_SCREEN_SEED = 54303
CONVERSION_PROMOTION_LIMIT = 4
SLOT_BLUEPRINT = (
    ("slot-01", "mechanism_composer", "DIAGNOSIS_DRIVEN"),
    ("slot-02", "lineage_refiner", "DIAGNOSIS_DRIVEN"),
    ("slot-03", "falsification_designer", "DIAGNOSIS_DRIVEN"),
    ("slot-04", "frontier_architect", "DIAGNOSIS_DRIVEN"),
    ("slot-05", "mechanism_composer", "FRONTIER_HYPOTHESIS"),
    ("slot-06", "lineage_refiner", "FRONTIER_HYPOTHESIS"),
    ("slot-07", "falsification_designer", "FRONTIER_HYPOTHESIS"),
    ("slot-08", "frontier_architect", "FRONTIER_HYPOTHESIS"),
)

Q5A_AGGREGATE_CONTEXT = {
    "source": "Q5A_DEVELOPMENT_ONLY_AGGREGATE",
    "provider_denominator": 24,
    "construct_count": 17,
    "qualified_count": 10,
    "resource_admitted_count": 8,
    "full_episode_count": 2,
    "timeout_cluster": {
        "count": 6,
        "frozen_training_deadline_seconds": 900,
        "observed_elapsed_seconds": (901, 902),
    },
    "development_signals": {
        "complete_episode_count": 2,
        "signal_summary": "two near-zero negative development signals; no scientific effect claim",
    },
    "candidate_specific_results": False,
    "held_out_reads": 0,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Q5AIdeaFeasibilityError(f"JSON root is not an object: {path}")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        data = canonical_json_bytes(canonical_value(value))
        handle.write(data)
    return bytes_sha256(data)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _q5_contract_instruction(mode: str) -> str:
    realization = (
        "Use realization_mode PARENT_PRESERVING and make mechanism_off_definition an executable switch. "
        if mode == "DIAGNOSIS_DRIVEN"
        else "Use realization_mode NON_NESTED and state the matched parent/control explicitly. "
    )
    return (
        _shared_contract_instruction(mode)
        + " Quote closest_parent from the supplied executable profile/catalog parent references; a natural-language hypothesis is not a parent. "
        + realization
        + " Make minimal_testable_wedge concrete: name the tensor/data flow, loss or scoring path, off implementation, and rough memory/time complexity. "
        + " Give a minimal credible resource estimate instead of copying the maximum budget. "
    )


def _pool_mechanism_signatures(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "closest_parent": str(row.get("closest_parent", "")),
            "mechanism_change": str(row.get("mechanism_change", "")),
            "causal_operator": str(row.get("causal_chain", "")),
        }
        for row in rows
    ]


def _render_q5_prompt(
    template: str,
    *,
    slot: str,
    role: str,
    mode: str,
    seed: int,
    context: Mapping[str, Any],
    catalog: list[dict[str, str]],
    active: Any,
    context_digest: str,
    pool_signatures: Sequence[Mapping[str, Any]] = (),
) -> str:
    role_instruction = ROLE_INSTRUCTIONS[role] + (
        " Prefer a single parent-preserving operator change on one executable parent."
        if mode == "DIAGNOSIS_DRIVEN"
        else " Keep the frontier mechanism genuinely non-nested and distinguish its causal operator from prior pool signatures."
    ) + (
        " Copy compatibility_requirements only from the frozen protocol list: "
        + ", ".join(PROTOCOL_REQUIREMENTS)
        + ". Copy required_dependencies only from the frozen dependency list: "
        + ", ".join(AVAILABLE_DEPENDENCIES)
        + ". Estimate the minimum credible implementation tokens, GPU minutes, and wall minutes; do not copy the ceiling: "
        + json.dumps(BUDGET_LIMITS, sort_keys=True)
        + "."
    )
    replacements = {
        "{{LOGICAL_SLOT_ID}}": slot,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": role,
        "{{CONTRACT_INSTRUCTION}}": _q5_contract_instruction(mode),
        "{{ROLE_INSTRUCTION}}": role_instruction,
        "{{POOL_MECHANISM_SIGNATURES}}": json.dumps(
            list(pool_signatures), sort_keys=True, separators=(",", ":")
        ),
        "{{RESEARCH_CONTEXT_JSON}}": json.dumps(context, sort_keys=True, separators=(",", ":")),
        "{{PROFILE_CATALOG_JSON}}": json.dumps(catalog, sort_keys=True, separators=(",", ":")),
        "{{PROTOCOL_REF}}": active.protocol_ref,
        "{{PROTOCOL_DIGEST}}": active.protocol_digest,
        "{{CONTEXT_REF}}": Q1_CONTEXT_REF,
        "{{CONTEXT_DIGEST}}": context_digest,
        "{{PROFILE_REF}}": active.profile_ref,
        "{{PROFILE_DIGEST}}": active.profile_digest,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", rendered):
        raise ProspectivePolicyComparisonError("Q5-A Provider prompt has unresolved placeholders")
    lowered = rendered.lower()
    for forbidden in ("expected winner", "candidate_ndcg", "observed_ndcg", "qualification result"):
        if forbidden in lowered:
            raise ProspectivePolicyComparisonError("Q5-A Provider prompt leaked an outcome")
    return rendered


def _set_runtime_environment(args: argparse.Namespace) -> None:
    if _RUNTIME_BINDING is None:
        raise RuntimeBindingError("campaign runtime binding is not active")
    expected = {
        "projects_root": _RUNTIME_BINDING.projects_root,
        "search_data_root": _RUNTIME_BINDING.search_data_root,
        "recbole_root": _RUNTIME_BINDING.recbole_root,
        "python_executable": _RUNTIME_BINDING.python_executable,
        "api_config": _RUNTIME_BINDING.api_config,
    }
    for name, value in expected.items():
        if Path(str(getattr(args, name))).resolve() != value:
            raise RuntimeBindingError(f"campaign runtime argument drift: {name}")


def _require_prefreeze(output_root: Path) -> dict[str, Any]:
    path = output_root / "PREFREEZE_MANIFEST.json"
    if not path.is_file():
        raise Q5AIdeaFeasibilityError("Q5-A prefreeze manifest is missing")
    value = _read(path)
    if value.get("schema") != Q5A_PREFREEZE_SCHEMA:
        raise Q5AIdeaFeasibilityError("Q5-A prefreeze schema drift")
    return value


def prefreeze(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise Q5AIdeaFeasibilityError(f"output root already exists: {output_root}")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", FOUNDATION_COMMIT, head],
        check=False,
    ).returncode != 0:
        raise Q5AIdeaFeasibilityError("Q5-A runner is not a child of the accepted foundation")
    if _git("status", "--porcelain"):
        raise Q5AIdeaFeasibilityError("Q5-A prefreeze source worktree is not clean")
    tree = _git("rev-parse", "HEAD^{tree}")
    source_tree_digest = sha256_digest({"commit": head, "git_tree": tree})
    manifest = build_q5a_prefreeze_manifest(
        campaign_id=CAMPAIGN_ID,
        foundation_commit=FOUNDATION_COMMIT,
        foundation_package_digest=FOUNDATION_PACKAGE_DIGEST,
        source_tree_digest=source_tree_digest,
        common_execution={
            **COMMON_EXECUTION,
            "provider_temperature": 0,
            "implementer_temperature": 0,
            "provider_schema_consumer": "fresh_r1.bounded_provider_call",
        },
        pool_seeds=POOL_SEEDS,
    )
    output_root.mkdir(parents=True)
    _write_new(output_root / "PREFREEZE_MANIFEST.json", manifest)
    _write_new(
        output_root / "POOL_GENERATION_PLAN.json",
        {
            "schema": "recclaw.research-line.q5a-pool-generation-plan.v1",
            "campaign_id": CAMPAIGN_ID,
            "slot_blueprint": SLOT_BLUEPRINT,
            "pool_seeds": POOL_SEEDS,
            "provider_calls": Q5A_POOL_COUNT * Q5A_POOL_SIZE,
            "retries": 0,
            "replacement": False,
            "success_filtering": False,
            "held_out_reads": 0,
            "development_only": True,
        },
    )
    print(json.dumps({"status": "PREFROZEN", "output_root": str(output_root)}, sort_keys=True))


def pool(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    prefreeze_manifest = _require_prefreeze(output_root)
    _set_runtime_environment(args)
    context = canonical_value(
        {
            **build_research_context(args.repo_root.resolve()),
            "q5a_aggregate_facts": Q5A_AGGREGATE_CONTEXT,
        }
    )
    artifacts, _receipt = load_registered_r1_artifacts(args.repo_root.resolve())
    registry = build_r1_registry(artifacts)
    _current, _build, _next, _build_receipt, active = build_active_r2_profile(registry)
    catalog = public_active_profile_catalog(active, artifacts, seed=57000)
    bindings = canonical_value(
        {**_r2_bindings(active), "context_ref": Q1_CONTEXT_REF, "context_digest": sha256_digest(context)}
    )
    environment = _r2_environment(active)
    resource_root = args.repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources"
    template_path = resource_root / "idea_quality_producer_prompt_v1.txt"
    schema = derive_enriched_proposal_schema()
    started = []
    for pool_index, pool_seed in enumerate(POOL_SEEDS, 1):
        pool_root = output_root / "pools" / f"{pool_index:02d}"
        pool_root.mkdir(parents=True)
        schema_path = pool_root / "shared.schema.json"
        _write_new(schema_path, schema)
        _write_new(pool_root / "RESEARCH_CONTEXT.json", context)
        template = template_path.read_text(encoding="utf-8")
        rows: list[dict[str, Any]] = []
        pool_signatures: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        attempts: list[list[Mapping[str, Any]]] = []
        provider_attempt_ledger: list[dict[str, Any]] = []
        for slot_offset, (slot, role, mode) in enumerate(SLOT_BLUEPRINT, 1):
            seed = pool_seed * 100 + slot_offset
            if role not in ALLOWED_DIRECTIONS:
                raise Q5AIdeaFeasibilityError("Q5-A slot direction is outside F1 authority")
            ledger_recorded = False
            try:
                prompt = _render_q5_prompt(
                    template,
                    slot=slot,
                    role=role,
                    mode=mode,
                    seed=seed,
                    context=context,
                    catalog=catalog,
                    active=active,
                    context_digest=sha256_digest(context),
                    pool_signatures=pool_signatures,
                )
                call = bounded_provider_call(
                    call_root=pool_root / "provider" / slot,
                    schema_path=schema_path,
                    logical_call_id=f"{CAMPAIGN_ID}:pool-{pool_index}:{slot}",
                    session_id=f"{CAMPAIGN_ID}:pool-{pool_index}:{slot}:session",
                    prompt=prompt,
                    token_ceiling=Q1_PROPOSAL_TOKEN_CEILING,
                    maximum_physical_attempts=1,
                )
                attempt_rows = list(call.attempts)
                attempts.append(attempt_rows)
                call_root = (pool_root / "provider" / slot).resolve()
                provider_attempt_ledger.append(
                    {
                        "slot": slot,
                        "call_root": str(call_root),
                        "attempt_paths": [
                            str((call_root / f"physical_attempt_{int(row['ordinal']):02d}").resolve())
                            for row in attempt_rows
                        ],
                        "attempt_count": len(attempt_rows),
                        "attempt_record_digest": sha256_digest(attempt_rows),
                    }
                )
                ledger_recorded = True
                if call.call is None:
                    failures.append({"slot": slot, "stage": "PROVIDER", "failure": call.failure})
                    continue
                validate_v4_response_contract(call.call.response, provider_schema=schema)
                draft = call.call.response["proposals"][0]
                if draft.get("producer_role") != role or draft.get("idea_mode") != mode:
                    failures.append({"slot": slot, "stage": "CONTRACT", "failure": "role_or_mode_drift"})
                    continue
                spec, facts = project_open_producer_draft(
                    draft, bindings=bindings, strict_resolution_contract=True
                )
                resolution = resolve_capability(spec, resolution_facts=facts, environment=environment)
                feasible = resolution.resolution in {
                    CapabilityResolutionResultV1.SEARCH_READY,
                    CapabilityResolutionResultV1.INNOVATION_REQUIRED,
                }
                score = score_preoutcome_testability(
                    spec,
                    q0r2_resource_feasible=feasible,
                    structural_context={
                        "parent_catalog": catalog,
                        "pool_signatures": pool_signatures,
                        "required_budget": facts.get("required_budget"),
                        "mode": mode,
                        "role": role,
                    },
                )
                rows.append(
                    canonical_value(
                        {
                            "pool_index": pool_index,
                            "slot": slot,
                            "producer_role": role,
                            "proposal_seed": seed,
                            "provider_attempts": call.attempts,
                            "proposal_response_digest": call.call.response_digest,
                            "research_spec": spec.canonical_dict(),
                            "resolution_facts": facts,
                            "resolution": resolution.canonical_dict(),
                            "preoutcome_score": score,
                            "manual_candidate_patches": 0,
                            "stage": "OPENSPEC_FROZEN",
                            "outcome_fields_consumed": [],
                        }
                    )
                )
                pool_signatures.append(
                    {
                        "closest_parent": str(spec.closest_parent),
                        "mechanism_change": str(spec.mechanism_change),
                        "causal_operator": str(spec.causal_chain),
                    }
                )
            except Exception as error:
                if not ledger_recorded:
                    call_root = (pool_root / "provider" / slot).resolve()
                    observed_attempts = sorted(
                        str(path.resolve())
                        for path in call_root.glob("physical_attempt_*")
                        if path.is_dir()
                    )
                    provider_attempt_ledger.append(
                        {
                            "slot": slot,
                            "call_root": str(call_root),
                            "attempt_paths": observed_attempts,
                            "attempt_count": len(observed_attempts),
                            "attempt_record_digest": None,
                        }
                    )
                failures.append({"slot": slot, "stage": "POOL_SLOT", "failure": repr(error)})
        usage = _provider_usage(attempts)
        raw_pool = canonical_value(
            {
                "schema": "recclaw.research-line.q5a-raw-pool.v1",
                "pool_index": pool_index,
                "pool_seed": pool_seed,
                "candidate_pools": {"shared": rows},
                "candidate_count": len(rows),
                "provider_denominator": Q5A_POOL_SIZE,
                "implementation_or_qualification_outcomes_present_when_written": 0,
                "outcome_fields_consumed": [],
                "held_out_reads": 0,
                "retries": 0,
            }
        )
        raw_path = pool_root / "RAW_POOL_BEFORE_SELECTION.json"
        raw_sha = _write_new(raw_path, raw_pool)
        status = "POOL_COMPLETE" if len(rows) == Q5A_POOL_SIZE and not failures else "POOL_FAILED_MISSING"
        manifest = None
        if status == "POOL_COMPLETE":
            manifest = build_q5a_pool_manifest(
                pool_index=pool_index,
                pool_seed=pool_seed,
                pool=raw_pool,
                provider_usage=usage,
            )
            _write_new(pool_root / "POOL_MANIFEST.json", manifest)
        receipt = {
            "schema": "recclaw.research-line.q5a-pool-receipt.v1",
            "status": status,
            "pool_index": pool_index,
            "pool_seed": pool_seed,
            "raw_pool_file": str(raw_path),
            "raw_pool_file_sha256": raw_sha,
            "candidate_count": len(rows),
            "provider_usage": usage,
            "failures": failures,
            "manual_candidate_patches": 0,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
        receipt_path = pool_root / "POOL_RECEIPT.json"
        receipt["provider_attempt_ledger"] = provider_attempt_ledger
        receipt["provider_attempt_ledger_digest"] = sha256_digest(provider_attempt_ledger)
        receipt_sha = _write_new(receipt_path, receipt)
        started.append(
            {
                "pool_index": pool_index,
                "status": status,
                "candidate_count": len(rows),
                "pool_receipt_file": str(receipt_path.resolve()),
                "pool_receipt_sha256": receipt_sha,
                "provider_attempt_ledger_digest": receipt["provider_attempt_ledger_digest"],
            }
        )
    aggregate = {
        "schema": "recclaw.research-line.q5a-pool-generation-receipt.v1",
        "campaign_id": prefreeze_manifest["campaign_id"],
        "pool_count": Q5A_POOL_COUNT,
        "pool_size": Q5A_POOL_SIZE,
        "pools": started,
        "all_pools_complete": all(row["status"] == "POOL_COMPLETE" for row in started),
        "provider_calls": sum(Q5A_POOL_SIZE for _ in started),
        "provider_attempt_ledger_digest": sha256_digest(
            [
                {
                    "pool_index": row["pool_index"],
                    "provider_attempt_ledger_digest": row["provider_attempt_ledger_digest"],
                }
                for row in started
            ]
        ),
        "retries": 0,
        "replacement": False,
        "held_out_reads": 0,
        "development_only": True,
    }
    _write_new(output_root / "POOL_GENERATION_RECEIPT.json", aggregate)
    print(json.dumps(aggregate, sort_keys=True))


def _ranked_selection(
    pool: Mapping[str, Any],
    policy: str,
    *,
    f1_policy: Mapping[str, Any] | None,
    outcome_policy: Mapping[str, Any] | None,
    outcome_activation: Mapping[str, Any] | None,
    seed: int,
) -> tuple[list[str], list[str], dict[str, str], dict[str, float], float | None, str | None, str, float | None, str]:
    rows = [row for group in sorted(pool["candidate_pools"]) for row in pool["candidate_pools"][group]]
    if policy == "STATIC":
        ordered = sorted(rows, key=lambda row: str(row["preoutcome_score"]["spec_digest"]))
        source_digest = None
        draw = None
        semantics = "ACCEPTED_STATIC_HIGH_CHANGE_IDENTITY_ORDER_TOP_TWO"
        tie_set: list[str] = []
        selected_by_label = "STATIC_HIGH_CHANGE_IDENTITY_ORDER"
        selected = [str(row["preoutcome_score"]["spec_digest"]) for row in ordered[:Q5A_POLICY_SELECTION_BUDGET]]
        probabilities = {
            str(row["preoutcome_score"]["spec_digest"]): (1.0 if str(row["preoutcome_score"]["spec_digest"]) in selected else 0.0)
            for row in rows
        }
        selected_by = {candidate_id: selected_by_label for candidate_id in selected}
        return selected, tie_set, selected_by, probabilities, draw, source_digest, semantics, None, "NONE"
    elif policy == "CURRENT_F1":
        if f1_policy is None:
            raise Q5AIdeaFeasibilityError("F1 policy snapshot is missing")
        rank = {str(value): index for index, value in enumerate(f1_policy["direction_order"])}
        ordered = sorted(
            rows,
            key=lambda row: (rank[str(row["producer_role"])], str(row["preoutcome_score"]["spec_digest"])),
        )
        source_digest = str(f1_policy["policy_digest"])
        draw = None
        semantics = "ACCEPTED_F1_IDEA_DIRECTION_ORDER_THEN_CANDIDATE_ID_TOP_TWO"
        tie_set = []
        selected = [str(row["preoutcome_score"]["spec_digest"]) for row in ordered[:Q5A_POLICY_SELECTION_BUDGET]]
        probabilities = {
            str(row["preoutcome_score"]["spec_digest"]): (1.0 if str(row["preoutcome_score"]["spec_digest"]) in selected else 0.0)
            for row in rows
        }
        selected_by = {candidate_id: "ACCEPTED_F1_DIRECTION_ORDER" for candidate_id in selected}
        return selected, tie_set, selected_by, probabilities, draw, source_digest, semantics, None, "NONE"
    else:
        if outcome_policy is None or outcome_activation is None:
            raise Q5AIdeaFeasibilityError("outcome-aware policy snapshot is missing")
        acquisition = select_outcome_aware(
            _q5a_outcome_pool_view(pool),
            pool_digest=sha256_digest(pool),
            policy=outcome_policy,
            activation=outcome_activation,
            random_seed=seed,
        )
        ordered = list(acquisition["candidates"])
        source_digest = str(outcome_policy["policy_digest"])
        draw = float(acquisition["random_draw"])
        semantics = "ACCEPTED_Q3_SCORE_TOP_TWO_WITH_FROZEN_TIE_RULE"
        tie_tolerance = float(acquisition.get("top_score_tie_tolerance", 1e-12))
        scores = [float(row["selection_score"]) for row in ordered]
        cutoff = scores[Q5A_POLICY_SELECTION_BUDGET - 1]
        higher = [row for row in ordered if float(row["selection_score"]) > cutoff + tie_tolerance]
        boundary = [
            row
            for row in ordered
            if abs(float(row["selection_score"]) - cutoff) <= tie_tolerance
        ]
        slots = Q5A_POLICY_SELECTION_BUDGET - len(higher)
        if slots < 0 or len(boundary) < slots:
            raise Q5AIdeaFeasibilityError("outcome-aware top-two boundary is inconsistent")
        rng = random.Random(seed)
        chosen_boundary = rng.sample(boundary, slots) if len(boundary) > slots else boundary
        selected_rows = higher + chosen_boundary
        selected = [str(row["candidate_id"]) for row in selected_rows]
        tie_set = [str(row["candidate_id"]) for row in boundary] if len(boundary) > slots else []
        probabilities = {}
        selected_by = {}
        boundary_probability = slots / len(boundary) if boundary else 0.0
        for row in ordered:
            candidate_id = str(row["candidate_id"])
            if row in higher:
                probabilities[candidate_id] = 1.0
                selected_by[candidate_id] = "LEARNED_SCORE"
            elif row in boundary:
                probabilities[candidate_id] = boundary_probability
                if candidate_id in selected:
                    selected_by[candidate_id] = "UNIFORM_TIE" if len(boundary) > slots else "LEARNED_SCORE"
            else:
                probabilities[candidate_id] = 0.0
        return selected, tie_set, selected_by, probabilities, None, source_digest, semantics, tie_tolerance, "NONE"


def _validate_outcome_policy_activation_pair(
    outcome_policy: Mapping[str, Any], outcome_activation: Mapping[str, Any]
) -> None:
    """Reject an outcome policy unless its declared active pair is exact."""

    if outcome_activation.get("status") != "ACTIVE_DEVELOPMENT_ONLY":
        raise Q5AIdeaFeasibilityError(
            "OUTCOME_AWARE activation is not ACTIVE_DEVELOPMENT_ONLY"
        )
    if outcome_activation.get("policy_digest") != outcome_policy.get("policy_digest"):
        raise Q5AIdeaFeasibilityError(
            "OUTCOME_AWARE policy/activation policy_digest mismatch"
        )


def _q5a_outcome_pool_view(pool: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the frozen Q5-A envelope to the accepted Q3 projection metadata."""

    if "selection_rule" in pool:
        return dict(pool)
    return {
        **pool,
        "selection_rule": "NONE_POOL_ONLY_POLICIES_SELECT_AFTER_BYTE_FREEZE",
    }


def select(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    select_artifacts = (
        output_root / "F1_POLICY_SNAPSHOT.json",
        output_root / "OUTCOME_POLICY_SNAPSHOT.json",
        output_root / "OUTCOME_ACTIVATION_SNAPSHOT.json",
        output_root / "selections",
        output_root / "FROZEN_SELECTIONS_BEFORE_REALIZATION.json",
        output_root / "SELECTION_STAGE_RECEIPT.json",
    )
    if any(path.exists() for path in select_artifacts):
        raise Q5AIdeaFeasibilityError(
            "Q5-A SELECT output already exists; write-once stage cannot be re-entered"
        )
    prefreeze_manifest = _require_prefreeze(output_root)
    pool_receipt = _read(output_root / "POOL_GENERATION_RECEIPT.json")
    if not pool_receipt.get("all_pools_complete"):
        raise Q5AIdeaFeasibilityError("Q5-A cannot select from an incomplete pool set")
    gate = read_signed_execution_gate(output_root / "EXECUTION_GATE.json")
    pool_release = read_signed_stage_release(
        output_root / "EXECUTION_STAGE_POOL_RELEASED.json", stage="POOL"
    )
    if pool_release.get("status") != "COMPLETED" or pool_release.get("artifact_path") != str((output_root / "POOL_GENERATION_RECEIPT.json").resolve()):
        raise Q5AIdeaFeasibilityError("Q5-A SELECT requires a completed POOL stage release")
    if pool_release.get("artifact_sha256") != bytes_sha256((output_root / "POOL_GENERATION_RECEIPT.json").read_bytes()):
        raise Q5AIdeaFeasibilityError("Q5-A POOL receipt digest drifted before SELECT")
    f1_policy = _read(args.f1_policy.resolve())
    outcome_policy = _read(args.outcome_policy.resolve())
    outcome_activation = _read(args.outcome_activation.resolve())
    _validate_outcome_policy_activation_pair(outcome_policy, outcome_activation)
    selections: dict[str, list[dict[str, Any]]] = {policy: [] for policy in Q5A_POLICIES}
    selection_files: dict[tuple[int, str], dict[str, Any]] = {}
    explorations: list[dict[str, Any]] = []
    pool_values: list[dict[str, Any]] = []
    for pool_index in range(1, Q5A_POOL_COUNT + 1):
        pool_root = output_root / "pools" / f"{pool_index:02d}"
        pool_manifest = _read(pool_root / "POOL_MANIFEST.json")
        raw_pool = _read(pool_root / "RAW_POOL_BEFORE_SELECTION.json")
        pool_values.append(pool_manifest)
        pool_digest = str(pool_manifest["pool_digest"])
        for policy_index, policy in enumerate(Q5A_POLICIES):
            selected, tie_set, selected_by, probabilities, draw, source_digest, semantics, tie_tolerance, prior_usage = _ranked_selection(
                raw_pool,
                policy,
                f1_policy=f1_policy,
                outcome_policy=outcome_policy,
                outcome_activation=outcome_activation,
                seed=73000 + pool_index * 10 + policy_index,
            )
            selection = build_q5a_selection_manifest(
                pool=pool_manifest,
                pool_digest=pool_digest,
                policy_name=policy,
                selected_candidate_ids=selected,
                candidate_probabilities=probabilities,
                tie_set=tie_set,
                selected_by=selected_by,
                selection_seed=73000 + pool_index * 10 + policy_index,
                exploration_draw=draw,
                source_policy_digest=source_digest,
                selection_semantics=semantics,
                tie_tolerance=tie_tolerance,
                prior_usage=prior_usage,
            )
            selections[policy].append(selection)
            selection_files[(pool_index, policy)] = selection
        exploration = build_q5a_shared_exploration(
            pool=raw_pool,
            pool_digest=pool_digest,
            random_seed=EXPLORATION_SEEDS[pool_index - 1],
        )
        explorations.append(exploration)
    union = build_q5a_realization_union(
        prefreeze=prefreeze_manifest,
        pools=pool_values,
        selections=selections,
        explorations=explorations,
    )
    frozen_path = output_root / "FROZEN_SELECTIONS_BEFORE_REALIZATION.json"
    frozen_value = {"selections": selections, "explorations": explorations, "union": union}
    frozen_sha = sha256_digest(frozen_value)
    selection_payload = {
        "schema": "recclaw.research-line.q5a-selection-stage-receipt.v1",
        "status": "SELECT_COMPLETE",
        "campaign_root": str(output_root),
        "gate_digest": gate["gate_digest"],
        "pool_release_digest": pool_release["release_digest"],
        "pool_generation_receipt_sha256": bytes_sha256((output_root / "POOL_GENERATION_RECEIPT.json").read_bytes()),
        "frozen_selection_path": str(frozen_path.resolve()),
        "frozen_selection_sha256": frozen_sha,
        "selection_union_digest": union["union_digest"],
        "held_out_reads": 0,
        "retries": 0,
        "development_only": True,
        "scientific_effect_claim": False,
    }
    staged_relative_paths: list[Path] = [
        Path("F1_POLICY_SNAPSHOT.json"),
        Path("OUTCOME_POLICY_SNAPSHOT.json"),
        Path("OUTCOME_ACTIVATION_SNAPSHOT.json"),
    ]
    staged_relative_paths.extend(
        Path("selections") / f"pool-{pool_index:02d}" / f"{policy}.json"
        for pool_index in range(1, Q5A_POOL_COUNT + 1)
        for policy in Q5A_POLICIES
    )
    staged_relative_paths.extend(
        Path("selections") / f"pool-{pool_index:02d}" / "SHARED_EXPLORATION.json"
        for pool_index in range(1, Q5A_POOL_COUNT + 1)
    )
    staged_relative_paths.extend(
        (Path("FROZEN_SELECTIONS_BEFORE_REALIZATION.json"), Path("SELECTION_STAGE_RECEIPT.json"))
    )
    with tempfile.TemporaryDirectory(
        prefix=".q5a-select-staging-", dir=output_root
    ) as staging_value:
        staging_root = Path(staging_value)
        _write_new(staging_root / "F1_POLICY_SNAPSHOT.json", f1_policy)
        _write_new(staging_root / "OUTCOME_POLICY_SNAPSHOT.json", outcome_policy)
        _write_new(staging_root / "OUTCOME_ACTIVATION_SNAPSHOT.json", outcome_activation)
        for (pool_index, policy), selection in selection_files.items():
            _write_new(
                staging_root
                / "selections"
                / f"pool-{pool_index:02d}"
                / f"{policy}.json",
                selection,
            )
        for pool_index, exploration in enumerate(explorations, 1):
            _write_new(
                staging_root
                / "selections"
                / f"pool-{pool_index:02d}"
                / "SHARED_EXPLORATION.json",
                exploration,
            )
        staged_frozen_sha = _write_new(
            staging_root / "FROZEN_SELECTIONS_BEFORE_REALIZATION.json", frozen_value
        )
        if staged_frozen_sha != frozen_sha:
            raise Q5AIdeaFeasibilityError("Q5-A frozen selection digest changed during staging")
        selection_payload["frozen_selection_sha256"] = frozen_sha
        _write_new(
            staging_root / "SELECTION_STAGE_RECEIPT.json",
            {**selection_payload, "selection_stage_digest": sha256_digest(selection_payload)},
        )
        published: list[Path] = []
        try:
            for relative_path in staged_relative_paths:
                destination = output_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging_root / relative_path, destination)
                published.append(destination)
        except Exception:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            for directory in sorted(
                {
                    path.parent
                    for path in published
                    if path.parent != output_root
                },
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            raise
    print(json.dumps({"status": "SELECTIONS_FROZEN", "unique_spec_count": union["selected_unique_spec_count"]}, sort_keys=True))


REALIZATION_STAGE_ORDER = (
    "preflight",
    "selected-resolver",
    "implementer",
    "materialize-qualifier",
    "resource-admission",
    "mechanism-probe",
    "matched-execution",
    "episode",
)
Q5_STAGE_RECEIPTS = {
    "MATERIALIZE": ("IMPLEMENTER_RECEIPT.json", {"IMPLEMENTATION_PROVIDER_SUCCESS"}),
    "CONSTRUCT": ("RESOLVER_RECEIPT.json", {"RESOLUTION_CONFIRMED"}),
    "QUALIFY": ("MATERIALIZE_QUALIFIER_RECEIPT.json", {"QUALIFICATION_PASS"}),
    "RESOURCE_ADMITTED": ("RESOURCE_ADMISSION_RECEIPT.json", {"RESOURCE_ADMITTED"}),
    "FULL_EPISODE": ("EPISODE_RECEIPT.json", {"EPISODE_CREATED"}),
}
Q5_STAGE_COSTS = {
    "MATERIALIZE": "implementer",
    "CONSTRUCT": "selected-resolver",
    "QUALIFY": "materialize-qualifier",
    "RESOURCE_ADMITTED": "resource-admission",
    "FULL_EPISODE": "episode",
}


def _frozen_selection(output_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefreeze = _require_prefreeze(output_root)
    frozen = _read(output_root / "FROZEN_SELECTIONS_BEFORE_REALIZATION.json")
    union = frozen.get("union")
    if not isinstance(union, dict) or union.get("schema") != "recclaw.research-line.q5a-realization-union.v1":
        raise Q5AIdeaFeasibilityError("Q5-A frozen selection union is missing or invalid")
    return prefreeze, frozen, union


def _candidate_index(output_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for pool_index in range(1, Q5A_POOL_COUNT + 1):
        raw_path = output_root / "pools" / f"{pool_index:02d}" / "RAW_POOL_BEFORE_SELECTION.json"
        raw_pool = _read(raw_path)
        for group in sorted(raw_pool.get("candidate_pools", {})):
            for row in raw_pool["candidate_pools"][group]:
                candidate_id = str(row["preoutcome_score"]["spec_digest"])
                if candidate_id in index:
                    raise Q5AIdeaFeasibilityError("Q5-A realization index has duplicate OpenSpec identity")
                index[candidate_id] = (raw_path, row)
    return index


def _q4_realization_manifest(
    *,
    prefreeze: Mapping[str, Any],
    union: Mapping[str, Any],
    candidate_id: str,
    raw_pool_path: Path,
    round_index: int,
    conversion_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution = dict(prefreeze["common_execution"])
    execution.update(
        {
            "provider_temperature": 0,
            "implementer_temperature": 0,
            "provider_schema_consumer": "fresh_r1.bounded_provider_call",
        }
    )
    payload_value: dict[str, Any] = {
            "schema": "recclaw.research-line.q4-prospective-arm-manifest.v1",
            "campaign_id": f"{CAMPAIGN_ID}-shared-realization",
            "round_index": round_index,
            "policy_name": "Q5A_SHARED_REALIZATION",
            "physical_order": Q5A_POLICIES,
            "frozen_inputs": {
                "full_pool_file": str(raw_pool_path.resolve()),
                "full_pool_file_sha256": bytes_sha256(raw_pool_path.read_bytes()),
                "selection_digest": str(union["union_digest"]),
            },
            "frozen_execution": execution,
            "selected_candidate": {
                "candidate_id": candidate_id,
                "selection_probability": 1.0,
                "selection_score": None,
                "exploration_probability": 0.0,
                "exploration_draw": None,
                "exploration_selected": False,
            },
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    if conversion_plan is not None:
        payload_value["conversion_efficiency"] = {
            **dict(conversion_plan),
            "implementation": {
                "schema": "recclaw.research-line.q5-conversion-implementer.v1",
                "max_revision_turns": MAX_REPAIR_TURNS,
                "candidate_local_multi_file": True,
                "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
                "held_out_reads": 0,
            },
            "execution_contract": RECBole_INTERFACE_CONTRACT,
            "shared_parent_root": str(
                (Path(str(raw_pool_path)).resolve().parents[2] / "shared_parents").resolve()
            ),
        }
    payload = canonical_value(payload_value)
    return {**payload, "manifest_digest": sha256_digest(payload)}


def _invoke_realization_stage(
    args: argparse.Namespace,
    *,
    realization_root: Path,
    manifest_path: Path,
    raw_pool_path: Path,
    stage: str,
) -> bool:
    command = [
        str(args.python_executable.resolve()),
        str((args.repo_root / "scripts/run_multiround_soak_stage.py").resolve()),
        stage,
        "--repo-root",
        str(args.repo_root.resolve()),
        "--round-root",
        str(realization_root.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--input-pool",
        str(raw_pool_path.resolve()),
        "--projects-root",
        str(args.projects_root.resolve()),
        "--search-data-root",
        str(args.search_data_root.resolve()),
        "--recbole-root",
        str(args.recbole_root.resolve()),
        "--python-executable",
        str(args.python_executable.resolve()),
        "--api-config",
        str(args.api_config.resolve()),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    invocation = {
        "schema": "recclaw.research-line.q5a-stage-invocation.v1",
        "stage": stage,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "retries": 0,
        "held_out_reads": 0,
        "development_only": True,
    }
    _write_new(realization_root / "stage_invocations" / f"{stage}.json", invocation)
    if completed.returncode != 0:
        _write_new(
            realization_root / "stage_failures" / f"{stage}.json",
            {
                **invocation,
                "failure_class": "REAL_STAGE_PROCESS_FAILURE",
                "subsequent_stages_not_attempted": True,
            },
        )
        return False
    return True


def _read_optional(path: Path) -> dict[str, Any] | None:
    return _read(path) if path.is_file() else None


def _realization_contract(
    *,
    candidate_id: str,
    row: Mapping[str, Any],
    realization_root: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    package_digest = str(qualification["candidate_package_digest"])
    source_digest = str(qualification["candidate_source_tree_digest"])
    mechanism_path = realization_root / "MECHANISM_PROBE_RECEIPT.json"
    mechanism = _read_optional(mechanism_path) or {
        "schema": "recclaw.research-line.q5a-missing-mechanism-evidence.v1",
        "status": "MISSING",
    }
    contract = build_shared_realization_contract(
        research_spec_ref=f"recclaw-open-research-spec-v1:{candidate_id}",
        research_spec_digest=candidate_id,
        candidate_package_ref=f"recclaw-candidate-package-v1:{package_digest}",
        candidate_package_digest=package_digest,
        source_tree_ref=f"recclaw-candidate-source-tree-v1:{source_digest}",
        source_tree_digest=source_digest,
        equivalence_ref=f"recclaw-q5a-mechanism-evidence-v1:{sha256_digest(mechanism)}",
        equivalence_digest=sha256_digest(mechanism),
        protocol_ref=str(row["research_spec"]["protocol_ref"]),
        protocol_digest=str(row["research_spec"]["protocol_digest"]),
        realization_class="NEW_CANDIDATE",
    )
    realization_mode = str(row["research_spec"].get("realization_mode"))
    realization_typing = (
        "NESTED_MECHANISM" if realization_mode == "PARENT_PRESERVING" else "EFFECT_ONLY_NON_NESTED"
    )
    return canonical_value(
        {
            "research_spec_digest": candidate_id,
            "realization_contract": contract,
            "realization_class": "NEW_CANDIDATE",
            "realization_typing": realization_typing,
            "realization_mode": realization_mode,
            "realization_root": str(realization_root.resolve()),
            "qualification_receipt_digest": qualification.get("qualification_receipt_digest"),
            "candidate_package_digest": package_digest,
            "candidate_source_tree_digest": source_digest,
        }
    )


def _stage_rows(realization_root: Path, candidate_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q5_stage in Q5A_STAGES:
        receipt_name, success_statuses = Q5_STAGE_RECEIPTS[q5_stage]
        receipt_path = realization_root / receipt_name
        receipt = _read_optional(receipt_path)
        cost_path = realization_root / "stage_costs" / f"{Q5_STAGE_COSTS[q5_stage]}.json"
        cost = _read_optional(cost_path)
        observed = receipt is not None
        status = str(receipt.get("status")) if receipt is not None else "MISSING_STAGE_RECEIPT"
        rows.append(
            {
                "research_spec_digest": candidate_id,
                "stage": q5_stage,
                "status": "SUCCESS" if status in success_statuses else status,
                "observed": observed,
                "missingness": None if observed else "STAGE_NOT_EXECUTED",
                "cost_ms": int(cost["wall_time_ms"]) if cost and cost.get("wall_time_ms") is not None else None,
                "receipt_file": str(receipt_path),
                "cost_file": str(cost_path),
            }
        )
    return rows


def _build_conversion_screen_results(
    entries: Sequence[Mapping[str, Any]],
    policy_owners: Mapping[str, set[str]],
    exploration_ids: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in entries:
        receipt = _read_optional(Path(entry["realization_root"]) / "MATCHED_EXECUTION_RECEIPT.json")
        candidate = (receipt.get("candidate") or {}) if receipt else {}
        candidate_id = str(entry["candidate_id"])
        results.append(
            {
                "candidate_id": candidate_id,
                "status": receipt.get("status") if receipt else "MISSING",
                "stable": bool(receipt and receipt.get("stable")),
                "screen_signal": receipt.get("screen_signal") if receipt else None,
                "screen_cost_ms": candidate.get("wall_time_ms") if receipt else None,
                "policy_owners": sorted(policy_owners.get(candidate_id, set())),
                "shared_exploration": candidate_id in exploration_ids,
            }
        )
    return results


def realize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    prefreeze, _frozen, union = _frozen_selection(output_root)
    index = _candidate_index(output_root)
    selected_ids = [str(value) for value in union["selected_unique_spec_digests"]]
    if set(selected_ids) != set(index).intersection(selected_ids):
        raise Q5AIdeaFeasibilityError("Q5-A selection union references an unknown OpenSpec")
    realization_records: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    conversion_plan = build_conversion_execution_plan(
        selected_ids,
        screen_seed=CONVERSION_SCREEN_SEED,
        full_seeds=FULL_DEVELOPMENT_SEEDS,
        promotion_limit=min(CONVERSION_PROMOTION_LIMIT, len(selected_ids)),
    )
    _write_new(output_root / "CONVERSION_EXECUTION_PLAN.json", conversion_plan)
    entries: list[dict[str, Any]] = []
    policy_owners: dict[str, set[str]] = {candidate_id: set() for candidate_id in selected_ids}
    for policy, pool_rows in union["policy_attribution"].items():
        for candidate_ids in pool_rows.values():
            for candidate_id in candidate_ids:
                policy_owners.setdefault(str(candidate_id), set()).add(str(policy))
    exploration_ids = {
        str(candidate_id) for candidate_id in union["shared_exploration_by_pool"].values()
    }
    for ordinal, candidate_id in enumerate(sorted(selected_ids), 1):
        raw_pool_path, row = index[candidate_id]
        realization_root = output_root / "realizations" / f"{ordinal:03d}_{candidate_id[:12]}"
        if realization_root.exists():
            raise Q5AIdeaFeasibilityError(f"shared realization already exists: {realization_root}")
        realization_root.mkdir(parents=True)
        manifest = _q4_realization_manifest(
            prefreeze=prefreeze,
            union=union,
            candidate_id=candidate_id,
            raw_pool_path=raw_pool_path,
            round_index=ordinal,
            conversion_plan=conversion_plan,
        )
        manifest_path = realization_root / "ARM_MANIFEST.json"
        _write_new(manifest_path, manifest)
        stage_ok = True
        for stage in REALIZATION_STAGE_ORDER[:-1]:
            if not stage_ok:
                break
            stage_ok = _invoke_realization_stage(
                args,
                realization_root=realization_root,
                manifest_path=manifest_path,
                raw_pool_path=raw_pool_path,
                stage=stage,
            )
        entries.append(
            {
                "candidate_id": candidate_id,
                "row": row,
                "raw_pool_path": raw_pool_path,
                "realization_root": realization_root,
                "manifest_path": manifest_path,
                "stage_ok": stage_ok,
            }
        )
    screen_results = _build_conversion_screen_results(
        entries,
        policy_owners,
        exploration_ids,
    )
    _write_new(
        output_root / "CONVERSION_SCREEN_RESULTS.json",
        {
            "schema": "recclaw.research-line.q5-conversion-screen-results.v1",
            "plan_digest": conversion_plan["plan_digest"],
            "results": screen_results,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )
    promoted = choose_stable_promotions(
        screen_results,
        promotion_limit=int(conversion_plan["promotion"]["limit"]),
    )
    promotion_plan = finalize_conversion_execution_plan(
        conversion_plan,
        screen_results,
        promoted,
    )
    _write_new(output_root / "CONVERSION_PROMOTION_PLAN.json", promotion_plan)
    promoted_set = set(promoted)
    for entry in entries:
        candidate_id = entry["candidate_id"]
        realization_root = entry["realization_root"]
        if entry["stage_ok"]:
            if candidate_id in promoted_set:
                _invoke_realization_stage(
                    args,
                    realization_root=realization_root,
                    manifest_path=entry["manifest_path"],
                    raw_pool_path=entry["raw_pool_path"],
                    stage="full-execution",
                )
            _invoke_realization_stage(
                args,
                realization_root=realization_root,
                manifest_path=entry["manifest_path"],
                raw_pool_path=entry["raw_pool_path"],
                stage="episode",
            )
        rows = _stage_rows(realization_root, candidate_id)
        stage_rows.extend(rows)
        qualification = _read_optional(realization_root / "MATERIALIZE_QUALIFIER_RECEIPT.json")
        if qualification and qualification.get("status") == "QUALIFICATION_PASS":
            record = _realization_contract(
                candidate_id=candidate_id,
                row=entry["row"],
                realization_root=realization_root,
                qualification=qualification,
            )
            _write_new(realization_root / "REALIZATION_CONTRACT.json", record)
            realization_records.append(record)
        else:
            failures.append(
                {
                    "research_spec_digest": candidate_id,
                    "status": "REALIZATION_MISSING",
                    "qualification_status": qualification.get("status") if qualification else None,
                    "stage_process_failure": not entry["stage_ok"],
                }
            )
    denominator = build_q5a_stage_denominator(
        realization_rows=stage_rows,
        realization_denominator=len(selected_ids),
    )
    _write_new(output_root / "Q5A_STAGE_DENOMINATOR.json", denominator)
    complete = len(realization_records) == len(selected_ids)
    if complete:
        final_union = build_q5a_realization_union(
            prefreeze=prefreeze,
            pools=[_read(output_root / "pools" / f"{index:02d}" / "POOL_MANIFEST.json") for index in range(1, Q5A_POOL_COUNT + 1)],
            selections={
                policy: [_read(output_root / "selections" / f"pool-{index:02d}" / f"{policy}.json") for index in range(1, Q5A_POOL_COUNT + 1)]
                for policy in Q5A_POLICIES
            },
            explorations=[_read(output_root / "selections" / f"pool-{index:02d}" / "SHARED_EXPLORATION.json") for index in range(1, Q5A_POOL_COUNT + 1)],
            realizations=realization_records,
        )
        _write_new(output_root / "REALIZATION_UNION_AFTER_EXECUTION.json", final_union)
    else:
        _write_new(
            output_root / "REALIZATION_UNION_AFTER_EXECUTION.json",
            {
                "schema": "recclaw.research-line.q5a-realization-union.v1",
                "status": "PARTIAL_REALIZATION_WITH_EXPLICIT_MISSINGNESS",
                "selected_unique_spec_count": len(selected_ids),
                "realized_unique_spec_count": len(realization_records),
                "missing_realizations": failures,
                "implementations_per_unique_openspec": 1,
                "outcome_key_rule": "ONE_OUTCOME_PER_CANDIDATE_PACKAGE_PER_SEED",
                "held_out_reads": 0,
                "retries": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    _write_new(
        output_root / "REALIZATION_EXECUTION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q5a-realization-execution-receipt.v1",
            "status": "REALIZATION_COMPLETE" if complete else "REALIZATION_PARTIAL_MISSINGNESS",
            "selected_unique_spec_count": len(selected_ids),
            "realized_unique_spec_count": len(realization_records),
            "missing_realizations": failures,
            "physical_implementer_calls": len(selected_ids),
            "one_realization_per_openspec": True,
            "retries": 0,
            "held_out_reads": 0,
            "candidate_replacement": False,
            "success_filtering": False,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )
    print(json.dumps({"status": "REALIZATION_COMPLETE" if complete else "REALIZATION_PARTIAL_MISSINGNESS", "selected_unique_spec_count": len(selected_ids), "realized_unique_spec_count": len(realization_records)}, sort_keys=True))


def _stage_success(rows: list[dict[str, Any]], stage: str, candidate_id: str) -> bool:
    return any(
        row["research_spec_digest"] == candidate_id
        and row["stage"] == stage
        and row["status"] == "SUCCESS"
        for row in rows
    )


def summarize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    prefreeze, frozen, union = _frozen_selection(output_root)
    stage_denominator = _read(output_root / "Q5A_STAGE_DENOMINATOR.json")
    stage_rows: list[dict[str, Any]] = []
    realization_by_id: dict[str, dict[str, Any]] = {}
    for contract_path in sorted((output_root / "realizations").glob("*/REALIZATION_CONTRACT.json")):
        record = _read(contract_path)
        candidate_id = str(record["research_spec_digest"])
        realization_root = Path(str(record["realization_root"]))
        realization_by_id[candidate_id] = record
        stage_rows.extend(_stage_rows(realization_root, candidate_id))
    qualification_by_id: dict[str, dict[str, Any] | None] = {}
    admission_by_id: dict[str, dict[str, Any] | None] = {}
    authority_rows = []
    for candidate_id in sorted(union["selected_unique_spec_digests"]):
        realization = realization_by_id.get(candidate_id)
        if realization is None:
            authority_rows.append({"research_spec_digest": candidate_id, "status": "MISSING_REALIZATION"})
            continue
        root = Path(str(realization["realization_root"]))
        qualification = _read_optional(root / "MATERIALIZE_QUALIFIER_RECEIPT.json")
        admission = _read_optional(root / "RESOURCE_ADMISSION_RECEIPT.json")
        qualification_by_id[candidate_id] = qualification
        admission_by_id[candidate_id] = admission
        authority = classify_realization_authority(
            realization=realization,
            qualification=qualification or {},
            admission=admission or {},
        )
        episode = _read_optional(root / "EPISODE_RECEIPT.json")
        matched = _read_optional(root / "MATCHED_EXECUTION_RECEIPT.json")
        full_episode_effect = bool(
            episode
            and episode.get("status") == "EPISODE_CREATED"
            and matched
            and matched.get("status") == "COMPLETED_MATCHED_PAIR"
        )
        authority_rows.append(
            {
                "research_spec_digest": candidate_id,
                "realization_digest": realization["realization_contract"]["realization_digest"],
                "realization_typing": realization["realization_typing"],
                "mechanism_information_authority": authority["mechanism_information_authority"],
                "mechanism_state": authority["mechanism_state"],
                "effect_authority": "EFFECT" if full_episode_effect else "NOT_ASSESSED",
                "effect_input_allowed": full_episode_effect,
                "effect_eligibility": "FULL_COMPARABLE_FRESH_MATCHED_DEVELOPMENT_EPISODE" if full_episode_effect else "ELIGIBLE_ONLY_AFTER_FULL_MATCHED_EPISODE",
                "authority_reason": authority["reason"],
                "no_off_switch_is_not_negative_mechanism_evidence": True,
            }
        )
    _write_new(output_root / "REALIZATION_AUTHORITY_SUMMARY.json", {"schema": "recclaw.research-line.q5a-realization-authority-summary.v1", "rows": authority_rows, "held_out_reads": 0, "retries": 0, "development_only": True, "scientific_effect_claim": False})
    policy_executable: dict[str, int] = {}
    for policy in Q5A_POLICIES:
        selected_ids = [
            str(candidate_id)
            for pool_rows in union["policy_attribution"][policy].values()
            for candidate_id in pool_rows
        ]
        policy_executable[policy] = len({candidate_id for candidate_id in selected_ids if candidate_id in realization_by_id})
    mechanism_count = sum(row.get("mechanism_information_authority") == "MECHANISM_INFORMATION" for row in authority_rows)
    effect_count = sum(row.get("effect_authority") == "EFFECT" for row in authority_rows)
    all_selection_files = [output_root / "selections" / f"pool-{index:02d}" / f"{policy}.json" for index in range(1, Q5A_POOL_COUNT + 1) for policy in Q5A_POLICIES]
    selection_values = [_read(path) for path in all_selection_files]
    tie_contract = all(
        value.get("selection_probability_sum") == float(Q5A_POLICY_SELECTION_BUDGET)
        and isinstance(value.get("top_tie_set"), list)
        and set(value.get("selected_by", {})) == set(value.get("selected_candidate_ids", []))
        for value in selection_values
    )
    total_cost_ms = sum(int(row["cost_ms"]) for row in stage_rows if row.get("cost_ms") is not None)
    executable_count = len(realization_by_id)
    informative_count = mechanism_count + effect_count
    exploration_ids = set(str(value) for value in union["shared_exploration_by_pool"].values())
    exploitation_ids = {
        str(candidate_id)
        for policy_rows in union["policy_attribution"].values()
        for pool_rows in policy_rows.values()
        for candidate_id in pool_rows
    }
    exploration_summary = {
        "shared_exploration_spec_count": len(exploration_ids),
        "exploitation_unique_spec_count": len(exploitation_ids),
        "overlap_count": len(exploration_ids & exploitation_ids),
        "exploration_stage_completion": {
            stage: sum(_stage_success(stage_rows, stage, candidate_id) for candidate_id in exploration_ids)
            for stage in Q5A_STAGES
        },
        "exploitation_stage_completion": {
            stage: sum(_stage_success(stage_rows, stage, candidate_id) for candidate_id in exploitation_ids)
            for stage in Q5A_STAGES
        },
    }
    score_values = []
    for pool_index in range(1, Q5A_POOL_COUNT + 1):
        raw = _read(output_root / "pools" / f"{pool_index:02d}" / "RAW_POOL_BEFORE_SELECTION.json")
        score_values.extend(int(row["preoutcome_score"]["total"]) for row in raw["candidate_pools"]["shared"])
    idea_score_discriminates = len(set(score_values)) > 1
    package = {
        "schema": "recclaw.research-line.q5a-idea-feasibility-package.v1",
        "campaign_id": CAMPAIGN_ID,
        "prefreeze_digest": prefreeze["prefreeze_digest"],
        "selection_union_digest": union["union_digest"],
        "stage_denominator_digest": stage_denominator["stage_digest"],
        "policy_executable_package_count": policy_executable,
        "stage_empirical_completion": stage_denominator["stage_stats"],
        "cost": {
            "total_cost_ms": total_cost_ms,
            "cost_per_executable_package_ms": total_cost_ms / executable_count if executable_count else None,
            "cost_per_informative_package_ms": total_cost_ms / informative_count if informative_count else None,
            "executable_denominator": executable_count,
            "informative_denominator": informative_count,
        },
        "full_episode_completion": {"count": sum(_stage_success(stage_rows, "FULL_EPISODE", candidate_id) for candidate_id in union["selected_unique_spec_digests"]), "denominator": len(union["selected_unique_spec_digests"]), "rate": sum(_stage_success(stage_rows, "FULL_EPISODE", candidate_id) for candidate_id in union["selected_unique_spec_digests"]) / len(union["selected_unique_spec_digests"]) if union["selected_unique_spec_digests"] else None},
        "calibration": {
            "status": "NOT_COMPUTABLE_NO_STAGE_CONDITIONAL_PREOUTCOME_PROBABILITIES",
            "brier_score": None,
            "reliability_bins": [],
            "denominator": len(union["selected_unique_spec_digests"]),
        },
        "tie_and_prior": {
            "tie_contract_executed": tie_contract,
            "tie_selection_count": sum(bool(value.get("top_tie_set")) for value in selection_values),
            "prior_usage_count": sum(value.get("prior_usage") != "NONE" for value in selection_values),
        },
        "exploration_vs_exploitation": exploration_summary,
        "idea_score_discriminates": idea_score_discriminates,
        "authority_counts": {"nested_mechanism_information": mechanism_count, "effect_episode": effect_count},
        "denominator_and_missingness_complete": (
            stage_denominator.get("stage_order") == list(Q5A_STAGES)
            and all(value.get("denominator") == len(union["selected_unique_spec_digests"]) for value in stage_denominator.get("stage_stats", []))
        ),
        "shared_realization_byte_drift": len(realization_by_id) != len(set(realization["realization_contract"]["realization_digest"] for realization in realization_by_id.values())),
        "held_out_reads": 0,
        "retries": 0,
        "candidate_replacement": False,
        "success_filtering": False,
        "development_only": True,
        "scientific_effect_claim": False,
    }
    gate = {
        "schema": "recclaw.research-line.q5a-gate.v1",
        "campaign_id": CAMPAIGN_ID,
        "status": "Q5A_PASS" if all((
            sum(value > 0 for value in policy_executable.values()) >= 2,
            package["shared_realization_byte_drift"] is False,
            mechanism_count >= 2,
            package["denominator_and_missingness_complete"],
            tie_contract,
            idea_score_discriminates,
            package["held_out_reads"] == 0,
            package["retries"] == 0,
            package["candidate_replacement"] is False,
            package["success_filtering"] is False,
        )) else "Q5A_NEGATIVE_GATE",
        "criteria": {
            "at_least_two_policy_nonempty_executable": sum(value > 0 for value in policy_executable.values()) >= 2,
            "shared_realization_no_byte_drift": package["shared_realization_byte_drift"] is False,
            "at_least_two_nested_off_equivalent": mechanism_count >= 2,
            "stage_denominator_missingness_cost_complete": package["denominator_and_missingness_complete"],
            "tie_contract_real": tie_contract,
            "idea_score_not_all_same": idea_score_discriminates,
            "zero_heldout_retry_replacement_filtering": package["held_out_reads"] == 0 and package["retries"] == 0 and not package["candidate_replacement"] and not package["success_filtering"],
        },
        "negative_gate_stop_rule": "STOP_NO_POOL_EXPANSION_NO_RETRY_IF_NEGATIVE",
        "held_out_reads": 0,
        "retries": 0,
        "development_only": True,
        "scientific_effect_claim": False,
    }
    _write_new(output_root / "Q5A_IDEA_FEASIBILITY_PACKAGE.json", package)
    _write_new(output_root / "Q5A_GATE.json", gate)
    print(json.dumps({"status": gate["status"], "policy_executable": policy_executable, "nested_mechanism_information": mechanism_count, "full_episode_count": effect_count}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prefreeze")
    p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("pool")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--projects-root", type=Path, required=True)
    p.add_argument("--search-data-root", type=Path, required=True)
    p.add_argument("--recbole-root", type=Path, required=True)
    p.add_argument("--python-executable", type=Path, required=True)
    p.add_argument("--api-config", type=Path, required=True)
    p.add_argument("--deployment-manifest", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--prefreeze-manifest", type=Path, required=True)
    p = sub.add_parser("select")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--f1-policy", type=Path, required=True)
    p.add_argument("--outcome-policy", type=Path, required=True)
    p.add_argument("--outcome-activation", type=Path, required=True)
    p = sub.add_parser("realize")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--projects-root", type=Path, required=True)
    p.add_argument("--search-data-root", type=Path, required=True)
    p.add_argument("--recbole-root", type=Path, required=True)
    p.add_argument("--python-executable", type=Path, required=True)
    p.add_argument("--api-config", type=Path, required=True)
    p.add_argument("--deployment-manifest", type=Path, required=True)
    p.add_argument("--preflight-receipt", type=Path, required=True)
    p.add_argument("--prefreeze-manifest", type=Path, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prefreeze":
        prefreeze(args)
    elif args.command == "pool":
        pool(args)
    elif args.command == "select":
        select(args)
    elif args.command == "realize":
        realize(args)
    elif args.command == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        _CHILD_STATUS = "FAILED"
        raise
    else:
        _CHILD_STATUS = "COMPLETED"
    finally:
        if _RUNTIME_CHILD and _RUNTIME_BINDING is not None and _EXECUTION_OWNER is not None:
            release_execution_owner(_RUNTIME_BINDING, _EXECUTION_OWNER, status=_CHILD_STATUS)
    raise SystemExit(exit_code)
