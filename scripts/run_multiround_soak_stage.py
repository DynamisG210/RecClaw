#!/usr/bin/env python3
"""Execute exactly one real stage of a sealed Q4 multiround soak round."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RECBOLE_COMMIT = "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"


def _bootstrap(args: argparse.Namespace) -> None:
    binding_manifest = os.environ.get("RECCLAW_BINDING_MANIFEST")
    if not binding_manifest:
        raise RuntimeError("stage child is missing the frozen runtime binding manifest")
    sys.path.insert(0, str(args.repo_root / "src"))
    from recclaw_runtime_binding import RuntimeBindingV1

    binding = RuntimeBindingV1.from_manifest(
        Path(binding_manifest), repo_root=args.repo_root
    ).activate()
    expected_paths = {
        "projects_root": binding.projects_root,
        "search_data_root": binding.search_data_root,
        "recbole_root": binding.recbole_root,
        "python_executable": binding.python_executable,
        "api_config": binding.api_config,
    }
    for name, expected in expected_paths.items():
        if Path(getattr(args, name)).resolve() != expected:
            raise RuntimeError(f"stage runtime argument drift: {name}")
    os.environ["RECCLAW_PROJECTS_ROOT"] = str(args.projects_root)
    os.environ["RECCLAW_SEARCH_DATA_ROOT"] = str(args.search_data_root)
    os.environ["RECCLAW_RECBOLE_ROOT"] = str(args.recbole_root)
    os.environ["RECCLAW_PYTHON_EXECUTABLE"] = str(args.python_executable)
    os.environ["RECCLAW_API_CONFIG"] = str(args.api_config)
    runtime_paths = (
        (args.repo_root, args.repo_root / "src")
        if args.stage == "selected-resolver"
        else (args.recbole_root, args.repo_root, args.repo_root / "src")
    )
    for path in runtime_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import canonical_json_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
    return path


def _receipt(path: Path, value: Mapping[str, Any]) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import (
        canonical_value,
        sha256_digest,
    )

    payload = canonical_value(value)
    return _write_new(path, {**payload, "receipt_digest": sha256_digest(payload)})


def _manifest(args: argparse.Namespace) -> dict[str, Any]:
    from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest

    value = _read(args.manifest)
    expected = value.pop("manifest_digest")
    observed = sha256_digest(value)
    value["manifest_digest"] = expected
    if expected != observed or value.get("held_out_reads") != 0:
        raise RuntimeError("round manifest identity drift")
    return value


def _verify_file(path: Path, expected: str, label: str) -> None:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256

    observed = bytes_sha256(path.read_bytes())
    if observed != expected:
        raise RuntimeError(f"{label} SHA drift: {observed}")


def _read_upstream(args: argparse.Namespace, name: str) -> dict[str, Any]:
    """Read a sealed v1 upstream artifact through the v2 hash binding."""

    if args.upstream_root is None:
        return _read(args.round_root / name)
    binding = _read(args.round_root / "SEALED_UPSTREAM_BINDING.json")
    artifact = binding.get("sealed_artifacts", {}).get(name)
    if not isinstance(artifact, dict):
        raise RuntimeError(f"sealed upstream binding is missing {name}")
    path = args.upstream_root / name
    if Path(str(artifact.get("path"))).resolve() != path.resolve():
        raise RuntimeError(f"sealed upstream path drift for {name}")
    _verify_file(path, str(artifact.get("sha256")), f"sealed upstream {name}")
    return _read(path)


def _selected_record(
    manifest: Mapping[str, Any], pool_path: Path
) -> tuple[str, dict[str, Any]]:
    _verify_file(
        pool_path,
        str(manifest["frozen_inputs"]["full_pool_file_sha256"]),
        "full pool",
    )
    pool = _read(pool_path)
    selected_id = str(manifest["selected_candidate"]["candidate_id"])
    matches = []
    for arm, rows in pool["candidate_pools"].items():
        for row in rows:
            if row.get("preoutcome_score", {}).get("spec_digest") == selected_id:
                matches.append((str(arm), row))
    if len(matches) != 1:
        raise RuntimeError("selected candidate does not bind exactly one full-pool row")
    arm, record = matches[0]
    if record.get("stage") != "OPENSPEC_FROZEN":
        raise RuntimeError("selected candidate is not a frozen OpenSpec")
    return arm, record


def _rehydrate_spec(record: Mapping[str, Any]) -> Any:
    from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
    from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
        CurrentProfileExpressibilityV1,
        IdeaModeV1,
        OpenResearchSpecV1,
        RealizationModeV1,
    )

    payload = dict(record["research_spec"])
    payload.pop("schema", None)
    for name in (
        "implementation_requirements",
        "expected_evidence",
        "compatibility_requirements",
        "causal_chain",
        "discriminative_predictions",
    ):
        if name in payload:
            payload[name] = tuple(payload[name])
    payload["current_profile_expressibility_claim"] = CurrentProfileExpressibilityV1(
        payload["current_profile_expressibility_claim"]
    )
    if payload.get("idea_mode") is not None:
        payload["idea_mode"] = IdeaModeV1(payload["idea_mode"])
    if payload.get("realization_mode") is not None:
        payload["realization_mode"] = RealizationModeV1(payload["realization_mode"])
    spec = OpenResearchSpecV1(**payload)
    expected = str(record["preoutcome_score"]["spec_digest"])
    if spec.digest != expected or sha256_digest(record["research_spec"]) != expected:
        raise RuntimeError("selected OpenSpec semantic identity drift")
    return spec


def _round_identity(manifest: Mapping[str, Any]) -> str:
    return f"{manifest['campaign_id']}-round-{manifest['round_index']}"


def run_provider_resolver(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.idea_quality import run_idea_quality

    manifest = _manifest(args)
    seeds = manifest["frozen_execution"]["provider_proposal_seeds"]
    stage_root = args.round_root / "next_pool"
    result = run_idea_quality(
        args.repo_root,
        run_root=stage_root,
        stop_after_pool=True,
        provider_run_identity=f"{_round_identity(manifest)}-next-pool",
        proposal_seeds=seeds,
    )
    if result["held_out_reads"] != 0 or result["implementation_provider_calls"] != 0:
        raise RuntimeError("Provider/Resolver stage crossed its authority boundary")
    return stage_root / "PROVIDER_RESOLVER_RECEIPT.json"


def run_preflight(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (
        build_active_r2_profile,
        build_r1_registry,
        load_registered_r1_artifacts,
    )

    manifest = _manifest(args)
    import recbole
    import torch

    recbole_head = (args.recbole_root / ".git/HEAD").read_text(
        encoding="utf-8"
    ).strip()
    if recbole_head != REQUIRED_RECBOLE_COMMIT:
        raise RuntimeError("gpu35 RecBole commit identity drift")

    artifacts, _receipt_value = load_registered_r1_artifacts(args.repo_root)
    registry = build_r1_registry(artifacts)
    _current, _build_manifest, _next_profile, _build_receipt, active = (
        build_active_r2_profile(registry)
    )
    search_manifest = args.search_data_root / "search_partition_manifest.json"
    _verify_file(
        search_manifest,
        str(manifest["frozen_execution"]["data_digest"]),
        "SEARCH partition manifest",
    )
    _verify_file(
        args.api_config,
        str(manifest["frozen_execution"]["provider_endpoint_digest"]),
        "Provider config",
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("gpu35 CUDA capability is unavailable")
    return _receipt(
        args.round_root / "REMOTE_PREFLIGHT_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-remote-preflight.v1",
            "status": "PASS",
            "registered_r1_capability_count": len(artifacts),
            "active_profile_digest": active.profile_digest,
            "recbole_version": recbole.__version__,
            "recbole_commit": recbole_head,
            "recbole_root": str(args.recbole_root),
            "python_executable_sha256": bytes_sha256(args.python_executable.read_bytes()),
            "cuda_available": True,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "search_manifest_sha256": bytes_sha256(search_manifest.read_bytes()),
            "provider_config_sha256": bytes_sha256(args.api_config.read_bytes()),
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_selected_resolver(args: argparse.Namespace) -> Path:
    """Re-run the real Resolver for the policy-owned selected chain."""

    from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
    from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (
        _r2_environment,
        build_active_r2_profile,
        build_r1_registry,
        load_registered_r1_artifacts,
    )
    from recclaw_core.experiments.helix_abc_v1.open_spec import resolve_capability

    manifest = _manifest(args)
    arm, record = _selected_record(manifest, args.input_pool)
    spec = _rehydrate_spec(record)
    artifacts, _receipt_value = load_registered_r1_artifacts(args.repo_root)
    registry = build_r1_registry(artifacts)
    _current, _build_manifest, _next_profile, _build_receipt, active = (
        build_active_r2_profile(registry)
    )
    environment = _r2_environment(active)
    resolution = resolve_capability(
        spec,
        resolution_facts=record["resolution_facts"],
        environment=environment,
    )
    observed = resolution.canonical_dict()
    expected = record["resolution"]
    matches = observed == expected
    return _receipt(
        args.round_root / "RESOLVER_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-prospective-selected-resolver.v1",
            "status": "RESOLUTION_CONFIRMED" if matches else "RESOLUTION_DRIFT",
            "policy_name": manifest.get("policy_name"),
            "selected_arm": arm,
            "selected_candidate_id": spec.digest,
            "expected_resolution_digest": sha256_digest(expected),
            "observed_resolution_digest": sha256_digest(observed),
            "resolution": observed,
            "resolver_reexecuted": True,
            "qualification_or_resource_outcomes_consumed": False,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_implementer(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
        _shared_policy,
        bounded_provider_call,
        render_implementation_prompt,
    )
    from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
        build_shared_implementer_request,
    )
    from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
        CANDIDATE_LOCAL_ALLOWED_FILES,
    )

    manifest = _manifest(args)
    arm, record = _selected_record(manifest, args.input_pool)
    spec = _rehydrate_spec(record)
    resources = args.repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources"
    template_path = resources / "idea_quality_implementer_prompt_v1.txt"
    schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    template = template_path.read_text(encoding="utf-8")
    conversion_contract = manifest.get("conversion_efficiency")
    if conversion_contract is not None:
        template_path = resources / "q5_conversion_implementer_prompt_v1.txt"
        schema_path = resources / "q5_conversion_implementer_response_v1.schema.json"
        template = template_path.read_text(encoding="utf-8")
    policy = _shared_policy(
        bytes_sha256(template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
        **(
            {
                "allowed_files": CANDIDATE_LOCAL_ALLOWED_FILES,
                "execution_contract": conversion_contract,
            }
            if conversion_contract is not None
            else {}
        ),
    )
    prompt = render_implementation_prompt(
        template, build_shared_implementer_request(spec, policy=policy)
    )
    identity = _round_identity(manifest)
    call = bounded_provider_call(
        call_root=args.round_root / "provider/implementation",
        schema_path=schema_path,
        logical_call_id=f"{identity}:selected:implementation",
        session_id=f"{identity}:implementation-session",
        prompt=prompt,
        token_ceiling=int(manifest["frozen_execution"]["implementation_token_ceiling"]),
        maximum_physical_attempts=1,
    )
    status = "IMPLEMENTATION_PROVIDER_SUCCESS" if call.call is not None else "IMPLEMENTATION_PROVIDER_FAILURE"
    return _receipt(
        args.round_root / "IMPLEMENTER_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-implementer-stage.v1",
            "status": status,
            "round_identity": identity,
            "selected_arm": arm,
            "selected_candidate_id": spec.digest,
            "logical_call_id": f"{identity}:selected:implementation",
            "maximum_physical_attempts": 1,
            "attempts": call.attempts,
            "failure": call.failure,
            "implementation_response": (
                call.call.response["proposals"][0] if call.call is not None else None
            ),
            "manual_candidate_patches": 0,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_materialize_qualifier(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import (
        bytes_sha256,
        sha256_digest,
    )
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (
        _materialize_and_qualify,
        _shared_policy,
        bounded_provider_call,
        render_implementation_prompt,
    )
    from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
        build_shared_implementer_request,
    )
    from recclaw_core.experiments.helix_abc_v1.idea_quality import _q1_unit_check
    from recclaw_core.experiments.helix_abc_v1.vnext_contracts import QualificationStatusV1
    from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (
        CANDIDATE_LOCAL_ALLOWED_FILES,
        MAX_REPAIR_TURNS,
        build_mechanical_repair_request,
        is_mechanical_repair_failure,
    )
    from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
        InnovationSpineError,
    )

    manifest = _manifest(args)
    _arm, record = _selected_record(manifest, args.input_pool)
    spec = _rehydrate_spec(record)
    implementation = _read(args.round_root / "IMPLEMENTER_RECEIPT.json")
    if implementation["status"] != "IMPLEMENTATION_PROVIDER_SUCCESS":
        return _receipt(
            args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q4-materialize-qualifier-stage.v1",
                "status": "NOT_RUN_UPSTREAM_IMPLEMENTER_FAILURE",
                "selected_candidate_id": spec.digest,
                "candidate_root": None,
                "qualification_status": None,
                "manual_candidate_patches": 0,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    resources = args.repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources"
    template_path = resources / "idea_quality_implementer_prompt_v1.txt"
    schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    conversion_contract = manifest.get("conversion_efficiency")
    if conversion_contract is not None:
        template_path = resources / "q5_conversion_implementer_prompt_v1.txt"
        schema_path = resources / "q5_conversion_implementer_response_v1.schema.json"
    policy = _shared_policy(
        bytes_sha256(template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
        **(
            {
                "allowed_files": CANDIDATE_LOCAL_ALLOWED_FILES,
                "execution_contract": conversion_contract,
            }
            if conversion_contract is not None
            else {}
        ),
    )
    current_implementation = implementation["implementation_response"]
    materialized = None
    qualification = None
    behavior: dict[str, Any] = {}
    failure_detail: Mapping[str, Any] | None = None
    materialized_root = args.round_root / "materialized"
    revision_history: list[dict[str, Any]] = []
    for attempt in range(0, MAX_REPAIR_TURNS + 1):
        materialized_root = (
            args.round_root / "materialized"
            if attempt == 0
            else args.round_root / f"materialized_revision_{attempt:02d}"
        )
        try:
            materialized, qualification, behavior = _materialize_and_qualify(
                repo_root=args.repo_root,
                side_root=materialized_root,
                slot_id="selected",
                seed=int(manifest["frozen_execution"]["qualification_seed"]),
                spec=spec,
                implementation=current_implementation,
                implementation_prompt_digest=bytes_sha256(template_path.read_bytes()),
                tool_policy_digest=bytes_sha256(tool_policy_path.read_bytes()),
                run_identity=_round_identity(manifest),
                policy=policy,
                unit_check_factory=_q1_unit_check(spec),
            )
            failure_detail = qualification.failure_detail
            revision_history.append(
                {
                    "turn": attempt,
                    "root": str(materialized_root),
                    "status": "PASS" if qualification.receipt.status is QualificationStatusV1.PASS else "FAIL",
                    "failure": failure_detail,
                }
            )
        except InnovationSpineError as error:
            materialized = None
            qualification = None
            behavior = {}
            failure_detail = {
                "failure_class": error.failure_class,
                "reason_code": error.reason_code,
                "stage": "SCHEMA",
                "message": str(error),
                "traceback": f"InnovationSpineError: {error.reason_code}: {str(error)[:1000]}",
            }
            revision_history.append(
                {"turn": attempt, "root": str(materialized_root), "status": "FAIL", "failure": failure_detail}
            )
        except Exception as error:  # noqa: BLE001 - isolate one candidate failure.
            materialized = None
            qualification = None
            behavior = {}
            failure_detail = {
                "failure_class": "RUNTIME",
                "reason_code": type(error).__name__,
                "stage": "QUALIFY",
                "message": str(error),
                "traceback": f"{type(error).__name__}: {str(error)[:1000]}",
            }
            revision_history.append(
                {"turn": attempt, "root": str(materialized_root), "status": "FAIL", "failure": failure_detail}
            )
        if qualification is not None and qualification.receipt.status is QualificationStatusV1.PASS:
            break
        if (
            conversion_contract is None
            or attempt >= MAX_REPAIR_TURNS
            or not isinstance(failure_detail, Mapping)
            or not is_mechanical_repair_failure(failure_detail)
        ):
            break
        current_source = {
            str(item["path"]): str(item["content"])
            for item in current_implementation.get("files", [])
        }
        repair_request = build_mechanical_repair_request(
            build_shared_implementer_request(spec, policy=policy),
            failure_detail,
            current_source=current_source,
            failure_message=str(failure_detail.get("message", "")),
            short_trace=str(failure_detail.get("traceback", "")),
            repair_attempt=attempt + 1,
        )
        repair_prompt = render_implementation_prompt(
            template_path.read_text(encoding="utf-8"), repair_request
        )
        repair_call = bounded_provider_call(
            call_root=args.round_root / f"provider/implementation_revision_{attempt + 1:02d}",
            schema_path=schema_path,
            logical_call_id=f"{_round_identity(manifest)}:selected:implementation-revision-{attempt + 1}",
            session_id=f"{_round_identity(manifest)}:implementation-revision-session-{attempt + 1}",
            prompt=repair_prompt,
            token_ceiling=int(manifest["frozen_execution"]["implementation_token_ceiling"]),
            maximum_physical_attempts=1,
        )
        revision_history[-1]["revision_request_digest"] = sha256_digest(repair_request)
        revision_history[-1]["revision_provider_attempts"] = repair_call.attempts
        if repair_call.call is None:
            revision_history[-1]["revision_status"] = "PROVIDER_FAILURE"
            break
        current_implementation = repair_call.call.response["proposals"][0]
        revision_history[-1]["revision_status"] = "PROVIDER_SUCCESS"
    if materialized is None or qualification is None:
        return _receipt(
            args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q4-materialize-qualifier-stage.v1",
                "status": "QUALIFICATION_FAILURE",
                "selected_candidate_id": spec.digest,
                "candidate_root": None,
                "qualification_status": None,
                "failure_detail": failure_detail,
                "revision_history": revision_history,
                "manual_candidate_patches": 0,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    candidate_root = materialized_root / "candidates" / "selected" / str(
        materialized.shared_request["blind_candidate_id"]
    )
    passed = qualification.receipt.status is QualificationStatusV1.PASS
    return _receipt(
        args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-materialize-qualifier-stage.v1",
            "status": "QUALIFICATION_PASS" if passed else "QUALIFICATION_FAILURE",
            "selected_candidate_id": spec.digest,
            "candidate_root": str(candidate_root),
            "candidate_package": materialized.to_dict(),
            "candidate_package_digest": materialized.package.digest,
            "candidate_source_tree_digest": materialized.package.source_tree_digest,
            "entrypoint": materialized.package.executable_entrypoint,
            "qualification": qualification.to_dict(),
            "qualification_status": qualification.receipt.status.value,
            "qualification_receipt_ref": qualification.receipt.receipt_id,
            "qualification_receipt_digest": qualification.receipt.digest,
            "behavioral_evidence": behavior,
            "revision_history": revision_history,
            "manual_candidate_patches": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_resource_admission(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import run_development_training

    manifest = _manifest(args)
    qualification = _read_upstream(args, "MATERIALIZE_QUALIFIER_RECEIPT.json")
    if qualification["status"] != "QUALIFICATION_PASS":
        return _receipt(
            args.round_root / "RESOURCE_ADMISSION_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q4-resource-admission-stage.v1",
                "status": "RESOURCE_DEFERRED_UPSTREAM_FAILURE",
                "admitted": False,
                "future_eligible": True,
                "resource_probe": None,
                "effect_update_allowed": False,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    prefix = args.repo_root / (
        "results/research_line/q0r_type_preserving_resource_scheduling_20260802_01/"
        "FIXED_BATCH_PREFIX_CONTRACT.json"
    )
    _verify_file(
        prefix,
        str(manifest["frozen_execution"]["resource_prefix_contract_sha256"]),
        "resource prefix contract",
    )
    candidate_root = Path(qualification["candidate_root"])
    source = candidate_root / "recclaw_ext/candidate.py"
    probe = run_development_training(
        repo_root=args.repo_root,
        side_root=args.round_root / "resource_probe",
        run_id="selected-candidate-prefix",
        seed=int(manifest["frozen_execution"]["resource_probe_seed"]),
        candidate_root=candidate_root,
        entrypoint=str(qualification["entrypoint"]),
        source_sha256=bytes_sha256(source.read_bytes()),
        run_identity=_round_identity(manifest),
        authority="user-delegated-q4-multiround-resource-admission",
        timeout_seconds=int(manifest["frozen_execution"]["resource_deadline_seconds"]),
        epochs=int(manifest["frozen_execution"]["resource_probe_epochs"]),
        execution_purpose="RESOURCE_PROBE_ONLY",
        resource_telemetry=True,
        watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
        prefix_contract_path=prefix,
        recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
    )
    admitted = probe.get("exit_status") == "SUCCESS"
    status = "RESOURCE_ADMITTED" if admitted else (
        "RESOURCE_CENSORED" if probe.get("exit_status") == "RESOURCE_CENSORED" else "RESOURCE_DEFERRED"
    )
    return _receipt(
        args.round_root / "RESOURCE_ADMISSION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-resource-admission-stage.v1",
            "status": status,
            "admitted": admitted,
            "future_eligible": not admitted,
            "resource_probe": probe,
            "effect_update_allowed": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def _conversion_parent(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    *,
    kind: str,
    seed: int,
    epochs: int,
    timeout_seconds: int,
    execution_purpose: str,
) -> tuple[dict[str, Any], bool]:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import run_development_training

    contract = manifest["conversion_efficiency"]
    parent_root = Path(str(contract["shared_parent_root"])) / kind / f"seed-{seed}"
    receipt_path = parent_root / "PARENT_RECEIPT.json"
    source = args.recbole_root / "recbole/model/general_recommender/bpr.py"
    source_digest = bytes_sha256(source.read_bytes())
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if (
            receipt.get("seed") != int(seed)
            or receipt.get("epochs") != int(epochs)
            or receipt.get("source_sha256") != source_digest
        ):
            raise RuntimeError("conversion shared parent binding drift")
        return receipt, True
    parent = run_development_training(
        repo_root=args.repo_root,
        side_root=parent_root,
        run_id=f"matched-bpr-{kind}-{seed}",
        seed=int(seed),
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=source_digest,
        run_identity=_round_identity(manifest),
        authority="user-delegated-q5b-conversion-shared-parent",
        timeout_seconds=timeout_seconds,
        epochs=int(epochs),
        execution_purpose=execution_purpose,
        watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
        recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
    )
    receipt = {
        "schema": "recclaw.research-line.q5-conversion-shared-parent.v1",
        "kind": kind,
        "seed": int(seed),
        "epochs": int(epochs),
        "source_sha256": source_digest,
        "run": parent,
        "held_out_reads": 0,
        "development_only": True,
        "scientific_effect_claim": False,
    }
    _receipt(receipt_path, receipt)
    return _read(receipt_path), False


def _run_conversion_screen(args: argparse.Namespace, manifest: Mapping[str, Any]) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import run_development_training

    contract = manifest["conversion_efficiency"]
    screen = contract["screen"]
    seed = int(screen["seed"])
    epochs = int(screen["epochs"])
    admission = _read(args.round_root / "RESOURCE_ADMISSION_RECEIPT.json")
    if not admission.get("admitted"):
        return _receipt(
            args.round_root / "MATCHED_EXECUTION_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q5-conversion-screen-stage.v1",
                "status": "NO_LEGAL_ADMITTED_ARM",
                "baseline": None,
                "candidate": None,
                "matched_seed": seed,
                "screen_epochs": epochs,
                "screen_signal": None,
                "stable": False,
                "parent_reused": False,
                "physical_training_runs": 0,
                "retries": 0,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    parent, parent_reused = _conversion_parent(
        args,
        manifest,
        kind="screen",
        seed=seed,
        epochs=epochs,
        timeout_seconds=int(manifest["frozen_execution"]["training_deadline_seconds"]),
        execution_purpose="DEVELOPMENT_SCREEN_PARENT",
    )
    qualification = _read(args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json")
    candidate_root = Path(qualification["candidate_root"])
    source = candidate_root / "recclaw_ext/candidate.py"
    candidate = run_development_training(
        repo_root=args.repo_root,
        side_root=args.round_root / "matched_execution" / "screen",
        run_id="selected-candidate-screen",
        seed=seed,
        candidate_root=candidate_root,
        entrypoint=str(qualification["entrypoint"]),
        source_sha256=bytes_sha256(source.read_bytes()),
        run_identity=_round_identity(manifest),
        authority="user-delegated-q5b-conversion-screen",
        timeout_seconds=int(manifest["frozen_execution"]["training_deadline_seconds"]),
        epochs=epochs,
        execution_purpose="DEVELOPMENT_SCREEN_CANDIDATE",
        watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
        recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
    )
    closed = all(
        run.get("exit_status") == "SUCCESS" and "ndcg@10" in run.get("metrics", {})
        for run in (parent["run"], candidate)
    )
    signal = None
    if closed:
        signal = float(candidate["metrics"]["ndcg@10"]) - float(parent["run"]["metrics"]["ndcg@10"])
    return _receipt(
        args.round_root / "MATCHED_EXECUTION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q5-conversion-screen-stage.v1",
            "status": "COMPLETED_MATCHED_SCREEN" if closed else "MATCHED_PAIR_INCOMPLETE",
            "baseline": parent["run"],
            "candidate": candidate,
            "matched_seed": seed,
            "screen_epochs": epochs,
            "screen_signal": signal,
            "stable": closed,
            "parent_reused": parent_reused,
            "physical_training_runs": 1 + (0 if parent_reused else 1),
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_matched_execution(args: argparse.Namespace) -> Path:
    manifest = _manifest(args)
    if manifest.get("conversion_efficiency") is not None:
        return _run_conversion_screen(args, manifest)
    return _run_standard_matched_execution(args)


def _run_standard_matched_execution(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import run_development_training

    manifest = _manifest(args)
    admission = _read(args.round_root / "RESOURCE_ADMISSION_RECEIPT.json")
    qualification = _read_upstream(args, "MATERIALIZE_QUALIFIER_RECEIPT.json")
    if not admission["admitted"]:
        return _receipt(
            args.round_root / "MATCHED_EXECUTION_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q4-matched-execution-stage.v1",
                "status": "NO_LEGAL_ADMITTED_ARM",
                "baseline": None,
                "candidate": None,
                "physical_training_runs": 0,
                "retries": 0,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    seed = int(manifest["frozen_execution"]["training_seed"])
    deadline = int(manifest["frozen_execution"]["training_deadline_seconds"])
    epochs = int(manifest["frozen_execution"]["training_epochs"])
    identity = _round_identity(manifest)
    baseline_source = args.recbole_root / "recbole/model/general_recommender/bpr.py"
    baseline = run_development_training(
        repo_root=args.repo_root,
        side_root=args.round_root / "matched_execution",
        run_id="matched-bpr-control",
        seed=seed,
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=bytes_sha256(baseline_source.read_bytes()),
        run_identity=identity,
        authority="user-delegated-q4-multiround-matched-development",
        timeout_seconds=deadline,
        epochs=epochs,
        watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
        recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
    )
    candidate_root = Path(qualification["candidate_root"])
    source = candidate_root / "recclaw_ext/candidate.py"
    candidate = run_development_training(
        repo_root=args.repo_root,
        side_root=args.round_root / "matched_execution",
        run_id="selected-candidate",
        seed=seed,
        candidate_root=candidate_root,
        entrypoint=str(qualification["entrypoint"]),
        source_sha256=bytes_sha256(source.read_bytes()),
        run_identity=identity,
        authority="user-delegated-q4-multiround-matched-development",
        timeout_seconds=deadline,
        epochs=epochs,
        watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
        recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
    )
    closed = all(
        run.get("exit_status") == "SUCCESS" and "ndcg@10" in run.get("metrics", {})
        for run in (baseline, candidate)
    )
    return _receipt(
        args.round_root / "MATCHED_EXECUTION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-matched-execution-stage.v1",
            "status": "COMPLETED_MATCHED_PAIR" if closed else "MATCHED_PAIR_INCOMPLETE",
            "baseline": baseline,
            "candidate": candidate,
            "physical_training_runs": 2,
            "serial_execution": True,
            "matched_seed": seed,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_full_execution(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256
    from recclaw_core.experiments.helix_abc_v1.fresh_r1 import run_development_training

    manifest = _manifest(args)
    contract = manifest["conversion_efficiency"]
    candidate_id = str(manifest["selected_candidate"]["candidate_id"])
    campaign_root = args.round_root.parents[1]
    promotion_path = campaign_root / "CONVERSION_PROMOTION_PLAN.json"
    promotion = _read(promotion_path)
    promoted = tuple(str(value) for value in promotion["promotion"]["promoted_candidate_ids"])
    if candidate_id not in promoted:
        return _receipt(
            args.round_root / "FULL_MATCHED_EXECUTION_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q5-conversion-full-stage.v1",
                "status": "NOT_PROMOTED_AFTER_SCREEN",
                "candidate_id": candidate_id,
                "full_seed_results": [],
                "physical_training_runs": 0,
                "retries": 0,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    qualification = _read(args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json")
    candidate_root = Path(qualification["candidate_root"])
    source = candidate_root / "recclaw_ext/candidate.py"
    full = contract["full"]
    full_deadline = int(
        promotion["promotion"]["full_deadline_seconds_by_candidate"][candidate_id]
    )
    parent_deadline = int(promotion["promotion"]["shared_parent_deadline_seconds"])
    results: list[dict[str, Any]] = []
    physical_runs = 0
    for seed in tuple(int(value) for value in full["fresh_development_seeds"]):
        parent, parent_reused = _conversion_parent(
            args,
            manifest,
            kind="full",
            seed=seed,
            epochs=int(full["epochs"]),
            timeout_seconds=parent_deadline,
            execution_purpose="DEVELOPMENT_FULL_PARENT",
        )
        candidate = run_development_training(
            repo_root=args.repo_root,
            side_root=args.round_root / "matched_execution" / f"full_seed_{seed}",
            run_id=f"selected-candidate-full-{seed}",
            seed=seed,
            candidate_root=candidate_root,
            entrypoint=str(qualification["entrypoint"]),
            source_sha256=bytes_sha256(source.read_bytes()),
            run_identity=_round_identity(manifest),
            authority="user-delegated-q5b-conversion-full-development",
            timeout_seconds=full_deadline,
            epochs=int(full["epochs"]),
            execution_purpose="DEVELOPMENT_FULL_CANDIDATE",
            watchdog_seconds=int(manifest["frozen_execution"]["engineering_watchdog_seconds"]),
            recbole_commit_identity=REQUIRED_RECBOLE_COMMIT,
        )
        physical_runs += 1 + (0 if parent_reused else 1)
        results.append(
            {
                "seed": seed,
                "baseline": parent["run"],
                "candidate": candidate,
                "parent_reused": parent_reused,
                "status": (
                    "COMPLETED_MATCHED_PAIR"
                    if parent["run"].get("exit_status") == "SUCCESS"
                    and candidate.get("exit_status") == "SUCCESS"
                    and "ndcg@10" in parent["run"].get("metrics", {})
                    and "ndcg@10" in candidate.get("metrics", {})
                    else "MATCHED_PAIR_INCOMPLETE"
                ),
            }
        )
    complete = all(row["status"] == "COMPLETED_MATCHED_PAIR" for row in results)
    first = results[0] if results else None
    return _receipt(
        args.round_root / "FULL_MATCHED_EXECUTION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q5-conversion-full-stage.v1",
            "status": "COMPLETED_MATCHED_PAIR" if complete else "MATCHED_PAIR_INCOMPLETE",
            "candidate_id": candidate_id,
            "baseline": first["baseline"] if first else None,
            "candidate": first["candidate"] if first else None,
            "matched_seed": first["seed"] if first else None,
            "full_epochs": int(full["epochs"]),
            "full_seed_results": results,
            "physical_training_runs": physical_runs,
            "screen_outcomes_reused_as_effect": False,
            "fresh_development_seeds": [row["seed"] for row in results],
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_episode(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest
    from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
        EpisodeEvidenceClassV1,
        ResearchFailureClassV1,
        TypedResearchEpisodeV1,
    )

    manifest = _manifest(args)
    _arm, record = _selected_record(manifest, args.input_pool)
    spec = _rehydrate_spec(record)
    qualification = _read_upstream(args, "MATERIALIZE_QUALIFIER_RECEIPT.json")
    conversion = manifest.get("conversion_efficiency")
    matched_path = args.round_root / "MATCHED_EXECUTION_RECEIPT.json"
    if conversion is not None and (args.round_root / "FULL_MATCHED_EXECUTION_RECEIPT.json").is_file():
        matched_path = args.round_root / "FULL_MATCHED_EXECUTION_RECEIPT.json"
    matched = _read(matched_path)
    if matched["status"] != "COMPLETED_MATCHED_PAIR":
        return _receipt(
            args.round_root / "EPISODE_RECEIPT.json",
            {
                "schema": "recclaw.research-line.q4-episode-stage.v1",
                "status": "EPISODE_MISSING",
                "missingness": (
                    "RESOURCE_OR_UPSTREAM_DEFERRED"
                    if matched["status"] == "NO_LEGAL_ADMITTED_ARM"
                    else "SCREEN_NOT_PROMOTED"
                    if matched["status"] == "COMPLETED_MATCHED_SCREEN"
                    else "INCOMPLETE_MATCHED_DEVELOPMENT_EXECUTION"
                ),
                "typed_episode": None,
                "effect_update_allowed": False,
                "held_out_reads": 0,
                "development_only": True,
                "scientific_effect_claim": False,
            },
        )
    capability = canonical_value(
        {
            "candidate_package_digest": qualification["candidate_package_digest"],
            "source_tree_digest": qualification["candidate_source_tree_digest"],
            "entrypoint": qualification["entrypoint"],
            "qualification_receipt_digest": qualification["qualification_receipt_digest"],
        }
    )
    identity = _round_identity(manifest)
    seed_results = matched.get("full_seed_results")
    if not isinstance(seed_results, list) or not seed_results:
        seed_results = [
            {
                "seed": matched["matched_seed"],
                "baseline": matched["baseline"],
                "candidate": matched["candidate"],
            }
        ]
    episodes = []
    outcomes = []
    costs = []
    for result in seed_results:
        baseline = result["baseline"]
        candidate = result["candidate"]
        seed = int(result["seed"])
        outcome = canonical_value(
            {
                "baseline_metrics": baseline["metrics"],
                "candidate_metrics": candidate["metrics"],
                "metric": "ndcg@10",
                "partition": "DEVELOPMENT_VALIDATION",
                "seed": seed,
                "single_seed_interpretation": "INCONCLUSIVE",
            }
        )
        cost = canonical_value(
            {
                "baseline_wall_time_ms": baseline["wall_time_ms"],
                "candidate_wall_time_ms": candidate["wall_time_ms"],
                "physical_training_runs": 2,
            }
        )
        binding = canonical_value(
            {
                "baseline_binding_digest": baseline["binding_digest"],
                "candidate_binding_digest": candidate["binding_digest"],
                "matched_seed": seed,
                "protocol_digest": spec.protocol_digest,
            }
        )
        episode_identity = (
            identity if len(seed_results) == 1 else f"{identity}-seed-{seed}"
        )
        episode = TypedResearchEpisodeV1(
            campaign_id=episode_identity,
            context_ref=spec.context_ref,
            context_digest=spec.context_digest,
            hypothesis=spec.hypothesis,
            executable_capability_ref=f"{episode_identity}-qualified:{sha256_digest(capability)}",
            executable_capability_digest=sha256_digest(capability),
            executable_profile_ref=f"{episode_identity}-profile:{sha256_digest(capability)}",
            executable_profile_digest=sha256_digest(capability),
            experiment_binding_ref=f"{episode_identity}-binding:{sha256_digest(binding)}",
            experiment_binding_digest=sha256_digest(binding),
            comparator_ref=f"{episode_identity}-bpr:{baseline['binding_digest']}",
            comparator_digest=sha256_digest(baseline),
            outcome_ref=f"{episode_identity}-outcome:{sha256_digest(outcome)}",
            outcome_digest=sha256_digest(outcome),
            cost_ref=f"{episode_identity}-cost:{sha256_digest(cost)}",
            cost_digest=sha256_digest(cost),
            protocol_ref=spec.protocol_ref,
            protocol_digest=spec.protocol_digest,
            evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
            experiment_executed=True,
            mechanism_interpretation="NOT_ADJUDICATED",
            competing_explanation=spec.competing_explanation,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            mechanism_negative_evidence=False,
            next_discriminative_test=spec.falsifier,
            qualification_receipt_ref=qualification["qualification_receipt_ref"],
            qualification_receipt_digest=qualification["qualification_receipt_digest"],
            qualification_evidence_used_as_scientific=False,
        )
        episodes.append(episode.canonical_dict())
        outcomes.append(outcome)
        costs.append(cost)
    return _receipt(
        args.round_root / "EPISODE_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-episode-stage.v1",
            "status": "EPISODE_CREATED",
            "missingness": None,
            "typed_episode": episodes[0] if len(episodes) == 1 else None,
            "typed_episodes": episodes,
            "outcome_summary": outcomes[0] if len(outcomes) == 1 else None,
            "outcome_summaries": outcomes,
            "cost_summaries": costs,
            "full_seed_count": len(episodes),
            "effect_update_allowed": True,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        },
    )


def run_mechanism_probe(args: argparse.Namespace) -> Path:
    """Classify the real qualifier and Q0R2 probe before full outcome."""

    from recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison import (
        classify_mechanism_probe,
    )

    qualification = _read_upstream(args, "MATERIALIZE_QUALIFIER_RECEIPT.json")
    admission = _read(args.round_root / "RESOURCE_ADMISSION_RECEIPT.json")
    payload = classify_mechanism_probe(
        qualification=qualification,
        admission=admission,
    )
    return _receipt(args.round_root / "MECHANISM_PROBE_RECEIPT.json", payload)


def run_authority_update(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest
    from recclaw_core.experiments.helix_abc_v1.open_meta_q3 import (
        _projection_row,
        _resource_head_input,
        evaluate_q3_development_activation,
        fit_q3_three_head_policy,
        run_group_aware_offline_replay,
    )

    manifest = _manifest(args)
    _verify_file(
        args.previous_projection,
        str(manifest["frozen_execution"]["previous_projection_sha256"]),
        "previous projection",
    )
    _verify_file(
        args.previous_policy,
        str(manifest["frozen_inputs"]["policy_file_sha256"]),
        "previous policy",
    )
    previous = _read(args.previous_projection)
    previous_policy = _read(args.previous_policy)
    admission = _read(args.round_root / "RESOURCE_ADMISSION_RECEIPT.json")
    episode = _read(args.round_root / "EPISODE_RECEIPT.json")
    acquisition = manifest["task_acquisitions"][manifest["execution_task_type"]]
    selected_id = str(manifest["selected_candidate"]["candidate_id"])
    new_rows = []
    for candidate in acquisition["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        selected = candidate_id == selected_id
        feasibility_input = None
        feasibility_reason = "UNSELECTED_DENOMINATOR_ROW_NO_RESOURCE_OBSERVATION"
        if selected and admission.get("resource_probe") is not None:
            feasibility_input = _resource_head_input(
                admission["resource_probe"],
                resource_stage=f"Q4_ROUND_{manifest['round_index']}_RESOURCE_ADMISSION",
            )
            feasibility_reason = "OBSERVED_RESOURCE_ADMISSION_PROBE"
        elif selected and admission["status"] == "RESOURCE_DEFERRED_UPSTREAM_FAILURE":
            qualification = _read(args.round_root / "MATERIALIZE_QUALIFIER_RECEIPT.json")
            if qualification["status"] == "QUALIFICATION_FAILURE":
                feasibility_input = _resource_head_input(
                    {"status": "QUALIFICATION_FAILURE"},
                    resource_stage=f"Q4_ROUND_{manifest['round_index']}_QUALIFICATION",
                    physical_run=True,
                )
                feasibility_reason = "OBSERVED_QUALIFICATION_FAILURE"
        effect_input = None
        effect_reason = "NO_COMPLETE_COMPARABLE_MATCHED_EFFECT"
        if selected and episode["status"] == "EPISODE_CREATED":
            summary = episode["outcome_summary"]
            baseline_value = float(summary["baseline_metrics"]["ndcg@10"])
            candidate_value = float(summary["candidate_metrics"]["ndcg@10"])
            effect_input = canonical_value(
                {
                    "effect_target": "PARENT_RELATIVE_DEVELOPMENT_NDCG_AT_10",
                    "parent_relative_effect": candidate_value - baseline_value,
                    "control_metric": baseline_value,
                    "candidate_metric": candidate_value,
                    "evidence_weight": 0.5,
                    "evidence_class": "INCONCLUSIVE_EXPERIMENT",
                    "mechanism_interpretation": "NOT_ADJUDICATED",
                    "comparability": "FULL_MATCHED_FRESH_DEVELOPMENT_EPISODE",
                    "research_features": {
                        "direction": candidate["candidate_features"]["direction"],
                        "round_index": manifest["round_index"],
                    },
                }
            )
            effect_reason = "FULL_COMPARABLE_FRESH_MATCHED_DEVELOPMENT_EPISODE"
        new_rows.append(
            _projection_row(
                row_id=f"q4-round-{manifest['round_index']}/{candidate_id}",
                source_stage=f"Q4_ROUND_{manifest['round_index']}",
                group_id=f"q4-pool/{candidate_id}",
                feasibility_input=feasibility_input,
                feasibility_reason=feasibility_reason,
                mechanism_input=None,
                mechanism_reason="NO_REAL_MECHANISM_PROBE_OR_ABLATION_LABEL",
                effect_input=effect_input,
                effect_reason=effect_reason,
                audit={
                    "candidate_identity_feature": False,
                    "selected": selected,
                    "selection_probability": candidate["selection_probability"],
                    "missingness": None if effect_input is not None else episode["missingness"],
                },
            )
        )
    rows = [*previous["rows"], *new_rows]
    source_counts = Counter(str(row["source_stage"]) for row in rows)
    head_counts = {
        head: sum(bool(row["head_authority"][head]["allowed"]) for row in rows)
        for head in ("feasibility", "mechanism_information", "effect")
    }
    negative = dict(previous["negative_evidence_preserved"])
    if admission["status"] == "RESOURCE_CENSORED":
        negative["resource_censored_count"] += 1
    elif admission["status"].startswith("RESOURCE_DEFERRED"):
        negative["resource_deferred_count"] += 1
    projection_payload = canonical_value(
        {
            "schema": "recclaw.research-line.q4-multiround-projection.v1",
            "rows": rows,
            "row_count": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "head_update_counts": head_counts,
            "negative_evidence_preserved": negative,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    projection = {**projection_payload, "projection_digest": sha256_digest(projection_payload)}
    policy = fit_q3_three_head_policy(
        projection,
        parent_policy_digest=previous_policy["policy_digest"],
        policy_version=f"q4-multiround-round-{manifest['round_index']}",
    )
    replay = run_group_aware_offline_replay(
        projection, parent_policy_digest=previous_policy["policy_digest"]
    )
    shadow = {
        "same_pool_and_budget_within_each_task": True,
        "held_out_reads": 0,
        "policy_superiority_claim": False,
        "scientific_effect_claim": False,
    }
    promotion = evaluate_q3_development_activation(projection, replay, shadow, policy)
    output = args.round_root / "policy_update"
    _write_new(output / "denominator_projection.json", projection)
    _write_new(output / "versioned_policy.json", policy)
    _write_new(output / "group_aware_replay.json", replay)
    _write_new(output / "development_promotion.json", promotion)
    return _receipt(
        args.round_root / "AUTHORITY_UPDATE_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-authority-update-stage.v1",
            "status": promotion["status"],
            "projection_digest": projection["projection_digest"],
            "policy_digest": policy["policy_digest"],
            "parent_policy_digest": policy["parent_policy_digest"],
            "head_update_counts": head_counts,
            "round_denominator_count": len(new_rows),
            "round_effect_update_count": sum(
                bool(row["head_authority"]["effect"]["allowed"]) for row in new_rows
            ),
            "round_mechanism_information_update_count": 0,
            "resource_status_updates_effect": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
            "policy_superiority_claim": False,
        },
    )


def run_activation(args: argparse.Namespace) -> Path:
    from recclaw_core.experiments.helix_abc_v1.open_meta_q3 import build_q3_policy_activation

    manifest = _manifest(args)
    root = args.round_root / "policy_update"
    projection = _read(root / "denominator_projection.json")
    policy = _read(root / "versioned_policy.json")
    promotion = _read(root / "development_promotion.json")
    activation = build_q3_policy_activation(
        policy,
        promotion,
        projection_digest=projection["projection_digest"],
        activation_id=f"{_round_identity(manifest)}-activation",
    )
    _write_new(root / "active_policy.json", activation)
    return _receipt(
        args.round_root / "POLICY_ACTIVATION_RECEIPT.json",
        {
            "schema": "recclaw.research-line.q4-policy-activation-stage.v1",
            "status": activation["status"],
            "activation_digest": activation["activation_digest"],
            "policy_digest": activation["policy_digest"],
            "parent_policy_digest": activation["parent_policy_digest"],
            "rollback_policy_digest": activation["rollback_policy_digest"],
            "reversible": activation["reversible"],
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
            "policy_superiority_claim": False,
        },
    )


STAGES = {
    "preflight": run_preflight,
    "provider-resolver": run_provider_resolver,
    "selected-resolver": run_selected_resolver,
    "implementer": run_implementer,
    "materialize-qualifier": run_materialize_qualifier,
    "resource-admission": run_resource_admission,
    "mechanism-probe": run_mechanism_probe,
    "matched-execution": run_matched_execution,
    "full-execution": run_full_execution,
    "episode": run_episode,
    "authority-update": run_authority_update,
    "activation": run_activation,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-pool", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--previous-policy", type=Path)
    parser.add_argument("--previous-projection", type=Path)
    parser.add_argument("--projects-root", type=Path, required=True)
    parser.add_argument("--search-data-root", type=Path, required=True)
    parser.add_argument("--recbole-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "repo_root",
        "round_root",
        "manifest",
        "input_pool",
        "upstream_root",
        "previous_policy",
        "previous_projection",
        "projects_root",
        "search_data_root",
        "recbole_root",
        "python_executable",
        "api_config",
    ):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    _bootstrap(args)
    started_ns = time.monotonic_ns()
    result = STAGES[args.stage](args)
    wall_time_ms = (time.monotonic_ns() - started_ns) // 1_000_000
    manifest = _manifest(args)
    if manifest.get("schema") == "recclaw.research-line.q4-prospective-arm-manifest.v1":
        from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256

        _receipt(
            args.round_root / "stage_costs" / f"{args.stage}.json",
            {
                "schema": "recclaw.research-line.q4-prospective-stage-cost.v1",
                "stage": args.stage,
                "policy_name": manifest.get("policy_name"),
                "wall_time_ms": wall_time_ms,
                "artifact": str(result),
                "artifact_sha256": bytes_sha256(result.read_bytes()),
                "held_out_reads": 0,
                "development_only": True,
            },
        )
    print(json.dumps({"stage": args.stage, "artifact": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
