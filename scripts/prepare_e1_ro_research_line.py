#!/usr/bin/env python3
"""Prepare the native RO E1 campaign; optionally execute smoke or authorized search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

E1_BASELINE_NDCG_AT_10 = 0.20394805
E1_BASELINE_RECALL_AT_10 = 0.15099746
E1_PARENT_MODEL_SHA256 = (
    "b35353db98222be1fd3117a5255e3be4e95f7fd94d9d59cfda10a605e1187339"
)
E1_SEARCH_MANIFEST_SHA256 = (
    "662f65d9ca5eb813cd2f7e705d11228fd828ab439215044023f879411b94b3bf"
)
E1_TRAINING_SEED = 54201
E1_CONTROLLER_SEED = 54202
E1_ROUND_COUNT = 12
E1_UNIT_TIMEOUT_SECONDS = 1800
E1_PROVIDER_PHYSICAL_CALL_CAP = 64
E1_PROVIDER_TOTAL_TOKEN_CAP = 512_000
E1_PROVIDER_TOKEN_CAP_PER_CALL = 64_000
E1_TRAINING_START_CAP = 12
E1_GPU_TIME_CAP_SECONDS = 4 * 60 * 60


class E1PreparationError(RuntimeError):
    """One explicit E1 preparation input is absent or identity-incompatible."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_sha256(value: str, *, field_name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise E1PreparationError(f"{field_name} must be one SHA-256 digest")
    return normalized


def _existing_file(path: Path, *, field_name: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise E1PreparationError(f"{field_name} is unavailable: {path}") from error
    if not resolved.is_file():
        raise E1PreparationError(f"{field_name} is not a file: {resolved}")
    return resolved


def _existing_directory(path: Path, *, field_name: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise E1PreparationError(f"{field_name} is unavailable: {path}") from error
    if not resolved.is_dir():
        raise E1PreparationError(f"{field_name} is not a directory: {resolved}")
    return resolved


def _configure_runtime(args: argparse.Namespace) -> Mapping[str, Path]:
    """Bind import-time worker paths before importing any Research modules."""

    data_root = _existing_directory(args.data_root, field_name="data_root")
    search_manifest = _existing_file(
        args.search_manifest,
        field_name="search_manifest",
    )
    baseline = _existing_file(args.baseline, field_name="baseline")
    worker_python = _existing_file(
        args.worker_python,
        field_name="worker_python",
    )
    recbole_root = _existing_directory(
        args.recbole_root,
        field_name="recbole_root",
    )
    recbole_init = _existing_file(
        recbole_root / "recbole" / "__init__.py",
        field_name="recbole package",
    )
    driver_dir = _existing_directory(args.driver_dir, field_name="driver_dir")
    nvidia_smi = _existing_file(
        driver_dir / "nvidia-smi",
        field_name="driver nvidia-smi",
    )
    api_config = _existing_file(args.api_config, field_name="api_config")

    os.environ["RECCLAW_PROJECTS_ROOT"] = str(REPO_ROOT.parent)
    os.environ["RECCLAW_SEARCH_DATA_ROOT"] = str(data_root)
    os.environ["RECCLAW_PYTHON_EXECUTABLE"] = str(worker_python)
    os.environ["RECCLAW_RECBOLE_ROOT"] = str(recbole_root)
    os.environ["RECCLAW_API_CONFIG"] = str(api_config)
    os.environ["RECCLAW_LAB_API_WIRE_API"] = "responses"
    os.environ["LD_LIBRARY_PATH"] = str(driver_dir)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    return {
        "api_config": api_config,
        "baseline": baseline,
        "data_root": data_root,
        "driver_dir": driver_dir,
        "nvidia_smi": nvidia_smi,
        "recbole_init": recbole_init,
        "recbole_root": recbole_root,
        "search_manifest": search_manifest,
        "worker_python": worker_python,
    }


def _read_fixed_baseline(
    path: Path,
    *,
    expected_sha256: str,
    manifest_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    expected = _normalized_sha256(
        expected_sha256,
        field_name="baseline_sha256",
    )
    observed = _sha256(path)
    if observed != expected:
        raise E1PreparationError("baseline artifact SHA-256 differs from input")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E1PreparationError("baseline artifact is not readable JSON") from error
    if not isinstance(payload, Mapping):
        raise E1PreparationError("baseline artifact must be a JSON object")
    if payload.get("status") != "ok":
        raise E1PreparationError("baseline artifact is not a successful result")
    if payload.get("source_sha256") != E1_PARENT_MODEL_SHA256:
        raise E1PreparationError("baseline does not use the frozen E1 parent source")
    ndcg = payload.get("ndcg_at_10")
    recall = payload.get("recall_at_10")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (ndcg, recall)
    ):
        raise E1PreparationError("baseline development metrics are unavailable")
    if (
        float(ndcg) != E1_BASELINE_NDCG_AT_10
        or float(recall) != E1_BASELINE_RECALL_AT_10
    ):
        raise E1PreparationError("baseline development metrics differ from calibration")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get(
        "search_manifest_sha256"
    ) != manifest_sha256:
        raise E1PreparationError("baseline search-manifest identity differs")
    return payload, observed


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def preview_initial_requests(composition, output: Path, adapter) -> Mapping[str, Any]:
    """Construct the selected mode's initial prompts without physical work."""
    import tiktoken
    from recclaw_core.research_line.interfaces import research_producer_roles
    from recclaw_core.research_line.runtime import (
        _exact_lineage_parent_bundle, _lineage_parent_mechanism_program,
        _latest_completed_execution,
    )
    from recclaw_core.research_line.e1_native_bridge import E1_ALLOWED_FILES
    from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1

    class Captured(BaseException):
        pass

    actor, state = composition.provider, composition.campaign.state
    context = state.context
    parent = _lineage_parent_mechanism_program(
        context, state.carryover_open_candidates, search_space_adapter=adapter)
    bundle = _exact_lineage_parent_bundle(
        context, state.carryover_open_candidates,
        candidate_root_by_capability=state.candidate_root_by_capability,
        allowed_files=E1_ALLOWED_FILES, search_space_adapter=adapter)
    if parent is not None and bundle is not None:
        parent = {**parent, 'source_bundle': bundle}
    latest = _latest_completed_execution(context, allowed_files=E1_ALLOWED_FILES)
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise Captured()

    original_call = actor.provider_call
    actor.provider_call = capture
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for role in research_producer_roles(context.budget):
            view = {**context.producer_view(role), 'active_task_directive': None}
            if parent is not None:
                view['lineage_parent_mechanism_program'] = parent
            if latest is not None:
                view['latest_completed_execution'] = latest
            if role == 'frontier_architect' and composition.config.research_mode == 'portfolio':
                view['research_portfolio'] = ()
            captured.clear()
            try:
                actor(role, view)
            except Captured:
                pass
            broker = object.__new__(LabApiCanaryBrokerV1)
            broker.max_total_tokens_per_call = 64000
            broker.wire_api = 'responses'
            broker.model = 'gpt-5.6-terra' if role == 'frontier_architect' else 'gpt-5.6-luna'
            broker.reasoning_effort = 'medium'
            broker.schema = json.loads(Path(captured['schema_path']).read_text())
            broker.request_budget = object()
            payload = broker.sealed_request_identity(
                proposal_generation_session_id='offline-preview',
                prompt=captured['prompt'], expected_proposal_count=1,
                max_total_tokens=64000,
                max_output_tokens=captured['output_token_ceiling'])['request_payload']
            _write_json(output / f'{role}.json', payload)
            count = len(tiktoken.encoding_for_model('gpt-5').encode(
                json.dumps(payload, ensure_ascii=False), disallowed_special=()))
            rows.append(dict(role=role, input_tokens_estimate=count,
                output_ceiling=payload['max_output_tokens'],
                evidence_class=('INITIAL_PROMPT_WITHOUT_UNKNOWN_PEER_PORTFOLIO'
                    if role == 'frontier_architect' and composition.config.research_mode == 'portfolio'
                    else 'INITIAL_CONSTRUCTED_REQUEST'),
                request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()))
    finally:
        actor.provider_call = original_call
    result = dict(api_calls=0, training_calls=0, context_digest=context.digest,
                  research_mode=composition.config.research_mode,
                  tokenizer_version=tiktoken.__version__, requests=rows,
                  note='Input estimates, not reported usage. Later-round context, output lengths '
                       'and revision needs are not yet known; portfolio-mode architect peer outputs '
                       'are unavailable at initial preparation.')
    _write_json(output / 'preview.json', result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-id", default="e1-ro-research-line")
    parser.add_argument("--research-mode", choices=("portfolio", "director_sequential"),
                        default="portfolio", help="Explicit frozen research-call topology")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--search-manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-context", type=Path, required=True)
    parser.add_argument("--upstream-release", required=True)
    parser.add_argument("--preview-prompts", type=Path,
                        help="Prepare and export initial requests offline; never invoke a provider or worker")
    parser.add_argument(
        "--baseline-sha256",
        required=True,
        help="exact fixed calibration-result artifact SHA-256",
    )
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--recbole-root", type=Path, required=True)
    parser.add_argument("--driver-dir", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--credential-index", type=int, default=1,
                        help="zero-based credential entry; use 0 for endpoint1-only files")
    parser.add_argument("--comparison-root", type=Path, required=True,
                        help="RecHarness E1 checkout containing the common experiment supervisor")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--smoke-run", action="store_true")
    parser.add_argument("--search-run", action="store_true")
    parser.add_argument("--canary-run", action="store_true",
                        help="Bounded engineering check, not a controller comparison")
    parser.add_argument("--canary-rounds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--resume", action="store_true",
                        help="resume the selected smoke or search from its native checkpoint")
    parser.add_argument("--smoke-provider-only", action="store_true",
                        help="one request from the first active research role, no campaign round")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sum((args.smoke_run, args.smoke_provider_only, args.search_run, args.canary_run)) > 1:
        raise E1PreparationError('Choose one execution scope')
    if args.preview_prompts and any((args.smoke_run, args.smoke_provider_only, args.search_run, args.canary_run)):
        raise E1PreparationError('Prompt preview is preparation-only')
    if args.resume and not (args.smoke_run or args.search_run or args.canary_run):
        raise E1PreparationError('--resume requires --smoke-run or --search-run')
    if args.gpu_id < 0:
        raise E1PreparationError("gpu_id must be non-negative")
    runtime_paths = _configure_runtime(args)
    comparison_root = _existing_directory(args.comparison_root, field_name="comparison_root")
    sys.path.insert(0, str(comparison_root))
    from gagc.e1_runtime import Budget, MeteredQualification, load_pilot_context
    from gagc.e1_ro_runtime import MeteredROTraining, MeteredROResourceProbe, metered_ro_provider
    from gagc.e1_worker import WorkerBoundary, resolve_gpu_uuid

    # These modules bind worker paths and wire API at import time.
    from recclaw_core.experiments.helix_abc_v1 import fresh_r1
    from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
    from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
        DEVELOPMENT_EVALUATOR,
        DEVELOPMENT_SPLIT,
    )
    from recclaw_core.experiments.helix_abc_v1.provider_model_routing import (
        model_routing_manifest,
    )
    from recclaw_core.research_line.e1_multvae_search_space_adapter import (
        E1_DIRECTOR_PROMPT_PATH,
        E1_PARENT_ID,
        E1_PARENT_PROGRAM_DIGEST,
        E1MultVAESearchSpaceAdapterV1,
        e1_baseline_context,
        e1_frozen_profile_ref,
    )
    from recclaw_core.research_line.e1_native_bridge import (
        E1_PROTOCOL_DIGEST,
        E1_PROTOCOL_REF,
        E1_RECBOLE_COMMIT,
        parent_package_identity,
        verify_search_manifest,
    )
    from recclaw_core.research_line.single_round import ResearchBaselineSourceV1
    from recclaw_core.research_line.interfaces import research_producer_roles
    from recclaw_core.research_line.standalone import (
        StandaloneResearchConfig,
        compose_standalone_campaign,
    )

    expected_import_paths = {
        "API_CONFIG": runtime_paths["api_config"],
        "PYTHON_EXECUTABLE": runtime_paths["worker_python"],
        "RECBole_ROOT": runtime_paths["recbole_root"],
        "SEARCH_DATA_ROOT": runtime_paths["data_root"],
    }
    for constant_name, expected_path in expected_import_paths.items():
        if Path(getattr(fresh_r1, constant_name)).resolve() != expected_path:
            raise E1PreparationError(
                f"fresh_r1 import-time {constant_name} differs from explicit runtime"
            )
    if fresh_r1.LAB_API_WIRE_API != "responses":
        raise E1PreparationError("RO Provider wire API is not Responses")

    manifest = verify_search_manifest(
        manifest_path=runtime_paths["search_manifest"],
        data_root=runtime_paths["data_root"],
    )
    if manifest["manifest_sha256"] != E1_SEARCH_MANIFEST_SHA256:
        raise E1PreparationError("E1 runtime search manifest bytes differ")
    baseline, baseline_sha256 = _read_fixed_baseline(
        runtime_paths["baseline"],
        expected_sha256=args.baseline_sha256,
        manifest_sha256=manifest["manifest_sha256"],
    )
    pilot_context = load_pilot_context(
        args.pilot_context,
        (REPO_ROOT / "e1_native_parent/recclaw_ext/models/e1_multvae.py").read_text(),
    )
    baseline_context = e1_baseline_context(
        ndcg_at_10=E1_BASELINE_NDCG_AT_10,
        recall_at_10=E1_BASELINE_RECALL_AT_10,
        result_ref=str(runtime_paths["baseline"]),
        result_sha256=baseline_sha256,
        seed=E1_TRAINING_SEED,
    )
    baseline_source = ResearchBaselineSourceV1.from_identity(
        source_ref=str(runtime_paths["baseline"]),
        source_sha256=baseline_sha256,
        comparator_ref=E1_PARENT_ID,
        comparator_digest=E1_PARENT_PROGRAM_DIGEST,
        frozen_ndcg_at_10=E1_BASELINE_NDCG_AT_10,
        protocol_digest=E1_PROTOCOL_DIGEST,
        seed=E1_TRAINING_SEED,
    )

    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    campaign_root = run_root / "campaign"
    gpu_uuid = resolve_gpu_uuid(runtime_paths['driver_dir'], args.gpu_id)
    boundary = WorkerBoundary(runtime_paths['worker_python'], REPO_ROOT,
        runtime_paths['recbole_root'], runtime_paths['driver_dir'], run_root / 'worker_control', gpu_uuid)
    api_key = None
    if args.smoke_run or args.smoke_provider_only or args.search_run or args.canary_run:
        from recclaw_core.experiments.helix_abc_v1.lab_api_broker import load_lab_api_credentials
        _base_url, api_key = load_lab_api_credentials(runtime_paths['api_config'], args.credential_index)
    budget = Budget(run_root / "trusted_budget.json", **(
        dict(max_calls=12, max_tokens=192000, max_starts=args.canary_rounds, max_gpu_secs=3600)
        if args.canary_run else {}
    ))
    round_count = args.canary_rounds if args.canary_run else E1_ROUND_COUNT
    if not args.resume and (budget.state["calls"] or budget.state["starts"]):
        raise E1PreparationError("prepare-only requires an unused experiment budget")
    budget.save()
    config = StandaloneResearchConfig(
        repo_root=REPO_ROOT,
        run_root=campaign_root,
        api_config_source=runtime_paths["api_config"],
        campaign_id=str(args.campaign_id),
        research_mode=args.research_mode,
        native_resource_observation=True,
        baseline_source=baseline_source,
        seed=E1_TRAINING_SEED,
        search_seed=E1_CONTROLLER_SEED,
        epochs=100,
        timeout_seconds=E1_UNIT_TIMEOUT_SECONDS,
        watchdog_seconds=E1_UNIT_TIMEOUT_SECONDS,
        final_worker_ceiling_seconds=E1_UNIT_TIMEOUT_SECONDS,
        observation_seed_schedule=(E1_TRAINING_SEED,) * round_count,
        cuda_visible_devices=gpu_uuid,
        round_count=round_count,
        attempt_scheduler=True,
        max_attempts_per_round=1,
        provider_maximum_physical_attempts=fresh_r1.MAX_PHYSICAL_ATTEMPTS,
        implementation_total_token_ceiling_per_call=(
            E1_PROVIDER_TOKEN_CAP_PER_CALL
        ),
        proposal_output_token_ceiling_total_per_slot=(
            E1_PROVIDER_TOKEN_CAP_PER_CALL
        ),
        implementation_output_token_ceiling_total_per_candidate=(
            E1_PROVIDER_TOKEN_CAP_PER_CALL
        ),
        evaluator=DEVELOPMENT_EVALUATOR,
        split=DEVELOPMENT_SPLIT,
        frozen_profile_ref=e1_frozen_profile_ref(),
        protocol_ref=E1_PROTOCOL_REF,
        protocol_digest=E1_PROTOCOL_DIGEST,
        execution_purpose="DEVELOPMENT_MAIN_OFFLINE_TOPN",
        baseline_context=baseline_context,
        allow_resume_source_sha256_drift=args.resume and args.smoke_run,
    )
    adapter = E1MultVAESearchSpaceAdapterV1(worker_boundary=boundary,
        resource_probe_runner=MeteredROResourceProbe(budget), pilot_context=pilot_context)
    if config.research_mode == "director_sequential":
        adapter.proposal_prompt_source = E1_DIRECTOR_PROMPT_PATH
    adapter.qualification_executor = MeteredQualification(budget, boundary.qualify)
    composition = compose_standalone_campaign(
        config,
        resume=args.resume,
        provider_call=metered_ro_provider(budget, credential_index=args.credential_index),
        launch=MeteredROTraining(budget, worker_boundary=boundary),
        search_space_adapter=adapter,
    )
    if (
        composition.manifest.get("controller") != "ResearchCampaign"
        or composition.manifest.get("research_only") is not True
        or composition.manifest.get("frozen_profile_ref") != e1_frozen_profile_ref()
        or (not args.resume and (
            composition.profile.profile_ref != e1_frozen_profile_ref()["profile_id"]
            or composition.profile.entries != ()
        ))
    ):
        raise E1PreparationError("campaign identity mismatch or fresh E1 profile is not empty")
    if not args.resume and (budget.state["calls"] or budget.state["starts"]):
        raise E1PreparationError("prepare-only unexpectedly performed physical work")

    routing = model_routing_manifest()
    expected_role_models = {
        "critic": "gpt-5.6-luna",
        "falsification_designer": "gpt-5.6-luna",
        "frontier_architect": "gpt-5.6-terra",
        "implementer": "gpt-5.6-terra",
        "lineage_refiner": "gpt-5.6-luna",
        "mechanism_composer": "gpt-5.6-luna",
    }
    if any(
        routing["role_models"].get(role) != model
        or routing["role_reasoning_effort"].get(role) != "medium"
        for role, model in expected_role_models.items()
    ):
        raise E1PreparationError("E1 Provider model/reasoning routing differs")

    parent_identity = parent_package_identity()
    preparation = {
        "schema": "recclaw.e1.ro-research-preparation.v1",
        "status": "prepared_not_started",
        "evidence_class": "ENGINEERING_PREPARATION_ONLY",
        "formal_search_started": False,
        "provider_calls": 0,
        "physical_training_starts": 0,
        "controller": "ResearchCampaign",
        "research_only": True,
        "research_mode": config.research_mode,
        "campaign_composed": True,
        "campaign_id": config.campaign_id,
        "campaign_root": str(campaign_root),
        "campaign_manifest_digest": sha256_digest(composition.manifest),
        "profile_ref": composition.profile.profile_ref,
        "profile_digest": composition.profile.profile_digest,
        "protocol_ref": E1_PROTOCOL_REF,
        "protocol_digest": E1_PROTOCOL_DIGEST,
        "parent_identity": parent_identity,
        "baseline": {
            "path": str(runtime_paths["baseline"]),
            "sha256": baseline_sha256,
            "source_sha256": baseline["source_sha256"],
            "ndcg_at_10": E1_BASELINE_NDCG_AT_10,
            "recall_at_10": E1_BASELINE_RECALL_AT_10,
        },
        "search": {
            "data_root": str(runtime_paths["data_root"]),
            "manifest_path": str(runtime_paths["search_manifest"]),
            "manifest_sha256": manifest["manifest_sha256"],
            "verified_files": manifest["verified_files"],
            "outer_heldout_access": "FORBIDDEN",
        },
        "runtime": {
            "worker_python": str(runtime_paths["worker_python"]),
            "worker_python_sha256": _sha256(runtime_paths["worker_python"]),
            "recbole_root": str(runtime_paths["recbole_root"]),
            "recbole_init_sha256": _sha256(runtime_paths["recbole_init"]),
            "recbole_commit": E1_RECBOLE_COMMIT,
            "driver_dir": str(runtime_paths["driver_dir"]),
            "nvidia_smi_sha256": _sha256(runtime_paths["nvidia_smi"]),
            "gpu_id": args.gpu_id,
            "gpu_uuid": gpu_uuid,
            "execution_boundary": "EXPLICIT_NATIVE_WORKER",
            "api_config_path": str(runtime_paths["api_config"]),
            "api_config_sha256": _sha256(runtime_paths["api_config"]),
        },
        "provider_policy": {
            "active_research_roles": list(research_producer_roles(
                composition.campaign.state.context.budget)),
            "credential_index": args.credential_index,
            "wire_api": "responses",
            "reasoning_effort": "medium",
            "temperature_field": "OMITTED_FOR_RESPONSES",
            "model_routing": routing,
        },
        "requested_budget": {
            "provider_physical_calls": budget.limits['calls'],
            "provider_total_tokens": budget.limits['tokens'],
            "provider_total_tokens_per_call": E1_PROVIDER_TOKEN_CAP_PER_CALL,
            "physical_training_starts": budget.limits['starts'],
            "gpu_seconds": budget.limits['gpu_secs'],
            "unit_timeout_seconds": E1_UNIT_TIMEOUT_SECONDS,
        },
        "native_budget_binding": {
            "round_count": config.round_count,
            "max_attempts_per_round": config.max_attempts_per_round,
            "maximum_candidate_starts": (
                config.round_count * int(config.max_attempts_per_round or 0)
            ),
            "unit_timeout_seconds": config.timeout_seconds,
            "per_call_token_ceiling": (
                config.implementation_total_token_ceiling_per_call
            ),
            "status": "SHARED_DURABLE_SUPERVISOR_BOUND_NOT_EXECUTED",
            "ledger_path": str(budget.path),
            "ledger_limits": budget.limits,
            "unbound_campaign_global_limits": [],
        },
        "launch_binding": "METERED_SUPERVISOR_PREPARED_NOT_EXECUTED",
        "formal_blocker": None,
        "upstream_release": args.upstream_release,
        "pilot_context_sha256": _sha256(args.pilot_context),
        "execution_scope": "PREPARATION_ONLY_NO_FORMAL_SEARCH_REQUESTED",
    }
    output = run_root / "preparation.json"
    if not args.resume:
        _write_json(output, preparation)
    print(json.dumps({"status": "resumed" if args.resume else "prepared_not_started",
                      "output": str(output)}))
    if args.preview_prompts:
        preview_initial_requests(composition, args.preview_prompts, adapter)
    if args.search_run or args.canary_run:
        from gagc.e1_search import run_ro_search, execute_search
        return execute_search(run_root, budget, lambda: run_ro_search(composition), api_key=api_key,
            evidence_class='ENGINEERING_CANARY' if args.canary_run else 'PILOT_CONTROLLER_SEARCH')
    if args.smoke_run or args.smoke_provider_only:
        from gagc.e1_smoke import write_smoke_result
        try:
            if args.smoke_provider_only:
                composition.provider.maximum_physical_attempts = 1
                context = composition.campaign.state.context
                role = research_producer_roles(context.budget)[0]
                proposal = composition.provider(role, context.producer_view(role))
                result = dict(status='single_provider_returned',
                              proposal_type=type(proposal).__name__)
            else:
                results = composition.run(round_count=1)
                result = dict(status='native_round_returned', returned_rounds=len(results),
                              next_round_index=composition.campaign.state.next_round_index)
        except Exception as error:
            result = dict(status='smoke_stopped', error_type=type(error).__name__,
                          error=str(error).replace(api_key, '[REDACTED]'))
            write_smoke_result(run_root, budget, result)
            return 1
        write_smoke_result(run_root, budget, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
