#!/usr/bin/env python3
"""Run a bounded, standalone Research/BLICF campaign with clean resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1 import fresh_r1  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (  # noqa: E402
    campaign_development_validation_profile_manifest,
    campaign_development_validation_profile_ref,
    campaign_scientific_profile_ref,
)
from recclaw_core.experiments.helix_abc_v1.conversion_efficiency import (  # noqa: E402
    MAX_REPAIR_TURNS,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (  # noqa: E402
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
)
from recclaw_core.research_line.standalone import (  # noqa: E402
    MAX_STANDALONE_ROUNDS,
    STRICT_BL_ICF_PROTOCOL_DIGEST,
    STRICT_BL_ICF_PROTOCOL_REF,
    StandaloneCampaignError,
    StandaloneResearchConfig,
    compose_standalone_campaign,
    load_portfolio_candidates,
    load_research_profile_source,
)
from recclaw_core.research_line.single_round import (  # noqa: E402
    ResearchBaselineSourceV1,
)
from recclaw_core.research_line.original_matched import (  # noqa: E402
    OriginalMatched264Config,
    compose_original_matched_264_campaign,
)
from recclaw_core.helix.standalone_bridge import (  # noqa: E402
    StandaloneFullHelixBridgeV31,
    standalone_guard_protocol,
)
from recclaw_core.research_line.gpu_reservation_provider import (  # noqa: E402
    DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    GpuReservationProviderError,
    make_nvidia_smi_gpu_reservation_provider,
)


_FORMAL_RUNTIME_PATHS = (
    ("RECCLAW_PROJECTS_ROOT", "directory", "PROJECTS_ROOT"),
    ("RECCLAW_SEARCH_DATA_ROOT", "directory", "SEARCH_DATA_ROOT"),
    ("RECCLAW_RECBOLE_ROOT", "directory", "RECBole_ROOT"),
    ("RECCLAW_PYTHON_EXECUTABLE", "executable", "PYTHON_EXECUTABLE"),
)


def _validate_formal_runtime_environment() -> None:
    """Fail before Provider work when formal child roots would fall back."""

    failures: list[str] = []
    for variable, kind, constant_name in _FORMAL_RUNTIME_PATHS:
        raw = os.environ.get(variable)
        if raw is None or not raw.strip():
            failures.append(f"{variable}=MISSING")
            continue
        configured = Path(raw).expanduser().resolve()
        effective = Path(getattr(fresh_r1, constant_name)).resolve()
        if configured != effective:
            failures.append(f"{variable}=IMPORT_MISMATCH")
            continue
        if kind == "directory":
            usable = configured.is_dir() and os.access(configured, os.R_OK | os.X_OK)
        else:
            usable = configured.is_file() and os.access(configured, os.R_OK | os.X_OK)
        if not usable:
            failures.append(f"{variable}=INACCESSIBLE")
    if failures:
        raise StandaloneCampaignError(
            "formal runtime environment is not child-safe: " + ",".join(failures)
        )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _read_resource_profiles(path: Path | None) -> Mapping[str, Mapping[str, Any]]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StandaloneCampaignError(
            f"cannot read resource profile JSON: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise StandaloneCampaignError("resource profile JSON must be an object")
    result: dict[str, Mapping[str, Any]] = {}
    for capability_ref, profile in payload.items():
        if not isinstance(capability_ref, str) or not capability_ref.strip():
            raise StandaloneCampaignError(
                "resource profile capability refs must be non-empty strings"
            )
        if not isinstance(profile, Mapping):
            raise StandaloneCampaignError(
                f"resource profile for {capability_ref} must be an object"
            )
        result[capability_ref] = canonical_value(dict(profile))
    return result


def _read_baseline_context(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StandaloneCampaignError(
            f"cannot read baseline context JSON: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise StandaloneCampaignError("baseline context JSON must be an object")
    return canonical_value(dict(payload))


def _read_observation_seed_schedule(path: Path | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StandaloneCampaignError(
            f"cannot read observation seed schedule JSON: {path}"
        ) from error
    if isinstance(payload, (str, bytes)) or not isinstance(payload, list):
        raise StandaloneCampaignError(
            "observation seed schedule JSON must be an array"
        )
    result: list[int] = []
    for index, seed in enumerate(payload):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
            raise StandaloneCampaignError(
                f"observation seed schedule item {index} must be a positive integer"
            )
        result.append(seed)
    if not result:
        raise StandaloneCampaignError("observation seed schedule must not be empty")
    return tuple(result)


def _build_gpu_reservation_provider(args: argparse.Namespace) -> Any | None:
    """Build formal GPU evidence only when the CLI explicitly enables it."""

    authority_ref = args.gpu_reservation_authority_ref
    timeout_seconds = args.gpu_reservation_probe_timeout_seconds
    if not args.require_gpu_reservation_evidence:
        if args.allow_existing_gpu_processes:
            raise StandaloneCampaignError(
                "--allow-existing-gpu-processes requires "
                "--require-gpu-reservation-evidence"
            )
        if authority_ref is not None:
            raise StandaloneCampaignError(
                "--gpu-reservation-authority-ref requires "
                "--require-gpu-reservation-evidence"
            )
        if timeout_seconds != DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS:
            raise StandaloneCampaignError(
                "--gpu-reservation-probe-timeout-seconds requires "
                "--require-gpu-reservation-evidence"
            )
        return None
    selector = (
        str(args.gpu_id)
        if args.gpu_id is not None
        else args.cuda_visible_devices
    )
    if selector is None:
        raise StandaloneCampaignError(
            "--require-gpu-reservation-evidence requires explicit "
            "--cuda-visible-devices or --gpu-id"
        )
    if authority_ref is None:
        raise StandaloneCampaignError(
            "--require-gpu-reservation-evidence requires "
            "--gpu-reservation-authority-ref"
        )
    try:
        return make_nvidia_smi_gpu_reservation_provider(
            cuda_visible_devices=selector,
            reservation_authority_ref=authority_ref,
            timeout_seconds=timeout_seconds,
            allow_existing_compute_processes=args.allow_existing_gpu_processes,
        )
    except GpuReservationProviderError as error:
        raise StandaloneCampaignError(
            f"invalid formal GPU reservation provider configuration: {error}"
        ) from error


def _gpu_reservation_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    provider = _build_gpu_reservation_provider(args)
    return {
        "gpu_reservation_evidence_provider": provider,
        "require_gpu_reservation_evidence": (
            args.require_gpu_reservation_evidence
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Research-only standalone BLICF ResearchCampaign. "
            "The round count is bounded and the run root is durable."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--search-policy-mode", choices=("fixed", "adaptive"), default="fixed",
        help="default: fixed explicit policy with scientific memory; adaptive is opt-in; one campaign per process",
    )
    parser.add_argument(
        "--round-count",
        "--rounds",
        dest="round_count",
        type=_positive_int,
        required=True,
        help="number of rounds to attempt in this invocation (maximum 100)",
    )
    parser.add_argument(
        "--max-rounds-this-invocation",
        type=_positive_int,
        help=(
            "optional bound on new rounds started by this process; the durable "
            "campaign target remains --round-count, enabling matched interleaving"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-resume-source-sha256-drift",
        action="store_true",
        help=(
            "explicitly authorize resume when only source_sha256 changes, or "
            "when source_ref and source_sha256 migrate together as one source "
            "artifact identity; all other sealed identities remain strict"
        ),
    )
    parser.add_argument(
        "--allow-resume-endpoint2-to-endpoint1",
        action="store_true",
        help="authorize endpoint2-to-endpoint1 resume only; model/reasoning/budgets stay fixed",
    )
    parser.add_argument(
        "--allow-resume-endpoint1-to-endpoint2-and-reasoning-drift",
        action="store_true",
        help=(
            "explicitly authorize the one-way endpoint1-to-endpoint2 config "
            "path migration plus None-to-low reasoning routing migration; "
            "model IDs and every scientific execution input remain strict"
        ),
    )
    parser.add_argument(
        "--allow-resume-exhausted-engineering-generation",
        action="store_true",
        help=(
            "authorize exactly one additional same-round discovery generation "
            "only for a resumed checkpoint with typed engineering-plus-duplicate "
            "exhaustion evidence"
        ),
    )
    parser.add_argument("--seed", type=int, default=54201)
    parser.add_argument(
        "--search-seed",
        type=int,
        help=(
            "Provider proposal/search seed; defaults to --seed while all "
            "training and observation behavior remains bound to --seed"
        ),
    )
    parser.add_argument(
        "--original-matched-264",
        action="store_true",
        help=(
            "use Original proposal/selection policy over the shared strict "
            "BLICF264 implementation and execution substrate"
        ),
    )
    parser.add_argument(
        "--shared-implementation-root",
        type=Path,
        help=(
            "paired-seed implementation store shared by A/B/C; identical "
            "canonical specs reuse one implementation identity"
        ),
    )
    parser.add_argument(
        "--shared-implementation-arm",
        choices=("A", "B", "C"),
        help="arm label for the paired shared implementation store",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=fresh_r1.EXPERIMENT_EPOCHS,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_int,
        default=fresh_r1.MAX_WORKER_CEILING_SECONDS,
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=_positive_int,
        default=fresh_r1.MAX_WORKER_CEILING_SECONDS,
    )
    parser.add_argument(
        "--final-worker-ceiling-seconds",
        type=_positive_int,
        default=fresh_r1.MAX_WORKER_CEILING_SECONDS,
        help=(
            "final Fresh worker safety ceiling; independent from watchdog-seconds"
        ),
    )
    parser.add_argument(
        "--implementation-total-token-ceiling-per-call",
        type=_positive_int,
        default=32_000,
        help=(
            "total input-plus-output ceiling for each implementation call; "
            "the implementation output ceiling remains independently bounded"
        ),
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cuda-visible-devices",
        help="one explicit CUDA device token passed to every Fresh worker",
    )
    device_group.add_argument(
        "--gpu-id",
        type=_nonnegative_int,
        help=(
            "one explicit non-negative physical GPU id passed to FreshRunner; "
            "does not export CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument(
        "--require-gpu-reservation-evidence",
        action="store_true",
        help=(
            "require caller-authorized, owner-bound GPU reservation evidence "
            "before each formal worker"
        ),
    )
    parser.add_argument(
        "--allow-existing-gpu-processes",
        action="store_true",
        help=(
            "allow pre-existing compute contexts in the formal reservation "
            "probe; strict rejection remains the default"
        ),
    )
    parser.add_argument(
        "--gpu-reservation-authority-ref",
        help=(
            "normalized external exclusive-reservation claim reference; "
            "only consumed with --require-gpu-reservation-evidence"
        ),
    )
    parser.add_argument(
        "--gpu-reservation-probe-timeout-seconds",
        type=_positive_int,
        default=DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
        help="bounded nvidia-smi evidence probe timeout",
    )
    parser.add_argument(
        "--observation-seed-schedule-json",
        type=Path,
        help=(
            "optional frozen JSON array of per-round observation seeds; "
            "the first item must equal --seed"
        ),
    )
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--baseline-digest", required=True)
    parser.add_argument("--baseline-ndcg-at-10", type=float, required=True)
    parser.add_argument("--baseline-protocol-digest", required=True)
    parser.add_argument(
        "--baseline-context-json",
        type=Path,
        help=(
            "optional search-space baseline pack and target semantics injected "
            "into the sealed Research Context"
        ),
    )
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument(
        "--proposal-replay-run-root",
        type=Path,
        help=(
            "optional prior successful Research run whose four strict-program "
            "Provider responses are replayed once without API calls; candidate "
            "implementation is always regenerated"
        ),
    )
    parser.add_argument(
        "--development-validation-v31",
        action="store_true",
        help=(
            "bind the V31 train/dev/dev online metric contract; the worker "
            "cannot evaluate outer heldout in this mode"
        ),
    )
    parser.add_argument(
        "--paired-budget-slot-canary",
        action="store_true",
        help=(
            "run the preregistered C=HELIX seed-54201, 15-slot development "
            "canary; every successful slot contains exactly one unique "
            "metric-bearing experiment, while the explicit fixed-slot option "
            "may close four genuine failures without fabricating a metric"
        ),
    )
    parser.add_argument(
        "--paired-arm",
        choices=("B_RESEARCH", "C_FULL_HELIX"),
        help=(
            "explicit treatment identity for a paired canary: B is the full "
            "Research Line and C is that same Research Line plus the full Helix "
            "V31 evidence/allocation/fusion bridge"
        ),
    )
    parser.add_argument(
        "--attempt-scheduler",
        action="store_true",
        help="enable the existing bounded attempt scheduler",
    )
    parser.add_argument(
        "--max-attempts-per-round",
        type=int,
    )
    parser.add_argument(
        "--close-exhausted-no-metric-slot",
        action="store_true",
        help=(
            "close a fixed formal slot only after its genuine candidate-local "
            "attempt budget is exhausted without a metric; retain failure "
            "evidence and preserve the frontier"
        ),
    )
    parser.add_argument(
        "--portfolio-profile-json",
        type=Path,
        help=(
            "optional frozen complete PortfolioCandidateV2 profile set; "
            "missing fields are rejected"
        ),
    )
    parser.add_argument(
        "--resource-profile-json",
        type=Path,
        help="optional capability-to-resource prediction mapping",
    )
    parser.add_argument(
        "--profile-source-json",
        type=Path,
        help=(
            "optional pre-round role/capability Research profile policy bundle "
            "(exact candidate-record JSON remains a compatibility form); its "
            "schema, source_ref, and source_digest are sealed into resume identity"
        ),
    )
    parser.add_argument(
        "--evidence-guard-expected-dataset-manifest-digest",
        help=(
            "optional exact SHA-256 dataset-manifest identity for the Evidence "
            "Guard; requires the validation seed schedule"
        ),
    )
    parser.add_argument(
        "--evidence-guard-validation-seed-schedule-json",
        type=Path,
        help=(
            "optional frozen JSON array of Guard validation seeds; evidence "
            "informs discovery, while physical verification requires an "
            "explicit request and separately scheduled budgeted work"
        ),
    )
    parser.add_argument(
        "--evidence-guard-required-seed-count",
        type=_positive_int,
        default=3,
    )
    parser.add_argument(
        "--evidence-guard-comparator",
        help="optional Guard comparator identity (defaults to baseline comparator)",
    )
    parser.add_argument(
        "--evidence-guard-minimum-effect-delta",
        type=float,
        default=0.0,
        help=(
            "explicit Guard positive-signal threshold; default 0.0 is sealed "
            "into the bridge identity"
        ),
    )
    parser.add_argument(
        "--evidence-guard-replication-trigger-delta",
        type=float,
        default=0.0,
        help=(
            "positive development delta for explicitly requested next-seed work; "
            "discovery does not automatically reserve verification actions"
        ),
    )
    parser.add_argument(
        "--evidence-guard-max-allocation-actions",
        type=_nonnegative_int,
        help=(
            "required hard cap for explicitly requested REPLICATE/CONTROL work; "
            "neither preempts discovery slots nor authorizes extra automatic workers"
        ),
    )
    parser.add_argument(
        "--evidence-guard-max-open-allocation-actions",
        type=_positive_int,
        default=1,
        help="maximum concurrently unresolved Evidence Guard actions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.require_gpu_reservation_evidence:
        try:
            _validate_formal_runtime_environment()
        except StandaloneCampaignError as error:
            parser.error(str(error))
    if args.round_count > MAX_STANDALONE_ROUNDS:
        parser.error(f"--round-count must be <= {MAX_STANDALONE_ROUNDS}")
    if (args.shared_implementation_root is None) != (
        args.shared_implementation_arm is None
    ):
        parser.error(
            "--shared-implementation-root and --shared-implementation-arm "
            "must be supplied together"
        )
    if args.original_matched_264:
        if args.round_count not in {100, 200}:
            parser.error(
                "Original-Matched-264 requires --round-count 100 or 200"
            )
        if args.seed != 54201:
            parser.error("Original-Matched-264 requires --seed 54201")
        if args.search_seed not in {54201, 54202, 54203}:
            parser.error(
                "Original-Matched-264 requires --search-seed 54201, 54202, or 54203"
            )
        if args.epochs != 100:
            parser.error("Original-Matched-264 requires --epochs 100")
        if not args.attempt_scheduler or args.max_attempts_per_round != 4:
            parser.error(
                "Original-Matched-264 requires --attempt-scheduler "
                "--max-attempts-per-round 4"
            )
        if not args.close_exhausted_no_metric_slot:
            parser.error(
                "Original-Matched-264 requires --close-exhausted-no-metric-slot"
            )
        if args.shared_implementation_root is None:
            parser.error(
                "Original-Matched-264 requires --shared-implementation-root"
            )
        if args.shared_implementation_arm != "A":
            parser.error(
                "Original-Matched-264 requires --shared-implementation-arm A"
            )
        if any(
            value is not None
            for value in (
                args.portfolio_profile_json,
                args.profile_source_json,
                args.resource_profile_json,
                args.evidence_guard_expected_dataset_manifest_digest,
                args.evidence_guard_validation_seed_schedule_json,
            )
        ):
            parser.error(
                "Original-Matched-264 forbids Research/Helix portfolio and Guard inputs"
            )
    if args.attempt_scheduler and args.max_attempts_per_round is None:
        parser.error("--attempt-scheduler requires --max-attempts-per-round")
    if args.development_validation_v31 and args.paired_budget_slot_canary:
        parser.error("development validation modes are mutually exclusive")
    if args.paired_arm is not None and not args.paired_budget_slot_canary:
        parser.error("--paired-arm requires --paired-budget-slot-canary")
    if args.paired_budget_slot_canary:
        if args.paired_arm is None:
            parser.error("paired budget-slot canary requires --paired-arm")
        if args.round_count != 15:
            parser.error("paired budget-slot canary requires --round-count 15")
        if args.seed != 54201:
            parser.error("paired budget-slot canary requires --seed 54201")
        if args.epochs != 100:
            parser.error("paired budget-slot canary requires --epochs 100")
        if not args.attempt_scheduler or args.max_attempts_per_round != 4:
            parser.error(
                "paired budget-slot canary requires --attempt-scheduler "
                "--max-attempts-per-round 4"
            )
        if args.cuda_visible_devices is not None or args.gpu_id is None:
            parser.error(
                "paired budget-slot canary requires native --gpu-id and "
                "forbids --cuda-visible-devices"
            )
        if not args.require_gpu_reservation_evidence:
            parser.error(
                "paired budget-slot canary requires "
                "--require-gpu-reservation-evidence"
            )
        if args.portfolio_profile_json is not None:
            parser.error("paired budget-slot canary forbids --portfolio-profile-json")
        if args.profile_source_json is not None:
            parser.error("paired budget-slot canary forbids --profile-source-json")
        if args.resource_profile_json is not None:
            parser.error("paired budget-slot canary forbids --resource-profile-json")
        if args.paired_arm == "B_RESEARCH":
            if any(
                value is not None
                for value in (
                    args.evidence_guard_expected_dataset_manifest_digest,
                    args.evidence_guard_validation_seed_schedule_json,
                    args.evidence_guard_max_allocation_actions,
                    args.evidence_guard_comparator,
                )
            ):
                parser.error("B_RESEARCH forbids Helix/Evidence Guard inputs")
        elif args.paired_arm == "C_FULL_HELIX":
            if args.evidence_guard_expected_dataset_manifest_digest is None:
                parser.error(
                    "C_FULL_HELIX requires the exact dataset manifest identity"
                )
            if args.evidence_guard_validation_seed_schedule_json is None:
                parser.error(
                    "C_FULL_HELIX requires the exact validation seed schedule"
                )
            if args.evidence_guard_max_allocation_actions is None:
                parser.error(
                    "C_FULL_HELIX requires an explicit positive "
                    "--evidence-guard-max-allocation-actions"
                )
    try:
        portfolio_candidates = (
            load_portfolio_candidates(args.portfolio_profile_json)
            if args.portfolio_profile_json is not None
            else ()
        )
        resource_profiles = _read_resource_profiles(args.resource_profile_json)
        baseline_context = _read_baseline_context(args.baseline_context_json)
        observation_seed_schedule = _read_observation_seed_schedule(
            args.observation_seed_schedule_json
        )
        if args.paired_budget_slot_canary and observation_seed_schedule != (
            54201,
        ) * 15:
            raise StandaloneCampaignError(
                "paired budget-slot canary requires 15 repeated seed 54201 slots"
            )
        evidence_guard_schedule = _read_observation_seed_schedule(
            args.evidence_guard_validation_seed_schedule_json
        )
        profile_source = (
            load_research_profile_source(args.profile_source_json)
            if args.profile_source_json is not None
            else None
        )
        gpu_reservation_config = _gpu_reservation_config_kwargs(args)
        if args.development_validation_v31 or args.paired_budget_slot_canary:
            scientific_profile = campaign_development_validation_profile_manifest()
            evaluator = DEVELOPMENT_EVALUATOR
            split = DEVELOPMENT_SPLIT
            frozen_profile_ref = campaign_development_validation_profile_ref()
            protocol_ref = str(scientific_profile["protocol"]["protocol_ref"])
            protocol_digest = str(
                scientific_profile["protocol"]["protocol_digest"]
            )
            execution_purpose = (
                "DEVELOPMENT_PILOT_OFFLINE_TOPN"
                if args.paired_budget_slot_canary
                else "DEVELOPMENT_MAIN_OFFLINE_TOPN"
            )
            if args.paired_budget_slot_canary and (
                protocol_ref != STRICT_BL_ICF_PROTOCOL_REF
                or protocol_digest != STRICT_BL_ICF_PROTOCOL_DIGEST
            ):
                raise StandaloneCampaignError(
                    "paired budget-slot canary requires the frozen strict "
                    "BL-ICF development protocol identity"
                )
        else:
            scientific_profile = campaign_development_validation_profile_manifest()
            evaluator = DEVELOPMENT_EVALUATOR
            split = DEVELOPMENT_SPLIT
            frozen_profile_ref = campaign_development_validation_profile_ref()
            protocol_ref = str(scientific_profile["protocol"]["protocol_ref"])
            protocol_digest = str(
                scientific_profile["protocol"]["protocol_digest"]
            )
            execution_purpose = "DEVELOPMENT_MAIN_OFFLINE_TOPN"
        if args.baseline_protocol_digest != protocol_digest:
            raise StandaloneCampaignError(
                "--baseline-protocol-digest does not match the selected protocol"
            )
        baseline_source = ResearchBaselineSourceV1.from_identity(
            source_ref=args.source_ref,
            source_sha256=args.source_digest,
            comparator_ref=args.baseline_ref,
            comparator_digest=args.baseline_digest,
            frozen_ndcg_at_10=args.baseline_ndcg_at_10,
            protocol_digest=args.baseline_protocol_digest,
            seed=args.seed,
        )
        if bool(args.evidence_guard_expected_dataset_manifest_digest) != bool(
            args.evidence_guard_validation_seed_schedule_json
        ):
            raise StandaloneCampaignError(
                "Evidence Guard dataset-manifest digest and validation seed "
                "schedule must be supplied together"
            )
        if args.evidence_guard_expected_dataset_manifest_digest is not None:
            if observation_seed_schedule is None:
                raise StandaloneCampaignError(
                    "Evidence Guard requires the ordinary observation seed "
                    "schedule as the shared frozen seed universe"
                )
            discovery_seeds = {str(seed) for seed in observation_seed_schedule}
            verification_seeds = {
                str(seed) for seed in (evidence_guard_schedule or ())
            }
            if not discovery_seeds.issubset(verification_seeds):
                raise StandaloneCampaignError(
                    "Evidence Guard verification schedule must include every "
                    "ordinary discovery seed"
                )
            if (
                args.evidence_guard_comparator is not None
                and args.evidence_guard_comparator != baseline_source.comparator_ref
            ):
                raise StandaloneCampaignError(
                    "Evidence Guard comparator must equal the baseline comparator"
                )
            if args.evidence_guard_max_allocation_actions is None:
                raise StandaloneCampaignError(
                    "Evidence Guard requires an explicit matched allocation-action cap"
                )
            if args.evidence_guard_max_allocation_actions < 1:
                raise StandaloneCampaignError(
                    "Evidence Guard allocation-action cap must be positive"
                )
            if args.round_count >= 50:
                if args.portfolio_profile_json is not None:
                    raise StandaloneCampaignError(
                        "formal Evidence Guard runs cannot use portfolio-profile JSON"
                    )
                if profile_source is None:
                    raise StandaloneCampaignError(
                        "formal Evidence Guard runs require the v27 profile source"
                    )
        evidence_port = None
        if args.evidence_guard_expected_dataset_manifest_digest is not None:
            guard_schedule = _read_observation_seed_schedule(
                args.evidence_guard_validation_seed_schedule_json
            )
            if guard_schedule is None:
                raise StandaloneCampaignError(
                    "Evidence Guard validation seed schedule is required"
                )
            evidence_port = StandaloneFullHelixBridgeV31(
                run_root=args.run_root,
                campaign_id=args.campaign_id,
                protocol=standalone_guard_protocol(
                    epochs=args.epochs,
                    protocol_digest=protocol_digest,
                ),
                comparator=(
                    args.evidence_guard_comparator
                    or baseline_source.comparator_ref
                ),
                comparator_ndcg_at_10=baseline_source.frozen_ndcg_at_10,
                expected_dataset_manifest_digest=(
                    args.evidence_guard_expected_dataset_manifest_digest
                ),
                expected_split=split,
                expected_evaluator=evaluator,
                required_seed_count=args.evidence_guard_required_seed_count,
                validation_seed_schedule=tuple(str(seed) for seed in guard_schedule),
                minimum_effect_delta=args.evidence_guard_minimum_effect_delta,
                replication_trigger_delta=(
                    args.evidence_guard_replication_trigger_delta
                ),
                max_allocation_actions=(
                    args.evidence_guard_max_allocation_actions
                ),
                max_open_allocation_actions=(
                    args.evidence_guard_max_open_allocation_actions
                ),
            )
        config = StandaloneResearchConfig(
            search_policy_mode=args.search_policy_mode,
            repo_root=REPO_ROOT,
            run_root=args.run_root,
            api_config_source=args.api_config,
            campaign_id=args.campaign_id,
            baseline_source=baseline_source,
            allow_resume_source_sha256_drift=(
                args.allow_resume_source_sha256_drift
            ),
            allow_resume_endpoint1_to_endpoint2_and_reasoning_drift=(
                args.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
            ),
            allow_resume_endpoint2_to_endpoint1=args.allow_resume_endpoint2_to_endpoint1,
            allow_resume_exhausted_engineering_generation=(
                args.allow_resume_exhausted_engineering_generation
            ),
            baseline_context=baseline_context,
            seed=args.seed,
            search_seed=args.search_seed,
            epochs=args.epochs,
            timeout_seconds=args.timeout_seconds,
            watchdog_seconds=args.watchdog_seconds,
            final_worker_ceiling_seconds=args.final_worker_ceiling_seconds,
            cuda_visible_devices=args.cuda_visible_devices,
            gpu_id=args.gpu_id,
            observation_seed_schedule=observation_seed_schedule,
            round_count=args.round_count,
            attempt_scheduler=args.attempt_scheduler,
            max_attempts_per_round=args.max_attempts_per_round,
            prebinding_token_ceiling_retry=True,
            close_exhausted_no_metric_slot=(
                args.close_exhausted_no_metric_slot
            ),
            provider_maximum_physical_attempts=fresh_r1.MAX_PHYSICAL_ATTEMPTS,
            implementation_total_token_ceiling_per_call=(
                args.implementation_total_token_ceiling_per_call
            ),
            proposal_output_token_ceiling_total_per_slot=(
                64_000 if args.paired_budget_slot_canary else None
            ),
            implementation_output_token_ceiling_total_per_candidate=(
                64_000 if args.paired_budget_slot_canary else None
            ),
            max_implementation_calls_per_candidate=MAX_REPAIR_TURNS + 1,
            portfolio_candidates=portfolio_candidates,
            research_profile_source=profile_source,
            resource_profile_by_capability=resource_profiles,
            evidence_port=evidence_port,
            evaluator=evaluator,
            split=split,
            frozen_profile_ref=(
                frozen_profile_ref
                if frozen_profile_ref is not None
                else campaign_scientific_profile_ref()
            ),
            protocol_ref=protocol_ref,
            protocol_digest=(
                protocol_digest if protocol_ref is not None else None
            ),
            execution_purpose=execution_purpose,
            proposal_replay_run_root=args.proposal_replay_run_root,
            arm_code=args.shared_implementation_arm,
            shared_implementation_root=args.shared_implementation_root,
            **gpu_reservation_config,
        )
        composition = (
            compose_original_matched_264_campaign(
                OriginalMatched264Config(config),
                resume=args.resume,
            )
            if args.original_matched_264
            else compose_standalone_campaign(
                config,
                resume=args.resume,
            )
        )
        if evidence_port is not None:
            experiment = composition.manifest.get("experiment")
            actual_dataset_digest = (
                experiment.get("dataset_manifest_sha256")
                if isinstance(experiment, Mapping)
                else None
            )
            if actual_dataset_digest != args.evidence_guard_expected_dataset_manifest_digest:
                raise StandaloneCampaignError(
                    "Evidence Guard dataset-manifest identity differs from the "
                    "sealed composition manifest"
                )
        completed_before = composition.campaign.state.next_round_index - 1
        remaining_rounds = max(0, args.round_count - completed_before)
        invocation_round_limit = (
            remaining_rounds
            if args.max_rounds_this_invocation is None
            else min(remaining_rounds, args.max_rounds_this_invocation)
        )
        results = composition.run(invocation_round_limit)
    except StandaloneCampaignError as error:
        parser.error(str(error))
    completed_records = tuple(
        composition.campaign._load_round(round_index)
        for round_index in range(1, args.round_count + 1)
        if composition.campaign.round_checkpoint_path(round_index).is_file()
    )
    metric_round_count = sum(
        1
        for record in completed_records
        if record.result.has_metric_bearing_attempt
        and sum(attempt.metric_bearing for attempt in record.result.attempts) == 1
    )
    no_metric_typed_failure_slot_count = sum(
        1
        for record in completed_records
        if record.status == "TYPED_FAILURE_NO_METRIC"
        and not record.result.has_metric_bearing_attempt
        and not any(attempt.metric_bearing for attempt in record.result.attempts)
    )
    completed_slot_count = metric_round_count + no_metric_typed_failure_slot_count
    allocation_audit = None
    guard_actions_closed = True
    if evidence_port is not None:
        allocation_actions = evidence_port.allocation_actions()
        allocation_budget = evidence_port.allocation_budget_snapshot()
        open_action_count = sum(
            action.get("status") == "RESERVED" for action in allocation_actions
        )
        guard_actions_closed = open_action_count == 0
        allocation_audit = canonical_value(
            {
                "schema": "recclaw.helix.frontier-allocation-run-audit.v31",
                "policy_digest": evidence_port.allocation_policy.digest,
                "action_count": len(allocation_actions),
                "open_action_count": open_action_count,
                "all_actions_closed": guard_actions_closed,
                "actions_digest": sha256_digest(allocation_actions),
                "budget": allocation_budget,
                "guard_provider_calls": 0,
            }
        )
    campaign_complete = bool(
        composition.campaign.state.next_round_index == args.round_count + 1
        and len(completed_records) == args.round_count
        and completed_slot_count == args.round_count
        and all(record.status != "INCOMPLETE" for record in completed_records)
        and guard_actions_closed
    )
    exit_code = 0 if campaign_complete else 3
    summary = canonical_value(
        {
            "schema": "recclaw.research-line.standalone-run-summary.v1",
            "campaign_id": composition.campaign.state.campaign_id,
            "paired_arm": args.paired_arm,
            "run_root": str(config.run_root),
            "resumed": args.resume,
            "requested_round_count": args.round_count,
            "max_rounds_this_invocation": args.max_rounds_this_invocation,
            "rounds_started": len(results),
            "next_round_index": composition.campaign.state.next_round_index,
            "metric_round_count": metric_round_count,
            "no_metric_typed_failure_slot_count": (
                no_metric_typed_failure_slot_count
            ),
            "completed_slot_count": completed_slot_count,
            "campaign_complete": campaign_complete,
            "exit_code": exit_code,
            "state_digest": composition.campaign.state.digest,
            "context_ref": composition.campaign.state.context.context_ref,
            "context_digest": composition.campaign.state.context.digest,
            "last_round_result_digest": (
                composition.campaign.state.last_round_result_digest
            ),
            "portfolio_enabled": bool(
                config.portfolio_candidates
                or config.research_profile_source is not None
            ),
            "evidence_guard_allocation_audit": allocation_audit,
        }
    )
    if evidence_port is not None:
        evidence_port.close()
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
