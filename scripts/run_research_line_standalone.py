#!/usr/bin/env python3
"""Run a bounded, standalone Research/BLICF campaign with clean resume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value  # noqa: E402
from recclaw_core.experiments.helix_abc_v1 import fresh_r1  # noqa: E402
from recclaw_core.experiments.helix_abc_v1.search_adapter import (  # noqa: E402
    adapt_current_search_profile,
)
from recclaw_core.research_line.standalone import (  # noqa: E402
    StandaloneCampaignError,
    StandaloneResearchConfig,
    compose_standalone_campaign,
    load_portfolio_candidates,
    load_research_profile_source,
)
from recclaw_core.research_line.single_round import (  # noqa: E402
    ResearchBaselineSourceV1,
)
from recclaw_core.research_line.gpu_reservation_provider import (  # noqa: E402
    DEFAULT_NVIDIA_SMI_TIMEOUT_SECONDS,
    GpuReservationProviderError,
    make_nvidia_smi_gpu_reservation_provider,
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
        "--round-count",
        "--rounds",
        dest="round_count",
        type=_positive_int,
        required=True,
        help="number of rounds to attempt in this invocation (maximum 50)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=54303)
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
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--source-digest", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.round_count > 50:
        parser.error("--round-count must be <= 50")
    if args.attempt_scheduler and args.max_attempts_per_round is None:
        parser.error("--attempt-scheduler requires --max-attempts-per-round")
    try:
        portfolio_candidates = (
            load_portfolio_candidates(args.portfolio_profile_json)
            if args.portfolio_profile_json is not None
            else ()
        )
        resource_profiles = _read_resource_profiles(args.resource_profile_json)
        observation_seed_schedule = _read_observation_seed_schedule(
            args.observation_seed_schedule_json
        )
        profile_source = (
            load_research_profile_source(args.profile_source_json)
            if args.profile_source_json is not None
            else None
        )
        gpu_reservation_config = _gpu_reservation_config_kwargs(args)
        profile = adapt_current_search_profile(campaign_id=args.campaign_id)
        if args.baseline_protocol_digest != profile.protocol_digest:
            raise StandaloneCampaignError(
                "--baseline-protocol-digest does not match the active Research profile"
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
        config = StandaloneResearchConfig(
            repo_root=REPO_ROOT,
            run_root=args.run_root,
            api_config_source=args.api_config,
            campaign_id=args.campaign_id,
            baseline_source=baseline_source,
            seed=args.seed,
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
            portfolio_candidates=portfolio_candidates,
            research_profile_source=profile_source,
            resource_profile_by_capability=resource_profiles,
            **gpu_reservation_config,
        )
        composition = compose_standalone_campaign(
            config,
            resume=args.resume,
        )
        results = composition.run(args.round_count)
    except StandaloneCampaignError as error:
        parser.error(str(error))
    summary = canonical_value(
        {
            "schema": "recclaw.research-line.standalone-run-summary.v1",
            "campaign_id": composition.campaign.state.campaign_id,
            "run_root": str(config.run_root),
            "resumed": args.resume,
            "requested_round_count": args.round_count,
            "rounds_started": len(results),
            "next_round_index": composition.campaign.state.next_round_index,
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
        }
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
