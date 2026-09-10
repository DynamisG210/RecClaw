"""Thin external two-arm paired scheduler for the P9 pilot boundary.

Arm adapters own their roots and all endogenous state.  The scheduler passes
only a fresh snapshot of frozen exogenous conditions, round identity, seed,
device, and AB/BA order.  It never receives or forwards proposals,
candidates, implementations, outcomes, history, memory, policy, frontier, or
context.

Adapter interface::

    def __call__(self, request: ArmRequest) -> ArmResult: ...

The adapter returns public metrics only.  A physical adapter can keep its
private trace and state inside its own closure or object.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


RESEARCH_LINE_ARM = "research_line"
ORIGINAL_RECCLAW_ARM = "original_recclaw"
ARM_NAMES = (RESEARCH_LINE_ARM, ORIGINAL_RECCLAW_ARM)
ARM_ORDER_AB = "AB"
ARM_ORDER_BA = "BA"
SCHEDULING_INDEPENDENT_ARM_QUEUES = "INDEPENDENT_ARM_QUEUES"
SCHEDULING_ROUND_BARRIER = "ROUND_BARRIER"

STATUS_COMPLETED = "COMPLETED"
STATUS_MISSING = "MISSING"
STATUS_CENSORED = "CENSORED"
_STATUSES = {STATUS_COMPLETED, STATUS_MISSING, STATUS_CENSORED}

_CONDITIONS_SCHEMA = "recclaw.paired.frozen-exogenous.v1"
_REQUEST_SCHEMA = "recclaw.paired.arm-request.v1"
_RECEIPT_SCHEMA = "recclaw.paired.arm-receipt.v1"
_MANIFEST_SCHEMA = "recclaw.paired.round-manifest.v1"
_ATTEMPT_SCHEMA = "recclaw.paired.arm-attempt.v1"


class PairedSchedulerError(ValueError):
    """Raised when frozen inputs or create-once artifacts disagree."""


def _clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise PairedSchedulerError("value must be standard JSON") from error


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PairedSchedulerError(f"{name} must be normalized and non-empty")


def _require_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PairedSchedulerError(f"{name} must be a lowercase SHA-256 digest")


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedSchedulerError(f"{name} must be a mapping")
    copied = _clone(dict(value))
    if not isinstance(copied, dict):  # pragma: no cover
        raise PairedSchedulerError(f"{name} must be a JSON object")
    return copied


@dataclass(frozen=True, slots=True)
class FrozenExogenousConditions:
    """Frozen inputs for one arm request; paired composition enforces fixed-66."""

    dataset: str
    split: str
    evaluator: str
    seed_schedule: tuple[int, ...]
    provider: Mapping[str, Any]
    model: str
    token_budget: int
    call_budget: int
    gpu_budget: Mapping[str, Any]
    round_budget: Mapping[str, Any]
    runtime: str
    tool: str
    profile_ref: str
    profile_digest: str
    profile_manifest_digest: str
    profile_entry_count: int = 66

    schema = _CONDITIONS_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "split",
            "evaluator",
            "model",
            "runtime",
            "tool",
            "profile_ref",
        ):
            _require_text(getattr(self, name), name)
        for name in ("profile_digest", "profile_manifest_digest"):
            _require_digest(getattr(self, name), name)
        seeds = tuple(self.seed_schedule)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise PairedSchedulerError("seed_schedule must contain integer seeds")
        object.__setattr__(self, "seed_schedule", seeds)
        for name in ("provider", "gpu_budget", "round_budget"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        for name in ("token_budget", "call_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PairedSchedulerError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.profile_entry_count, bool)
            or not isinstance(self.profile_entry_count, int)
            or self.profile_entry_count < 1
        ):
            raise PairedSchedulerError("profile_entry_count must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return _clone(
            {
                "schema": self.schema,
                "dataset": self.dataset,
                "split": self.split,
                "evaluator": self.evaluator,
                "seed_schedule": list(self.seed_schedule),
                "provider": self.provider,
                "model": self.model,
                "token_budget": self.token_budget,
                "call_budget": self.call_budget,
                "gpu_budget": self.gpu_budget,
                "round_budget": self.round_budget,
                "runtime": self.runtime,
                "tool": self.tool,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "profile_manifest_digest": self.profile_manifest_digest,
                "profile_entry_count": self.profile_entry_count,
            }
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def copy_for_request(self) -> "FrozenExogenousConditions":
        data = self.to_dict()
        return type(self)(
            dataset=data["dataset"],
            split=data["split"],
            evaluator=data["evaluator"],
            seed_schedule=tuple(data["seed_schedule"]),
            provider=data["provider"],
            model=data["model"],
            token_budget=data["token_budget"],
            call_budget=data["call_budget"],
            gpu_budget=data["gpu_budget"],
            round_budget=data["round_budget"],
            runtime=data["runtime"],
            tool=data["tool"],
            profile_ref=data["profile_ref"],
            profile_digest=data["profile_digest"],
            profile_manifest_digest=data["profile_manifest_digest"],
            profile_entry_count=data["profile_entry_count"],
        )


@dataclass(frozen=True, slots=True)
class ArmRequest:
    """State-free request handed to exactly one arm opportunity."""

    campaign_id: str
    round_index: int
    opportunity_id: str
    arm: str
    arm_order: str
    seed: int
    conditions: FrozenExogenousConditions
    device: str | None = None

    schema = _REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _require_text(self.campaign_id, "campaign_id")
        _require_text(self.opportunity_id, "opportunity_id")
        if self.arm not in ARM_NAMES or self.arm_order not in (ARM_ORDER_AB, ARM_ORDER_BA):
            raise PairedSchedulerError("unknown arm or arm order")
        if not isinstance(self.conditions, FrozenExogenousConditions):
            raise PairedSchedulerError("conditions must be FrozenExogenousConditions")
        if self.round_index < 1 or self.round_index > len(self.conditions.seed_schedule):
            raise PairedSchedulerError("round_index is outside the seed schedule")
        if self.seed != self.conditions.seed_schedule[self.round_index - 1]:
            raise PairedSchedulerError("request seed does not match seed schedule")
        if self.device is not None:
            _require_text(self.device, "device")

    def to_dict(self) -> dict[str, Any]:
        return _clone(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "opportunity_id": self.opportunity_id,
                "arm": self.arm,
                "arm_order": self.arm_order,
                "seed": self.seed,
                "device": self.device,
                "conditions": self.conditions.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Public result returned by an arm adapter."""

    status: str
    public_metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None

    def __post_init__(self) -> None:
        status = str(self.status).upper()
        if status not in _STATUSES:
            raise PairedSchedulerError(f"unknown arm status: {self.status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "public_metrics", _mapping(self.public_metrics, "public_metrics"))
        if self.failure_code is None and status != STATUS_COMPLETED:
            object.__setattr__(self, "failure_code", f"ARM_REPORTED_{status}")

    @classmethod
    def completed(cls, public_metrics: Mapping[str, Any]) -> "ArmResult":
        return cls(STATUS_COMPLETED, public_metrics)

    @classmethod
    def missing(
        cls,
        failure_code: str = "ARM_REPORTED_MISSING",
        public_metrics: Mapping[str, Any] | None = None,
    ) -> "ArmResult":
        return cls(STATUS_MISSING, {} if public_metrics is None else public_metrics, failure_code)

    @classmethod
    def censored(
        cls,
        failure_code: str = "ARM_REPORTED_CENSORED",
        public_metrics: Mapping[str, Any] | None = None,
    ) -> "ArmResult":
        return cls(STATUS_CENSORED, {} if public_metrics is None else public_metrics, failure_code)


class ArmRunner(Protocol):
    """Callable adapter; its closure/object owns its arm root and state."""

    def __call__(self, request: ArmRequest) -> ArmResult: ...


@dataclass(frozen=True, slots=True)
class ArmReceipt:
    """Durable receipt containing one opportunity's public result."""

    campaign_id: str
    round_index: int
    arm: str
    opportunity_id: str
    manifest_digest: str
    status: str
    public_metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_code: str | None = None
    device: str | None = None

    schema = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.arm not in ARM_NAMES or self.round_index < 1:
            raise PairedSchedulerError("invalid receipt identity")
        _require_digest(self.manifest_digest, "manifest_digest")
        status = str(self.status).upper()
        if status not in _STATUSES:
            raise PairedSchedulerError(f"unknown receipt status: {self.status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "public_metrics", _mapping(self.public_metrics, "public_metrics"))
        if self.failure_code is None and status != STATUS_COMPLETED:
            object.__setattr__(self, "failure_code", f"ARM_{status}")
        if self.device is not None:
            _require_text(self.device, "device")

    def to_dict(self) -> dict[str, Any]:
        return _clone(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "arm": self.arm,
                "opportunity_id": self.opportunity_id,
                "manifest_digest": self.manifest_digest,
                "status": self.status,
                "public_metrics": self.public_metrics,
                "failure_code": self.failure_code,
                "device": self.device,
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArmReceipt":
        data = _mapping(value, "arm_receipt")
        if data.get("schema") != cls.schema:
            raise PairedSchedulerError("unexpected arm receipt schema")
        return cls(
            campaign_id=data["campaign_id"],
            round_index=data["round_index"],
            arm=data["arm"],
            opportunity_id=data["opportunity_id"],
            manifest_digest=data["manifest_digest"],
            status=data["status"],
            public_metrics=data.get("public_metrics", {}),
            failure_code=data.get("failure_code"),
            device=data.get("device"),
        )


def _number(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _reported(receipt: ArmReceipt | None, key: str) -> float | None:
    if receipt is None or receipt.status != STATUS_COMPLETED:
        return None
    value = receipt.public_metrics.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mechanisms(metrics: Mapping[str, Any]) -> set[str]:
    value = metrics.get("mechanism_ids", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {item for item in value if isinstance(item, str) and item.strip() == item}


@dataclass(frozen=True, slots=True)
class ArmAggregate:
    arm: str
    opportunities: int
    completed_opportunities: int
    missing_opportunities: int
    censored_opportunities: int
    completion_rate: float
    ndcg_trajectory: tuple[float | None, ...]
    frontier_trajectory: tuple[float | None, ...]
    cumulative_ndcg: float
    cumulative_frontier: float
    cumulative_incumbent_relative_improvement: float
    own_incumbent_improvement_count: int
    own_incumbent_improvement_magnitude: float
    total_cost: float
    total_hours: float
    total_episodes: float
    episodes_per_hour: float
    mechanism_diversity: int
    mechanism_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PairedAggregate:
    opportunities: int
    completed_opportunities: int
    missing_opportunities: int
    censored_opportunities: int
    completion_rate: float
    ndcg_trajectory: Mapping[str, tuple[float | None, ...]]
    frontier_trajectory: Mapping[str, tuple[float | None, ...]]
    cumulative_ndcg: float
    cumulative_frontier: float
    cumulative_incumbent_relative_improvement: float
    own_incumbent_improvement_count: int
    own_incumbent_improvement_magnitude: float
    own_incumbent_improvement_count_by_arm: Mapping[str, int]
    own_incumbent_improvement_magnitude_by_arm: Mapping[str, float]
    total_cost: float
    total_hours: float
    total_episodes: float
    episodes_per_hour: float
    mechanism_diversity: int
    mechanism_ids: tuple[str, ...]
    by_arm: Mapping[str, ArmAggregate]


def _trajectory(
    receipts: Sequence[ArmReceipt],
    key: str,
    rounds: int,
) -> tuple[float | None, ...]:
    by_round = {receipt.round_index: receipt for receipt in receipts}
    return tuple(_reported(by_round.get(round_index), key) for round_index in range(1, rounds + 1))


def _aggregate_arm(arm: str, receipts: Sequence[ArmReceipt], rounds: int) -> ArmAggregate:
    ndcg = _trajectory(receipts, "ndcg", rounds)
    frontier = _trajectory(receipts, "frontier", rounds)
    improvement = _trajectory(receipts, "incumbent_relative_improvement", rounds)
    positive = [value for value in improvement if value is not None and value > 0]
    mechanisms = sorted(
        set().union(*(_mechanisms(receipt.public_metrics) for receipt in receipts))
        if receipts
        else set()
    )
    opportunities = len(receipts)
    completed = sum(receipt.status == STATUS_COMPLETED for receipt in receipts)
    missing = sum(receipt.status == STATUS_MISSING for receipt in receipts)
    censored = sum(receipt.status == STATUS_CENSORED for receipt in receipts)
    hours = sum(_number(receipt.public_metrics, "elapsed_hours") for receipt in receipts)
    episodes = sum(_number(receipt.public_metrics, "episodes") for receipt in receipts)
    return ArmAggregate(
        arm=arm,
        opportunities=opportunities,
        completed_opportunities=completed,
        missing_opportunities=missing,
        censored_opportunities=censored,
        completion_rate=completed / opportunities if opportunities else 0.0,
        ndcg_trajectory=ndcg,
        frontier_trajectory=frontier,
        cumulative_ndcg=sum(value for value in ndcg if value is not None),
        cumulative_frontier=sum(value for value in frontier if value is not None),
        cumulative_incumbent_relative_improvement=sum(
            value for value in improvement if value is not None
        ),
        own_incumbent_improvement_count=len(positive),
        own_incumbent_improvement_magnitude=sum(positive),
        total_cost=sum(_number(receipt.public_metrics, "cost") for receipt in receipts),
        total_hours=hours,
        total_episodes=episodes,
        episodes_per_hour=episodes / hours if hours else 0.0,
        mechanism_diversity=len(mechanisms),
        mechanism_ids=tuple(mechanisms),
    )


def aggregate_paired_receipts(
    receipts: Iterable[ArmReceipt],
    *,
    rounds: int | None = None,
) -> PairedAggregate:
    """Aggregate public metrics and retain nulls for missing/censored outcomes."""

    items = list(receipts)
    if rounds is None:
        rounds = max((receipt.round_index for receipt in items), default=0)
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise PairedSchedulerError("rounds must be a non-negative integer")
    grouped: dict[str, list[ArmReceipt]] = {arm: [] for arm in ARM_NAMES}
    seen: set[tuple[str, int]] = set()
    for receipt in items:
        if not isinstance(receipt, ArmReceipt) or receipt.arm not in grouped:
            raise PairedSchedulerError("aggregate input must contain paired ArmReceipt objects")
        if receipt.round_index > rounds:
            raise PairedSchedulerError("receipt is outside the aggregate round range")
        key = (receipt.arm, receipt.round_index)
        if key in seen:
            raise PairedSchedulerError("duplicate arm opportunity")
        seen.add(key)
        grouped[receipt.arm].append(receipt)
    by_arm = {arm: _aggregate_arm(arm, grouped[arm], rounds) for arm in ARM_NAMES}
    ndcg = {arm: value.ndcg_trajectory for arm, value in by_arm.items()}
    frontier = {arm: value.frontier_trajectory for arm, value in by_arm.items()}
    all_receipts = [receipt for arm in ARM_NAMES for receipt in grouped[arm]]
    mechanisms = sorted(
        set().union(*(_mechanisms(receipt.public_metrics) for receipt in all_receipts))
        if all_receipts
        else set()
    )
    opportunities = len(all_receipts)
    completed = sum(receipt.status == STATUS_COMPLETED for receipt in all_receipts)
    missing = sum(receipt.status == STATUS_MISSING for receipt in all_receipts)
    censored = sum(receipt.status == STATUS_CENSORED for receipt in all_receipts)
    hours = sum(value.total_hours for value in by_arm.values())
    episodes = sum(value.total_episodes for value in by_arm.values())
    count_by_arm = {
        arm: value.own_incumbent_improvement_count for arm, value in by_arm.items()
    }
    magnitude_by_arm = {
        arm: value.own_incumbent_improvement_magnitude for arm, value in by_arm.items()
    }
    return PairedAggregate(
        opportunities=opportunities,
        completed_opportunities=completed,
        missing_opportunities=missing,
        censored_opportunities=censored,
        completion_rate=completed / opportunities if opportunities else 0.0,
        ndcg_trajectory=ndcg,
        frontier_trajectory=frontier,
        cumulative_ndcg=sum(value.cumulative_ndcg for value in by_arm.values()),
        cumulative_frontier=sum(value.cumulative_frontier for value in by_arm.values()),
        cumulative_incumbent_relative_improvement=sum(
            value.cumulative_incumbent_relative_improvement for value in by_arm.values()
        ),
        own_incumbent_improvement_count=sum(count_by_arm.values()),
        own_incumbent_improvement_magnitude=sum(magnitude_by_arm.values()),
        own_incumbent_improvement_count_by_arm=count_by_arm,
        own_incumbent_improvement_magnitude_by_arm=magnitude_by_arm,
        total_cost=sum(value.total_cost for value in by_arm.values()),
        total_hours=hours,
        total_episodes=episodes,
        episodes_per_hour=episodes / hours if hours else 0.0,
        mechanism_diversity=len(mechanisms),
        mechanism_ids=tuple(mechanisms),
        by_arm=by_arm,
    )


@dataclass(frozen=True, slots=True)
class PairedRoundResult:
    round_index: int
    arm_order: str
    receipts: tuple[ArmReceipt, ...]


@dataclass(frozen=True, slots=True)
class PairedCampaignResult:
    campaign_id: str
    root: Path
    rounds: tuple[PairedRoundResult, ...]
    aggregate: PairedAggregate

    @property
    def receipts(self) -> tuple[ArmReceipt, ...]:
        return tuple(receipt for result in self.rounds for receipt in result.receipts)


@dataclass(frozen=True, slots=True)
class _PreparedArm:
    request: ArmRequest
    receipt_path: Path


def _write_once(path: Path, value: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _clone(dict(value))
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PairedSchedulerError(f"cannot read existing artifact: {path}") from error
        if existing != payload:
            raise PairedSchedulerError(f"create-once artifact differs: {path}")
        return False


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedSchedulerError(f"cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PairedSchedulerError(f"JSON artifact must be an object: {path}")
    return value


def _round_order(round_index: int) -> tuple[str, tuple[str, str]]:
    return (
        (ARM_ORDER_AB, (RESEARCH_LINE_ARM, ORIGINAL_RECCLAW_ARM))
        if round_index % 2
        else (ARM_ORDER_BA, (ORIGINAL_RECCLAW_ARM, RESEARCH_LINE_ARM))
    )


class PairedScheduler:
    """Schedule the two independent arm adapters and nothing else."""

    def __init__(
        self,
        *,
        root: Path,
        campaign_id: str,
        conditions: FrozenExogenousConditions,
        research_line: ArmRunner,
        original_recclaw: ArmRunner,
        parallel_arms: bool = False,
        independent_arm_queues: bool = False,
    ) -> None:
        self.root = Path(root)
        _require_text(campaign_id, "campaign_id")
        if not isinstance(conditions, FrozenExogenousConditions):
            raise PairedSchedulerError("conditions must be FrozenExogenousConditions")
        if not callable(research_line) or not callable(original_recclaw):
            raise PairedSchedulerError("both arm adapters must be callable")
        self.campaign_id = campaign_id
        self.conditions = conditions
        self.parallel_arms = parallel_arms
        self.independent_arm_queues = independent_arm_queues
        self._runners = {
            RESEARCH_LINE_ARM: research_line,
            ORIGINAL_RECCLAW_ARM: original_recclaw,
        }

    def run(
        self,
        rounds: int | None = None,
        *,
        parallel_arms: bool | None = None,
    ) -> PairedCampaignResult:
        target = len(self.conditions.seed_schedule) if rounds is None else rounds
        if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= len(self.conditions.seed_schedule):
            raise PairedSchedulerError("rounds must be within the frozen seed schedule")
        parallel = self.parallel_arms if parallel_arms is None else parallel_arms
        if parallel and len(set(self._devices())) < 2:
            parallel = False
        if self.independent_arm_queues and parallel:
            return self._run_independent_arm_queues(target)
        results: list[PairedRoundResult] = []
        for round_index in range(1, target + 1):
            arm_order, execution_order = _round_order(round_index)
            manifest = self._manifest(round_index)
            device_assignments = manifest["device_assignments"]
            round_directory = self.root / "rounds" / f"round-{round_index:04d}"
            _write_once(round_directory / "manifest.json", manifest)
            manifest_digest = _digest(manifest)
            if parallel:
                prepared: list[ArmReceipt | _PreparedArm] = [
                    self._prepare_arm(
                        round_index,
                        arm_order,
                        arm,
                        device_assignments[arm],
                        manifest_digest,
                        round_directory,
                    )
                    for arm in execution_order
                ]
                receipts = self._finish_arms(prepared, parallel=True)
            else:
                serial_receipts: list[ArmReceipt] = []
                for arm in execution_order:
                    prepared = self._prepare_arm(
                        round_index,
                        arm_order,
                        arm,
                        device_assignments[arm],
                        manifest_digest,
                        round_directory,
                    )
                    serial_receipts.append(
                        prepared
                        if isinstance(prepared, ArmReceipt)
                        else self._finish_one(prepared, False)
                    )
                receipts = tuple(serial_receipts)
            results.append(PairedRoundResult(round_index, arm_order, receipts))
        receipts = tuple(receipt for result in results for receipt in result.receipts)
        return PairedCampaignResult(
            self.campaign_id,
            self.root,
            tuple(results),
            aggregate_paired_receipts(receipts, rounds=target),
        )

    def _run_independent_arm_queues(self, target: int) -> PairedCampaignResult:
        """Advance each arm sequentially without a cross-arm round barrier."""

        for round_index in range(1, target + 1):
            _write_once(
                self.root / "rounds" / f"round-{round_index:04d}" / "manifest.json",
                self._manifest(round_index),
            )

        def run_arm(arm: str) -> tuple[ArmReceipt, ...]:
            arm_receipts: list[ArmReceipt] = []
            for round_index in range(1, target + 1):
                arm_order, _execution_order = _round_order(round_index)
                round_directory = self.root / "rounds" / f"round-{round_index:04d}"
                manifest_digest = _digest(self._manifest(round_index))
                prepared = self._prepare_arm(
                    round_index,
                    arm_order,
                    arm,
                    self._device_for(round_index, arm),
                    manifest_digest,
                    round_directory,
                )
                arm_receipts.append(
                    prepared
                    if isinstance(prepared, ArmReceipt)
                    else self._finish_one(prepared, True)
                )
            return tuple(arm_receipts)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paired-arm-queue") as pool:
            futures = {arm: pool.submit(run_arm, arm) for arm in ARM_NAMES}
            by_arm = {arm: future.result() for arm, future in futures.items()}

        results: list[PairedRoundResult] = []
        for round_index in range(1, target + 1):
            arm_order, execution_order = _round_order(round_index)
            receipts = tuple(
                by_arm[arm][round_index - 1] for arm in execution_order
            )
            results.append(PairedRoundResult(round_index, arm_order, receipts))
        receipts = tuple(receipt for result in results for receipt in result.receipts)
        return PairedCampaignResult(
            self.campaign_id,
            self.root,
            tuple(results),
            aggregate_paired_receipts(receipts, rounds=target),
        )

    def _devices(self) -> tuple[str, ...]:
        raw = self.conditions.gpu_budget.get("devices")
        if raw is None:
            raw = self.conditions.gpu_budget.get("gpu")
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, Sequence):
            return tuple(value for value in raw if isinstance(value, str) and value.strip() == value)
        return ()

    def _device_for(self, round_index: int, arm: str) -> str | None:
        devices = self._devices()
        if not devices:
            return None
        arm_slot = ARM_NAMES.index(arm)
        if self.independent_arm_queues:
            return devices[arm_slot % len(devices)]
        return devices[(round_index - 1 + arm_slot) % len(devices)]

    def _manifest(self, round_index: int) -> dict[str, Any]:
        arm_order, execution_order = _round_order(round_index)
        return {
            "schema": _MANIFEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "round_index": round_index,
            "arm_order": arm_order,
            "execution_order": list(execution_order),
            "scheduling_mode": (
                SCHEDULING_INDEPENDENT_ARM_QUEUES
                if self.independent_arm_queues
                else SCHEDULING_ROUND_BARRIER
            ),
            "opportunity_ids": {
                arm: f"{self.campaign_id}:round-{round_index}:{arm}"
                for arm in ARM_NAMES
            },
            "device_assignments": {
                arm: self._device_for(round_index, arm) for arm in ARM_NAMES
            },
            "seed": self.conditions.seed_schedule[round_index - 1],
            "conditions": self.conditions.to_dict(),
        }

    def _prepare_arm(
        self,
        round_index: int,
        arm_order: str,
        arm: str,
        device: str | None,
        manifest_digest: str,
        round_directory: Path,
    ) -> ArmReceipt | _PreparedArm:
        opportunity_id = f"{self.campaign_id}:round-{round_index}:{arm}"
        receipt_path = round_directory / f"{arm}.receipt.json"
        if receipt_path.exists():
            return self._load_receipt(receipt_path, opportunity_id, manifest_digest, device)
        created = _write_once(
            round_directory / f"{arm}.attempt.json",
            {
                "schema": _ATTEMPT_SCHEMA,
                "campaign_id": self.campaign_id,
                "round_index": round_index,
                "arm": arm,
                "opportunity_id": opportunity_id,
                "manifest_digest": manifest_digest,
                "device": device,
            },
        )
        if receipt_path.exists():
            return self._load_receipt(receipt_path, opportunity_id, manifest_digest, device)
        if not created:
            receipt = ArmReceipt(
                self.campaign_id,
                round_index,
                arm,
                opportunity_id,
                manifest_digest,
                STATUS_CENSORED,
                failure_code="INTERRUPTED_BEFORE_RECEIPT",
                device=device,
            )
            _write_once(receipt_path, receipt.to_dict())
            return receipt
        return _PreparedArm(
            request=ArmRequest(
                self.campaign_id,
                round_index,
                opportunity_id,
                arm,
                arm_order,
                self.conditions.seed_schedule[round_index - 1],
                self.conditions.copy_for_request(),
                device,
            ),
            receipt_path=receipt_path,
        )

    def _finish_arms(
        self,
        prepared: Sequence[ArmReceipt | _PreparedArm],
        *,
        parallel: bool,
    ) -> tuple[ArmReceipt, ...]:
        existing = [item for item in prepared if isinstance(item, ArmReceipt)]
        fresh = [item for item in prepared if isinstance(item, _PreparedArm)]
        if parallel and fresh:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paired-arm") as pool:
                futures = [pool.submit(self._finish_one, item, True) for item in fresh]
                fresh_receipts = [future.result() for future in futures]
        else:
            fresh_receipts = [self._finish_one(item, False) for item in fresh]
        by_arm = {receipt.arm: receipt for receipt in (*existing, *fresh_receipts)}
        return tuple(by_arm[item.request.arm] if isinstance(item, _PreparedArm) else item for item in prepared)

    def _finish_one(self, prepared: _PreparedArm, catch_base: bool) -> ArmReceipt:
        request = prepared.request
        started_ns = time.monotonic_ns()
        try:
            result = self._runners[request.arm](request)
            if not isinstance(result, ArmResult):
                raise TypeError("adapter must return ArmResult")
            elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
            public_metrics = dict(result.public_metrics)
            public_metrics["cost"] = elapsed_seconds
            public_metrics["elapsed_hours"] = elapsed_seconds / 3600
            public_metrics.setdefault("episodes", 0)
            receipt = ArmReceipt(
                self.campaign_id,
                request.round_index,
                request.arm,
                request.opportunity_id,
                _digest(
                    self._manifest_for_request(request)
                ),
                result.status,
                public_metrics,
                result.failure_code,
                request.device,
            )
        except Exception as error:
            receipt = self._failure_receipt(
                request,
                STATUS_CENSORED,
                error,
                elapsed_seconds=(time.monotonic_ns() - started_ns) / 1_000_000_000,
            )
        except BaseException as error:
            if not catch_base:
                raise
            receipt = self._failure_receipt(
                request,
                STATUS_CENSORED,
                error,
                elapsed_seconds=(time.monotonic_ns() - started_ns) / 1_000_000_000,
            )
        _write_once(prepared.receipt_path, receipt.to_dict())
        return receipt

    def _failure_receipt(
        self,
        request: ArmRequest,
        status: str,
        error: BaseException,
        *,
        elapsed_seconds: float,
    ) -> ArmReceipt:
        traceback.print_exception(error)
        detail = " ".join(str(error).split())[:240]
        return ArmReceipt(
            self.campaign_id,
            request.round_index,
            request.arm,
            request.opportunity_id,
            _digest(self._manifest_for_request(request)),
            status,
            public_metrics={
                "cost": elapsed_seconds,
                "elapsed_hours": elapsed_seconds / 3600,
                "episodes": 0,
            },
            failure_code=(
                f"ARM_EXCEPTION_{type(error).__name__}"
                + (f":{detail}" if detail else "")
            ),
            device=request.device,
        )

    def _manifest_for_request(self, request: ArmRequest) -> dict[str, Any]:
        return self._manifest(request.round_index)

    def _load_receipt(
        self,
        path: Path,
        opportunity_id: str,
        manifest_digest: str,
        device: str | None,
    ) -> ArmReceipt:
        receipt = ArmReceipt.from_dict(_read_object(path))
        if (
            receipt.campaign_id != self.campaign_id
            or receipt.opportunity_id != opportunity_id
            or receipt.manifest_digest != manifest_digest
            or receipt.device != device
        ):
            raise PairedSchedulerError(f"receipt identity mismatch: {path}")
        return receipt


def run_paired_campaign(
    *,
    root: Path,
    campaign_id: str,
    conditions: FrozenExogenousConditions,
    research_line: ArmRunner,
    original_recclaw: ArmRunner,
    rounds: int | None = None,
    parallel_arms: bool = False,
    independent_arm_queues: bool = False,
) -> PairedCampaignResult:
    return PairedScheduler(
        root=root,
        campaign_id=campaign_id,
        conditions=conditions,
        research_line=research_line,
        original_recclaw=original_recclaw,
        parallel_arms=parallel_arms,
        independent_arm_queues=independent_arm_queues,
    ).run(rounds)


__all__ = [
    "ARM_NAMES",
    "ARM_ORDER_AB",
    "ARM_ORDER_BA",
    "SCHEDULING_INDEPENDENT_ARM_QUEUES",
    "SCHEDULING_ROUND_BARRIER",
    "ORIGINAL_RECCLAW_ARM",
    "RESEARCH_LINE_ARM",
    "STATUS_CENSORED",
    "STATUS_COMPLETED",
    "STATUS_MISSING",
    "ArmAggregate",
    "ArmReceipt",
    "ArmRequest",
    "ArmResult",
    "ArmRunner",
    "FrozenExogenousConditions",
    "PairedAggregate",
    "PairedCampaignResult",
    "PairedRoundResult",
    "PairedScheduler",
    "PairedSchedulerError",
    "aggregate_paired_receipts",
    "run_paired_campaign",
]
