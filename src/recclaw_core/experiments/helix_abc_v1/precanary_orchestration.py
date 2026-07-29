"""Minimum-sufficient, no-training M4 three-arm orchestration.

The neutral scheduler is the sole state-store writer. Arm workers receive only
opaque identities, arm-private filesystem capabilities, a frozen proposal
session, and public fused feedback. This module intentionally does not provide
real LLM or training backends.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space.canonical import deep_thaw

from recclaw_core.helix.composition import SameSlateHelixSelectorV1
from recclaw_core.helix.contracts import (
    CandidateEnvelope,
    GuardContext,
    RawResultEnvelope,
)
from recclaw_core.helix.guard_adapter import EvidenceGuardPortV1
from recclaw_core.helix.ledger import EvidenceGuardLedgerWriterV1
from recclaw_core.helix.ports import NullEvidencePortV1
from recclaw_core.helix.scientific_attribution import (
    DeterministicHelixAdmissionV13,
    FrontierEligibilityV2,
    FusedSearchFeedbackV2,
    NOT_AVAILABLE,
    ResearchTaskQueueV1,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchFeedbackClassV2,
    SearchUtilityEventV2,
)

from .canonical import canonical_json_bytes, canonical_value, content_id, sha256_digest
from .broker_failure_closure import (
    BrokerFailureClosureV1,
    close_broker_failure,
)
from .canary_broker import (
    CanaryBrokerError,
    PostProviderSemanticRejectionV1,
)
from .campaign_runtime import (
    CampaignRuntimeError,
    campaign_runtime_profile,
    executable_mechanism,
    execution_recipe_for_program,
    program_from_proposal as campaign_program_from_proposal,
    root_parent_mechanism_id,
)
from .common_execution_guard import CommonExecutionGuardV1
from .compilation_cache import compile_campaign_program as compile_program
from .contracts import (
    ArmCode,
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from .controllers import OriginalControllerV1
from .integrated_state_core import (
    IntegratedCampaignStateCoreV1,
    ObservationPathV1,
    ProposalSourceV1,
    RoundBoundaryEventV1,
    canonical_observation_path,
)
from .fake_runner import (
    PackageOwnedLauncherV1,
    register_raw_result_envelope,
)
from .materialization import (
    DeterministicMaterializerV1,
    build_binding_v2,
    classify_execution_trust,
    development_execution_gate,
)
from .research_capability import (
    DISCOVERY_PRODUCERS,
    FixtureProducerBrokerV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    initial_research_policy,
)
from .research_contracts import (
    CandidateProposalV2,
    CandidateProposalV3,
    CandidateProposalV4,
    DevelopmentalMechanismBeliefV1,
    DevelopmentalMechanismBeliefV2,
    ProducerSessionResultV1,
    ResearchTaskRefV1,
)
from .research_controller import (
    ResearchLineControllerV1,
    ResearchRoundPlanV1,
)
from .runtime_release import (
    common_release_projection_digest,
    development_protocol,
)
from .state_store import (
    ClaimExecutionCommand,
    CloseRoundCommand,
    OpenRoundCommand,
    RegisterArtifactCommand,
    ResourceDebitV1,
    SingleWriterExperimentStoreV1,
)
from .training_runtime_contracts import ExecutionSeedBindingV1


class PreCanaryInvariantError(RuntimeError):
    pass


class BrokerRoundFailureError(PreCanaryInvariantError):
    def __init__(self, closure: BrokerFailureClosureV1) -> None:
        super().__init__(
            "Pilot Broker round closed as "
            f"{closure.failure_class}/COMMON_NO_EXECUTION"
        )
        self.closure = closure


def m4_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=200,
        total_output_tokens=200,
        total_billed_token_debit=400,
        total_proposal_count=4,
        wall_time_ms=1000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=0,
        gpu_cost_microunits=0,
    )


def _round_execution_budget_debits(
    *,
    gpu_device_time_ms: int,
    gpu_cost_microunits: int,
    wall_time_ms: int,
    ceilings: ResourceCeilingsV1,
    resource_ceiling_rejected: bool,
) -> tuple[ResourceDebitV1, ...]:
    """Debit the allocation while preserving actual use in result artifacts."""

    if resource_ceiling_rejected:
        gpu_device_time_ms = min(
            gpu_device_time_ms,
            ceilings.gpu_device_time_ms,
        )
        gpu_cost_microunits = min(
            gpu_cost_microunits,
            ceilings.gpu_cost_microunits,
        )
        wall_time_ms = min(wall_time_ms, ceilings.wall_time_ms)
    return (
        ResourceDebitV1("GPU_DEVICE_TIME_MS", gpu_device_time_ms),
        ResourceDebitV1("GPU_COST_MICROUNITS", gpu_cost_microunits),
        ResourceDebitV1("WALL_TIME_MS", wall_time_ms),
    )


@dataclass(frozen=True, slots=True)
class TreatmentAssignmentEnvelopeV1:
    experiment_id: str
    opaque_instance_ids: tuple[str, str, str]
    assignment_commitment: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PrivateTreatmentAssignmentV1:
    experiment_id: str
    nonce_digest: str
    arm_to_instance: tuple[tuple[ArmCode, str], ...]

    @classmethod
    def create(
        cls, experiment_id: str, *, nonce: str
    ) -> "PrivateTreatmentAssignmentV1":
        nonce_digest = sha256_digest({"nonce": nonce})
        mapping = tuple(
            (
                arm,
                "inst-"
                + sha256_digest(
                    {
                        "arm": arm.value,
                        "experiment_id": experiment_id,
                        "nonce_digest": nonce_digest,
                    }
                )[:24],
            )
            for arm in ArmCode
        )
        return cls(experiment_id, nonce_digest, mapping)

    @property
    def mapping(self) -> dict[ArmCode, str]:
        return dict(self.arm_to_instance)

    @property
    def commitment(self) -> str:
        return sha256_digest(
            {
                "experiment_id": self.experiment_id,
                "mapping": [
                    {"arm": arm.value, "opaque_instance_id": instance}
                    for arm, instance in self.arm_to_instance
                ],
                "nonce_digest": self.nonce_digest,
            }
        )

    def neutral_envelope(self) -> TreatmentAssignmentEnvelopeV1:
        return TreatmentAssignmentEnvelopeV1(
            experiment_id=self.experiment_id,
            opaque_instance_ids=tuple(sorted(self.mapping.values())),
            assignment_commitment=self.commitment,
        )


_ARM_PRIVATE_NAMESPACES = (
    "cache",
    "candidate",
    "memory",
    "prompt",
    "raw",
    "registry",
    "runtime",
    "tmp",
)


@dataclass(frozen=True, slots=True)
class ArmPrivateRootsV1:
    opaque_instance_id: str
    root: Path
    namespaces: tuple[tuple[str, Path], ...]

    def __post_init__(self) -> None:
        if not self.opaque_instance_id:
            raise PreCanaryInvariantError("opaque instance identity is required")
        root = self.root.resolve()
        mapping = dict(self.namespaces)
        if set(mapping) != set(_ARM_PRIVATE_NAMESPACES):
            raise PreCanaryInvariantError("Arm capability namespace set is not exact")
        identities: set[tuple[int, int]] = set()
        for path in mapping.values():
            if path.is_symlink() or path.resolve().parent != root:
                raise PreCanaryInvariantError(
                    "Arm namespace must be a direct private-root child"
                )
            stat = path.stat()
            identities.add((stat.st_dev, stat.st_ino))
        if len(identities) != len(mapping):
            raise PreCanaryInvariantError("Arm namespace aliases another root")

    def namespace(self, name: str) -> Path:
        try:
            return dict(self.namespaces)[name]
        except KeyError as error:
            raise PreCanaryInvariantError("namespace is not in the Arm capability") from error


@dataclass(frozen=True, slots=True)
class RuntimeLayoutV1:
    root: Path
    neutral_root: Path
    stability_root: Path
    arm_roots: tuple[ArmPrivateRootsV1, ...]
    evidence_root_by_instance: tuple[tuple[str, Path], ...]

    @classmethod
    def materialize(
        cls,
        root: Path,
        assignment: PrivateTreatmentAssignmentV1,
    ) -> "RuntimeLayoutV1":
        root = root.resolve()
        neutral = root / "neutral"
        stability = root / "post_selection_stability"
        neutral.mkdir(parents=True, exist_ok=False)
        stability.mkdir(parents=True, exist_ok=False)
        arms: list[ArmPrivateRootsV1] = []
        resolved: set[Path] = {neutral.resolve(), stability.resolve()}
        for opaque_id in assignment.mapping.values():
            arm_root = root / "instances" / opaque_id
            namespaces = []
            for name in _ARM_PRIVATE_NAMESPACES:
                path = arm_root / name
                path.mkdir(parents=True, exist_ok=False)
                namespaces.append((name, path))
                resolved.add(path.resolve())
            arms.append(
                ArmPrivateRootsV1(
                    opaque_instance_id=opaque_id,
                    root=arm_root,
                    namespaces=tuple(namespaces),
                )
            )
        evidence: list[tuple[str, Path]] = []
        c_id = assignment.mapping[ArmCode.C]
        c_evidence = root / "evidence_audit" / c_id
        c_evidence.mkdir(parents=True, exist_ok=False)
        evidence.append((c_id, c_evidence))
        if len(resolved) != 2 + 3 * len(_ARM_PRIVATE_NAMESPACES):
            raise PreCanaryInvariantError("runtime roots alias each other")
        return cls(root, neutral, stability, tuple(arms), tuple(evidence))

    def arm(self, opaque_instance_id: str) -> ArmPrivateRootsV1:
        try:
            return next(
                item
                for item in self.arm_roots
                if item.opaque_instance_id == opaque_instance_id
            )
        except StopIteration as error:
            raise PreCanaryInvariantError("unknown opaque instance") from error

    def evidence_root(self, opaque_instance_id: str) -> Path | None:
        return dict(self.evidence_root_by_instance).get(opaque_instance_id)


@dataclass(frozen=True, slots=True)
class ArmFilesystemCapabilityV1:
    roots: ArmPrivateRootsV1

    def _target(self, namespace: str, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise PreCanaryInvariantError("path is outside the Arm capability")
        base = self.roots.namespace(namespace)
        if base.is_symlink():
            raise PreCanaryInvariantError("namespace root must not be a symlink")
        target = base.joinpath(*relative.parts)
        cursor = base
        for part in relative.parts[:-1]:
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                raise PreCanaryInvariantError("symlink traversal is forbidden")
        if target.exists():
            stat = target.lstat()
            if target.is_symlink():
                raise PreCanaryInvariantError("symlink targets are forbidden")
            if stat.st_nlink != 1:
                raise PreCanaryInvariantError("hardlinked targets are forbidden")
        return target

    def write_bytes(self, namespace: str, relative_path: str, data: bytes) -> Path:
        target = self._target(namespace, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
        return target

    def read_bytes(self, namespace: str, relative_path: str) -> bytes:
        return self._target(namespace, relative_path).read_bytes()


def probe_uid_isolation(
    *,
    own_root: Path,
    sibling_root: Path,
    worker_uid: int,
) -> dict[str, bool]:
    """Exercise kernel DAC with an account-free numeric UID."""

    if os.geteuid() != 0:
        raise PreCanaryInvariantError("UID isolation probe requires the root test host")
    own_root.chmod(0o700)
    sibling_root.chmod(0o700)
    os.chown(own_root, worker_uid, worker_uid)
    os.chown(sibling_root, worker_uid + 1, worker_uid + 1)
    own = own_root / "owned.txt"
    sibling = sibling_root / "sibling.txt"
    own.write_text("own", encoding="utf-8")
    sibling.write_text("sibling", encoding="utf-8")
    os.chown(own, worker_uid, worker_uid)
    os.chown(sibling, worker_uid + 1, worker_uid + 1)
    script = (
        "from pathlib import Path; import sys; "
        "own=Path(sys.argv[1]); sibling=Path(sys.argv[2]); "
        "print(int(own.read_text()==\"own\")); "
        "\ntry:\n sibling.read_text(); print(0)\n"
        "except PermissionError:\n print(1)\n"
        "try:\n sibling.joinpath(\"write.txt\").write_text(\"x\"); print(0)\n"
        "except PermissionError:\n print(1)\n"
    )
    completed = subprocess.run(
        [
            "setpriv",
            f"--reuid={worker_uid}",
            f"--regid={worker_uid}",
            "--clear-groups",
            "/usr/bin/python3",
            "-c",
            script,
            str(own),
            str(sibling_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() == "1" for line in completed.stdout.splitlines()]
    if len(values) != 3:
        raise PreCanaryInvariantError("isolation probe produced an invalid result")
    return {
        "own_read": values[0],
        "sibling_read_denied": values[1],
        "sibling_write_denied": values[2],
    }


@dataclass(frozen=True, slots=True)
class FakeProposalSessionV1:
    validation_programs: tuple[Mapping[str, Any], ...]
    ordered_programs: tuple[Mapping[str, Any], ...]
    selected_candidate_id: str
    physical_call_count: int
    input_tokens: int
    output_tokens: int
    billed_tokens: int
    proposal_count: int
    proposal_session_digest: str
    route_trace_digest: str | None
    research_plan: ResearchRoundPlanV1 | None
    ordered_proposal_candidate_ids: tuple[str, ...] = ()
    research_proposals: tuple[
        CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4,
        ...,
    ] = ()
    producer_session: ProducerSessionResultV1 | None = None
    research_task: ResearchTaskV1 | None = None
    broker_call_latencies_ms: tuple[int, ...] = ()
    proposal_session_wall_time_ms: int = 0


@dataclass(slots=True)
class ThreeArmFakeBrokerV1:
    """No-network broker using the activated Original and M2 components."""

    research_controllers: dict[ArmCode, ResearchLineControllerV1]

    @classmethod
    def create(cls) -> "ThreeArmFakeBrokerV1":
        def controller() -> ResearchLineControllerV1:
            return ResearchLineControllerV1(
                producer_mode=(
                    ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
                ),
                policy=initial_research_policy(),
                broker=FixtureProducerBrokerV1(),
                router=StrongStaticRouterV1(),
                memory_writer=SearchMemoryWriterV1(
                    "DEVELOPMENT_ONLY/SEARCH_MEMORY"
                ),
            )

        return cls({ArmCode.B: controller(), ArmCode.C: controller()})

    @property
    def bc_controller_identity_digest(self) -> str:
        identities = {
            item.identity_digest for item in self.research_controllers.values()
        }
        if len(identities) != 1:
            raise PreCanaryInvariantError("B/C Research controller identity diverged")
        return next(iter(identities))

    def generate(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> FakeProposalSessionV1:
        if len(drafts) != 4:
            raise PreCanaryInvariantError("M4 fake batch must contain four frozen drafts")
        if arm is ArmCode.A:
            programs_by_id = {
                compile_program(item["mechanism_program"]).candidate_id: item[
                    "mechanism_program"
                ]
                for item in drafts
            }
            fixture = tuple(
                {
                    "candidate_id": compile_program(
                        item["mechanism_program"]
                    ).candidate_id,
                    "original_score": float(4 - index),
                }
                for index, item in enumerate(drafts)
            )
            controller = OriginalControllerV1()
            proposals = controller.propose(
                {
                    "execution_mode": "M0_FIXTURE_ONLY",
                    "fixture_proposals": fixture,
                },
                    {"space": "BL_ICF_EXECUTABLE_PROFILE_V2"},
                {"proposal_count": 4},
            )
            selected = controller.select(
                proposals,
                {"execution_mode": "M0_FIXTURE_ONLY"},
                {"proposal_count": 4},
            )
            ordered = tuple(programs_by_id[str(item["candidate_id"])] for item in proposals)
            session_digest = sha256_digest(
                {
                    "broker": "ThreeArmFakeBrokerV1",
                    "mode": "ORIGINAL_SINGLE_INVOCATION",
                    "ordered_programs": ordered,
                    "resource_envelope": ceilings,
                }
            )
            return FakeProposalSessionV1(
                validation_programs=ordered,
                ordered_programs=ordered,
                selected_candidate_id=str(selected["candidate_id"]),
                physical_call_count=1,
                input_tokens=ceilings.total_input_tokens,
                output_tokens=ceilings.total_output_tokens,
                billed_tokens=ceilings.total_billed_token_debit,
                proposal_count=4,
                proposal_session_digest=session_digest,
                route_trace_digest=None,
                research_plan=None,
                broker_call_latencies_ms=(0,),
            )
        controller = self.research_controllers[arm]
        role_memory = {
            role: {"prior_round_digest": sha256_digest({"role": role})}
            for role in DISCOVERY_PRODUCERS
        }
        session = controller.broker.dispatch(
            session_id=f"m4-research-{search_seed}-{round_index}",
            mode=controller.producer_mode,
            drafts=drafts,
            context={"round_index": round_index, "search_seed": search_seed},
            role_memory=role_memory,
            seed=search_seed,
            ceilings=ceilings,
            policy_projection=controller.policy.to_dict(),
        )
        route = controller.router.route(
            session.proposals, policy_projection=controller.policy.to_dict()
        )
        if route.selected_candidate_id is None:
            raise PreCanaryInvariantError("M4 fake Research route selected nothing")
        by_id = {item.candidate_id: item for item in session.proposals}
        ordered_ids = list(route.ranked_candidate_ids)
        ordered = tuple(by_id[item].mechanism_program for item in ordered_ids)
        plan = ResearchRoundPlanV1(
            round_index=round_index,
            proposal_session_digest=session.digest,
            route_trace_digest=route.digest,
            selected_candidate_id=route.selected_candidate_id,
            physical_call_count=session.physical_call_count,
            proposal_count=session.proposal_count,
            ordinary_execution_opportunities=1,
            plan_status="SELECTED",
            policy_digest=controller.policy.digest,
        )
        return FakeProposalSessionV1(
            validation_programs=tuple(
                item.mechanism_program for item in session.proposals
            ),
            ordered_programs=ordered,
            selected_candidate_id=route.selected_candidate_id,
            physical_call_count=session.physical_call_count,
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            billed_tokens=session.billed_tokens,
            proposal_count=session.proposal_count,
            proposal_session_digest=session.digest,
            route_trace_digest=route.digest,
            research_plan=plan,
            ordered_proposal_candidate_ids=tuple(ordered_ids),
            research_proposals=session.proposals,
            producer_session=session,
            broker_call_latencies_ms=tuple(
                item.latency_ms for item in session.calls
            ),
            proposal_session_wall_time_ms=session.session_latency_ms,
        )


def _guard_protocol() -> dict[str, Any]:
    return {
        "protocol_id": "PROTO-ML1M-FULL-001",
        "profile_family": "OFFLINE_TOPN",
        "dataset": "ml-1m",
        "dataset_snapshot": "ml-1m-snapshot-001",
        "split": {"strategy": "random_user_holdout", "ratio": [0.8, 0.1, 0.1]},
        "training_sampling": {"mode": "uniform_negative"},
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {"seen_items": "exclude"},
        "metric": {"name": "ndcg", "cutoff": 10},
        "training_procedure": {"optimizer": "adam", "max_epochs": 300},
    }


def _guard_context() -> GuardContext:
    return GuardContext(
        claim={
            "claim_id": "CLAIM-M4-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "claim_kind": "LOCAL_IMPROVEMENT",
            "target_model": "CandidateModel",
            "comparator": "LightGCN",
            "metric": "ndcg",
            "required_seed_count": 3,
            "scope": {"dataset": "ml-1m"},
        },
        protocol=_guard_protocol(),
        current_evidence={
            "snapshot_id": "M4-EMPTY",
            "claim_id": "CLAIM-M4-001",
            "protocol_id": "PROTO-ML1M-FULL-001",
            "observation_ids": [],
        },
    )


def _register_materialization_artifacts_m4(
    store: SingleWriterExperimentStoreV1,
    *,
    binding: Any,
    report: Any,
    opaque_instance_id: str,
) -> tuple[dict[str, Any], ...]:
    root = Path(str(binding.arm_private_root))
    type_by_suffix = {
        "handler_config.json": "M1_HANDLER_CONFIG",
        "implementation_manifest.json": "M1_IMPLEMENTATION_MANIFEST",
        "program.json": "M1_MECHANISM_PROGRAM",
    }
    rows: list[dict[str, Any]] = []
    for item in report.files:
        source_relative = str(item["path"])
        artifact_type = next(
            value
            for suffix, value in type_by_suffix.items()
            if source_relative.endswith(suffix)
        )
        stored_relative = f"instances/{opaque_instance_id}/{source_relative}"
        rows.append(
            store.register_artifact(
                RegisterArtifactCommand(
                    round_id=str(binding.round_id),
                    artifact_type=artifact_type,
                    relative_path=stored_relative,
                    producer="M4NeutralMaterializationIndexerV1",
                    idempotency_key=(
                        f"m4-materialized:{binding.round_id}:{artifact_type}"
                    ),
                ),
                root.joinpath(*source_relative.split("/")).read_bytes(),
            )
        )
    report_relative = (
        f"instances/{opaque_instance_id}/artifacts/{binding.run_id}/"
        "materialization_report.v1.json"
    )
    rows.append(
        store.register_artifact(
            RegisterArtifactCommand(
                round_id=str(binding.round_id),
                artifact_type="MATERIALIZATION_REPORT_V1",
                relative_path=report_relative,
                producer="M4NeutralMaterializationIndexerV1",
                idempotency_key=f"m4-materialization-report:{binding.round_id}",
            ),
            canonical_json_bytes(report.to_dict()),
        )
    )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class ArmRoundResultV1:
    opaque_instance_id: str
    round_id: str
    candidate_id: str
    terminal_class: str
    physical_call_count: int
    proposal_count: int
    input_tokens: int
    output_tokens: int
    billed_tokens: int
    ordinary_execution_count: int
    gpu_device_time_ms: int
    gpu_cost_microunits: int
    feedback_digest: str
    evidence_port_status: str
    training_backend_started: bool
    broker_call_latencies_ms: tuple[int, ...]
    proposal_session_wall_time_ms: int
    training_wall_time_ms: int
    round_total_wall_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PreExecutionRejectionClosureV1:
    schema: str
    round_id: str
    failure_class: str
    cause_type: str
    detail_digest: str
    known_semantic_rejection: bool
    proposal_generation_session_consumed: bool
    proposal_response_present: bool
    execution_claim_present: bool
    training_started: bool
    guard_called: bool
    search_memory_updated: bool
    meta_observation_updated: bool
    frontier_updated: bool
    retry_count: int
    physical_call_count: int
    proposal_count: int
    input_tokens_actual: int
    output_tokens_actual: int
    billed_tokens_actual: int
    wall_time_ms_actual: int
    call_latencies_ms: tuple[int, ...]
    response_digests: tuple[str, ...]
    allocated_resource_debits: tuple[tuple[str, int], ...]
    allocation_ceiling_exceeded: tuple[str, ...]
    round_terminal_class: str
    feedback_class: str
    closure_digest: str

    @classmethod
    def create(
        cls,
        *,
        round_id: str,
        failure_class: str,
        cause_type: str,
        detail_digest: str,
        known_semantic_rejection: bool,
        usage: Mapping[str, Any],
        ceilings: ResourceCeilingsV1,
    ) -> "PreExecutionRejectionClosureV1":
        actual = {
            "INPUT_TOKEN": max(0, int(usage.get("input_tokens", 0))),
            "OUTPUT_TOKEN": max(0, int(usage.get("output_tokens", 0))),
            "BILLED_TOKEN_DEBIT": max(
                0, int(usage.get("billed_tokens", 0))
            ),
            "PROPOSAL": max(0, int(usage.get("proposal_count", 0))),
            "WALL_TIME_MS": max(0, int(usage.get("wall_time_ms", 0))),
        }
        allocation_ceilings = {
            "INPUT_TOKEN": ceilings.total_input_tokens,
            "OUTPUT_TOKEN": ceilings.total_output_tokens,
            "BILLED_TOKEN_DEBIT": ceilings.total_billed_token_debit,
            "PROPOSAL": ceilings.total_proposal_count,
            "WALL_TIME_MS": ceilings.wall_time_ms,
        }
        physical_call_count = max(
            0, int(usage.get("physical_call_count", 0))
        )
        proposal_attempt_actual = (
            actual["PROPOSAL"]
            if actual["PROPOSAL"] > 0
            else physical_call_count
        )
        allocated = (
            ("PHYSICAL_LLM_CALL", physical_call_count),
            (
                "INPUT_TOKEN",
                min(actual["INPUT_TOKEN"], allocation_ceilings["INPUT_TOKEN"]),
            ),
            (
                "OUTPUT_TOKEN",
                min(
                    actual["OUTPUT_TOKEN"],
                    allocation_ceilings["OUTPUT_TOKEN"],
                ),
            ),
            (
                "BILLED_TOKEN_DEBIT",
                min(
                    actual["BILLED_TOKEN_DEBIT"],
                    allocation_ceilings["BILLED_TOKEN_DEBIT"],
                ),
            ),
            (
                "PROPOSAL",
                min(actual["PROPOSAL"], allocation_ceilings["PROPOSAL"]),
            ),
            (
                "PROPOSAL_ATTEMPT",
                min(
                    proposal_attempt_actual,
                    ceilings.proposal_attempt_debit,
                ),
            ),
            (
                "WALL_TIME_MS",
                min(
                    actual["WALL_TIME_MS"],
                    allocation_ceilings["WALL_TIME_MS"],
                ),
            ),
            ("RETRY", 0),
            ("ORDINARY_EXECUTION", 0),
            ("COMMON_VALIDATION", 0),
            ("GPU_DEVICE_TIME_MS", 0),
            ("GPU_COST_MICROUNITS", 0),
        )
        exceeded = tuple(
            sorted(
                dimension
                for dimension, quantity in actual.items()
                if quantity > allocation_ceilings[dimension]
            )
        )
        if proposal_attempt_actual > ceilings.proposal_attempt_debit:
            exceeded = tuple(
                sorted((*exceeded, "PROPOSAL_ATTEMPT"))
            )
        call_latencies = tuple(
            max(0, int(item))
            for item in usage.get("call_latencies_ms", ())
        )
        response_digests = tuple(
            str(item) for item in usage.get("response_digests", ())
        )
        payload = {
            "allocated_resource_debits": allocated,
            "allocation_ceiling_exceeded": exceeded,
            "billed_tokens_actual": actual["BILLED_TOKEN_DEBIT"],
            "call_latencies_ms": call_latencies,
            "cause_type": cause_type,
            "detail_digest": detail_digest,
            "execution_claim_present": False,
            "failure_class": failure_class,
            "feedback_class": "ENGINEERING_ONLY",
            "frontier_updated": False,
            "guard_called": False,
            "input_tokens_actual": actual["INPUT_TOKEN"],
            "known_semantic_rejection": known_semantic_rejection,
            "meta_observation_updated": False,
            "output_tokens_actual": actual["OUTPUT_TOKEN"],
            "physical_call_count": physical_call_count,
            "proposal_count": actual["PROPOSAL"],
            "proposal_generation_session_consumed": physical_call_count > 0,
            "proposal_response_present": bool(response_digests),
            "response_digests": response_digests,
            "retry_count": 0,
            "round_id": round_id,
            "round_terminal_class": "NO_EXECUTION",
            "schema": (
                "recclaw.m6i.pre-execution-rejection-closure.v1"
            ),
            "search_memory_updated": False,
            "training_started": False,
            "wall_time_ms_actual": actual["WALL_TIME_MS"],
        }
        return cls(**payload, closure_digest=sha256_digest(payload))

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class ThreeArmPreCanaryOrchestratorV1:
    """One neutral writer and deterministic fake A/B/C round execution."""

    def __init__(
        self,
        root: Path,
        *,
        assignment_nonce: str = "M4-PRECANARY-NONCE",
        broker: Any | None = None,
        contract: Any | None = None,
        resource_ceilings: ResourceCeilingsV1 | None = None,
        guard_context: GuardContext | None = None,
    ) -> None:
        self.contract = contract or default_experiment_contract()
        self.assignment = PrivateTreatmentAssignmentV1.create(
            self.contract.experiment_id, nonce=assignment_nonce
        )
        self.layout = RuntimeLayoutV1.materialize(root, self.assignment)
        self.store = self._create_store(
            self.layout.neutral_root / "experiment.sqlite3",
            self.layout.neutral_root / "artifacts",
        )
        self.store.initialize_experiment(
            self.contract, arm_instance_ids=self.assignment.mapping
        )
        self.broker = broker or ThreeArmFakeBrokerV1.create()
        self.integrated_state = IntegratedCampaignStateCoreV1(
            experiment_id=self.contract.experiment_id
        )
        self.integrated_state.bind_arms(self.assignment.mapping)
        bind_arm_instances = getattr(self.broker, "bind_arm_instances", None)
        if bind_arm_instances is not None:
            bind_arm_instances(
                experiment_id=self.contract.experiment_id,
                arm_to_instance=self.assignment.mapping,
            )
        self.resource_ceilings = resource_ceilings or m4_budget()
        self.guard_context = guard_context or _guard_context()
        self.admission = DeterministicHelixAdmissionV13()
        self.ports: dict[ArmCode, Any] = {
            ArmCode.A: NullEvidencePortV1(),
            ArmCode.B: NullEvidencePortV1(),
        }
        c_id = self.assignment.mapping[ArmCode.C]
        c_evidence = self.layout.evidence_root(c_id)
        if c_evidence is None:
            raise PreCanaryInvariantError("Arm C evidence root is missing")
        self.guard_ledger = EvidenceGuardLedgerWriterV1(
            c_evidence,
        )
        self.ports[ArmCode.C] = EvidenceGuardPortV1(
            context=self.guard_context,
            ledger=self.guard_ledger,
            opaque_arm_instance_id=c_id,
        )
        self.research_task_queues = {
            ArmCode.B: ResearchTaskQueueV1(
                self.assignment.mapping[ArmCode.B]
            ),
            ArmCode.C: ResearchTaskQueueV1(
                self.assignment.mapping[ArmCode.C]
            ),
        }
        self._matched_control_sources: dict[
            ArmCode, dict[str, CandidateProposalV4]
        ] = {
            ArmCode.B: {},
            ArmCode.C: {},
        }
        self._completed: dict[tuple[int, int], tuple[ArmRoundResultV1, ...]] = {}
        private_containers = (
            self.research_task_queues[ArmCode.B],
            self.research_task_queues[ArmCode.C],
        )
        if len({id(item) for item in private_containers}) != len(
            private_containers
        ):
            raise PreCanaryInvariantError(
                "Research task queues must be Arm-private objects"
            )

    def _create_store(
        self, db_path: Path, artifact_root: Path
    ) -> SingleWriterExperimentStoreV1:
        return SingleWriterExperimentStoreV1(db_path, artifact_root)

    def close(self) -> None:
        self.guard_ledger.close()
        self.store.close()

    def __enter__(self) -> "ThreeArmPreCanaryOrchestratorV1":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _genesis(self) -> str:
        return sha256_digest(
            {
                "experiment_contract_digest": self.contract.identity_digest,
                "state": "GENESIS",
            }
        )

    def _controller_state_before(
        self, *, opaque_instance_id: str, search_seed: int
    ) -> str:
        connection = sqlite3.connect(self.store.db_path)
        try:
            row = connection.execute(
                """
                SELECT controller_state_digest
                FROM arm_state
                WHERE experiment_id=? AND arm_instance_id=? AND search_seed=?
                """,
                (self.contract.experiment_id, opaque_instance_id, search_seed),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise PreCanaryInvariantError("committed controller state is missing")
        return str(row[0])

    def _build_helix_raw(
        self,
        *,
        selected: CandidateEnvelope,
        opaque_instance_id: str,
        common_result: Any,
        observation_seed: str,
    ) -> RawResultEnvelope:
        return RawResultEnvelope(
            candidate_id=selected.candidate_id,
            opaque_arm_instance_id=opaque_instance_id,
            raw_result_digest=str(common_result.raw_output_digest),
            common_result_closure_digest=str(
                common_result.common_result_closure_digest
            ),
            observed_protocol=_guard_protocol(),
            target_model="CandidateModel",
            comparator="LightGCN",
            seed_runs=(
                {
                    "seed_id": observation_seed,
                    "run_id": str(common_result.run_id),
                    "artifact_sha256": str(common_result.raw_output_digest),
                },
            ),
            observation_kind="METRIC_EVALUATION",
            run_status="SUCCESS",
            artifact_identity_status="EXACT",
            normalized_metrics={"ndcg": 0.20},
        )

    def _planned_guard_protocol(self) -> Mapping[str, Any]:
        return _guard_protocol()

    def _session_for_research_task(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        task: ResearchTaskV1,
    ) -> FakeProposalSessionV1:
        controller = self.broker.research_controllers[arm]
        program = deep_thaw(task.mechanism_program)
        compiled = compile_program(program)
        if (
            str(compiled.candidate_id) != task.candidate_id
            or str(compiled.mechanism_program_digest)
            != task.mechanism_program_digest
            or str(compiled.mechanism_semantics_digest)
            != task.candidate_semantic_digest
        ):
            raise PreCanaryInvariantError(
                "Research task program identity changed"
            )
        route_trace_digest = sha256_digest(
            {
                "policy": "ResearchTaskQueueV1",
                "round_index": round_index,
                "task_digest": task.digest,
            }
        )
        session_digest = sha256_digest(
            {
                "mode": "NORMAL_BUDGET_RESEARCH_TASK",
                "round_index": round_index,
                "task_digest": task.digest,
            }
        )
        plan = ResearchRoundPlanV1(
            round_index=round_index,
            proposal_session_digest=session_digest,
            route_trace_digest=route_trace_digest,
            selected_candidate_id=task.candidate_id,
            physical_call_count=0,
            proposal_count=0,
            ordinary_execution_opportunities=1,
            plan_status="SELECTED_RESEARCH_TASK",
            policy_digest=controller.policy.digest,
        )
        return FakeProposalSessionV1(
            validation_programs=(program,),
            ordered_programs=(program,),
            selected_candidate_id=task.candidate_id,
            physical_call_count=0,
            input_tokens=0,
            output_tokens=0,
            billed_tokens=0,
            proposal_count=0,
            proposal_session_digest=session_digest,
            route_trace_digest=route_trace_digest,
            research_plan=plan,
            research_task=task,
        )

    def _search_utility_event(
        self,
        *,
        selected: CandidateEnvelope,
        selected_plan: Any,
        selected_recipe: Mapping[str, Any],
        helix_raw: RawResultEnvelope,
        gpu_device_time_ms: int,
        gpu_cost_microunits: int,
        execution_wall_time_ms: int,
        comparator_delta: float | str = NOT_AVAILABLE,
    ) -> SearchUtilityEventV2:
        metrics = helix_raw.to_dict()["normalized_metrics"]
        return SearchUtilityEventV2(
            candidate_semantic_digest=str(
                selected_plan.mechanism_semantics_digest
            ),
            candidate_id=selected.candidate_id,
            mechanism_axis=str(
                selected_recipe.get("mechanism_axis")
                or selected_recipe.get("axis")
                or (
                    executable_mechanism(
                        str(selected_recipe["mechanism_id"])
                    ).mechanism_axis
                    if selected_recipe.get("mechanism_id")
                    != "LEGACY_FIXTURE"
                    else None
                )
                or "legacy_fixture"
            ),
            common_outcome_class=str(helix_raw.run_status),
            runnable_observation=(
                "RUNNABLE"
                if str(helix_raw.run_status)
                in {"SUCCESS", "SMOKE_PASS", "COMPLETED"}
                else "NOT_RUNNABLE"
            ),
            comparator_delta=comparator_delta,
            metric_contract_digest=sha256_digest(
                {
                    "metric_keys": sorted(metrics),
                    "observed_protocol": helix_raw.to_dict()[
                        "observed_protocol"
                    ],
                }
            ),
            resource_cost_projection={
                "gpu_cost_microunits": gpu_cost_microunits,
                "gpu_device_time_ms": gpu_device_time_ms,
                "wall_time_ms": execution_wall_time_ms,
            },
            typed_blocker_class=(
                "NONE"
                if str(helix_raw.run_status)
                in {"SUCCESS", "SMOKE_PASS", "COMPLETED"}
                else str(helix_raw.run_status)
            ),
            observation_seed=str(helix_raw.seed_runs[0]["seed_id"]),
        )

    def _research_task(
        self,
        *,
        task_type: ResearchTaskTypeV1,
        round_index: int,
        selected: CandidateEnvelope,
        selected_plan: Any,
        program: Mapping[str, Any],
    ) -> ResearchTaskV1 | None:
        missing_seed_count = 0
        if task_type is ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE:
            observed = {
                item.observation_seed
                for item in self.guard_ledger.evidence_snapshot().observations
                if item.candidate_semantic_digest
                == selected.candidate_semantic_digest
            }
            observed.update(selected.seed_ids)
            stability_seeds = getattr(
                self.contract,
                "post_selection_stability_seeds",
                (2026, 2027, 2028),
            )
            remaining = tuple(
                str(seed)
                for seed in stability_seeds
                if str(seed) not in observed
            )
            if not remaining:
                return None
            required = remaining[0]
            missing_seed_count = len(remaining)
        else:
            required = "FROZEN_PROTOCOL_BRANCH_DIAGNOSTIC"
        identity = {
            "candidate_semantic_digest": (
                selected_plan.mechanism_semantics_digest
            ),
            "created_round": round_index,
            "opaque_arm_instance_id": selected.opaque_arm_instance_id,
            "required_seed_or_control": required,
            "task_type": task_type.value,
        }
        return ResearchTaskV1(
            task_id=sha256_digest(identity),
            task_type=task_type,
            candidate_id=selected.candidate_id,
            candidate_semantic_digest=str(
                selected_plan.mechanism_semantics_digest
            ),
            mechanism_program_digest=str(
                selected_plan.mechanism_program_digest
            ),
            parent_candidate_id=None,
            comparator_identity=selected.comparator,
            protocol_digest=sha256_digest(
                selected.to_dict()["planned_protocol"]
            ),
            required_seed_or_control=required,
            task_status=ResearchTaskStatusV1.PENDING,
            created_round=round_index,
            utility_priority=1.0,
            missing_seed_count=missing_seed_count,
            mechanism_program=program,
            owner_arm_instance_id=selected.opaque_arm_instance_id,
        )

    def _matched_control_task(
        self,
        *,
        arm: ArmCode,
        proposal: CandidateProposalV4,
        round_index: int,
        execution_seed: int,
    ) -> ResearchTaskV1 | None:
        plan = proposal.matched_control_plan
        if plan.plan_status != "QUEUE_MATCHED_CONTROL":
            return None
        root_mechanism_id = root_parent_mechanism_id(
            proposal.mechanism_id
        )
        program = campaign_program_from_proposal(
            {"mechanism_id": root_mechanism_id}
        )
        compiled = compile_program(program)
        if (
            plan.comparator_candidate_id != str(compiled.candidate_id)
            or plan.comparator_program_digest
            != str(compiled.mechanism_program_digest)
        ):
            raise PreCanaryInvariantError(
                "matched-control task does not bind the planned comparator"
            )
        identity = {
            "opaque_arm_instance_id": self.assignment.mapping[arm],
            "mechanism_question_digest": plan.mechanism_question_digest,
            "primary_candidate_id": proposal.candidate_id,
            "required_seed": str(execution_seed),
            "task_type": ResearchTaskTypeV1.RUN_MATCHED_CONTROL.value,
        }
        return ResearchTaskV1(
            task_id=sha256_digest(identity),
            task_type=ResearchTaskTypeV1.RUN_MATCHED_CONTROL,
            candidate_id=str(compiled.candidate_id),
            candidate_semantic_digest=str(
                compiled.mechanism_semantics_digest
            ),
            mechanism_program_digest=str(
                compiled.mechanism_program_digest
            ),
            parent_candidate_id=proposal.candidate_id,
            comparator_identity=root_mechanism_id,
            protocol_digest=plan.protocol_digest,
            required_seed_or_control=str(execution_seed),
            task_status=ResearchTaskStatusV1.PENDING,
            created_round=round_index,
            utility_priority=float(
                proposal.utility_features.information_gain
            ),
            missing_seed_count=1,
            mechanism_program=program,
            owner_arm_instance_id=self.assignment.mapping[arm],
        )

    def _matched_control_belief(
        self,
        *,
        arm: ArmCode,
        source: CandidateProposalV4,
        task: ResearchTaskV1,
        control_result: RawResultEnvelope,
        control_candidate_instance_id: str,
    ) -> DevelopmentalMechanismBeliefV2:
        lookup = getattr(self.broker, "lineage_record_for", None)
        primary = (
            lookup(
                arm=arm,
                proposal_candidate_id=source.candidate_id,
                protocol_digest=task.protocol_digest,
            )
            if lookup is not None
            else None
        )
        metrics = control_result.to_dict()["normalized_metrics"]
        control_metric = next(
            (
                float(metrics[name])
                for name in ("ndcg@10", "ndcg")
                if isinstance(metrics.get(name), (int, float))
            ),
            None,
        )
        if (
            primary is None
            or primary.metric_value is None
            or control_metric is None
            or primary.observation_seed
            != str(task.required_seed_or_control)
        ):
            raise PreCanaryInvariantError(
                "matched-control completion lacks the exact primary result"
            )
        delta = float(primary.metric_value) - control_metric
        observation = (
            "development_matched_comparison:"
            + sha256_digest(
                {
                    "control_result": control_result.raw_result_digest,
                    "primary_result": primary.result_digest,
                    "task": task.digest,
                }
            )
        )
        return DevelopmentalMechanismBeliefV2(
            hypothesis_id=source.candidate_id,
            mechanism_axis=source.mechanism_axis,
            mechanism_question_digest=(
                source.matched_control_plan.mechanism_question_digest
            ),
            exact_parent_candidate_id=source.candidate_id,
            exact_comparator_candidate_id=control_candidate_instance_id,
            protocol_digest=task.protocol_digest,
            comparator_delta=delta,
            evidence_for=(observation,) if delta > 1e-4 else (),
            evidence_against=(observation,) if delta < -1e-4 else (),
            unresolved_confounds=("single_training_seed",),
            next_discriminative_task=None,
        )

    def _common_execution_protocol(self) -> Any:
        return development_protocol()

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: ResearchLineControllerV1,
        feedback: FusedSearchFeedbackV2,
        source_proposal_candidate_id: str,
    ) -> None:
        del (
            arm,
            round_index,
            controller,
            feedback,
            source_proposal_candidate_id,
        )

    def _observation_path(
        self, feedback: FusedSearchFeedbackV2
    ) -> ObservationPathV1:
        return canonical_observation_path(
            meta_update_allowed=feedback.meta_update_allowed,
            search_feedback_class=feedback.search_feedback_class.value,
        )

    def _apply_integrated_meta_boundary(
        self,
        *,
        boundary: RoundBoundaryEventV1,
        controller: ResearchLineControllerV1 | None,
        feedback: FusedSearchFeedbackV2 | None,
        source_proposal_candidate_id: str | None,
    ) -> None:
        """Apply every B/C terminal boundary through one canonical adapter."""

        arm = boundary.owner.arm
        if arm not in {ArmCode.B, ArmCode.C}:
            raise PreCanaryInvariantError(
                "integrated Meta boundary escaped Research Arms"
            )
        runtime = getattr(self.broker, "campaign_meta_runtime", None)
        record_boundary = getattr(runtime, "record_round_boundary", None)
        if record_boundary is not None:
            utility_event = (
                feedback.search_utility_event
                if feedback is not None
                else None
            )
            utility_value = (
                None
                if utility_event is None
                or utility_event.comparator_delta == NOT_AVAILABLE
                else float(utility_event.comparator_delta)
            )
            record_boundary(
                arm=arm,
                round_index=boundary.round_index,
                proposal_source=boundary.proposal_source.value,
                observation_path=boundary.observation_path.value,
                candidate_id=source_proposal_candidate_id,
                runtime_candidate_id=(
                    str(utility_event.candidate_id)
                    if utility_event is not None
                    else None
                ),
                run_status=(
                    str(utility_event.common_outcome_class)
                    if utility_event is not None
                    else boundary.terminal_class
                ),
                ndcg=utility_value,
                wall_time_ms=(
                    int(
                        utility_event.resource_cost_projection[
                            "wall_time_ms"
                        ]
                    )
                    if utility_event is not None
                    else 0
                ),
                source_search_utility_event_digest=(
                    utility_event.digest
                    if utility_event is not None
                    else boundary.digest
                ),
            )
            return
        if (
            controller is not None
            and feedback is not None
            and feedback.meta_update_allowed
            and source_proposal_candidate_id is not None
        ):
            self._after_research_close(
                arm=arm,
                round_index=boundary.round_index,
                controller=controller,
                feedback=feedback,
                source_proposal_candidate_id=(
                    source_proposal_candidate_id
                ),
            )

    def _terminalize_integrated_round(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        terminal_class: str,
        controller: ResearchLineControllerV1 | None = None,
        feedback: FusedSearchFeedbackV2 | None = None,
        source_proposal_candidate_id: str | None = None,
    ) -> RoundBoundaryEventV1:
        return self.integrated_state.terminalize(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            terminal_class=terminal_class,
            meta_boundary=(
                (
                    lambda boundary: self._apply_integrated_meta_boundary(
                        boundary=boundary,
                        controller=controller,
                        feedback=feedback,
                        source_proposal_candidate_id=(
                            source_proposal_candidate_id
                        ),
                    )
                )
                if arm in {ArmCode.B, ArmCode.C}
                else None
            ),
        )

    def _execute_selected(
        self,
        *,
        permit: Any,
        binding: Any,
        runtime_context: Any | None,
        gate: Any,
        pre_execution: Any,
        materialization_artifacts: tuple[dict[str, Any], ...],
    ) -> tuple[Any, Any]:
        del runtime_context
        receipt, raw_output, run_artifacts = PackageOwnedLauncherV1(
            self.store
        ).launch(
            permit=permit,
            binding=binding,
            gate=gate,
            pre_execution=pre_execution,
        )
        _closure, common_result = CommonExecutionGuardV1().close_result(
            permit=permit,
            binding=binding,
            claim=self.store.get_execution_claim(binding.round_id),
            receipt=receipt,
            raw_output=raw_output,
            artifact_closure=list(materialization_artifacts + run_artifacts),
        )
        if common_result is None:
            raise PreCanaryInvariantError("M4 common result closure failed")
        return raw_output, common_result

    def _prepare_runtime_execution(
        self,
        *,
        base_plan: Any,
        base_permit: Any,
        base_binding: Any,
        eligible: Any,
    ) -> tuple[Any, Any, Any | None]:
        del base_plan, eligible
        return base_permit, base_binding, None

    def _claim_runtime_execution(
        self,
        *,
        permit: Any,
        binding: Any,
        runtime_context: Any | None,
    ) -> None:
        del runtime_context
        self.store.claim_execution(
            ClaimExecutionCommand(
                round_id=binding.round_id,
                permit_digest=permit.digest,
                binding_digest=binding.digest,
                idempotency_key=f"m4:claim:{binding.round_id}",
            )
        )

    def _execution_resource_projection(
        self, raw_output: Any
    ) -> tuple[int, int, int]:
        del raw_output
        return 0, 0, 0

    def _research_belief(
        self,
        *,
        selected: CandidateEnvelope,
        event: SearchUtilityEventV2,
        proposal: CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4,
        next_task: ResearchTaskV1 | None = None,
    ) -> DevelopmentalMechanismBeliefV1 | DevelopmentalMechanismBeliefV2:
        if isinstance(proposal, CandidateProposalV4):
            comparator_delta = event.comparator_delta
            matched_plan = proposal.matched_control_plan
            matched = (
                comparator_delta != NOT_AVAILABLE
                and matched_plan.comparator_candidate_id is not None
            )
            observation = "development_observation:" + event.digest
            return DevelopmentalMechanismBeliefV2(
                hypothesis_id=str(selected.candidate_id),
                mechanism_axis=proposal.mechanism_axis,
                mechanism_question_digest=(
                    matched_plan.mechanism_question_digest
                ),
                exact_parent_candidate_id=(
                    (
                        proposal.parent_candidate_id
                        or matched_plan.comparator_candidate_id
                    )
                    if matched
                    else None
                ),
                exact_comparator_candidate_id=(
                    matched_plan.comparator_candidate_id
                    if matched
                    else None
                ),
                protocol_digest=(
                    matched_plan.protocol_digest
                ),
                comparator_delta=(
                    float(comparator_delta)
                    if matched
                    else NOT_AVAILABLE
                ),
                evidence_for=(
                    (observation,)
                    if matched and float(comparator_delta) > 1e-4
                    else ()
                ),
                evidence_against=(
                    (observation,)
                    if matched and float(comparator_delta) < -1e-4
                    else ()
                ),
                unresolved_confounds=(
                    ("single_training_seed",)
                    if matched
                    else ("matched_comparator_not_available",)
                ),
                next_discriminative_task=(
                    ResearchTaskRefV1(
                        task_id=next_task.task_id,
                        task_type=next_task.task_type.value,
                        task_status=next_task.task_status.value,
                    )
                    if next_task is not None
                    else (
                        ResearchTaskRefV1(
                            task_id=sha256_digest(
                                {
                                    "candidate_id": proposal.candidate_id,
                                    "task_type": "RUN_MATCHED_CONTROL",
                                }
                            ),
                            task_type="RUN_MATCHED_CONTROL",
                            task_status=(
                                "UNEXECUTED_NO_EXACT_CONTROL"
                            ),
                        )
                        if not matched
                        else None
                    )
                ),
            )
        return DevelopmentalMechanismBeliefV1(
            hypothesis_id=str(selected.candidate_id),
            mechanism_axis=proposal.mechanism_axis,
            competing_hypotheses=("fake_null",),
            predicted_outcome_signature=(
                f"{proposal.proposal_intent.value}:{proposal.mechanism_axis}:"
                "synthetic closure"
            ),
            evidence_for=(event.digest,),
            evidence_against=(),
            unresolved_confounds=("synthetic_fixture",),
            next_discriminative_test=(
                f"real {proposal.mechanism_axis} canary under the frozen protocol"
            ),
        )

    def run_fake_triplet(
        self,
        *,
        search_seed: int,
        round_index: int,
        drafts: Sequence[Mapping[str, Any]],
        execution_order: Sequence[ArmCode] | None = None,
    ) -> tuple[ArmRoundResultV1, ...]:
        key = (search_seed, round_index)
        if key in self._completed:
            return self._completed[key]
        ceilings = self.resource_ceilings
        results: list[ArmRoundResultV1] = []
        order = (
            tuple(execution_order)
            if execution_order is not None
            else tuple(
                sorted(
                    ArmCode,
                    key=lambda arm: sha256_digest(
                        {
                            "arm": arm.value,
                            "round": round_index,
                            "seed": search_seed,
                        }
                    ),
                )
            )
        )
        if len(order) != 3 or set(order) != set(ArmCode):
            raise PreCanaryInvariantError(
                "execution order must be one exact A/B/C permutation"
            )
        for arm in order:
            results.append(
                self._run_arm(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                    drafts=drafts,
                    ceilings=ceilings,
                )
            )
        ordered_results = tuple(
            next(item for item in results if item.opaque_instance_id == self.assignment.mapping[arm])
            for arm in ArmCode
        )
        self.integrated_state.close_triplet(
            search_seed=search_seed,
            round_index=round_index,
        )
        self._persist_triplet_behavioral_checkpoint(
            search_seed=search_seed,
            round_index=round_index,
            ordered_results=ordered_results,
        )
        self._completed[key] = ordered_results
        return ordered_results

    def _persist_triplet_behavioral_checkpoint(
        self,
        *,
        search_seed: int,
        round_index: int,
        ordered_results: tuple[ArmRoundResultV1, ...],
    ) -> None:
        """Persist private current-byte evidence before any final report."""

        call_audit_reader = getattr(
            self.broker, "call_sharing_audit", None
        )
        call_records = (
            tuple(call_audit_reader().get("records", ()))
            if call_audit_reader is not None
            else ()
        )
        meta_runtime = getattr(
            self.broker, "campaign_meta_runtime", None
        )
        private_digests: list[str] = []
        result_digests: list[str] = []
        terminal_counts: dict[str, int] = {}
        for arm, result in zip(ArmCode, ordered_results, strict=True):
            opaque_id = self.assignment.mapping[arm]
            task_queue = self.research_task_queues.get(arm)
            controller = getattr(
                self.broker, "research_controllers", {}
            ).get(arm)
            lineage = getattr(
                self.broker, "lineage_indexes", {}
            ).get(arm)
            seed_path = (
                self.store.artifact_root
                / "instances"
                / opaque_id
                / "audit"
                / sha256_digest({"round_id": result.round_id})
                / "execution_seed_binding.v1.json"
            )
            seed_binding: Mapping[str, Any] | str
            if seed_path.exists():
                seed_binding = canonical_value(
                    json.loads(
                        seed_path.read_text(encoding="utf-8")
                    )
                )
            else:
                seed_binding = "ABSENT_NO_EXECUTION"
            private_payload = canonical_value(
                {
                    "schema": (
                        "recclaw.m6i.arm-private-behavioral-checkpoint.v1"
                    ),
                    "experiment_id": self.contract.experiment_id,
                    "opaque_arm_instance_id": opaque_id,
                    "search_seed": int(search_seed),
                    "round_index": int(round_index),
                    "integrated_round": (
                        self.integrated_state.round_projection(
                            arm=arm,
                            search_seed=search_seed,
                            round_index=round_index,
                        )
                    ),
                    "round_result": result.to_dict(),
                    "execution_seed_binding": seed_binding,
                    "call_identity_records": [
                        record
                        for record in call_records
                        if record.get("owner", {}).get(
                            "opaque_arm_instance_id"
                        )
                        == opaque_id
                    ],
                    "research_task_queue": (
                        [item.to_dict() for item in task_queue.tasks]
                        if task_queue is not None
                        else []
                    ),
                    "research_task_queue_digest": (
                        task_queue.digest
                        if task_queue is not None
                        else "ABSENT"
                    ),
                    "search_memory_head_digest": (
                        controller.memory_writer.head.digest
                        if controller is not None
                        and controller.memory_writer.head is not None
                        else "ABSENT"
                    ),
                    "lineage_digest": (
                        lineage.digest
                        if lineage is not None
                        else "ABSENT"
                    ),
                    "meta_arm_private_context_digest": (
                        meta_runtime.arm_private_context_digest(arm)
                        if meta_runtime is not None
                        and arm in {ArmCode.B, ArmCode.C}
                        else "ABSENT"
                    ),
                }
            )
            private_digest = sha256_digest(private_payload)
            private_record = {
                **private_payload,
                "checkpoint_digest": private_digest,
            }
            relative = (
                f"behavioral_checkpoints/{search_seed}/"
                f"{round_index:04d}.arm-private.v1.json"
            )
            roots = self.layout.arm(opaque_id)
            target = roots.namespace("registry").joinpath(
                *PurePosixPath(relative).parts
            )
            private_bytes = (
                canonical_json_bytes(private_record) + b"\n"
            )
            if target.exists():
                if target.read_bytes() != private_bytes:
                    raise PreCanaryInvariantError(
                        "Arm-private behavioral checkpoint changed"
                    )
            else:
                ArmFilesystemCapabilityV1(roots).write_bytes(
                    "registry",
                    relative,
                    private_bytes,
                )
            private_digests.append(private_digest)
            result_digests.append(sha256_digest(result.to_dict()))
            terminal_counts[result.terminal_class] = (
                terminal_counts.get(result.terminal_class, 0) + 1
            )

        neutral_core = canonical_value(
            {
                "schema": (
                    "recclaw.m6i.incremental-neutral-behavioral-"
                    "checkpoint.v1"
                ),
                "experiment_id": self.contract.experiment_id,
                "search_seed": int(search_seed),
                "round_index": int(round_index),
                "triplet_barrier_closed": True,
                "private_checkpoint_digests": sorted(private_digests),
                "round_result_digests": sorted(result_digests),
                "terminal_class_counts": terminal_counts,
                "physical_call_count": sum(
                    item.physical_call_count for item in ordered_results
                ),
                "ordinary_execution_count": sum(
                    item.ordinary_execution_count
                    for item in ordered_results
                ),
                "training_backend_started_count": sum(
                    item.training_backend_started
                    for item in ordered_results
                ),
                "integrated_state_projection_digest": sha256_digest(
                    self.integrated_state.audit_projection()
                ),
                "store_integrity_digest": sha256_digest(
                    self.store.integrity_report()
                ),
            }
        )
        encoded = str(neutral_core).lower()
        forbidden = (
            "arm_code",
            "arm_instance",
            "assignment_key",
            "candidate_id",
            "metric",
            "ndcg",
            "treatment",
        )
        if any(token in encoded for token in forbidden):
            raise PreCanaryInvariantError(
                "neutral incremental checkpoint exposes an association"
            )
        neutral_record = {
            **neutral_core,
            "checkpoint_digest": sha256_digest(neutral_core),
        }
        self.store.register_artifact(
            RegisterArtifactCommand(
                round_id=None,
                artifact_type=(
                    "INCREMENTAL_NEUTRAL_BEHAVIORAL_CHECKPOINT_V1"
                ),
                relative_path=(
                    "behavioral_checkpoints/"
                    f"{sha256_digest({'search_seed': search_seed, 'round_index': round_index})}/"
                    "triplet_checkpoint.v1.json"
                ),
                producer="M6IIncrementalBehavioralCheckpointWriterV1",
                idempotency_key=(
                    f"m6i:incremental-behavioral-checkpoint:"
                    f"{search_seed}:{round_index}"
                ),
            ),
            canonical_json_bytes(neutral_record) + b"\n",
        )

    def _provider_usage_for_round(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        error: Exception,
    ) -> dict[str, Any]:
        usage_reader = getattr(
            self.broker, "provider_usage_for_round", None
        )
        raw = (
            usage_reader(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
            )
            if usage_reader is not None
            else {}
        )
        usage = {
            "billed_tokens": max(
                0, int(raw.get("billed_tokens", 0))
            ),
            "call_latencies_ms": tuple(
                max(0, int(item))
                for item in raw.get("call_latencies_ms", ())
            ),
            "input_tokens": max(
                0, int(raw.get("input_tokens", 0))
            ),
            "output_tokens": max(
                0, int(raw.get("output_tokens", 0))
            ),
            "physical_call_count": max(
                0, int(raw.get("physical_call_count", 0))
            ),
            "proposal_count": max(
                0, int(raw.get("proposal_count", 0))
            ),
            "response_digests": tuple(
                str(item)
                for item in raw.get("response_digests", ())
            ),
            "wall_time_ms": max(
                0, int(raw.get("wall_time_ms", 0))
            ),
        }
        if isinstance(error, CanaryBrokerError):
            usage["physical_call_count"] = max(
                usage["physical_call_count"],
                max(0, int(error.physical_call_count)),
            )
            usage["input_tokens"] = max(
                usage["input_tokens"],
                max(0, int(error.input_tokens)),
            )
            usage["output_tokens"] = max(
                usage["output_tokens"],
                max(0, int(error.output_tokens)),
            )
            usage["billed_tokens"] = max(
                usage["billed_tokens"],
                max(0, int(error.billed_tokens)),
            )
            usage["wall_time_ms"] = max(
                usage["wall_time_ms"],
                max(0, int(error.wall_time_ms)),
            )
        return canonical_value(usage)

    def _close_pre_execution_rejection(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        round_row: Mapping[str, Any],
        ceilings: ResourceCeilingsV1,
        error: Exception,
        known_semantic_rejection: bool,
    ) -> ArmRoundResultV1:
        round_id = str(round_row["round_id"])
        usage = self._provider_usage_for_round(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            error=error,
        )
        failure_class = (
            error.failure_class
            if isinstance(error, PostProviderSemanticRejectionV1)
            else "BROKER_ERROR_WITHOUT_PROCESS_RECEIPT"
            if isinstance(error, CanaryBrokerError)
            else "PRE_EXECUTION_IMPLEMENTATION_FAILURE"
        )
        cause_type = (
            error.cause_type
            if isinstance(error, PostProviderSemanticRejectionV1)
            else type(error).__name__
        )
        detail_digest = (
            error.detail_digest
            if isinstance(error, PostProviderSemanticRejectionV1)
            else sha256_digest(
                {
                    "cause_type": cause_type,
                    "detail": str(error),
                    "failure_class": failure_class,
                }
            )
        )
        closure = PreExecutionRejectionClosureV1.create(
            round_id=round_id,
            failure_class=str(failure_class),
            cause_type=str(cause_type),
            detail_digest=str(detail_digest),
            known_semantic_rejection=known_semantic_rejection,
            usage=usage,
            ceilings=ceilings,
        )
        integrated = self.integrated_state.round_projection(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
        )
        if integrated["proposal_source"] is None:
            self.integrated_state.bind_proposal_source(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                source=ProposalSourceV1.NO_PROPOSAL_TERMINAL,
            )
        self.integrated_state.close_no_execution(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
        )
        if arm in self.research_task_queues:
            active_tasks = tuple(
                task
                for task in self.research_task_queues[arm].tasks
                if task.task_status is ResearchTaskStatusV1.ACTIVE
            )
            for task in active_tasks:
                self.research_task_queues[arm].cancel_without_execution(
                    task.task_id
                )
                if (
                    task.task_type
                    is ResearchTaskTypeV1.RUN_MATCHED_CONTROL
                ):
                    self._matched_control_sources[arm].pop(
                        task.task_id, None
                    )
        artifact_path = (
            "pre_execution_rejections/"
            + sha256_digest({"round_id": round_id})
            + "/PRE_EXECUTION_REJECTION_V1.json"
        )
        self.store.register_artifact(
            RegisterArtifactCommand(
                round_id=round_id,
                artifact_type="PRE_EXECUTION_REJECTION_V1",
                relative_path=artifact_path,
                producer="M6I_PRE_EXECUTION_REJECTION_CLOSER_V1",
                idempotency_key=(
                    f"m6i:pre-execution-rejection-artifact:{round_id}"
                ),
            ),
            canonical_json_bytes(closure.to_dict()) + b"\n",
        )
        closed = self.store.close_round(
            CloseRoundCommand(
                round_id=round_id,
                terminal_class="NO_EXECUTION",
                feedback_payload={
                    "closure_digest": closure.closure_digest,
                    "failure_class": closure.failure_class,
                    "feedback_class": closure.feedback_class,
                    "frontier_updated": False,
                    "guard_called": False,
                    "meta_update_allowed": False,
                    "search_memory_updated": False,
                },
                controller_state_after_digest=str(
                    round_row["controller_state_before_digest"]
                ),
                resource_debits=tuple(
                    ResourceDebitV1(dimension, quantity)
                    for dimension, quantity in (
                        closure.allocated_resource_debits
                    )
                ),
                idempotency_key=(
                    f"m6i:pre-execution-rejection-close:{round_id}"
                ),
            )
        )
        self._terminalize_integrated_round(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            terminal_class="NO_EXECUTION",
        )
        return ArmRoundResultV1(
            opaque_instance_id=str(round_row["arm_instance_id"]),
            round_id=round_id,
            candidate_id="NO_CANDIDATE_PRE_EXECUTION_REJECTION",
            terminal_class=str(closed["terminal_class"]),
            physical_call_count=closure.physical_call_count,
            proposal_count=closure.proposal_count,
            input_tokens=closure.input_tokens_actual,
            output_tokens=closure.output_tokens_actual,
            billed_tokens=closure.billed_tokens_actual,
            ordinary_execution_count=0,
            gpu_device_time_ms=0,
            gpu_cost_microunits=0,
            feedback_digest=str(closed["feedback_digest"]),
            evidence_port_status="NOT_CALLED_PRE_EXECUTION_REJECTION",
            training_backend_started=False,
            broker_call_latencies_ms=closure.call_latencies_ms,
            proposal_session_wall_time_ms=closure.wall_time_ms_actual,
            training_wall_time_ms=0,
            round_total_wall_time_ms=closure.wall_time_ms_actual,
        )

    def _run_arm(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> ArmRoundResultV1:
        try:
            return self._run_arm_body(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                drafts=drafts,
                ceilings=ceilings,
            )
        except Exception as error:
            row = self.store._connection.execute(
                """
                SELECT * FROM rounds
                WHERE experiment_id = ? AND arm_instance_id = ?
                  AND search_seed = ? AND round_index = ?
                """,
                (
                    self.contract.experiment_id,
                    self.assignment.mapping[arm],
                    search_seed,
                    round_index,
                ),
            ).fetchone()
            if row is None or str(row["status"]) != "OPEN":
                raise
            claim_count = int(
                self.store._connection.execute(
                    """
                    SELECT COUNT(*) FROM execution_claims
                    WHERE round_id = ?
                    """,
                    (str(row["round_id"]),),
                ).fetchone()[0]
            )
            if claim_count:
                raise
            result = self._close_pre_execution_rejection(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                round_row=dict(row),
                ceilings=ceilings,
                error=error,
                known_semantic_rejection=isinstance(
                    error, PostProviderSemanticRejectionV1
                ),
            )
            if isinstance(error, PostProviderSemanticRejectionV1):
                return result
            raise

    def _run_arm_body(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> ArmRoundResultV1:
        round_started_ns = time.monotonic_ns()
        opaque_id = self.assignment.mapping[arm]
        controller_state_before_digest = self._controller_state_before(
            opaque_instance_id=opaque_id,
            search_seed=search_seed,
        )
        opened = self.store.open_round(
            OpenRoundCommand(
                experiment_id=self.contract.experiment_id,
                arm_instance_id=opaque_id,
                arm_code=arm,
                search_seed=search_seed,
                round_index=round_index,
                budget_snapshot=ceilings,
                controller_state_before_digest=controller_state_before_digest,
                idempotency_key=f"m4:open:{search_seed}:{round_index}:{opaque_id}",
            )
        )
        self.integrated_state.open_round(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
        )
        active_task: ResearchTaskV1 | None = None
        if arm in self.research_task_queues:
            pending_task = self.research_task_queues[arm].select_next(
                allowed_types=frozenset(ResearchTaskTypeV1)
            )
            if pending_task is not None:
                active_task = self.research_task_queues[arm].activate(
                    pending_task.task_id
                )
        prepare_consumer_context = getattr(
            self.broker, "prepare_round_consumer_context", None
        )
        if prepare_consumer_context is not None:
            prepare_consumer_context(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                active_task_digest=(
                    active_task.digest if active_task is not None else None
                ),
                research_task_queue_digest=(
                    self.research_task_queues[arm].digest
                    if arm in self.research_task_queues
                    else None
                ),
            )
        if active_task is not None:
            self.integrated_state.bind_proposal_source(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                source=ProposalSourceV1.ACTIVE_BOUND_TASK,
                active_task_digest=active_task.digest,
            )
            session = self._session_for_research_task(
                arm=arm,
                round_index=round_index,
                task=active_task,
            )
        else:
            if arm is ArmCode.A:
                self.integrated_state.bind_proposal_source(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                    source=ProposalSourceV1.ORIGINAL_CONTROLLER_PATH,
                )
            try:
                session = self.broker.generate(
                    arm=arm,
                    round_index=round_index,
                    search_seed=search_seed,
                    drafts=drafts,
                    ceilings=ceilings,
                )
            except CanaryBrokerError as error:
                if error.outcome is None or error.receipt is None:
                    raise
                if arm is not ArmCode.A:
                    self.integrated_state.bind_proposal_source(
                        arm=arm,
                        search_seed=search_seed,
                        round_index=round_index,
                        source=ProposalSourceV1.NO_PROPOSAL_TERMINAL,
                    )
                self.integrated_state.close_no_execution(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                )
                closure = close_broker_failure(
                    store=self.store,
                    experiment_id=self.contract.experiment_id,
                    search_seed=search_seed,
                    round_index=round_index,
                    round_id=str(opened["round_id"]),
                    controller_state_digest=(
                        controller_state_before_digest
                    ),
                    ceilings=ceilings,
                    receipt=error.receipt,
                    outcome=error.outcome,
                    physical_call_count=error.physical_call_count,
                    input_tokens=error.input_tokens,
                    output_tokens=error.output_tokens,
                    billed_tokens=error.billed_tokens,
                    wall_time_ms=error.wall_time_ms,
                    stop_campaign=False,
                )
                self._terminalize_integrated_round(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                    terminal_class=str(closure.round_terminal_class),
                )
                round_row = self.store.get_round(str(opened["round_id"]))
                return ArmRoundResultV1(
                    opaque_instance_id=opaque_id,
                    round_id=str(opened["round_id"]),
                    candidate_id="NO_CANDIDATE_BROKER_FAILURE",
                    terminal_class=str(closure.round_terminal_class),
                    physical_call_count=closure.physical_call_count,
                    proposal_count=0,
                    input_tokens=closure.input_token_debit,
                    output_tokens=closure.output_token_debit,
                    billed_tokens=closure.billed_token_debit,
                    ordinary_execution_count=0,
                    gpu_device_time_ms=0,
                    gpu_cost_microunits=0,
                    feedback_digest=str(round_row["feedback_digest"]),
                    evidence_port_status="NOT_CALLED_BROKER_FAILURE",
                    training_backend_started=False,
                    broker_call_latencies_ms=(closure.wall_time_ms,),
                    proposal_session_wall_time_ms=closure.wall_time_ms,
                    training_wall_time_ms=0,
                    round_total_wall_time_ms=closure.wall_time_ms,
                )
        common_guard = CommonExecutionGuardV1()
        eligible: list[tuple[Mapping[str, Any], Any, Any]] = []
        denials: list[tuple[str, tuple[str, ...]]] = []
        common_eligible_by_id: dict[str, tuple[Mapping[str, Any], Any, Any]] = {}
        for frozen_program in session.validation_programs:
            program = deep_thaw(frozen_program)
            compiled = compile_program(program)
            plan, action = common_guard.plan_check(
                program=program,
                caller_compile_report=compiled,
                protocol=self._common_execution_protocol(),
                budget=ceilings,
            )
            if action is not None:
                common_eligible_by_id[str(action.candidate_id)] = (
                    program,
                    plan,
                    action,
                )
            else:
                denials.append((str(compiled.candidate_id), tuple(plan.reason_codes)))
        finalize_common_route = getattr(
            self.broker, "finalize_common_route", None
        )
        if finalize_common_route is not None and active_task is None:
            session = finalize_common_route(
                arm=arm,
                round_index=round_index,
                session=session,
                common_eligible_candidate_ids=tuple(
                    common_eligible_by_id
                ),
            )
        if arm in {ArmCode.B, ArmCode.C} and active_task is None:
            if session.research_plan is None:
                raise PreCanaryInvariantError(
                    "Research proposal path lacks its canonical route"
                )
            self.integrated_state.bind_proposal_source(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                source=ProposalSourceV1.NORMAL_ROUTED_PROPOSAL,
                route_digest=(
                    session.route_trace_digest
                    or session.research_plan.route_trace_digest
                ),
            )
        for frozen_program in session.ordered_programs:
            candidate_id = str(compile_program(deep_thaw(frozen_program)).candidate_id)
            item = common_eligible_by_id.get(candidate_id)
            if item is not None:
                eligible.append(item)
        if not eligible:
            raise PreCanaryInvariantError(
                f"M4 fake slate has no COMMON_PASS candidate: {denials}"
            )
        envelope_rows = []
        campaign_profile_digest = str(
            campaign_runtime_profile()["profile_digest"]
        )
        if active_task is None:
            execution_seed = int(self.contract.ordinary_execution_seed)
        else:
            try:
                execution_seed = int(
                    active_task.required_seed_or_control
                )
            except ValueError:
                execution_seed = int(
                    self.contract.ordinary_execution_seed
                )
        for program, plan, action in eligible:
            if action.release_projection_digest == campaign_profile_digest:
                recipe = execution_recipe_for_program(program)
            else:
                recipe = {
                    "mechanism_id": "LEGACY_FIXTURE",
                    "model": "CandidateModel",
                }
            is_campaign_candidate = (
                action.release_projection_digest == campaign_profile_digest
            )
            envelope_rows.append(
                CandidateEnvelope(
                    candidate_id=str(action.candidate_id),
                    candidate_semantic_digest=str(
                        plan.mechanism_semantics_digest
                    ),
                    opaque_arm_instance_id=opaque_id,
                    common_status="COMMON_PASS",
                    mechanism_program_digest=str(
                        plan.mechanism_program_digest
                    ),
                    common_plan_digest=plan.digest,
                    action_family="RUN_OFFLINE_TOPN",
                    planned_protocol=self._planned_guard_protocol(),
                    target_model=str(recipe["model"]),
                    comparator=(
                        root_parent_mechanism_id(recipe["mechanism_id"])
                        if is_campaign_candidate
                        else "LightGCN"
                    ),
                    seed_ids=(str(execution_seed),),
                    purpose=(
                        "development comparison for exact mechanism "
                        f"{recipe['mechanism_id']} in round {round_index}"
                        if is_campaign_candidate
                        else "development comparison"
                    ),
                )
            )
        envelopes = tuple(envelope_rows)
        selection = SameSlateHelixSelectorV1(self.admission).select(
            envelopes, self.ports[arm]
        )
        if selection.selected_candidate is None:
            if arm is not ArmCode.C or selection.last_adjudication is None:
                raise PreCanaryInvariantError(
                    "non-Guard Arm unexpectedly exhausted the slate"
                )
            last_candidate_id = selection.inspected_candidate_ids[-1]
            plan = session.research_plan
            if plan is None:
                raise PreCanaryInvariantError(
                    "Guard PRE exhaustion lacks its Research plan"
                )
            fused_feedback = self.admission.no_search_update(
                last_candidate_id
            )
            round_total_wall_time_ms = max(
                session.proposal_session_wall_time_ms,
                int(
                    (time.monotonic_ns() - round_started_ns)
                    / 1_000_000
                ),
            )
            accounted_wall_time_ms = (
                round_total_wall_time_ms
                if bool(getattr(self.broker, "v13_mode", False))
                else 0
            )
            self.integrated_state.close_no_execution(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
            )
            if active_task is not None:
                self.research_task_queues[
                    arm
                ].cancel_without_execution(active_task.task_id)
                if (
                    active_task.task_type
                    is ResearchTaskTypeV1.RUN_MATCHED_CONTROL
                ):
                    self._matched_control_sources[arm].pop(
                        active_task.task_id, None
                    )
            closed = self.store.close_round(
                CloseRoundCommand(
                    round_id=opened["round_id"],
                    terminal_class="NO_EXECUTION",
                    feedback_payload=fused_feedback.to_dict(),
                    controller_state_after_digest=(
                        controller_state_before_digest
                    ),
                    resource_debits=(
                        ResourceDebitV1(
                            "PHYSICAL_LLM_CALL",
                            session.physical_call_count,
                        ),
                        ResourceDebitV1(
                            "INPUT_TOKEN", session.input_tokens
                        ),
                        ResourceDebitV1(
                            "OUTPUT_TOKEN", session.output_tokens
                        ),
                        ResourceDebitV1(
                            "BILLED_TOKEN_DEBIT",
                            session.billed_tokens,
                        ),
                        ResourceDebitV1(
                            "PROPOSAL", session.proposal_count
                        ),
                        ResourceDebitV1(
                            "PROPOSAL_ATTEMPT", session.proposal_count
                        ),
                        ResourceDebitV1(
                            "COMMON_VALIDATION",
                            len(session.validation_programs),
                        ),
                        ResourceDebitV1(
                            "WALL_TIME_MS",
                            accounted_wall_time_ms,
                        ),
                    ),
                    idempotency_key=f"m4:close:{opened['round_id']}",
                )
            )
            self._terminalize_integrated_round(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                terminal_class=str(closed["terminal_class"]),
                controller=getattr(
                    self.broker, "research_controllers", {}
                ).get(arm),
                feedback=fused_feedback,
                source_proposal_candidate_id=(
                    plan.selected_candidate_id
                    if arm in {ArmCode.B, ArmCode.C}
                    else None
                ),
            )
            return ArmRoundResultV1(
                opaque_instance_id=opaque_id,
                round_id=opened["round_id"],
                candidate_id=last_candidate_id,
                terminal_class=str(closed["terminal_class"]),
                physical_call_count=session.physical_call_count,
                proposal_count=session.proposal_count,
                input_tokens=session.input_tokens,
                output_tokens=session.output_tokens,
                billed_tokens=session.billed_tokens,
                ordinary_execution_count=0,
                gpu_device_time_ms=0,
                gpu_cost_microunits=0,
                feedback_digest=str(closed["feedback_digest"]),
                evidence_port_status=(
                    selection.last_adjudication.status.value
                ),
                training_backend_started=False,
                broker_call_latencies_ms=(
                    session.broker_call_latencies_ms
                ),
                proposal_session_wall_time_ms=(
                    session.proposal_session_wall_time_ms
                ),
                training_wall_time_ms=0,
                round_total_wall_time_ms=round_total_wall_time_ms,
            )
        selected = selection.selected_candidate
        selected_slate_index = (
            len(selection.inspected_candidate_ids) - 1
        )
        program, selected_plan, action = next(
            item
            for index, item in enumerate(eligible)
            if index == selected_slate_index
        )
        if str(action.candidate_id) != selected.candidate_id:
            raise PreCanaryInvariantError(
                "selected slate position changed during execution binding"
            )
        if (
            session.ordered_proposal_candidate_ids
            and len(session.ordered_proposal_candidate_ids)
            != len(eligible)
        ):
            raise PreCanaryInvariantError(
                "ordered proposal instances do not align with the executable slate"
            )
        selected_proposal_candidate_id = (
            session.ordered_proposal_candidate_ids[selected_slate_index]
            if session.ordered_proposal_candidate_ids
            else None
        )
        selected_research_proposal = (
            next(
                (
                    proposal
                    for proposal in session.research_proposals
                    if proposal.candidate_id
                    == selected_proposal_candidate_id
                ),
                None,
            )
            if (
                arm in {ArmCode.B, ArmCode.C}
                and active_task is None
                and selected_proposal_candidate_id is not None
            )
            else None
        )
        if (
            selected_proposal_candidate_id is not None
            and selected_research_proposal is None
        ):
            raise PreCanaryInvariantError(
                "selected proposal instance is absent from its typed session"
            )
        if selected_research_proposal is not None:
            candidate_instance_id = str(
                selected_research_proposal.candidate_id
            )
        else:
            register_candidate_instance = getattr(
                self.broker,
                "register_execution_candidate_instance",
                None,
            )
            producer_role = (
                f"research_task:{active_task.task_type.value}"
                if active_task is not None
                else "original_controller_execution"
            )
            local_identity = (
                active_task.task_id
                if active_task is not None
                else sha256_digest(
                    {
                        "arm": arm.value,
                        "round_index": round_index,
                        "search_seed": search_seed,
                        "source": "ORIGINAL_CONTROLLER_PATH",
                    }
                )
            )
            candidate_instance_id = (
                str(
                    register_candidate_instance(
                        arm=arm,
                        round_index=round_index,
                        producer_role=producer_role,
                        semantic_program_digest=str(
                            selected_plan.mechanism_semantics_digest
                        ),
                        local_parent_or_task_identity=local_identity,
                    )
                )
                if register_candidate_instance is not None
                else content_id(
                    "candidate-instance-v1",
                    {
                        "active_task_digest": (
                            active_task.digest
                            if active_task is not None
                            else None
                        ),
                        "opaque_arm_instance_id": opaque_id,
                        "round_index": round_index,
                        "semantic_program_digest": str(
                            selected_plan.mechanism_semantics_digest
                        ),
                        "source": (
                            "ACTIVE_BOUND_TASK"
                            if active_task is not None
                            else "ORIGINAL_CONTROLLER_PATH"
                        ),
                    },
                )
            )
        self.integrated_state.select_candidate(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            candidate_instance_id=candidate_instance_id,
        )
        try:
            if (
                action.release_projection_digest
                != campaign_profile_digest
            ):
                raise CampaignRuntimeError("legacy execution profile")
            selected_recipe = execution_recipe_for_program(program)
        except CampaignRuntimeError:
            selected_recipe = {
                "mechanism_id": "LEGACY_FIXTURE",
                "mechanism_semantics_digest": (
                    selected_plan.mechanism_semantics_digest
                ),
            }
        runtime_root = self.layout.arm(opaque_id).namespace("runtime")
        report = DeterministicMaterializerV1().materialize(
            action, program=program, arm_runtime_root=runtime_root
        )
        trust = classify_execution_trust(report, arm_runtime_root=runtime_root)
        base_binding = build_binding_v2(
            eligible=action,
            report=report,
            trust=trust,
            opaque_arm_instance_id=opaque_id,
            arm_private_root=runtime_root,
            round_id=opened["round_id"],
            search_seed=execution_seed,
        )
        materialization_artifacts = _register_materialization_artifacts_m4(
            self.store,
            binding=base_binding,
            report=report,
            opaque_instance_id=opaque_id,
        )
        gate = development_execution_gate(
            binding=base_binding,
            eligible=action,
            report=report,
            trust=trust,
            task_authorization_ref=(
                "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md#M1"
            ),
        )
        pre, permit = common_guard.pre_execute(
            eligible=action,
            report=report,
            trust=trust,
            binding=base_binding,
            gate=gate,
        )
        if permit is None:
            raise PreCanaryInvariantError(
                "M4 common PRE unexpectedly denied: "
                f"{tuple(pre.reason_codes)} gate={gate.decision} trust={trust.to_dict()}"
            )
        permit, binding, runtime_context = self._prepare_runtime_execution(
            base_plan=selected_plan,
            base_permit=permit,
            base_binding=base_binding,
            eligible=action,
        )
        source_kind = (
            "ACTIVE_BOUND_TASK"
            if active_task is not None
            else (
                "NORMAL_ROUTED_PROPOSAL"
                if arm in {ArmCode.B, ArmCode.C}
                else "ORIGINAL_CONTROLLER_PATH"
            )
        )
        seed_binding = ExecutionSeedBindingV1(
            {
                "active_task_digest": (
                    active_task.digest
                    if active_task is not None
                    else "ABSENT"
                ),
                "active_task_id": (
                    active_task.task_id
                    if active_task is not None
                    else "ABSENT"
                ),
                "active_task_type": (
                    active_task.task_type.value
                    if active_task is not None
                    else "ABSENT"
                ),
                "base_binding_digest": base_binding.digest,
                "binding_digest": binding.digest,
                "candidate_instance_id": candidate_instance_id,
                "execution_seed": str(execution_seed),
                "opaque_arm_instance_id": opaque_id,
                "required_seed_or_control": (
                    active_task.required_seed_or_control
                    if active_task is not None
                    else str(self.contract.ordinary_execution_seed)
                ),
                "round_id": str(binding.round_id),
                "source_kind": source_kind,
            }
        )
        seed_binding_artifact = self.store.register_artifact(
            RegisterArtifactCommand(
                round_id=str(binding.round_id),
                artifact_type="EXECUTION_SEED_BINDING_V1",
                relative_path=(
                    f"instances/{opaque_id}/audit/"
                    f"{sha256_digest({'round_id': str(binding.round_id)})}/"
                    "execution_seed_binding.v1.json"
                ),
                producer="M6IExecutionSeedBindingWriterV1",
                idempotency_key=(
                    f"m6i:execution-seed-binding:{binding.round_id}"
                ),
            ),
            canonical_json_bytes(seed_binding.to_dict()) + b"\n",
        )
        materialization_artifacts = (
            materialization_artifacts + (seed_binding_artifact,)
        )
        self._claim_runtime_execution(
            permit=permit,
            binding=binding,
            runtime_context=runtime_context,
        )
        self.integrated_state.start_execution(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
        )
        raw_output, common_result = self._execute_selected(
            permit=permit,
            binding=binding,
            runtime_context=runtime_context,
            gate=gate,
            pre_execution=pre,
            materialization_artifacts=materialization_artifacts,
        )
        register_raw_result_envelope(self.store, common_result)
        helix_raw = self._build_helix_raw(
            selected=selected,
            opaque_instance_id=opaque_id,
            common_result=common_result,
            observation_seed=str(execution_seed),
        )
        gpu_device_time_ms, gpu_cost_microunits, execution_wall_time_ms = (
            self._execution_resource_projection(raw_output)
        )
        actual_proposal = selected_research_proposal
        comparator_delta: float | str = NOT_AVAILABLE
        if isinstance(actual_proposal, CandidateProposalV4):
            protocol_digest = str(
                campaign_runtime_profile()["development_protocol_digest"]
            )
            matched_comparator_for = getattr(
                self.broker,
                "matched_comparator_for",
                None,
            )
            matched = (
                matched_comparator_for(
                    arm=arm,
                    proposal=actual_proposal,
                    protocol_digest=protocol_digest,
                    observation_seed=str(execution_seed),
                )
                if matched_comparator_for is not None
                else None
            )
            metrics = helix_raw.to_dict()["normalized_metrics"]
            metric = next(
                (
                    float(metrics[name])
                    for name in ("ndcg@10", "ndcg")
                    if isinstance(metrics.get(name), (int, float))
                ),
                None,
            )
            if matched is not None and metric is not None:
                comparator_delta = metric - float(
                    matched.comparator_metric
                )
        matched_control_task = (
            self._matched_control_task(
                arm=arm,
                proposal=actual_proposal,
                round_index=round_index,
                execution_seed=execution_seed,
            )
            if isinstance(actual_proposal, CandidateProposalV4)
            else None
        )
        post = self.ports[arm].post_run(helix_raw)
        search_utility_event = self._search_utility_event(
            selected=selected,
            selected_plan=selected_plan,
            selected_recipe=selected_recipe,
            helix_raw=helix_raw,
            gpu_device_time_ms=gpu_device_time_ms,
            gpu_cost_microunits=gpu_cost_microunits,
            execution_wall_time_ms=execution_wall_time_ms,
            comparator_delta=comparator_delta,
        )
        validation_task = self._research_task(
            task_type=ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE,
            round_index=round_index,
            selected=selected,
            selected_plan=selected_plan,
            program=program,
        )
        branch_task = self._research_task(
            task_type=ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC,
            round_index=round_index,
            selected=selected,
            selected_plan=selected_plan,
            program=program,
        )
        if (
            post.recommended_validation == "REQUIRES_CONFIRMATION"
            and validation_task is None
        ):
            # The frozen validation schedule is exhausted.  This is a typed
            # withheld/no-observation terminal, not a missing-object error.
            fused_feedback = self.admission.no_search_update(
                post.candidate_id
            )
            _private_compact_feedback = None
        else:
            fused_feedback, _private_compact_feedback = (
                self.admission.admit_post(
                    adjudication=post,
                    search_utility_event=search_utility_event,
                    validation_task=validation_task,
                    protocol_branch_task=branch_task,
                )
            )
        successful_result = str(helix_raw.run_status) in {
            "SUCCESS",
            "SMOKE_PASS",
            "COMPLETED",
        }
        if not successful_result:
            fused_feedback = FusedSearchFeedbackV2(
                candidate_id=selected.candidate_id,
                search_feedback_class=(
                    SearchFeedbackClassV2.COMMON_FAILED_EXECUTION
                ),
                search_utility_event=search_utility_event,
                frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                research_task=None,
                controller_update_allowed=True,
                meta_update_allowed=False,
                search_memory_update_allowed=True,
            )
        elif (
            active_task is not None
            and active_task.task_type
            is ResearchTaskTypeV1.RUN_MATCHED_CONTROL
            and fused_feedback.controller_update_allowed
        ):
            fused_feedback = FusedSearchFeedbackV2(
                candidate_id=selected.candidate_id,
                search_feedback_class=(
                    SearchFeedbackClassV2.DIAGNOSTIC_ONLY
                ),
                search_utility_event=search_utility_event,
                frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                research_task=None,
                controller_update_allowed=True,
                meta_update_allowed=False,
                search_memory_update_allowed=True,
            )
        elif (
            active_task is not None
            and active_task.task_type
            in {
                ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC,
                ResearchTaskTypeV1.REPAIR_IMPLEMENTATION,
            }
        ):
            fused_feedback = FusedSearchFeedbackV2(
                candidate_id=selected.candidate_id,
                search_feedback_class=(
                    SearchFeedbackClassV2.PROTOCOL_BRANCH_TASK
                    if active_task.task_type
                    is ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC
                    else SearchFeedbackClassV2.ENGINEERING_ONLY
                ),
                search_utility_event=search_utility_event,
                frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                research_task=None,
                controller_update_allowed=True,
                meta_update_allowed=False,
                search_memory_update_allowed=True,
            )
        self.integrated_state.close_result(
            arm=arm,
            search_seed=search_seed,
            round_index=round_index,
            observation_path=self._observation_path(fused_feedback),
        )
        if active_task is not None:
            if successful_result:
                self.research_task_queues[arm].complete(
                    active_task.task_id
                )
            else:
                self.research_task_queues[arm].cancel(
                    active_task.task_id
                )
        if (
            arm in self.research_task_queues
            and fused_feedback.research_task is not None
        ):
            self.research_task_queues[arm].enqueue(
                fused_feedback.research_task
            )
        if (
            matched_control_task is not None
            and successful_result
            and fused_feedback.controller_update_allowed
        ):
            self.research_task_queues[arm].enqueue(
                matched_control_task
            )
            self._matched_control_sources[arm][
                matched_control_task.task_id
            ] = actual_proposal
        if (
            isinstance(actual_proposal, CandidateProposalV4)
            and successful_result
            and fused_feedback.controller_update_allowed
        ):
            record_lineage_outcome = getattr(
                self.broker,
                "record_lineage_outcome",
                None,
            )
            if record_lineage_outcome is not None:
                record_lineage_outcome(
                    arm=arm,
                    proposal=actual_proposal,
                    runtime_candidate_id=selected.candidate_id,
                    mechanism_program_digest=str(
                        selected_plan.mechanism_program_digest
                    ),
                    mechanism_semantics_digest=str(
                        selected_plan.mechanism_semantics_digest
                    ),
                    protocol_digest=str(
                        campaign_runtime_profile()[
                            "development_protocol_digest"
                        ]
                    ),
                    observation_seed=str(execution_seed),
                    run_status=helix_raw.run_status,
                    normalized_metrics=helix_raw.to_dict()[
                        "normalized_metrics"
                    ],
                    result_digest=helix_raw.raw_result_digest,
                    round_index=round_index,
                )
        matched_control_belief: DevelopmentalMechanismBeliefV2 | None = None
        if (
            active_task is not None
            and active_task.task_type
            is ResearchTaskTypeV1.RUN_MATCHED_CONTROL
            and successful_result
            and fused_feedback.controller_update_allowed
        ):
            source = self._matched_control_sources[arm].get(
                active_task.task_id
            )
            if source is None:
                raise PreCanaryInvariantError(
                    "matched-control task lost its source proposal"
                )
            record_support_outcome = getattr(
                self.broker, "record_support_outcome", None
            )
            if record_support_outcome is None:
                raise PreCanaryInvariantError(
                    "V13 broker cannot record matched-control lineage"
                )
            record_support_outcome(
                arm=arm,
                proposal_candidate_id=candidate_instance_id,
                runtime_candidate_id=selected.candidate_id,
                mechanism_id=str(selected_recipe["mechanism_id"]),
                mechanism_axis="control",
                mechanism_program_digest=str(
                    selected_plan.mechanism_program_digest
                ),
                mechanism_semantics_digest=str(
                    selected_plan.mechanism_semantics_digest
                ),
                protocol_digest=active_task.protocol_digest,
                observation_seed=str(execution_seed),
                run_status=helix_raw.run_status,
                normalized_metrics=helix_raw.to_dict()[
                    "normalized_metrics"
                ],
                result_digest=helix_raw.raw_result_digest,
                round_index=round_index,
                mechanism_program=program,
                parent_candidate_id=active_task.parent_candidate_id,
            )
            matched_control_belief = self._matched_control_belief(
                arm=arm,
                source=source,
                task=active_task,
                control_result=helix_raw,
                control_candidate_instance_id=candidate_instance_id,
            )
        post_result_learning_error: Exception | None = None
        boundary_controller: ResearchLineControllerV1 | None = None
        boundary_source_candidate_id: str | None = None
        if arm is ArmCode.A:
            original_feedback = {
                "candidate_id": selected.candidate_id,
                "mechanism_id": selected_recipe["mechanism_id"],
                "mechanism_semantics_digest": (
                    selected_plan.mechanism_semantics_digest
                ),
                "round_index": round_index,
                "search_outcome": {
                    "gpu_cost_microunits": gpu_cost_microunits,
                    "gpu_device_time_ms": gpu_device_time_ms,
                    "normalized_metrics": helix_raw.to_dict()[
                        "normalized_metrics"
                    ],
                    "run_status": helix_raw.run_status,
                    "wall_time_ms": execution_wall_time_ms,
                },
            }
            original_runtime = getattr(
                self.broker, "original_controller", None
            )
            if original_runtime is None:
                transition = OriginalControllerV1().close_round(
                    original_feedback, "NO_WRITE"
                )
            else:
                transition = original_runtime.close_round(
                    original_feedback
                )
            after_digest = str(transition["transition_digest"])
        else:
            plan = session.research_plan
            if plan is None:
                raise PreCanaryInvariantError("Research Arm is missing its plan")
            if actual_proposal is None and active_task is None:
                raise PreCanaryInvariantError(
                    "Research execution is missing selected proposal lineage"
                )
            beliefs = (
                (
                    self._research_belief(
                        selected=selected,
                        event=fused_feedback.search_utility_event,
                        proposal=actual_proposal,
                        next_task=matched_control_task,
                    ),
                )
                if (
                    successful_result
                    and fused_feedback.controller_update_allowed
                    and fused_feedback.search_utility_event is not None
                    and actual_proposal is not None
                )
                else ()
            )
            if matched_control_belief is not None:
                beliefs = (matched_control_belief,)
            boundary_controller = self.broker.research_controllers[arm]
            transition = boundary_controller.close_round_v13(
                plan=plan,
                feedback=fused_feedback,
                beliefs=beliefs,
            )
            after_digest = (
                str(transition["round_transition_digest"])
                if transition["state_changed"]
                else controller_state_before_digest
            )
            record_feedback = getattr(
                self.broker, "record_search_feedback", None
            )
            if record_feedback is not None and transition["state_changed"]:
                record_feedback(
                    arm,
                    transition["search_memory_projection"],
                    executed_mechanism_id=str(
                        selected_recipe["mechanism_id"]
                    )
                    if active_task is None
                    else None,
                    execution_succeeded=(
                        search_utility_event.runnable_observation
                        == "RUNNABLE"
                    ),
                )
            boundary_source_candidate_id = (
                actual_proposal.candidate_id
                if actual_proposal is not None
                else active_task.candidate_id
            )
        try:
            self._terminalize_integrated_round(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                terminal_class="COMPLETED",
                controller=boundary_controller,
                feedback=fused_feedback,
                source_proposal_candidate_id=(
                    boundary_source_candidate_id
                ),
            )
        except Exception as error:
            post_result_learning_error = error
        round_total_wall_time_ms = max(
            session.proposal_session_wall_time_ms
            + execution_wall_time_ms,
            int(
                (time.monotonic_ns() - round_started_ns)
                / 1_000_000
            ),
        )
        accounted_wall_time_ms = (
            round_total_wall_time_ms
            if bool(getattr(self.broker, "v13_mode", False))
            else execution_wall_time_ms
        )
        execution_debits = _round_execution_budget_debits(
            gpu_device_time_ms=gpu_device_time_ms,
            gpu_cost_microunits=gpu_cost_microunits,
            wall_time_ms=accounted_wall_time_ms,
            ceilings=self.resource_ceilings,
            resource_ceiling_rejected=(
                common_result.exit_status == "COMMON_EXECUTION_FAILURE"
                and common_result.metric_source
                == "NOT_ADMITTED_RESOURCE_CEILING"
            ),
        )
        closed = self.store.close_round(
            CloseRoundCommand(
                round_id=opened["round_id"],
                terminal_class="COMPLETED",
                feedback_payload=fused_feedback.to_dict(),
                controller_state_after_digest=after_digest,
                resource_debits=(
                    ResourceDebitV1(
                        "PHYSICAL_LLM_CALL", session.physical_call_count
                    ),
                    ResourceDebitV1("INPUT_TOKEN", session.input_tokens),
                    ResourceDebitV1("OUTPUT_TOKEN", session.output_tokens),
                    ResourceDebitV1("BILLED_TOKEN_DEBIT", session.billed_tokens),
                    ResourceDebitV1("PROPOSAL", session.proposal_count),
                    ResourceDebitV1(
                        "PROPOSAL_ATTEMPT", session.proposal_count
                    ),
                    ResourceDebitV1(
                        "COMMON_VALIDATION", len(session.validation_programs)
                    ),
                )
                + execution_debits,
                idempotency_key=f"m4:close:{opened['round_id']}",
            )
        )
        if post_result_learning_error is not None:
            raise post_result_learning_error
        if (
            active_task is not None
            and active_task.task_type
            is ResearchTaskTypeV1.RUN_MATCHED_CONTROL
        ):
            self._matched_control_sources[arm].pop(
                active_task.task_id, None
            )
        return ArmRoundResultV1(
            opaque_instance_id=opaque_id,
            round_id=opened["round_id"],
            candidate_id=selected.candidate_id,
            terminal_class=str(closed["terminal_class"]),
            physical_call_count=session.physical_call_count,
            proposal_count=session.proposal_count,
            input_tokens=session.input_tokens,
            output_tokens=session.output_tokens,
            billed_tokens=session.billed_tokens,
            ordinary_execution_count=1,
            gpu_device_time_ms=gpu_device_time_ms,
            gpu_cost_microunits=gpu_cost_microunits,
            feedback_digest=str(closed["feedback_digest"]),
            evidence_port_status=post.status.value,
            training_backend_started=bool(raw_output.training_backend_started),
            broker_call_latencies_ms=session.broker_call_latencies_ms,
            proposal_session_wall_time_ms=(
                session.proposal_session_wall_time_ms
            ),
            training_wall_time_ms=execution_wall_time_ms,
            round_total_wall_time_ms=round_total_wall_time_ms,
        )

    def neutral_audit_projection(
        self, results: Sequence[ArmRoundResultV1]
    ) -> dict[str, Any]:
        self._validate_triplet_results(results)
        opaque_results = tuple(
            {
                "budget_closed": (
                    item.input_tokens <= self.resource_ceilings.total_input_tokens
                    and item.output_tokens <= self.resource_ceilings.total_output_tokens
                    and item.billed_tokens
                    <= self.resource_ceilings.total_billed_token_debit
                    and item.proposal_count
                    <= self.resource_ceilings.total_proposal_count
                    and item.ordinary_execution_count <= 1
                ),
                "feedback_present": bool(item.feedback_digest),
                "opaque_instance_id": item.opaque_instance_id,
                "round_closed": item.terminal_class == "COMPLETED",
            }
            for item in sorted(results, key=lambda result: result.opaque_instance_id)
        )
        projection = {
            "assignment_commitment": self.assignment.commitment,
            "common_execution_policy_digest": common_release_projection_digest(),
            "deterministic_fusion_digest": self.admission.policy_digest,
            "instances": opaque_results,
            "triplet_closed": all(item["round_closed"] for item in opaque_results),
        }
        forbidden = (
            "arm",
            "controller",
            "evidence_port",
            "guard",
            "physical_call",
            "producer",
            "research",
            "treatment",
        )
        lowered = canonical_json_bytes(projection).decode("utf-8").lower()
        if any(token in lowered for token in forbidden):
            raise PreCanaryInvariantError("neutral projection contains treatment data")
        return projection

    def _validate_triplet_results(
        self, results: Sequence[ArmRoundResultV1]
    ) -> None:
        expected_ids = set(self.assignment.mapping.values())
        actual_ids = {item.opaque_instance_id for item in results}
        if len(results) != 3 or actual_ids != expected_ids:
            raise PreCanaryInvariantError("result set is not the exact opaque triplet")
        for item in results:
            if (
                item.terminal_class != "COMPLETED"
                or item.ordinary_execution_count != 1
                or item.training_backend_started
                or item.input_tokens > self.resource_ceilings.total_input_tokens
                or item.output_tokens > self.resource_ceilings.total_output_tokens
                or item.billed_tokens
                > self.resource_ceilings.total_billed_token_debit
                or item.proposal_count
                > self.resource_ceilings.total_proposal_count
            ):
                raise PreCanaryInvariantError("triplet result fails the M4 run gate")

    def four_axis_totals(
        self, results: Sequence[ArmRoundResultV1]
    ) -> dict[str, dict[str, int]]:
        self._validate_triplet_results(results)
        connection = sqlite3.connect(self.store.db_path)
        try:
            totals: dict[str, dict[str, int]] = {}
            for item in results:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(DISTINCT r.round_id) AS round_count,
                        COALESCE(SUM(CASE WHEN l.dimension='ORDINARY_EXECUTION'
                            THEN l.quantity ELSE 0 END), 0) AS execution_count,
                        COALESCE(SUM(CASE WHEN l.dimension='BILLED_TOKEN_DEBIT'
                            THEN l.quantity ELSE 0 END), 0) AS token_count,
                        COALESCE(SUM(CASE WHEN l.dimension='GPU_COST_MICROUNITS'
                            THEN l.quantity ELSE 0 END), 0) AS gpu_cost
                    FROM rounds r
                    LEFT JOIN resource_ledger l ON l.round_id = r.round_id
                    WHERE r.round_id = ? AND r.arm_instance_id = ?
                    """,
                    (item.round_id, item.opaque_instance_id),
                ).fetchone()
                if row is None or int(row[0]) != 1:
                    raise PreCanaryInvariantError(
                        "four-axis reconstruction is missing a closed round"
                    )
                totals[item.opaque_instance_id] = {
                    "execution_count": int(row[1]),
                    "gpu_cost_microunits": int(row[3]),
                    "round_count": int(row[0]),
                    "token_count": int(row[2]),
                }
            return totals
        finally:
            connection.close()

    def seal_canary_review_packet(
        self,
        output_root: Path,
        results: Sequence[ArmRoundResultV1],
    ) -> dict[str, Any]:
        self._validate_triplet_results(results)
        integrity = self.store.integrity_report()
        if (
            integrity["integrity_check"] != "ok"
            or integrity["foreign_key_violations"]
        ):
            raise PreCanaryInvariantError("state-store integrity gate failed")
        connection = sqlite3.connect(self.store.db_path)
        try:
            round_ids = tuple(item.round_id for item in results)
            barriers = connection.execute(
                f"""
                SELECT DISTINCT b.closed_bitmap, b.next_index_authorized
                FROM rounds r
                JOIN triplet_barrier b
                  ON b.experiment_id=r.experiment_id
                 AND b.search_seed=r.search_seed
                 AND b.round_index=r.round_index
                WHERE r.round_id IN ({','.join('?' for _ in round_ids)})
                """,
                round_ids,
            ).fetchall()
        finally:
            connection.close()
        if barriers != [(7, 1)]:
            raise PreCanaryInvariantError("triplet barrier is not closed and authorized")
        if self.guard_ledger.count() != 2:
            raise PreCanaryInvariantError("Arm C PRE/POST audit call closure is incomplete")
        output_root.mkdir(parents=True, exist_ok=False)
        neutral = self.neutral_audit_projection(results)
        private = {
            "assignment_commitment": self.assignment.commitment,
            "mapping": {
                arm.value: opaque_id
                for arm, opaque_id in self.assignment.arm_to_instance
            },
            "nonce_digest": self.assignment.nonce_digest,
        }
        neutral_path = output_root / "neutral_audit_projection.json"
        mapping_path = output_root / "sealed_treatment_mapping.json"
        neutral_path.write_bytes(canonical_json_bytes(neutral) + b"\n")
        mapping_path.write_bytes(canonical_json_bytes(private) + b"\n")
        mapping_path.chmod(0o600)
        packet = {
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "mapping_commitment": self.assignment.commitment,
            "neutral_projection_sha256": sha256_digest(neutral),
            "sealed_mapping_sha256": sha256_digest(private),
            "verdict": "READY_FOR_CANARY",
        }
        packet_path = output_root / "canary_review_packet.json"
        packet_path.write_bytes(canonical_json_bytes(packet) + b"\n")
        return packet


__all__ = [
    "ArmFilesystemCapabilityV1",
    "ArmPrivateRootsV1",
    "ArmRoundResultV1",
    "PreCanaryInvariantError",
    "PrivateTreatmentAssignmentV1",
    "RuntimeLayoutV1",
    "ThreeArmFakeBrokerV1",
    "ThreeArmPreCanaryOrchestratorV1",
    "TreatmentAssignmentEnvelopeV1",
    "m4_budget",
    "probe_uid_isolation",
]
