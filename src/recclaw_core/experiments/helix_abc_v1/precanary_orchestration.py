"""Minimum-sufficient, no-training M4 three-arm orchestration.

The neutral scheduler is the sole state-store writer. Arm workers receive only
opaque identities, arm-private filesystem capabilities, a frozen proposal
session, and public fused feedback. This module intentionally does not provide
real LLM or training backends.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space import compile_program
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
    FusedSearchFeedbackV2,
    NOT_AVAILABLE,
    ResearchTaskQueueV1,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchUtilityEventV2,
)

from .canonical import canonical_json_bytes, canonical_value, sha256_digest
from .broker_failure_closure import (
    BrokerFailureClosureV1,
    close_broker_failure,
)
from .canary_broker import CanaryBrokerError
from .campaign_runtime import (
    CampaignRuntimeError,
    campaign_runtime_profile,
    executable_mechanism,
    execution_recipe_for_program,
    root_parent_mechanism_id,
)
from .common_execution_guard import CommonExecutionGuardV1
from .contracts import (
    ArmCode,
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from .controllers import OriginalControllerV1
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
    DevelopmentalMechanismBeliefV1,
    ProducerSessionResultV1,
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
    research_proposals: tuple[CandidateProposalV2 | CandidateProposalV3, ...] = ()
    producer_session: ProducerSessionResultV1 | None = None
    research_task: ResearchTaskV1 | None = None


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
                {"space": "BL_ICF_EXECUTABLE_PROFILE_V1"},
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
            research_proposals=session.proposals,
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
            ArmCode.B: ResearchTaskQueueV1(),
            ArmCode.C: ResearchTaskQueueV1(),
        }
        self._completed: dict[tuple[int, int], tuple[ArmRoundResultV1, ...]] = {}

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
            comparator_delta=NOT_AVAILABLE,
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
        proposal: CandidateProposalV2,
    ) -> DevelopmentalMechanismBeliefV1:
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
    ) -> tuple[ArmRoundResultV1, ...]:
        key = (search_seed, round_index)
        if key in self._completed:
            return self._completed[key]
        ceilings = self.resource_ceilings
        results: list[ArmRoundResultV1] = []
        order = sorted(
            ArmCode,
            key=lambda arm: sha256_digest(
                {"arm": arm.value, "round": round_index, "seed": search_seed}
            ),
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
        self._completed[key] = ordered_results
        return ordered_results

    def _run_arm(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> ArmRoundResultV1:
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
        active_task: ResearchTaskV1 | None = None
        if arm in self.research_task_queues:
            pending_task = self.research_task_queues[arm].select_next(
                allowed_types=frozenset(
                    {ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE}
                )
            )
            if pending_task is not None:
                active_task = self.research_task_queues[arm].activate(
                    pending_task.task_id
                )
        if active_task is not None:
            session = self._session_for_research_task(
                arm=arm,
                round_index=round_index,
                task=active_task,
            )
        else:
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
                )
                raise BrokerRoundFailureError(closure) from error
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
        execution_seed = (
            int(active_task.required_seed_or_control)
            if active_task is not None
            else int(self.contract.ordinary_execution_seed)
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
            del plan
            fused_feedback = self.admission.no_search_update(
                last_candidate_id
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
                    ),
                    idempotency_key=f"m4:close:{opened['round_id']}",
                )
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
            )
        selected = selection.selected_candidate
        program, selected_plan, action = next(
            item
            for item in eligible
            if str(item[2].candidate_id) == selected.candidate_id
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
        self._claim_runtime_execution(
            permit=permit,
            binding=binding,
            runtime_context=runtime_context,
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
        post = self.ports[arm].post_run(helix_raw)
        search_utility_event = self._search_utility_event(
            selected=selected,
            selected_plan=selected_plan,
            selected_recipe=selected_recipe,
            helix_raw=helix_raw,
            gpu_device_time_ms=gpu_device_time_ms,
            gpu_cost_microunits=gpu_cost_microunits,
            execution_wall_time_ms=execution_wall_time_ms,
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
        fused_feedback, _private_compact_feedback = (
            self.admission.admit_post(
                adjudication=post,
                search_utility_event=search_utility_event,
                validation_task=validation_task,
                protocol_branch_task=branch_task,
            )
        )
        if active_task is not None:
            self.research_task_queues[arm].complete(
                active_task.task_id
            )
        if (
            arm in self.research_task_queues
            and fused_feedback.research_task is not None
        ):
            self.research_task_queues[arm].enqueue(
                fused_feedback.research_task
            )
        post_result_learning_error: Exception | None = None
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
            actual_proposal = (
                next(
                    (
                        proposal
                        for proposal in session.research_proposals
                        if str(
                            compile_program(
                                deep_thaw(proposal.mechanism_program)
                            ).candidate_id
                        )
                        == selected.candidate_id
                    ),
                    None,
                )
                if active_task is None
                else None
            )
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
                    ),
                )
                if (
                    fused_feedback.controller_update_allowed
                    and fused_feedback.search_utility_event is not None
                    and actual_proposal is not None
                )
                else ()
            )
            transition = self.broker.research_controllers[
                arm
            ].close_round_v13(
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
                    ),
                    execution_succeeded=(
                        search_utility_event.runnable_observation
                        == "RUNNABLE"
                    ),
                )
            if (
                fused_feedback.meta_update_allowed
                and active_task is None
            ):
                try:
                    self._after_research_close(
                        arm=arm,
                        round_index=round_index,
                        controller=(
                            self.broker.research_controllers[arm]
                        ),
                        feedback=fused_feedback,
                        source_proposal_candidate_id=(
                            actual_proposal.candidate_id
                            if actual_proposal is not None
                            else active_task.candidate_id
                        ),
                    )
                except Exception as error:
                    post_result_learning_error = error
        execution_debits = (
            ResourceDebitV1("GPU_DEVICE_TIME_MS", gpu_device_time_ms),
            ResourceDebitV1("GPU_COST_MICROUNITS", gpu_cost_microunits),
            ResourceDebitV1("WALL_TIME_MS", execution_wall_time_ms),
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
