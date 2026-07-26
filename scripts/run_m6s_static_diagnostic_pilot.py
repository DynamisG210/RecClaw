#!/usr/bin/env python3
"""Run one development-only static-policy A/B/C diagnostic Pilot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_m6_pilot import (  # noqa: E402
    broker_export,
    collect_rows,
    file_sha256,
    pilot_environment_preflight,
    resource_audit,
    runtime_identity_audit,
    source_snapshot,
    write_json,
)

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CodexCliCanaryBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.contracts import (  # noqa: E402
    ArmCode,
    ArmPolicyV1,
    MetaPolicyModeV1,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.m6e_conformance import (  # noqa: E402
    require_m6e_conformance_packet,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    ArmRoundResultV1,
    PreCanaryInvariantError,
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (  # noqa: E402
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    RealPilotOrchestratorV1,
    pilot_budget,
    pilot_guard_context,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (  # noqa: E402
    common_release_projection_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    TrainingExecutionPurposeV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    TRAINING_RUNNER_ABI,
    training_runtime_release_digest,
)


STATIC_DIAGNOSTIC_SEARCH_SEED = 9206
STATIC_DIAGNOSTIC_ROUNDS_PER_ARM = 1
STATIC_DIAGNOSTIC_EXPERIMENT_ID = (
    "HELIX-ABC-DEVELOPMENT-STATIC-DIAGNOSTIC-9206-V1"
)
STATIC_DIAGNOSTIC_NONCE = "M6S-STATIC-DIAGNOSTIC-9206-OPAQUE-V1"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6s"
    / "STATIC_DIAGNOSTIC_PILOT_CONTRACT_V1.json"
)
SEALED_PILOT_SEEDS = frozenset({9201, 9202, 9203, 9204, 9205})


def static_policy_identity_digest() -> str:
    return sha256_digest(
        {
            "controller": "ResearchLineControllerV1",
            "meta_mode": MetaPolicyModeV1.STATIC_RESEARCH_ROUTER.value,
            "policy": "RESEARCH_STATIC_V2",
            "round_boundary_update": "NO_OP",
            "router": "StrongStaticRouterV1",
        }
    )


def static_arm_policies() -> tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]:
    base = default_experiment_contract()
    policies = []
    for policy in base.arm_policies:
        if policy.arm in {ArmCode.B, ArmCode.C}:
            policy = replace(
                policy,
                meta_policy_mode=MetaPolicyModeV1.STATIC_RESEARCH_ROUTER,
                controller_policy_digest=static_policy_identity_digest(),
            )
        policies.append(policy)
    return tuple(policies)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class StaticDiagnosticPilotStoreContractV1:
    experiment_id: str
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "StaticDiagnosticPilotStoreContractV1":
        base = default_experiment_contract()
        policies = static_arm_policies()
        payload = {
            "arm_policies": [item.to_dict() for item in policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": STATIC_DIAGNOSTIC_EXPERIMENT_ID,
            "formal_acceptance": False,
            "main_eligibility": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": STATIC_DIAGNOSTIC_ROUNDS_PER_ARM,
            "search_seeds": [STATIC_DIAGNOSTIC_SEARCH_SEED],
        }
        return cls(
            experiment_id=STATIC_DIAGNOSTIC_EXPERIMENT_ID,
            arm_policies=policies,
            search_seeds=(STATIC_DIAGNOSTIC_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=STATIC_DIAGNOSTIC_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


def static_diagnostic_guard_context():
    context = pilot_guard_context()
    claim = canonical_value(context.claim)
    protocol = canonical_value(context.protocol)
    current_evidence = canonical_value(context.current_evidence)
    from recclaw_core.helix.contracts import GuardContext

    return GuardContext(
        claim={
            **claim,
            "claim_id": "CLAIM-M6S-STATIC-DIAGNOSTIC-9206-V1",
        },
        protocol=protocol,
        current_evidence={
            **current_evidence,
            "snapshot_id": "M6S-STATIC-DIAGNOSTIC-9206-V1-EMPTY",
            "claim_id": "CLAIM-M6S-STATIC-DIAGNOSTIC-9206-V1",
        },
    )


class StaticDiagnosticPilotOrchestratorV1(RealPilotOrchestratorV1):
    """One-round Pilot that keeps the Research policy static by construction."""

    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
        project_root: Path,
        recbole_root: Path,
        data_path: Path,
        python_executable: Path,
    ) -> None:
        require_m6e_conformance_packet(project_root)
        super().__init__(
            root,
            broker=broker,
            project_root=project_root,
            recbole_root=recbole_root,
            data_path=data_path,
            python_executable=python_executable,
            _contract=StaticDiagnosticPilotStoreContractV1.create(),
            _assignment_nonce=STATIC_DIAGNOSTIC_NONCE,
            _guard_context=static_diagnostic_guard_context(),
        )

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: Any,
        feedback_projection: Mapping[str, Any],
    ) -> None:
        del round_index, feedback_projection
        if arm not in {ArmCode.B, ArmCode.C}:
            raise PreCanaryInvariantError("static policy boundary escaped Research Arms")
        if controller.policy.version != 1:
            raise PreCanaryInvariantError("static diagnostic policy changed version")

    def run_static_diagnostic(
        self,
    ) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        return (
            self.run_fake_triplet(
                search_seed=STATIC_DIAGNOSTIC_SEARCH_SEED,
                round_index=1,
                drafts=(),
            ),
        )


def static_diagnostic_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_instance_ids: set[str],
    guard_call_count: int,
    meta_versions: Mapping[str, int],
) -> dict[str, Any]:
    grouped = {
        instance: [row for row in rows if row["opaque_instance_id"] == instance]
        for instance in expected_instance_ids
    }
    checks = {
        "one_row_per_arm": (
            len(rows) == 3
            and all(len(instance_rows) == 1 for instance_rows in grouped.values())
        ),
        "all_training_runs_successful": (
            len(rows) == 3
            and all(row.get("run_status") == "SUCCESS" for row in rows)
        ),
        "all_success_metrics_present": all(
            row.get("run_status") != "SUCCESS"
            or isinstance(row.get("ndcg"), (int, float))
            for row in rows
        ),
        "guard_pre_post_exact": guard_call_count == 2,
        "meta_not_activated": (
            set(meta_versions) == {"B", "C"}
            and all(int(version) == 1 for version in meta_versions.values())
        ),
    }
    return {
        "checks": checks,
        "meta_versions": dict(meta_versions),
        "row_count": len(rows),
        "verdict": "CHAIN_PASS" if all(checks.values()) else "NOT_READY",
    }


def descriptive_effect_summary(
    rows: Sequence[Mapping[str, Any]],
    arm_to_instance: Mapping[str, str],
) -> dict[str, Any]:
    by_instance = {str(row["opaque_instance_id"]): row for row in rows}
    by_arm = {
        arm: by_instance.get(instance)
        for arm, instance in sorted(arm_to_instance.items())
    }
    metrics = {
        arm: (
            float(row["ndcg"])
            if row is not None and isinstance(row.get("ndcg"), (int, float))
            else None
        )
        for arm, row in by_arm.items()
    }

    def delta(left: str, right: str) -> float | None:
        if metrics.get(left) is None or metrics.get(right) is None:
            return None
        return float(metrics[left]) - float(metrics[right])

    return {
        "arm_rows": {
            arm: (
                {
                    "candidate_id": row["candidate_id"],
                    "ndcg": metrics[arm],
                    "run_status": row["run_status"],
                }
                if row is not None
                else None
            )
            for arm, row in by_arm.items()
        },
        "contrasts": {
            "B_minus_A": delta("B", "A"),
            "C_minus_A": delta("C", "A"),
            "C_minus_B": delta("C", "B"),
        },
        "interpretation": "DESCRIPTIVE_SINGLE_ROUND_ONLY",
        "main_or_meta_evidence": False,
    }


def verify_contract(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    content = dict(contract)
    expected_digest = content.pop("content_digest")
    if sha256_digest(content) != expected_digest:
        raise RuntimeError("static diagnostic contract content digest mismatch")
    if contract["status"] != "FROZEN_PRE_OUTCOME":
        raise RuntimeError("static diagnostic contract is not frozen")
    if contract["record_schema"] != "recclaw.static-diagnostic-pilot-contract.v1":
        raise RuntimeError("wrong static diagnostic contract schema")
    for relative, expected_hash in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"static diagnostic source mismatch: {relative}")
    exact_files = {
        Path(contract["broker"]["response_schema_path"]): contract["broker"][
            "response_schema_sha256"
        ],
        Path(contract["bl_icf"]["template_fixture_path"]): contract["bl_icf"][
            "template_fixture_sha256"
        ],
        Path(contract["training"]["profile_path"]): contract["training"][
            "profile_sha256"
        ],
        Path(contract["broker"]["codex_executable"]): contract["broker"][
            "codex_executable_sha256"
        ],
        Path(contract["broker"]["models_cache_path"]): contract["broker"][
            "models_cache_sha256"
        ],
    }
    for path, expected_hash in exact_files.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"static diagnostic external identity mismatch: {path}")

    store = StaticDiagnosticPilotStoreContractV1.create()
    expected_pilot = {
        "experiment_id": store.experiment_id,
        "ordinary_execution_seed": store.ordinary_execution_seed,
        "rounds_per_arm": store.scheduled_slots_per_arm_seed,
        "search_seeds": list(store.search_seeds),
        "store_contract_identity_digest": store.identity_digest,
    }
    if contract["pilot"] != expected_pilot:
        raise RuntimeError("static diagnostic store identity mismatch")
    if SEALED_PILOT_SEEDS.intersection(store.search_seeds):
        raise RuntimeError("static diagnostic reuses a sealed Pilot seed")
    assignment = PrivateTreatmentAssignmentV1.create(
        store.experiment_id, nonce=STATIC_DIAGNOSTIC_NONCE
    )
    if contract["assignment"] != {
        "commitment": assignment.commitment,
        "opaque": True,
    }:
        raise RuntimeError("static diagnostic treatment assignment mismatch")
    if contract["research"] != {
        "agentization_gate": "PASS_INDEPENDENT_MULTI_AGENT",
        "formal_meta_required_for_main": True,
        "meta_activation": "NONE",
        "meta_mode": MetaPolicyModeV1.STATIC_RESEARCH_ROUTER.value,
        "policy_label": "RESEARCH_STATIC_V2",
        "producer_mode": "BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1",
        "static_policy_identity_digest": static_policy_identity_digest(),
    }:
        raise RuntimeError("static diagnostic Research treatment mismatch")
    expected_arms = {
        "A": {
            "controller": "OriginalControllerV1",
            "evidence_port": "NullEvidencePortV1",
            "physical_llm_call_ceiling_per_round": 1,
        },
        "B": {
            "controller": "ResearchLineControllerV1",
            "evidence_port": "NullEvidencePortV1",
            "physical_llm_call_ceiling_per_round": 4,
        },
        "C": {
            "controller": "ResearchLineControllerV1",
            "evidence_port": "EvidenceGuardPortV1",
            "physical_llm_call_ceiling_per_round": 4,
        },
    }
    if contract["arm_composition"] != expected_arms:
        raise RuntimeError("static diagnostic A/B/C composition mismatch")
    if contract["budget_per_arm_round"] != pilot_budget().to_dict():
        raise RuntimeError("static diagnostic changed the common Pilot budget")
    if (
        contract["bl_icf"]["common_release_projection_digest"]
        != common_release_projection_digest()
    ):
        raise RuntimeError("static diagnostic common BL/runtime identity mismatch")
    if (
        contract["training"]["runner_abi"] != TRAINING_RUNNER_ABI
        or contract["training"]["runtime_release_digest"]
        != training_runtime_release_digest()
        or contract["training"]["execution_purpose"]
        != TrainingExecutionPurposeV1.PILOT.value
    ):
        raise RuntimeError("static diagnostic training identity mismatch")
    packet = require_m6e_conformance_packet(ROOT)
    if (
        contract["m6e"]["conformance_packet_digest"] != packet["content_digest"]
        or contract["m6e"]["P0"] != 0
        or contract["m6e"]["P1"] != 0
    ):
        raise RuntimeError("static diagnostic does not bind passing M6E")
    if (
        contract["m6f"]["verdict"] != "PASS"
        or contract["m6f"]["P0"] != 0
        or contract["m6f"]["P1"] != 0
    ):
        raise RuntimeError("static diagnostic does not bind passing M6F")
    return contract


def execute(contract_path: Path, output_root: Path) -> int:
    if output_root.exists():
        raise RuntimeError("static diagnostic output root already exists")
    output_root.mkdir(parents=True)
    contract = verify_contract(contract_path)
    write_json(
        output_root / "FROZEN_CONTRACT_IDENTITY.json",
        {
            "contract_content_digest": contract["content_digest"],
            "contract_sha256": file_sha256(contract_path),
        },
    )
    preflight = pilot_environment_preflight(contract)
    write_json(output_root / "ENVIRONMENT_PREFLIGHT.json", preflight)
    before = source_snapshot(contract)
    upstream = CodexCliCanaryBrokerV1(
        output_root / "broker_private",
        schema_path=Path(contract["broker"]["response_schema_path"]),
        codex_executable=Path(contract["broker"]["codex_executable"]),
        model=contract["broker"]["model"],
        reasoning_effort=contract["broker"]["reasoning_effort"],
        service_tier=contract["broker"]["service_tier"],
        max_total_tokens_per_call=int(
            contract["broker"]["max_total_tokens_per_call"]
        ),
        cli_version=contract["broker"]["codex_cli_version"],
        login_mode=contract["broker"]["login_mode"],
        release_manifest_path=(
            ROOT
            / "src"
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "resources"
            / "broker_process_release_v2.json"
        ),
    )
    broker = RealCanaryProposalBrokerV1.create(
        upstream=upstream,
        template_path=Path(contract["bl_icf"]["template_fixture_path"]),
        call_prefix="m6s-static-",
        phase_name="Static Diagnostic Pilot",
        adaptive_memory=True,
    )
    try:
        with StaticDiagnosticPilotOrchestratorV1(
            output_root / "runtime",
            broker=broker,
            project_root=ROOT,
            recbole_root=Path(contract["runtime"]["recbole_root"]),
            data_path=Path(contract["dataset"]["root"]).parent,
            python_executable=Path(contract["runtime"]["python"]),
        ) as orchestrator:
            try:
                rounds = orchestrator.run_static_diagnostic()
            except Exception:
                write_json(
                    output_root / "IMMUTABLE_AUDIT_BUNDLE.json",
                    orchestrator.immutable_audit_bundle(
                        output_root / "audit_snapshots"
                    ),
                )
                raise
            audit_bundle = orchestrator.immutable_audit_bundle(
                output_root / "audit_snapshots"
            )
            write_json(
                output_root / "IMMUTABLE_AUDIT_BUNDLE.json", audit_bundle
            )
            state_db = (
                output_root / "audit_snapshots" / "neutral_state.audit.sqlite3"
            )
            broker_db = (
                output_root / "audit_snapshots" / "broker_state.audit.sqlite3"
            )
            guard_db = (
                output_root / "audit_snapshots" / "guard_state.audit.sqlite3"
            )
            audit = orchestrator.pilot_audit(state_db, guard_db)
            mapping = {
                arm.value: opaque
                for arm, opaque in orchestrator.assignment.arm_to_instance
            }
            write_json(
                output_root / "sealed" / "ROUND_RESULTS.json",
                [
                    [result.to_dict() for result in triplet]
                    for triplet in rounds
                ],
                mode=0o600,
            )
            write_json(
                output_root / "sealed" / "TREATMENT_MAPPING.json",
                {
                    "assignment_commitment": orchestrator.assignment.commitment,
                    "mapping": mapping,
                    "nonce_digest": orchestrator.assignment.nonce_digest,
                },
                mode=0o600,
            )
            write_json(
                output_root / "NEUTRAL_AUDIT.json",
                audit_bundle["neutral_projection"],
            )
            resource = resource_audit(state_db, contract)
            runtime_identity = runtime_identity_audit(state_db)
            rows = collect_rows(
                state_db,
                output_root / "runtime" / "neutral" / "artifacts",
            )
        upstream.close()
        calls = broker_export(broker_db)
        after = source_snapshot(contract)
        readiness = static_diagnostic_readiness(
            rows,
            expected_instance_ids=set(mapping.values()),
            guard_call_count=int(audit["guard_call_count"]),
            meta_versions=audit["meta_versions"],
        )
        effects = descriptive_effect_summary(rows, mapping)
        expected_rounds = 3
        expected_calls = int(contract["broker"]["expected_upstream_calls"])
        gates = {
            "barriers_closed": bool(audit["barriers_closed"]),
            "broker_call_count": sum(row["call_count"] for row in calls),
            "broker_failures": sum(
                row["call_count"]
                for row in calls
                if row["status"] != "SUCCESS"
            ),
            "budget_accounting_closed": bool(resource["closed"]),
            "chain_readiness": readiness["verdict"],
            "execution_count": int(audit["execution_count"]),
            "feedback_count": int(audit["feedback_count"]),
            "guard_call_count": int(audit["guard_call_count"]),
            "meta_not_activated": readiness["checks"]["meta_not_activated"],
            "no_source_mutation": before == after,
            "round_count": int(audit["round_count"]),
            "runtime_identity_equal": bool(
                runtime_identity["A_B_C_common_runtime_identity_equal"]
            ),
            "state_store_integrity": (
                audit["state_store_integrity"]["integrity_check"] == "ok"
                and not audit["state_store_integrity"]["foreign_key_violations"]
            ),
        }
        passed = (
            gates["barriers_closed"]
            and gates["broker_call_count"] == expected_calls
            and gates["broker_failures"] == 0
            and gates["budget_accounting_closed"]
            and gates["chain_readiness"] == "CHAIN_PASS"
            and gates["execution_count"] == expected_rounds
            and gates["feedback_count"] == expected_rounds
            and gates["guard_call_count"] == 2
            and gates["meta_not_activated"]
            and gates["no_source_mutation"]
            and gates["round_count"] == expected_rounds
            and gates["runtime_identity_equal"]
            and runtime_identity["closed_execution_counts"] == [1, 1, 1]
            and runtime_identity["runtime_binding_count"] == expected_rounds
            and gates["state_store_integrity"]
        )
        write_json(output_root / "BROKER_CALL_AUDIT.json", calls)
        write_json(
            output_root / "sealed" / "PILOT_ITT_ROWS.json", rows, mode=0o600
        )
        write_json(output_root / "STATIC_CHAIN_READINESS.json", readiness)
        write_json(
            output_root / "DESCRIPTIVE_EFFECT_SUMMARY.json", effects
        )
        write_json(
            output_root / "RUNTIME_IDENTITY_AUDIT.json", runtime_identity
        )
        result = {
            "authority": "NONE",
            "contract_content_digest": contract["content_digest"],
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "formal_meta_required_for_main": True,
            "gates": gates,
            "main_eligibility": False,
            "meta_activation": "NONE",
            "resource_audit": resource,
            "runtime_identity_audit": runtime_identity,
            "source_snapshot_digest": sha256_digest(after),
            "verdict": "CHAIN_PASS" if passed else "NOT_READY",
        }
        write_json(output_root / "STATIC_DIAGNOSTIC_RESULT.json", result)
        return 0 if passed else 2
    except Exception as error:
        try:
            upstream.close()
        except Exception:
            pass
        closure = getattr(error, "closure", None)
        write_json(
            output_root / "STATIC_DIAGNOSTIC_FAILURE.json",
            {
                "authority": "NONE",
                "broker_failure_closure_digest": (
                    closure.closure_digest if closure is not None else None
                ),
                "error_type": type(error).__name__,
                "evidence_class": "DEVELOPMENT_ONLY",
                "failure_class": (
                    closure.failure_class if closure is not None else None
                ),
                "formal_acceptance": False,
                "main_eligibility": False,
                "reason": (
                    "BROKER_PROCESS_FAILURE/COMMON_NO_EXECUTION"
                    if closure is not None
                    else "INTERNAL_STATIC_DIAGNOSTIC_FAILURE"
                ),
                "verdict": "NOT_READY",
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
