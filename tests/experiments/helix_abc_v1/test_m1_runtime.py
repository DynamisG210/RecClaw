from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.mechanism_space import compile_program  # noqa: E402
from recclaw_core.experiments.helix_abc_v1 import (  # noqa: E402
    ArmCode,
    CandidateExecutionBindingV2,
    ClaimExecutionCommand,
    CommonDecision,
    CommonExecutionGuardV1,
    CommonExecutionPermitV1,
    CommonPreExecutionDecisionV1,
    DeterministicMaterializerV1,
    DevelopmentRecSysProtocolV1,
    ExecutionStartReceiptV1,
    ExecutionTrustClassificationV1,
    FakeNonTrainingRunnerV1,
    GateStatus,
    MarkExecutionStartedCommand,
    OpenRoundCommand,
    PackageOwnedLauncherV1,
    RawResultEnvelopeV1,
    RegisterArtifactCommand,
    ResourceCeilingsV1,
    SingleWriterExperimentStoreV1,
    StartStatus,
    build_binding_v2,
    campaign_projection,
    classify_execution_trust,
    common_release_projection,
    common_release_projection_digest,
    coverage_manifest,
    default_experiment_contract,
    development_execution_gate,
    development_protocol,
    executable_profile,
    executable_profile_digest,
    instance_binding,
    register_materialization_artifacts,
    register_raw_result_envelope,
    runtime_release_contract,
    verify_binding_v2,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (  # noqa: E402
    InvariantViolation,
)


FIXTURES = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"


def fixture_program(name: str) -> dict[str, object]:
    document = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for fixture in document["fixtures"]:
        if fixture["anchor_name"] == name:
            return copy.deepcopy(fixture["program"])
    raise KeyError(name)


def budget(*, ordinary: int = 1, validation: int = 1) -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=0,
        total_output_tokens=0,
        total_billed_token_debit=0,
        total_proposal_count=1,
        wall_time_ms=1000,
        retry_debit=0,
        proposal_attempt_debit=1,
        ordinary_executions=ordinary,
        common_validation_count=validation,
        gpu_device_time_ms=0,
        gpu_cost_microunits=0,
    )


class M1RuntimeTest(unittest.TestCase):
    def _plan(self, name: str, *, round_budget: ResourceCeilingsV1 | None = None):
        program = fixture_program(name)
        compiled = compile_program(program)
        decision, eligible = CommonExecutionGuardV1().plan_check(
            program=program,
            caller_compile_report=compiled,
            protocol=development_protocol(),
            budget=round_budget or budget(),
        )
        return program, compiled, decision, eligible

    def _open_round(self, temp: Path):
        store = SingleWriterExperimentStoreV1(temp / "state.sqlite3", temp / "arm-a")
        contract = default_experiment_contract()
        arm_ids = store.initialize_experiment(contract)
        genesis = sha256_digest(
            {"experiment_contract_digest": contract.identity_digest, "state": "GENESIS"}
        )
        opened = store.open_round(
            OpenRoundCommand(
                experiment_id=contract.experiment_id,
                arm_instance_id=arm_ids[ArmCode.A],
                arm_code=ArmCode.A,
                search_seed=42,
                round_index=1,
                budget_snapshot=budget(),
                controller_state_before_digest=genesis,
                idempotency_key="m1-open-a-42-1",
            )
        )
        return store, arm_ids[ArmCode.A], opened

    def _prepared(self, temp: Path, name: str = "BPR_MF"):
        program, _compiled, plan, eligible = self._plan(name)
        self.assertEqual(plan.decision, CommonDecision.PASS.value)
        self.assertIsNotNone(eligible)
        store, arm_id, opened = self._open_round(temp)
        root = temp / "arm-a"
        report = DeterministicMaterializerV1().materialize(
            eligible, program=program, arm_runtime_root=root
        )
        trust = classify_execution_trust(report, arm_runtime_root=root)
        binding = build_binding_v2(
            eligible=eligible,
            report=report,
            trust=trust,
            opaque_arm_instance_id=arm_id,
            arm_private_root=root,
            round_id=opened["round_id"],
            search_seed=2026,
        )
        materialization_artifacts = register_materialization_artifacts(
            store, binding=binding, report=report
        )
        gate = development_execution_gate(
            binding=binding,
            eligible=eligible,
            report=report,
            trust=trust,
            task_authorization_ref=(
                "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md#M1"
            ),
        )
        pre, permit = CommonExecutionGuardV1().pre_execute(
            eligible=eligible,
            report=report,
            trust=trust,
            binding=binding,
            gate=gate,
        )
        self.assertEqual(pre.decision, CommonDecision.PASS.value)
        self.assertIsNotNone(permit)
        return {
            "binding": binding,
            "eligible": eligible,
            "gate": gate,
            "materialization_artifacts": materialization_artifacts,
            "opened": opened,
            "permit": permit,
            "plan": plan,
            "pre": pre,
            "program": program,
            "report": report,
            "root": root,
            "store": store,
            "trust": trust,
        }

    def test_release_projection_is_exactly_common_across_arm_instances(self) -> None:
        common = canonical_json_bytes(common_release_projection())
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bindings = [
                instance_binding(
                    opaque_arm_instance_id=f"opaque-{arm.lower()}",
                    arm_private_root=root / arm.lower(),
                )
                for arm in ("A", "B", "C")
            ]
        self.assertEqual(len({common_release_projection_digest() for _ in bindings}), 1)
        self.assertEqual(
            {item["common_release_projection_digest"] for item in bindings},
            {common_release_projection_digest()},
        )
        self.assertNotIn(b"opaque-", common)
        self.assertNotIn(b'"arm"', common)
        frozen = json.loads(
            (
                ROOT
                / "docs"
                / "research_line"
                / "m1"
                / "COMMON_EXECUTION_GUARD_RELEASE_PROJECTION_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            canonical_json_bytes(frozen),
            canonical_json_bytes(common_release_projection()),
        )

    def test_same_plan_input_is_byte_identical_for_all_three_arms(self) -> None:
        program = fixture_program("BPR_MF")
        compiled = compile_program(program)
        outputs = [
            CommonExecutionGuardV1().plan_check(
                program=program,
                caller_compile_report=compiled,
                protocol=development_protocol(),
                budget=budget(),
            )[0].to_dict()
            for _arm in ("A", "B", "C")
        ]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_common_guard_has_no_research_evidence_or_fusion_import(self) -> None:
        source = (
            SRC
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "common_execution_guard.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        joined = " ".join(sorted(imports)).lower()
        self.assertNotIn("research_line", joined)
        self.assertNotIn("evidence", joined)
        self.assertNotIn("fusion", joined)

    def test_profile_declares_all_scientific_floor_capabilities(self) -> None:
        evidence = executable_profile()["capability_evidence"]
        self.assertEqual(
            set(evidence),
            {
                "architecture_operator_template",
                "bpr_mf_anchor",
                "graph_propagation_aggregation",
                "lightgcn_anchor",
                "non_default_negative_sampling",
                "pairwise_ranking_objective",
                "regularization_geometry",
                "self_supervision_contrastive",
            },
        )
        self.assertEqual(coverage_manifest()["profile_id"], "BL_ICF_EXECUTABLE_PROFILE_V1")
        self.assertEqual(
            campaign_projection()["coverage_digest"], executable_profile_digest()
        )
        self.assertTrue(
            all("single_fault_negative" in item for item in evidence.values())
        )
        self.assertEqual(
            coverage_manifest()["supported_parameter_domains"]["domain"],
            "BL_ICF_PROVIDER_SCHEMA_VALID_VALUES",
        )
        frozen = json.loads(
            (
                ROOT
                / "docs"
                / "research_line"
                / "m1"
                / "BL_RUNTIME_COVERAGE_MANIFEST_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(frozen, coverage_manifest())

    def test_profile_positive_recipes_compile_materialize_and_smoke(self) -> None:
        expected_axes = {
            "BPR_MF": {"PAIRWISE_RANKING"},
            "LIGHTGCN": {"GRAPH_PROPAGATION_AGGREGATION"},
            "NGCF": {
                "GRAPH_PROPAGATION_AGGREGATION",
                "PACKAGE_ARCHITECTURE_OPERATOR",
            },
            "SGL": {"SELF_SUPERVISION_CONTRASTIVE"},
            "DIRECTAU": {"REGULARIZATION_GEOMETRY"},
            "ULTRAGCN": {
                "NON_DEFAULT_NEGATIVE_SAMPLING",
                "PACKAGE_ARCHITECTURE_OPERATOR",
            },
        }
        for name in expected_axes:
            with self.subTest(recipe=name), tempfile.TemporaryDirectory() as raw:
                prepared = self._prepared(Path(raw), name)
                store = prepared["store"]
                try:
                    store.claim_execution(
                        ClaimExecutionCommand(
                            round_id=prepared["binding"].round_id,
                            permit_digest=prepared["permit"].digest,
                            binding_digest=prepared["binding"].digest,
                            idempotency_key=f"m1-profile-claim-{name}",
                        )
                    )
                    receipt, raw_output, run_artifacts = PackageOwnedLauncherV1(
                        store
                    ).launch(
                        permit=prepared["permit"],
                        binding=prepared["binding"],
                        gate=prepared["gate"],
                        pre_execution=prepared["pre"],
                    )
                    closure, envelope = CommonExecutionGuardV1().close_result(
                        permit=prepared["permit"],
                        binding=prepared["binding"],
                        claim=store.get_execution_claim(
                            prepared["binding"].round_id
                        ),
                        receipt=receipt,
                        raw_output=raw_output,
                        artifact_closure=list(
                            prepared["materialization_artifacts"] + run_artifacts
                        ),
                    )
                    self.assertEqual(closure.decision, CommonDecision.PASS.value)
                    self.assertIsNotNone(envelope)
                    self.assertTrue(
                        expected_axes[name].issubset(
                            set(raw_output.mechanism_axes_exercised)
                        )
                    )
                    result_row = register_raw_result_envelope(store, envelope)
                    self.assertEqual(
                        result_row["artifact_type"], "RAW_RESULT_ENVELOPE_V1"
                    )
                finally:
                    store.close()

    def test_non_anchor_structural_mechanism_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw), "SGL")
            store = prepared["store"]
            try:
                claim = store.claim_execution(
                    ClaimExecutionCommand(
                        round_id=prepared["binding"].round_id,
                        permit_digest=prepared["permit"].digest,
                        binding_digest=prepared["binding"].digest,
                        idempotency_key="m1-claim-sgl",
                    )
                )
                launcher = PackageOwnedLauncherV1(store)
                receipt, raw_output, artifacts = launcher.launch(
                    permit=prepared["permit"],
                    binding=prepared["binding"],
                    gate=prepared["gate"],
                    pre_execution=prepared["pre"],
                )
                closure, envelope = CommonExecutionGuardV1().close_result(
                    permit=prepared["permit"],
                    binding=prepared["binding"],
                    claim=store.get_execution_claim(prepared["binding"].round_id),
                    receipt=receipt,
                    raw_output=raw_output,
                    artifact_closure=list(
                        prepared["materialization_artifacts"] + artifacts
                    ),
                )
                self.assertEqual(launcher.runner_launch_count, 1)
                self.assertEqual(claim["claim_state"], "CLAIMED")
                self.assertEqual(closure.decision, CommonDecision.PASS.value)
                self.assertIsNotNone(envelope)
                self.assertEqual(
                    register_raw_result_envelope(store, envelope)["artifact_type"],
                    "RAW_RESULT_ENVELOPE_V1",
                )
                self.assertIn(
                    "SELF_SUPERVISION_CONTRASTIVE",
                    raw_output.mechanism_axes_exercised,
                )
                self.assertEqual(raw_output.optimizer_steps, 0)
                self.assertFalse(raw_output.training_backend_started)
                self.assertEqual(raw_output.normalized_metrics, {})
            finally:
                store.close()

    def test_plan_check_recomputes_compile_and_rejects_substitution(self) -> None:
        program = fixture_program("BPR_MF")
        report = compile_program(program).to_dict()
        report["candidate_id"] = "bl1_substituted"
        decision, eligible = CommonExecutionGuardV1().plan_check(
            program=program,
            caller_compile_report=report,
            protocol=development_protocol(),
            budget=budget(),
        )
        self.assertEqual(decision.decision, CommonDecision.DENY.value)
        self.assertIn("BL_COMPILE_FAILED", decision.reason_codes)
        self.assertIsNone(eligible)

    def test_candidate_controlled_custom_program_fails_at_common_plan(self) -> None:
        from tests.test_bl_icf_mechanism_space import _custom_program

        program = _custom_program()
        compiled = compile_program(program)
        self.assertTrue(compiled.is_valid)
        decision, eligible = CommonExecutionGuardV1().plan_check(
            program=program,
            caller_compile_report=compiled,
            protocol=development_protocol(),
            budget=budget(),
        )
        self.assertEqual(decision.decision, CommonDecision.DENY.value)
        self.assertIn("CAPABILITY_UNSUPPORTED", decision.reason_codes)
        self.assertIsNone(eligible)

    def test_schema_protocol_capability_and_budget_single_faults_fail_closed(self) -> None:
        invalid = fixture_program("BPR_MF")
        del invalid["record_type"]
        invalid_report = compile_program(invalid)
        decision, _ = CommonExecutionGuardV1().plan_check(
            program=invalid,
            caller_compile_report=invalid_report,
            protocol=development_protocol(),
            budget=budget(),
        )
        self.assertIn("SCHEMA_INVALID", decision.reason_codes)

        unsupported = fixture_program("SIMPLEX")
        unsupported_report = compile_program(unsupported)
        decision, _ = CommonExecutionGuardV1().plan_check(
            program=unsupported,
            caller_compile_report=unsupported_report,
            protocol=development_protocol(),
            budget=budget(),
        )
        self.assertIn("CAPABILITY_UNSUPPORTED", decision.reason_codes)

        protocol_payload = development_protocol().to_dict()
        protocol_payload["metric"] = "RECALL@20"
        decision, _ = CommonExecutionGuardV1().plan_check(
            program=fixture_program("BPR_MF"),
            caller_compile_report=compile_program(fixture_program("BPR_MF")),
            protocol=DevelopmentRecSysProtocolV1(protocol_payload),
            budget=budget(),
        )
        self.assertIn("FROZEN_PROTOCOL_CONTRACT_MISMATCH", decision.reason_codes)

        _program, _compiled, decision, eligible = self._plan(
            "BPR_MF", round_budget=budget(ordinary=0)
        )
        self.assertIn("BUDGET_DENIED", decision.reason_codes)
        self.assertIsNone(eligible)

    def test_materialization_replays_byte_identically_across_private_roots(self) -> None:
        program, _compiled, _plan, eligible = self._plan("ULTRAGCN")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = DeterministicMaterializerV1().materialize(
                eligible, program=program, arm_runtime_root=root / "a"
            )
            second = DeterministicMaterializerV1().materialize(
                eligible, program=program, arm_runtime_root=root / "b"
            )
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertEqual(first.implementation_digest, second.implementation_digest)

    def test_materialized_byte_substitution_changes_identity_and_trust(self) -> None:
        program, _compiled, _plan, eligible = self._plan("BPR_MF")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "arm"
            report = DeterministicMaterializerV1().materialize(
                eligible, program=program, arm_runtime_root=root
            )
            config = next(
                row for row in report.files if row["path"].endswith("handler_config.json")
            )
            path = root.joinpath(*config["path"].split("/"))
            payload = json.loads(path.read_bytes())
            payload["template_id"] = "SUBSTITUTED"
            path.write_bytes(canonical_json_bytes(payload))
            trust = classify_execution_trust(report, arm_runtime_root=root)
            self.assertEqual(
                trust.classification, "CANDIDATE_CONTROLLED_EXECUTABLE"
            )
            self.assertIn("MATERIALIZED_FILE_DIGEST_MISMATCH", trust.reason_codes)

    def test_unmanifested_candidate_file_denies_package_owned_trust(self) -> None:
        program, _compiled, _plan, eligible = self._plan("BPR_MF")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "arm"
            report = DeterministicMaterializerV1().materialize(
                eligible, program=program, arm_runtime_root=root
            )
            candidate_root = (
                root / "recclaw_ext" / "generated" / str(eligible.candidate_id)
            )
            (candidate_root / "extra.json").write_text("{}", encoding="utf-8")
            trust = classify_execution_trust(report, arm_runtime_root=root)
            self.assertEqual(
                trust.classification, "CANDIDATE_CONTROLLED_EXECUTABLE"
            )
            self.assertIn("UNMANIFESTED_OR_MISSING_FILE", trust.reason_codes)

    def test_symlink_hardlink_and_root_widening_fail_closed(self) -> None:
        program, _compiled, _plan, eligible = self._plan("BPR_MF")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            actual = root / "actual"
            actual.mkdir()
            (root / "linked").symlink_to(actual, target_is_directory=True)
            with self.assertRaises(ValueError):
                DeterministicMaterializerV1().materialize(
                    eligible, program=program, arm_runtime_root=root / "linked"
                )

            report = DeterministicMaterializerV1().materialize(
                eligible, program=program, arm_runtime_root=actual
            )
            config = next(
                row for row in report.files if row["path"].endswith("handler_config.json")
            )
            path = actual.joinpath(*config["path"].split("/"))
            clone = path.with_name("hardlink.json")
            os.link(path, clone)
            trust = classify_execution_trust(report, arm_runtime_root=actual)
            self.assertIn("HARDLINK_DETECTED", trust.reason_codes)

    def test_binding_v2_recomputes_implementation_budget_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            try:
                payload = prepared["binding"].to_dict()
                payload["implementation_digest"] = "0" * 64
                substituted = CandidateExecutionBindingV2(payload)
                valid, reasons = verify_binding_v2(
                    substituted,
                    eligible=prepared["eligible"],
                    report=prepared["report"],
                    trust=prepared["trust"],
                )
                self.assertFalse(valid)
                self.assertIn(
                    "BINDING_FIELD_MISMATCH:implementation_digest", reasons
                )

                payload = prepared["binding"].to_dict()
                payload["arm_private_root"] = Path(raw).as_posix()
                widened = CandidateExecutionBindingV2(payload)
                valid, reasons = verify_binding_v2(
                    widened,
                    eligible=prepared["eligible"],
                    report=prepared["report"],
                    trust=prepared["trust"],
                )
                self.assertFalse(valid)
                self.assertIn("TRUST_CLASSIFICATION_SUBSTITUTION", reasons)
            finally:
                prepared["store"].close()

    def test_trust_classifier_rejects_self_label_and_gate_requires_scope(self) -> None:
        with self.assertRaises(ValueError):
            ExecutionTrustClassificationV1(
                {
                    "classification": "PACKAGE_OWNED_TYPED_TEMPLATE",
                    "classifier_digest": "0" * 64,
                    "implementation_digest": "0" * 64,
                    "materialization_digest": "0" * 64,
                    "reason_codes": [],
                    "self_label": "TRUST_ME",
                }
            )
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            try:
                denied = development_execution_gate(
                    binding=prepared["binding"],
                    eligible=prepared["eligible"],
                    report=prepared["report"],
                    trust=prepared["trust"],
                    task_authorization_ref="MISSING",
                )
                self.assertEqual(denied.decision, GateStatus.DENY.value)
            finally:
                prepared["store"].close()

    def test_runner_rechecks_gate_preexecute_binding_and_permit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            try:
                payload = prepared["gate"].to_dict()
                payload["decision"] = GateStatus.DENY.value
                denied_gate = type(prepared["gate"])(payload)
                with self.assertRaises(ValueError):
                    FakeNonTrainingRunnerV1().run(
                        permit=prepared["permit"],
                        binding=prepared["binding"],
                        gate=denied_gate,
                        pre_execution=prepared["pre"],
                    )
                with self.assertRaises(InvariantViolation):
                    prepared["store"].get_execution_claim(
                        prepared["binding"].round_id
                    )
            finally:
                prepared["store"].close()

    def test_execution_claim_is_single_use_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            store = prepared["store"]
            try:
                def claim(index: int):
                    return store.claim_execution(
                        ClaimExecutionCommand(
                            round_id=prepared["binding"].round_id,
                            permit_digest=prepared["permit"].digest,
                            binding_digest=prepared["binding"].digest,
                            idempotency_key=f"m1-concurrent-claim-{index}",
                        )
                    )

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(claim, index) for index in (1, 2)]
                outcomes = []
                for future in futures:
                    try:
                        future.result()
                        outcomes.append("PASS")
                    except InvariantViolation:
                        outcomes.append("DENY")
                self.assertEqual(sorted(outcomes), ["DENY", "PASS"])
            finally:
                store.close()

    def test_forged_start_receipt_cannot_activate_committed_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            store = prepared["store"]
            try:
                claim = store.claim_execution(
                    ClaimExecutionCommand(
                        round_id=prepared["binding"].round_id,
                        permit_digest=prepared["permit"].digest,
                        binding_digest=prepared["binding"].digest,
                        idempotency_key="m1-claim-forged-receipt",
                    )
                )
                artifact = store.register_artifact(
                    RegisterArtifactCommand(
                        round_id=prepared["binding"].round_id,
                        artifact_type="EXECUTION_START_RECEIPT_V1",
                        relative_path=(
                            f"artifacts/{prepared['binding'].run_id}/forged_receipt.json"
                        ),
                        producer="adversarial-test",
                        idempotency_key="m1-forged-receipt-artifact",
                    ),
                    b"{}",
                )
                with self.assertRaises(InvariantViolation):
                    store.mark_execution_started(
                        MarkExecutionStartedCommand(
                            round_id=prepared["binding"].round_id,
                            claim_id=claim["claim_id"],
                            receipt_artifact_id=artifact["artifact_id"],
                            idempotency_key="m1-forged-start",
                        )
                    )
                self.assertEqual(
                    store.get_execution_claim(prepared["binding"].round_id)[
                        "claim_state"
                    ],
                    "CLAIMED",
                )
            finally:
                store.close()

    def test_launcher_cannot_reuse_finished_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            store = prepared["store"]
            try:
                store.claim_execution(
                    ClaimExecutionCommand(
                        round_id=prepared["binding"].round_id,
                        permit_digest=prepared["permit"].digest,
                        binding_digest=prepared["binding"].digest,
                        idempotency_key="m1-claim-once",
                    )
                )
                launcher = PackageOwnedLauncherV1(store)
                launcher.launch(
                    permit=prepared["permit"],
                    binding=prepared["binding"],
                    gate=prepared["gate"],
                    pre_execution=prepared["pre"],
                )
                with self.assertRaises(ValueError):
                    launcher.launch(
                        permit=prepared["permit"],
                        binding=prepared["binding"],
                        gate=prepared["gate"],
                        pre_execution=prepared["pre"],
                    )
                self.assertEqual(launcher.runner_launch_count, 1)
                self.assertEqual(
                    store.get_execution_claim(prepared["binding"].round_id)[
                        "claim_state"
                    ],
                    "FINISHED",
                )
            finally:
                store.close()

    def test_close_result_rejects_tampered_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            prepared = self._prepared(Path(raw))
            store = prepared["store"]
            try:
                store.claim_execution(
                    ClaimExecutionCommand(
                        round_id=prepared["binding"].round_id,
                        permit_digest=prepared["permit"].digest,
                        binding_digest=prepared["binding"].digest,
                        idempotency_key="m1-claim-tamper",
                    )
                )
                receipt, raw_output, artifacts = PackageOwnedLauncherV1(store).launch(
                    permit=prepared["permit"],
                    binding=prepared["binding"],
                    gate=prepared["gate"],
                    pre_execution=prepared["pre"],
                )
                payload = receipt.to_dict()
                payload["binding_digest"] = "0" * 64
                tampered = ExecutionStartReceiptV1(payload)
                closure, envelope = CommonExecutionGuardV1().close_result(
                    permit=prepared["permit"],
                    binding=prepared["binding"],
                    claim=store.get_execution_claim(prepared["binding"].round_id),
                    receipt=tampered,
                    raw_output=raw_output,
                    artifact_closure=list(
                        prepared["materialization_artifacts"] + artifacts
                    ),
                )
                self.assertEqual(closure.decision, CommonDecision.DENY.value)
                self.assertIsNone(envelope)
            finally:
                store.close()

    def test_raw_result_is_immutable_and_forbids_evidence_search_fields(self) -> None:
        payload = {
            "artifact_closure": [],
            "binding_digest": "0" * 64,
            "candidate_id": "candidate",
            "common_result_closure_digest": "1" * 64,
            "evaluation_purpose": "NON_OUTCOME_BEARING_INTERFACE_SMOKE",
            "exit_status": "SUCCESS",
            "metric_source": "NONE_NON_OUTCOME_BEARING_SMOKE",
            "normalized_metrics": {},
            "ordinary_execution_start_index": 1,
            "partition_role": "NOT_APPLICABLE_NO_DATA_READ",
            "raw_output_digest": "2" * 64,
            "round_id": "round",
            "run_id": "run",
            "seed": 2026,
        }
        envelope = RawResultEnvelopeV1(payload)
        before = canonical_json_bytes(envelope.to_dict())
        with self.assertRaises(AttributeError):
            envelope.seed = 7
        self.assertEqual(before, canonical_json_bytes(envelope.to_dict()))
        for field in ("claim_ceiling", "evidence_use", "router_score", "meta_update"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    RawResultEnvelopeV1({**payload, field: "FORBIDDEN"})

    def test_runtime_handler_has_no_ambient_io_or_training_imports(self) -> None:
        path = (
            SRC
            / "recclaw_core"
            / "experiments"
            / "helix_abc_v1"
            / "runtime_handlers.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue(
            imports.isdisjoint(
                {"os", "pathlib", "socket", "subprocess", "requests", "torch", "recbole"}
            )
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("open", called)

    def test_package_import_and_smoke_work_outside_repository_cwd(self) -> None:
        program_path = FIXTURES.as_posix()
        code = (
            "import json;"
            "from recclaw_core.mechanism_space import compile_program;"
            "from recclaw_core.experiments.helix_abc_v1 import "
            "CommonExecutionGuardV1,development_protocol,ResourceCeilingsV1;"
            f"d=json.load(open({program_path!r}));"
            "p=d['fixtures'][0]['program'];r=compile_program(p);"
            "b=ResourceCeilingsV1(0,0,0,1,1000,0,1,1,1,0,0);"
            "q,e=CommonExecutionGuardV1().plan_check("
            "program=p,caller_compile_report=r,protocol=development_protocol(),budget=b);"
            "assert q.decision=='COMMON_PASS' and e is not None"
        )
        env = {"PYTHONPATH": str(SRC), "PATH": os.environ["PATH"]}
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_release_explicitly_excludes_training(self) -> None:
        contract = runtime_release_contract()
        self.assertEqual(contract["execution_config_contract"]["optimizer_steps"], 0)
        self.assertEqual(
            contract["execution_config_contract"]["training_backend"], "FORBIDDEN"
        )
        self.assertEqual(
            contract["recbole_usage"], "IDENTITY_BOUND_NOT_IMPORTED_BY_M1_FAKE_RUNNER"
        )

    def test_runtime_records_reject_evidence_and_search_authority_fields(self) -> None:
        with self.assertRaises(ValueError):
            CommonExecutionPermitV1(
                {
                    "backend_digest": "0" * 64,
                    "binding_digest": "1" * 64,
                    "budget_digest": "2" * 64,
                    "candidate_id": "candidate",
                    "gate_decision_digest": "3" * 64,
                    "ordinary_launch_attempt_ordinal": 1,
                    "pre_execution_decision_digest": "4" * 64,
                    "round_id": "round",
                    "run_id": "run",
                    "runner_abi": "runner",
                    "evidence_use": "FORBIDDEN",
                }
            )


if __name__ == "__main__":
    unittest.main()
