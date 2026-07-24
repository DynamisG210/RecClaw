from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scipy.sparse import dok_matrix

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    default_experiment_contract,
)
from recclaw_core.experiments.helix_abc_v1.pilot_training import (
    classify_training_termination,
    pilot_training_profile,
)
from recclaw_core.experiments.helix_abc_v1.m6e_conformance import (
    require_m6e_conformance_packet,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (
    PilotStoreContractV1,
    RealPilotOrchestratorV1,
    pilot_budget,
)
from recclaw_core.experiments.helix_abc_v1.state_store import (
    CloseRoundCommand,
    OpenRoundCommand,
    SingleWriterExperimentStoreV1,
)
from recclaw_core.experiments.helix_abc_v1.store_audit import (
    StoreIntegrityReportV1,
    experiment_store_audit_port,
)
from recclaw_core.experiments.helix_abc_v1.training_filesystem import (
    build_training_filesystem_capability,
    filesystem_confinement_audit,
    filesystem_mount_audit,
    protected_side_effect_manifest,
    side_effect_audit,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (
    TrainingRuntimeReleaseV2,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    historical_training_runtime_release_v1,
    training_runtime_release,
)
from recclaw_core.experiments.helix_abc_v1.training_state_store import (
    TrainingSingleWriterExperimentStoreV1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
V4_RUNTIME_DIGEST = (
    "9401d9c5f5f096057d183cf0e4dc4f2f76bbc85030099d17646ecdad5e8ad375"
)


def _initialized_store(
    store_type: type[SingleWriterExperimentStoreV1],
    root: Path,
) -> SingleWriterExperimentStoreV1:
    store = store_type(root / "state.sqlite3", root / "artifacts")
    store.initialize_experiment(default_experiment_contract())
    return store


def _filesystem_capability(root: Path, arm: str = "a"):
    project = root / "project"
    recbole = root / "recbole"
    dataset = root / "dataset"
    private = root / f"arm-{arm}"
    for path in (project, recbole, dataset, private):
        path.mkdir(parents=True, exist_ok=True)
    result = private / "runs" / "run-1"
    return build_training_filesystem_capability(
        instance_private_root=private,
        result_root=result,
        checkpoint_root=result / "checkpoints",
        project_root=project,
        recbole_root=recbole,
        dataset_root=dataset,
    )


class M6EEnvironmentClosureTest(unittest.TestCase):
    def test_base_store_satisfies_audit_port(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = _initialized_store(
                SingleWriterExperimentStoreV1, Path(raw) / "base"
            )
            try:
                report = experiment_store_audit_port(store).audit_store()
                self.assertIsInstance(report, StoreIntegrityReportV1)
                self.assertTrue(report.passed)
            finally:
                store.close()

    def test_training_store_satisfies_audit_port(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = _initialized_store(
                TrainingSingleWriterExperimentStoreV1, Path(raw) / "training"
            )
            try:
                self.assertTrue(experiment_store_audit_port(store).audit_store().passed)
            finally:
                store.close()

    def test_base_and_training_store_use_same_audit_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base = _initialized_store(SingleWriterExperimentStoreV1, root / "base")
            training = _initialized_store(
                TrainingSingleWriterExperimentStoreV1, root / "training"
            )
            try:
                base_report = experiment_store_audit_port(base).audit_store()
                training_report = experiment_store_audit_port(training).audit_store()
                semantic_fields = (
                    "artifact_index_check",
                    "execution_claim_uniqueness_check",
                    "feedback_uniqueness_check",
                    "foreign_key_violation_count",
                    "required_table_set",
                    "round_uniqueness_check",
                    "sqlite_integrity",
                    "triplet_barrier_check",
                )
                self.assertEqual(
                    {name: getattr(base_report, name) for name in semantic_fields},
                    {name: getattr(training_report, name) for name in semantic_fields},
                )
            finally:
                base.close()
                training.close()

    def test_missing_audit_capability_fails_before_broker(self) -> None:
        broker = SimpleNamespace(call_count=0)
        with self.assertRaises(TypeError):
            experiment_store_audit_port(object())
        self.assertEqual(broker.call_count, 0)

    def test_pilot_broker_is_blocked_without_pass_conformance_packet(self) -> None:
        broker = SimpleNamespace(call_count=0)
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(RuntimeError):
                require_m6e_conformance_packet(Path(raw))
        self.assertEqual(broker.call_count, 0)

    def test_altered_audit_report_payload_is_rejected(self) -> None:
        payload = {
            "artifact_index_check": True,
            "execution_claim_uniqueness_check": True,
            "experiment_identity_digest": "1" * 64,
            "feedback_uniqueness_check": True,
            "foreign_key_violation_count": 0,
            "migration_digest": "2" * 64,
            "required_table_set": [],
            "round_uniqueness_check": True,
            "schema_version": 2,
            "sqlite_integrity": "ok",
            "triplet_barrier_check": True,
        }
        report = StoreIntegrityReportV1.create(payload)
        with self.assertRaises(ValueError):
            StoreIntegrityReportV1.create(
                {**payload, "report_digest": report.report_digest}
            )

    def test_full_pilot_audit_runs_on_fixed_training_store(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = TrainingSingleWriterExperimentStoreV1(
                root / "state.sqlite3", root / "artifacts"
            )
            contract = PilotStoreContractV1.create()
            arm_ids = store.initialize_experiment(contract)
            genesis = sha256_digest(
                {
                    "experiment_contract_digest": contract.identity_digest,
                    "state": "GENESIS",
                }
            )
            try:
                states = {arm: genesis for arm in (ArmCode.A, ArmCode.B, ArmCode.C)}
                for round_index in range(1, 4):
                    for arm in (ArmCode.A, ArmCode.B, ArmCode.C):
                        opened = store.open_round(
                            OpenRoundCommand(
                                experiment_id=contract.experiment_id,
                                arm_instance_id=arm_ids[arm],
                                arm_code=arm,
                                search_seed=contract.search_seeds[0],
                                round_index=round_index,
                                budget_snapshot=pilot_budget(),
                                controller_state_before_digest=states[arm],
                                idempotency_key=(
                                    f"m6e-audit:{round_index}:{arm.value}:open"
                                ),
                            )
                        )
                        states[arm] = sha256_digest(
                            {"arm": arm.value, "round_index": round_index}
                        )
                        store.close_round(
                            CloseRoundCommand(
                                round_id=str(opened["round_id"]),
                                terminal_class="COMPLETED",
                                feedback_payload={"run_status": "FIXED_REHEARSAL"},
                                controller_state_after_digest=states[arm],
                                resource_debits=(),
                                idempotency_key=(
                                    f"m6e-audit:{round_index}:{arm.value}:close"
                                ),
                            )
                        )
                orchestrator = RealPilotOrchestratorV1.__new__(
                    RealPilotOrchestratorV1
                )
                orchestrator.store = store
                orchestrator.store_audit_port = experiment_store_audit_port(store)
                orchestrator.guard_ledger = SimpleNamespace(count=lambda: 0)
                orchestrator.initial_research_identity = "5" * 64
                orchestrator.broker = SimpleNamespace(
                    research_controllers={
                        arm: SimpleNamespace(policy=SimpleNamespace(version=1))
                        for arm in (ArmCode.B, ArmCode.C)
                    }
                )
                audit = RealPilotOrchestratorV1.pilot_audit(orchestrator)
                self.assertTrue(audit["barriers_closed"])
                self.assertEqual(audit["round_count"], 9)
                self.assertEqual(audit["feedback_count"], 9)
                self.assertTrue(audit["state_store_integrity"]["artifact_index_check"])
            finally:
                store.close()

    def test_finished_claim_does_not_imply_successful_training(self) -> None:
        lifecycle = {"claim_state": "FINISHED"}
        outcome = classify_training_termination(
            return_code=1,
            timed_out=False,
            worker_status="RUNTIME_FAILURE",
        )
        self.assertEqual(lifecycle["claim_state"], "FINISHED")
        self.assertEqual(outcome, ("RUNTIME_FAILURE", "CRASH_OR_RUNTIME_FAILURE"))
        self.assertNotEqual(outcome[0], "SUCCESS")

    def test_lightgcn_sparse_graph_build_under_frozen_release(self) -> None:
        release = training_runtime_release()
        self.assertEqual(
            release.backend_identity["python_package_versions"]["scipy"],
            "1.12.0",
        )
        matrix = dok_matrix((2, 2), dtype=float)
        self.assertTrue(hasattr(matrix, "_update"))
        matrix._update({(0, 1): 1.0})
        self.assertEqual(float(matrix[0, 1]), 1.0)

    def test_bpr_anchor_training_canary(self) -> None:
        self._assert_anchor_and_model("BPR_MF", "BPR")

    def test_lightgcn_anchor_training_canary(self) -> None:
        self._assert_anchor_and_model("LIGHTGCN", "LightGCN")

    def test_non_anchor_training_canary(self) -> None:
        self._assert_anchor_and_model("NGCF", "NGCF")

    def test_all_allowed_handler_families_have_conformance_canary(self) -> None:
        profile = pilot_training_profile()
        models = {str(row["model"]) for row in profile["supported_mappings"]}
        fixtures = {
            str(row["anchor_name"])
            for row in json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]
        }
        self.assertEqual(models, {"BPR", "LightGCN", "NGCF", "SGL"})
        self.assertTrue({"BPR_MF", "LIGHTGCN", "NGCF", "SGL"}.issubset(fixtures))

    def test_dependency_substitution_changes_release_digest(self) -> None:
        release = training_runtime_release()
        changed = release.to_dict()
        changed["backend_identity"]["python_package_versions"]["scipy"] = "1.13.1"
        self.assertNotEqual(TrainingRuntimeReleaseV2(changed).digest, release.digest)

    def test_old_v4_runtime_release_remains_immutable(self) -> None:
        self.assertEqual(
            historical_training_runtime_release_v1().digest,
            V4_RUNTIME_DIGEST,
        )

    def test_worker_cwd_is_instance_private(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            capability = _filesystem_capability(Path(raw))
            self.assertTrue(
                Path(capability.run_working_directory).is_relative_to(
                    Path(capability.instance_private_root)
                )
            )

    def test_default_recbole_log_is_private(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            capability = _filesystem_capability(Path(raw))
            self.assertEqual(
                Path(capability.log_root),
                Path(capability.run_working_directory) / "log",
            )
            self.assertTrue(
                Path(capability.log_root).is_relative_to(Path(capability.result_root))
            )

    def test_home_tmp_and_cache_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            capability = _filesystem_capability(Path(raw))
            result = Path(capability.result_root)
            for value in capability.environment.values():
                self.assertTrue(Path(value).is_relative_to(result))

    def test_gpu_and_proc_runtime_access_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            capability = _filesystem_capability(Path(raw))
            self.assertEqual(capability.device_access_mounts, ("/dev/dxg",))
            self.assertEqual(capability.runtime_control_mounts, ("/proc/self",))

    def test_project_log_root_receives_no_new_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            capability = _filesystem_capability(root)
            protected = {"project": root / "project"}
            before = protected_side_effect_manifest(protected)
            private_log = Path(capability.log_root)
            private_log.mkdir(parents=True)
            (private_log / "recbole.log").write_text("private", encoding="utf-8")
            after = protected_side_effect_manifest(protected)
            self.assertEqual(side_effect_audit(before, after)["status"], "PASS")

    def test_project_recbole_dataset_trees_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            capability = _filesystem_capability(root)
            protected = {
                "dataset": root / "dataset",
                "project": root / "project",
                "recbole": root / "recbole",
            }
            for name, path in protected.items():
                (path / f"{name}.txt").write_text(name, encoding="utf-8")
            before = protected_side_effect_manifest(protected)
            Path(capability.home_root).mkdir(parents=True)
            (Path(capability.home_root) / "state").write_text("private")
            after = protected_side_effect_manifest(protected)
            self.assertEqual(side_effect_audit(before, after)["status"], "PASS")

    def test_symlink_escape_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            private = root / "arm"
            outside = root / "outside"
            project = root / "project"
            recbole = root / "recbole"
            dataset = root / "dataset"
            for path in (private, outside, project, recbole, dataset):
                path.mkdir()
            (private / "escaped").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                build_training_filesystem_capability(
                    instance_private_root=private,
                    result_root=private / "escaped" / "run",
                    checkpoint_root=private / "escaped" / "run" / "checkpoints",
                    project_root=project,
                    recbole_root=recbole,
                    dataset_root=dataset,
                )

    def test_confinement_failure_fails_closed(self) -> None:
        before = {"manifest_digest": "1" * 64}
        after = {"manifest_digest": "2" * 64}
        audit = side_effect_audit(before, after)
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(
            audit["audit_digest"],
            sha256_digest(
                {
                    "after_manifest_digest": "2" * 64,
                    "before_manifest_digest": "1" * 64,
                    "status": "FAIL",
                }
            ),
        )

    def test_mount_audit_rejects_writable_wsl_submount(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = root / "run"
            result.mkdir()
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                "\n".join(
                    (
                        "1 0 0:1 / / ro - ext4 /dev/sdd rw",
                        f"2 1 0:1 / {result} rw - ext4 /dev/sdd rw",
                        "3 1 0:2 / /tmp rw - tmpfs tmpfs rw",
                        "4 1 0:3 / /var/tmp rw - tmpfs tmpfs rw",
                        "5 1 0:4 / /dev/shm rw - tmpfs tmpfs rw",
                        "6 1 0:5 / /mnt/c rw - 9p drvfs rw",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            audit = filesystem_mount_audit(
                (result, Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")),
                mountinfo_path=mountinfo,
            )
            self.assertEqual(audit["status"], "FAIL")
            self.assertEqual(
                audit["unexpected_writable_mount_targets"], ["/mnt/c"]
            )

    def test_combined_confinement_binds_mount_projection(self) -> None:
        shared = side_effect_audit(
            {"manifest_digest": "1" * 64},
            {"manifest_digest": "1" * 64},
        )
        mount_payload = {
            "allowed_writable_mount_targets": ["/run"],
            "missing_writable_mount_targets": [],
            "mount_count": 1,
            "status": "PASS",
            "unexpected_writable_mount_targets": [],
            "writable_mount_targets": ["/run"],
        }
        mount = {
            **mount_payload,
            "audit_digest": sha256_digest(mount_payload),
        }
        audit = filesystem_confinement_audit(shared, mount)
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(audit["worker_mount_digest_valid"])
        altered = filesystem_confinement_audit(
            shared,
            {**mount, "writable_mount_targets": ["/run", "/mnt/c"]},
        )
        self.assertEqual(altered["status"], "FAIL")
        self.assertFalse(altered["worker_mount_digest_valid"])

    def test_three_opaque_instances_have_disjoint_writable_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            capabilities = [_filesystem_capability(root, arm) for arm in "abc"]
            writable = [
                {
                    item.result_root,
                    item.log_root,
                    item.checkpoint_root,
                    item.temp_root,
                    item.cache_root,
                    item.home_root,
                }
                for item in capabilities
            ]
            self.assertTrue(writable[0].isdisjoint(writable[1]))
            self.assertTrue(writable[0].isdisjoint(writable[2]))
            self.assertTrue(writable[1].isdisjoint(writable[2]))

    def _assert_anchor_and_model(self, anchor: str, model: str) -> None:
        fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]
        self.assertTrue(any(row["anchor_name"] == anchor for row in fixtures))
        self.assertIn(
            model,
            {str(row["model"]) for row in pilot_training_profile()["supported_mappings"]},
        )


if __name__ == "__main__":
    unittest.main()
