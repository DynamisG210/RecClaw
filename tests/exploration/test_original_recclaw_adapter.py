from __future__ import annotations

import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from recclaw_core.exploration.original_recclaw_adapter import (
    build_postcheck_envelope,
    build_next_round_feedback_event,
    build_precheck_envelope,
    evaluate_postcheck_fail_open,
    evaluate_precheck_fail_open,
)
from recclaw_core.exploration.original_recclaw_guard_hook import (
    OriginalRecClawGuardHook,
)


def protocol() -> dict[str, object]:
    return {
        "protocol_id": "ORIGINAL-RECCLAW-ML1M-RANDOM-FULL-V1",
        "profile_family": "OFFLINE_TOPN",
        "dataset": "ml-1m",
        "dataset_snapshot": "sha256:532f0c05827ee06d1b5f81de9686bb8cf4288e67260a55853bc45d092c82f9dd",
        "split": {
            "strategy": "random_user_holdout",
            "ratio": [0.8, 0.1, 0.1],
            "group_by": "user",
            "order": "random",
        },
        "training_sampling": {
            "mode": "uniform_negative",
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False,
        },
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {
            "seen_items": "exclude",
            "repeat_items": "exclude_if_seen",
        },
        "metric": {"name": "ndcg", "cutoff": 10, "aggregation": "mean_by_user"},
        "training_procedure": {
            "train_batch_size": 2048,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "max_epochs": 100,
            "evaluation_step": 1,
            "early_stopping_patience": 10,
            "reproducibility": True,
        },
    }


def contract() -> dict[str, object]:
    return {
        "claim": {
            "claim_id": "ORIGINAL-RECCLAW-CANDIDATE-VS-LIGHTGCN-V1",
            "protocol_id": "ORIGINAL-RECCLAW-ML1M-RANDOM-FULL-V1",
            "claim_kind": "LOCAL_IMPROVEMENT",
            "target_model": "OriginalRecClawCandidate",
            "comparator": "LightGCN",
            "metric": "ndcg",
            "required_seed_count": 3,
            "scope": {"dataset": "ml-1m", "development_ab": True},
        },
        "protocol": protocol(),
        "protocol_implementation": {
            "reported_metrics": [
                "ndcg@10",
                "recall@10",
                "mrr@10",
                "hit@10",
                "precision@10",
            ],
            "evaluator_mode": {"valid": "full", "test": "full"},
            "evaluation_batch_size": 65536,
            "worker_count": 8,
            "repeatable": False,
        },
        "current_evidence": {
            "snapshot_id": "ORIGINAL-RECCLAW-EVIDENCE-EMPTY-V1",
            "claim_id": "ORIGINAL-RECCLAW-CANDIDATE-VS-LIGHTGCN-V1",
            "protocol_id": "ORIGINAL-RECCLAW-ML1M-RANDOM-FULL-V1",
            "observation_ids": [],
        },
        "baseline": {
            "model": "LightGCN",
            "reference_metric_value": 0.2671,
        },
        "comparison_policy": {"min_improvement_delta": 0.00001},
        "metric_projection": {"ndcg": "ndcg@10", "recall": "recall@10"},
        "seed_policy": {
            "default_training_seed": "2026",
            "validation_seeds": ["2026", "2027", "2028"],
        },
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def artifact(run_id: str) -> dict[str, str]:
    return {
        "sha256": hashlib.sha256(run_id.encode("utf-8")).hexdigest(),
        "source_path": f"historical/{run_id}.json",
    }


def write_recbole_log(
    path: Path,
    *,
    candidate_id: str,
    seed: int,
    command_model: str = "LiveModel",
    overrides: dict[str, object] | None = None,
) -> None:
    config: dict[str, object] = {
        "seed": seed,
        "reproducibility": True,
        "data_path": "/home/tingrangan/projects/RecBole/dataset/ml-1m",
        "epochs": 100,
        "train_batch_size": 2048,
        "learner": "adam",
        "learning_rate": 0.001,
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False,
        },
        "eval_step": 1,
        "stopping_step": 10,
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "RO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        },
        "repeatable": False,
        "metrics": ["Recall", "NDCG", "MRR", "Hit", "Precision"],
        "topk": [10],
        "valid_metric": "NDCG@10",
        "eval_batch_size": 65536,
        "worker": 8,
        "candidate_id": candidate_id,
        "valid_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": "none",
        },
        "test_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": "none",
        },
    }
    config.update(overrides or {})
    lines = [
        "19 Jul 00:00    INFO  "
        + repr(
            [
                "run_candidate.py",
                f"--model={command_model}",
                "--dataset=ml-1m",
            ]
        ),
        "General Hyper Parameters:",
    ]
    lines.extend(f"{key} = {value!r}" for key, value in config.items())
    lines.extend(["", "19 Jul 00:01    INFO  ml-1m", "The number of users: 6041"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_packet(
    *,
    candidate_id: str = "cand_lightgcn_example",
    seed_ids: list[str] | None = None,
    metric: float | None = 0.289,
    run_status: str = "SUCCESS",
) -> dict[str, object]:
    chosen_seeds = seed_ids or ["2026"]
    metrics: dict[str, float] = {}
    if metric is not None:
        metrics = {"ndcg@10": metric, "recall@10": 0.18}
    return {
        "candidate_id": candidate_id,
        "round_id": 165,
        "original_decision": "keep",
        "original_interpretation": {
            "reason": "candidate improves baseline and history best"
        },
        "planned_seed_ids": chosen_seeds,
        "proposal": {
            "candidate_id": candidate_id,
            "parent_candidate_id": "cand_lightgcn_parent",
            "scaffold_id": "recclaw_ext.models.example:Example",
            "minimal_experiment": "run the candidate under the unchanged fixed protocol",
        },
        "protocol_projection_evidence": {
            "explicit_fixed_dataset": True,
            "explicit_full_sort_evaluation": True,
            "explicit_unchanged_split": True,
            "explicit_unchanged_training_sampling": True,
            "recbole_core_change_required_false": True,
            "parent_bound": True,
            "scaffold_bound": True,
            "training_seed_bound": True,
        },
        "runner_observation": {
            "observation_id": f"OBS-{candidate_id}",
            "observation_kind": "METRIC_EVALUATION",
            "run_status": run_status,
            "artifact_identity_status": "EXACT",
            "observed_protocol": protocol(),
            "metrics": metrics,
            "seed_runs": [
                {
                    "seed_id": seed_id,
                    "run_id": f"RUN-{candidate_id}-{seed_id}",
                    "primary_artifact": artifact(f"RUN-{candidate_id}-{seed_id}"),
                }
                for seed_id in chosen_seeds
            ],
            "blocker": None,
        },
    }


class OriginalRecClawAdapterTests(unittest.TestCase):
    def test_exact_single_seed_maps_runner_metric_and_produces_valid_signal(self) -> None:
        packet = candidate_packet()
        before = copy.deepcopy(packet)
        pre = evaluate_precheck_fail_open(contract=contract(), candidate_packet=packet)
        self.assertTrue(pre["guard_succeeded"])
        self.assertEqual(
            "PROTOCOL_COMPATIBLE_FOR_DEVELOPMENT",
            pre["feedback"]["recommendation"],
        )
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre["envelope"],
        )
        self.assertTrue(post["guard_succeeded"])
        self.assertEqual(
            {"ndcg": 0.289, "recall": 0.18},
            post["envelope"]["guard_request"]["observation"]["metrics"],
        )
        self.assertEqual(
            "VALID_IMPROVEMENT", post["feedback"]["outcome_classification"]
        )
        self.assertTrue(post["feedback"]["may_update_primary_search_memory"])
        self.assertEqual(before, packet)

    def test_multiseed_projection_reaches_only_development_multiseed_ceiling(self) -> None:
        packet = candidate_packet(seed_ids=["2026", "2027", "2028"])
        pre = build_precheck_envelope(contract=contract(), candidate_packet=packet)
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre,
        )
        observation = post["envelope"]["guard_request"]["observation"]
        self.assertEqual(3, observation["seed_count"])
        self.assertEqual(
            "SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL",
            post["feedback"]["claim_ceiling"],
        )
        self.assertFalse(post["feedback"]["may_update_claim_state"])

    def test_runtime_crash_is_not_misclassified_as_mechanism_negative(self) -> None:
        packet = candidate_packet(metric=None, run_status="RUNTIME_BLOCKED")
        packet["original_decision"] = "crash"
        pre = build_precheck_envelope(contract=contract(), candidate_packet=packet)
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre,
        )
        self.assertEqual(
            "RUNTIME_BLOCKER", post["feedback"]["outcome_classification"]
        )
        self.assertFalse(post["feedback"]["may_update_primary_search_memory"])
        self.assertTrue(post["feedback"]["may_update_diagnostic_memory"])

    def test_observed_sampled_evaluation_routes_to_protocol_branch(self) -> None:
        packet = candidate_packet()
        flipped = protocol()
        flipped["protocol_id"] = "ORIGINAL-RECCLAW-ML1M-RANDOM-SAMPLED-V1"
        flipped["evaluation_candidate_universe"] = {
            "mode": "sampled",
            "sample_size": 100,
        }
        packet["runner_observation"]["observed_protocol"] = flipped
        pre = build_precheck_envelope(contract=contract(), candidate_packet=packet)
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre,
        )
        self.assertEqual(
            "PROTOCOL_MISMATCH", post["feedback"]["outcome_classification"]
        )
        self.assertTrue(post["feedback"]["protocol_branch_required"])
        self.assertFalse(post["feedback"]["may_update_primary_search_memory"])

    def test_missing_proposal_seed_binding_requests_revision_without_blocking(self) -> None:
        packet = candidate_packet()
        del packet["planned_seed_ids"]
        pre = evaluate_precheck_fail_open(contract=contract(), candidate_packet=packet)
        self.assertEqual("REVISE_BEFORE_RUN", pre["feedback"]["recommendation"])
        self.assertIn("planned_seed_ids", pre["feedback"]["missing_material_fields"])
        self.assertFalse(pre["feedback"]["blocks_original_execution"])

    def test_guard_exception_fails_open_and_preserves_original_decision(self) -> None:
        def broken_guard(**_: object) -> dict[str, object]:
            raise TimeoutError("simulated timeout")

        packet = candidate_packet()
        pre = evaluate_precheck_fail_open(
            contract=contract(), candidate_packet=packet, evaluator=broken_guard
        )
        self.assertFalse(pre["guard_succeeded"])
        self.assertEqual("TimeoutError", pre["failure"]["kind"])
        self.assertEqual(
            "PRESERVE_ORIGINAL_DECISION_GUARD_UNAVAILABLE",
            pre["feedback"]["recommendation"],
        )
        self.assertFalse(pre["feedback"]["blocks_original_execution"])

    def test_bad_artifact_identity_fails_open_instead_of_crashing_caller(self) -> None:
        packet = candidate_packet()
        packet["runner_observation"]["seed_runs"][0]["primary_artifact"][
            "sha256"
        ] = "not-a-digest"
        pre = build_precheck_envelope(contract=contract(), candidate_packet=packet)
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre,
        )
        self.assertFalse(post["guard_succeeded"])
        self.assertEqual(
            "INSUFFICIENT_EVIDENCE", post["feedback"]["outcome_classification"]
        )
        self.assertTrue(post["original_observation_preserved"])

    def test_postcheck_projection_is_deterministic(self) -> None:
        packet = candidate_packet()
        pre = build_precheck_envelope(contract=contract(), candidate_packet=packet)
        first = build_postcheck_envelope(
            contract=contract(), candidate_packet=packet, precheck_envelope=pre
        )
        second = build_postcheck_envelope(
            contract=contract(), candidate_packet=packet, precheck_envelope=pre
        )
        self.assertEqual(first, second)

    def test_next_round_event_separates_primary_and_diagnostic_memory(self) -> None:
        packet = candidate_packet()
        pre = evaluate_precheck_fail_open(contract=contract(), candidate_packet=packet)
        post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=packet,
            precheck_envelope=pre["envelope"],
        )
        event = build_next_round_feedback_event(
            precheck_run=pre, postcheck_run=post, original_decision="keep"
        )
        self.assertEqual("PRIMARY_RESEARCH_FEEDBACK", event["memory_channel"])
        self.assertEqual(
            "INGEST_BOUNDED_RESEARCH_FEEDBACK", event["next_iteration_effect"]
        )
        self.assertTrue(event["may_update_primary_search_memory"])
        self.assertFalse(event["may_authorize_execution"])

        blocked = candidate_packet(metric=None, run_status="RUNTIME_BLOCKED")
        blocked["original_decision"] = "crash"
        blocked_pre = evaluate_precheck_fail_open(
            contract=contract(), candidate_packet=blocked
        )
        blocked_post = evaluate_postcheck_fail_open(
            contract=contract(),
            candidate_packet=blocked,
            precheck_envelope=blocked_pre["envelope"],
        )
        blocked_event = build_next_round_feedback_event(
            precheck_run=blocked_pre,
            postcheck_run=blocked_post,
            original_decision="crash",
        )
        self.assertEqual("DIAGNOSTIC_FEEDBACK", blocked_event["memory_channel"])
        self.assertFalse(blocked_event["may_update_primary_search_memory"])
        self.assertTrue(blocked_event["may_update_diagnostic_memory"])

    def test_active_live_hook_preserves_original_result_and_emits_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_lightgcn_live",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
                "mechanism": "one-axis live integration test",
                "source": "llm",
            }
            pre, pre_event = hook.precheck(
                candidate=candidate,
                params={"embedding_size": 128, "n_layers": 2},
                round_id=201,
            )
            self.assertFalse(hook.should_defer(pre))
            self.assertEqual("PRECHECK", pre_event["phase"])
            self.assertFalse(pre_event["expose_to_next_iteration"])

            log_path = root / "RUN-LIVE-001.log"
            write_recbole_log(
                log_path,
                candidate_id="cand_lightgcn_live",
                seed=2026,
            )
            result = {
                "run_id": "RUN-LIVE-001",
                "status": "success",
                "log_path": str(log_path),
                "ndcg@10": 0.28,
                "recall@10": 0.18,
            }
            original_record = {
                "run_id": "RUN-LIVE-001",
                "status": "success",
                "decision": "keep",
                "reason": "original loop keep",
                "next_action": "follow up",
                "compare_baseline": {"delta": 0.0129},
                "compare_history_best": {"delta": -0.01},
            }
            post, post_event = hook.postcheck(
                candidate=candidate,
                candidate_result=result,
                original_record=original_record,
            )
            self.assertEqual(
                "VALID_IMPROVEMENT", post["feedback"]["outcome_classification"]
            )
            self.assertEqual("POSTCHECK", post_event["phase"])
            self.assertEqual("keep", post_event["original_decision"])
            self.assertTrue(post_event["expose_to_next_iteration"])
            self.assertEqual(
                "ADMIT_ORIGINAL_TRIAL",
                post_event["original_trial_memory_disposition"],
            )
            self.assertEqual(
                "EXACT",
                post_event["live_protocol_binding"]["artifact_identity_status"],
            )
            hook.persist_feedback(post_event, root / "agent_memory.jsonl")
            self.assertEqual(
                1,
                len((root / "guard_feedback.jsonl").read_text().splitlines()),
            )
            self.assertEqual(
                1,
                len((root / "agent_memory.jsonl").read_text().splitlines()),
            )

    def test_live_hook_existing_non_log_is_quarantined_not_false_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_live_false_log",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
            }
            hook.precheck(candidate=candidate, params={}, round_id=203)
            fake_log = root / "README.md"
            fake_log.write_text("not a RecBole run\n", encoding="utf-8")
            post, event = hook.postcheck(
                candidate=candidate,
                candidate_result={
                    "run_id": "README",
                    "status": "success",
                    "log_path": str(fake_log),
                    "ndcg@10": 0.99,
                },
                original_record={
                    "run_id": "README",
                    "status": "success",
                    "decision": "keep",
                },
            )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertEqual("PARTIAL", observation["artifact_identity_status"])
            self.assertEqual(
                "INSUFFICIENT_EVIDENCE",
                event["postcheck_outcome_classification"],
            )
            self.assertEqual(
                "QUARANTINE_ORIGINAL_TRIAL",
                event["original_trial_memory_disposition"],
            )
            self.assertFalse(event["may_update_primary_search_memory"])
            self.assertEqual("README", event["original_record_snapshot"]["run_id"])

    def test_live_hook_material_sampled_flip_routes_branch_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_live_sampled_flip",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
            }
            hook.precheck(candidate=candidate, params={}, round_id=204)
            run_id = "RUN-SAMPLED-FLIP"
            log_path = root / f"{run_id}.log"
            write_recbole_log(
                log_path,
                candidate_id=candidate["candidate_id"],
                seed=2026,
                overrides={
                    "eval_args": {
                        "split": {"RS": [0.8, 0.1, 0.1]},
                        "order": "RO",
                        "group_by": "user",
                        "mode": {"valid": "uni100", "test": "uni100"},
                    }
                },
            )
            post, event = hook.postcheck(
                candidate=candidate,
                candidate_result={
                    "run_id": run_id,
                    "status": "success",
                    "log_path": str(log_path),
                    "ndcg@10": 0.99,
                },
                original_record={
                    "run_id": run_id,
                    "status": "success",
                    "decision": "keep",
                },
            )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertEqual("EXACT", observation["artifact_identity_status"])
            self.assertEqual(
                "PROTOCOL_MISMATCH", event["postcheck_outcome_classification"]
            )
            self.assertTrue(event["protocol_branch_required"])
            self.assertEqual(
                "QUARANTINE_ORIGINAL_TRIAL",
                event["original_trial_memory_disposition"],
            )
            self.assertIn(
                "RECBOLE_EVALUATION_MODE_MISMATCH",
                event["live_protocol_binding"]["diagnostics"],
            )

    def test_live_hook_wrong_seed_is_single_fault_partial_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_live_wrong_seed",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
            }
            hook.precheck(candidate=candidate, params={}, round_id=205)
            run_id = "RUN-WRONG-SEED"
            log_path = root / f"{run_id}.log"
            write_recbole_log(
                log_path,
                candidate_id=candidate["candidate_id"],
                seed=2027,
            )
            post, event = hook.postcheck(
                candidate=candidate,
                candidate_result={
                    "run_id": run_id,
                    "status": "success",
                    "log_path": str(log_path),
                    "ndcg@10": 0.99,
                },
                original_record={
                    "run_id": run_id,
                    "status": "success",
                    "decision": "keep",
                },
            )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertEqual("PARTIAL", observation["artifact_identity_status"])
            self.assertEqual(
                "QUARANTINE_ORIGINAL_TRIAL",
                event["original_trial_memory_disposition"],
            )
            self.assertIn(
                "RECBOLE_TRAINING_SEED_MISMATCH",
                event["live_protocol_binding"]["diagnostics"],
            )

    def test_live_hook_wrong_command_model_is_single_fault_partial_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_live_wrong_model",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
            }
            hook.precheck(candidate=candidate, params={}, round_id=206)
            run_id = "RUN-WRONG-MODEL"
            log_path = root / f"{run_id}.log"
            write_recbole_log(
                log_path,
                candidate_id=candidate["candidate_id"],
                seed=2026,
                command_model="OtherModel",
            )
            post, event = hook.postcheck(
                candidate=candidate,
                candidate_result={
                    "run_id": run_id,
                    "status": "success",
                    "log_path": str(log_path),
                    "ndcg@10": 0.99,
                },
                original_record={
                    "run_id": run_id,
                    "status": "success",
                    "decision": "keep",
                },
            )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertEqual("PARTIAL", observation["artifact_identity_status"])
            self.assertEqual(
                "INSUFFICIENT_EVIDENCE",
                event["postcheck_outcome_classification"],
            )
            self.assertEqual(
                "QUARANTINE_ORIGINAL_TRIAL",
                event["original_trial_memory_disposition"],
            )
            self.assertIn(
                "RECBOLE_COMMAND_MODEL_MISMATCH",
                event["live_protocol_binding"]["diagnostics"],
            )

    def test_original_memory_disposition_fails_open_only_on_guard_failure(self) -> None:
        self.assertEqual(
            "FAIL_OPEN_ORIGINAL",
            OriginalRecClawGuardHook.original_trial_memory_disposition(
                {"guard_succeeded": False, "feedback": {}}
            ),
        )
        self.assertEqual(
            "QUARANTINE_ORIGINAL_TRIAL",
            OriginalRecClawGuardHook.original_trial_memory_disposition(
                {
                    "guard_succeeded": True,
                    "feedback": {
                        "outcome_classification": "INSUFFICIENT_EVIDENCE",
                        "may_update_primary_search_memory": False,
                    },
                }
            ),
        )

    def test_runtime_feedback_remains_diagnostic_blocker_signal(self) -> None:
        payload = OriginalRecClawGuardHook.prepare_feedback_for_original_memory(
            {
                "candidate_id": "cand_runtime_blocker",
                "original_run_status": "crash",
                "postcheck_outcome_classification": "INSUFFICIENT_EVIDENCE",
                "expose_to_next_iteration": False,
            }
        )
        self.assertEqual("crash", payload["decision"])
        self.assertTrue(payload["diagnostic_blocker_signal"])
        self.assertTrue(payload["expose_to_next_iteration"])
        self.assertFalse(payload["may_update_primary_search_memory"])
        self.assertTrue(payload["may_update_diagnostic_memory"])

    def test_precheck_feedback_without_run_status_is_not_rewritten_as_crash(self) -> None:
        payload = OriginalRecClawGuardHook.prepare_feedback_for_original_memory(
            {
                "candidate_id": "cand_needs_revision",
                "phase": "PRECHECK",
                "precheck_recommendation": "REVISE_BEFORE_RUN",
                "memory_channel": "PROPOSAL_REVISION_FEEDBACK",
                "next_iteration_effect": "REQUEST_PROPOSAL_REVISION",
                "expose_to_next_iteration": True,
                "may_update_primary_search_memory": False,
                "may_update_diagnostic_memory": True,
            }
        )
        self.assertNotIn("decision", payload)
        self.assertNotIn("diagnostic_blocker_signal", payload)
        self.assertEqual("PROPOSAL_REVISION_FEEDBACK", payload["memory_channel"])
        self.assertEqual("REQUEST_PROPOSAL_REVISION", payload["next_iteration_effect"])

    def test_active_live_hook_defers_explicit_material_protocol_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=Path(temporary) / "guard_feedback.jsonl",
                mode="active",
            )
            candidate = {
                "candidate_id": "cand_protocol_flip",
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
                "mechanism": "changes evaluation",
            }
            pre, _ = hook.precheck(
                candidate=candidate,
                params={"eval_args": {"mode": "uni100"}},
                round_id=202,
            )
            self.assertTrue(hook.should_defer(pre))
            self.assertEqual("REVISE_BEFORE_RUN", pre["feedback"]["recommendation"])

    def test_live_multiseed_validation_binds_exact_logs_and_reaches_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            runs = []
            result_rows = ["run_id,log_path"]
            candidate_id = "cand_lightgcn_multiseed_live"
            for seed in (2026, 2027, 2028):
                run_id = f"RUN-VALIDATION-{seed}"
                log_path = root / f"{run_id}.log"
                write_recbole_log(
                    log_path,
                    candidate_id=candidate_id,
                    seed=seed,
                )
                result_rows.append(f"{run_id},{log_path}")
                runs.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "status": "success",
                        "value": 0.28,
                    }
                )
            results_csv = root / "results.csv"
            results_csv.write_text("\n".join(result_rows) + "\n", encoding="utf-8")
            candidate = {
                "candidate_id": candidate_id,
                "parent_candidate_id": "cand_lightgcn_parent",
                "entrypoint": "recclaw_ext.models.live:LiveModel",
                "mechanism": "validate the selected candidate across three seeds",
            }
            validation = {
                "status": "passed",
                "metric": "ndcg@10",
                "mean": 0.28,
                "runs": runs,
            }
            with mock.patch.dict(
                os.environ,
                {"RECCLAW_RESULTS_CSV": str(results_csv)},
                clear=False,
            ):
                post, event = hook.postcheck_seed_validation(
                    candidate=candidate,
                    params={"embedding_size": 128},
                    round_id=209,
                    seed_validation=validation,
                )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertTrue(post["guard_succeeded"])
            self.assertEqual("EXACT", observation["artifact_identity_status"])
            self.assertEqual(3, observation["seed_count"])
            self.assertEqual(
                "SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL",
                post["feedback"]["claim_ceiling"],
            )
            self.assertEqual("SEED_VALIDATION_POSTCHECK", event["phase"])
            self.assertTrue(event["may_update_primary_search_memory"])

    def test_live_multiseed_validation_missing_log_is_not_counted_as_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook = OriginalRecClawGuardHook(
                contract=contract(),
                feedback_path=root / "guard_feedback.jsonl",
                mode="active",
            )
            results_csv = root / "results.csv"
            results_csv.write_text(
                "run_id,log_path\n"
                f"RUN-2026,{root / 'RUN-2026.log'}\n"
                f"RUN-2027,{root / 'RUN-2027.log'}\n"
                f"RUN-2028,{root / 'missing.log'}\n",
                encoding="utf-8",
            )
            write_recbole_log(
                root / "RUN-2026.log",
                candidate_id="cand_lightgcn_partial_validation",
                seed=2026,
            )
            write_recbole_log(
                root / "RUN-2027.log",
                candidate_id="cand_lightgcn_partial_validation",
                seed=2027,
            )
            validation = {
                "status": "passed",
                "metric": "ndcg@10",
                "mean": 0.28,
                "runs": [
                    {
                        "seed": seed,
                        "run_id": f"RUN-{seed}",
                        "status": "success",
                        "value": 0.28,
                    }
                    for seed in (2026, 2027, 2028)
                ],
            }
            with mock.patch.dict(
                os.environ,
                {"RECCLAW_RESULTS_CSV": str(results_csv)},
                clear=False,
            ):
                post, event = hook.postcheck_seed_validation(
                    candidate={
                        "candidate_id": "cand_lightgcn_partial_validation",
                        "parent_candidate_id": "cand_lightgcn_parent",
                        "entrypoint": "recclaw_ext.models.live:LiveModel",
                    },
                    params={"embedding_size": 128},
                    round_id=210,
                    seed_validation=validation,
                )
            observation = post["envelope"]["guard_request"]["observation"]
            self.assertTrue(post["guard_succeeded"])
            self.assertEqual("PARTIAL", observation["artifact_identity_status"])
            self.assertFalse(event["may_update_primary_search_memory"])
            self.assertEqual(
                "INSUFFICIENT_EVIDENCE",
                event["postcheck_outcome_classification"],
            )


if __name__ == "__main__":
    unittest.main()
