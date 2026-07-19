from __future__ import annotations

import copy
import hashlib
import math
import unittest

from recclaw_core.exploration.evidence_guard import (
    evaluate_evidence_guard,
    guard_router_candidates,
)


def protocol() -> dict[str, object]:
    return {
        "protocol_id": "PROTO-ML1M-RANDOM-FULL-001",
        "profile_family": "OFFLINE_TOPN",
        "dataset": "ml-1m",
        "dataset_snapshot": "ml-1m-local-snapshot-001",
        "split": {
            "strategy": "random_user_holdout",
            "ratio": [0.8, 0.1, 0.1],
            "group_by": "user",
        },
        "training_sampling": {"mode": "uniform_negative", "distribution": "uniform"},
        "evaluation_candidate_universe": {"mode": "full_sort"},
        "candidate_policy": {"seen_items": "exclude", "repeat_items": "allow"},
        "metric": {"name": "ndcg", "cutoff": 10, "aggregation": "mean_by_user"},
        "training_procedure": {
            "train_batch_size": 2048,
            "optimizer": "adam",
            "max_epochs": 300,
        },
    }


def claim() -> dict[str, object]:
    return {
        "claim_id": "CLAIM-CANDIDATE-VS-LIGHTGCN-001",
        "protocol_id": "PROTO-ML1M-RANDOM-FULL-001",
        "claim_kind": "LOCAL_IMPROVEMENT",
        "target_model": "CandidateModel",
        "comparator": "LightGCN",
        "metric": "ndcg",
        "required_seed_count": 3,
        "scope": {"dataset": "ml-1m", "population": "eligible_users"},
    }


def evidence() -> dict[str, object]:
    return {
        "snapshot_id": "EVIDENCE-SNAPSHOT-001",
        "claim_id": "CLAIM-CANDIDATE-VS-LIGHTGCN-001",
        "protocol_id": "PROTO-ML1M-RANDOM-FULL-001",
        "observation_ids": [],
    }


def proposal(
    *, planned: dict[str, object] | None = None, seed_count: int = 1
) -> dict[str, object]:
    return {
        "proposal_id": "PROPOSAL-001",
        "action_family": "RUN_OFFLINE_TOPN",
        "target_claim_id": "CLAIM-CANDIDATE-VS-LIGHTGCN-001",
        "protocol_id": "PROTO-ML1M-RANDOM-FULL-001",
        "planned_protocol": planned or protocol(),
        "target_model": "CandidateModel",
        "comparator": "LightGCN",
        "seed_count": seed_count,
        "seed_ids": [str(2026 + index) for index in range(seed_count)],
        "purpose": "development comparison",
    }


def observation(
    *,
    observed: dict[str, object] | None = None,
    seed_count: int = 1,
    kind: str = "METRIC_EVALUATION",
    status: str = "SUCCESS",
    identity: str = "EXACT",
) -> dict[str, object]:
    seed_runs = [
        {
            "seed_id": str(2026 + index),
            "run_id": f"RUN-{2026 + index}",
            "artifact_sha256": hashlib.sha256(
                f"RUN-{2026 + index}".encode("utf-8")
            ).hexdigest(),
        }
        for index in range(seed_count)
    ]
    return {
        "observation_id": "OBS-001",
        "proposal_id": "PROPOSAL-001",
        "claim_id": "CLAIM-CANDIDATE-VS-LIGHTGCN-001",
        "protocol_id": "PROTO-ML1M-RANDOM-FULL-001",
        "observed_protocol": observed or protocol(),
        "target_model": "CandidateModel",
        "comparator": "LightGCN",
        "seed_count": seed_count,
        "seed_runs": seed_runs,
        "observation_kind": kind,
        "run_status": status,
        "artifact_identity_status": identity,
        "evidence_class": "DEVELOPMENT_ONLY",
        "metrics": {"ndcg": 0.27} if kind == "METRIC_EVALUATION" else {},
        "notes": [],
    }


def guard(
    *,
    action: dict[str, object] | None = None,
    observed: dict[str, object] | None | object = ...,  # sentinel: use default observation
) -> dict[str, object]:
    chosen = observation() if observed is ... else observed
    return evaluate_evidence_guard(
        claim=claim(),
        protocol=protocol(),
        current_evidence=evidence(),
        action_proposal=action or proposal(),
        observation=chosen,
    )


class EvidenceGuardTests(unittest.TestCase):
    def test_exact_single_seed_is_only_a_local_preliminary_signal(self) -> None:
        result = guard()
        self.assertEqual("LEGAL", result["action_legality"]["development_verdict"])
        self.assertEqual(
            "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertEqual(
            "LOCAL_SINGLE_OR_FEW_SEED_SIGNAL", result["claim_ceiling"]["level"]
        )
        self.assertEqual("INCONCLUSIVE", result["evidence_admissibility"]["authoritative_verdict"])
        self.assertFalse(result["claim_ceiling"]["may_update_claim_state"])

    def test_exact_required_multiseed_reaches_only_development_ceiling(self) -> None:
        result = guard(
            action=proposal(seed_count=3), observed=observation(seed_count=3)
        )
        self.assertEqual(
            "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertEqual(
            "SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL",
            result["claim_ceiling"]["level"],
        )
        self.assertFalse(result["formal_acceptance"])

    def test_observation_cannot_inflate_proposal_seed_count(self) -> None:
        result = guard(observed=observation(seed_count=3))
        self.assertEqual(
            "EXCLUDE_FROM_CURRENT_CLAIM",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertIn(
            "GUARD_SEED_COUNT_MISMATCH",
            result["evidence_admissibility"]["reason_codes"],
        )

    def test_duplicate_seed_run_identity_fails_closed(self) -> None:
        bad = observation(seed_count=2)
        bad["seed_runs"][1]["run_id"] = bad["seed_runs"][0]["run_id"]
        result = guard(action=proposal(seed_count=2), observed=bad)
        self.assertEqual("INCONCLUSIVE", result["action_legality"]["development_verdict"])
        self.assertEqual(
            "GUARD_SEED_RUN_CLOSURE", result["protocol_diagnostics"][0]["code"]
        )

    def test_temporal_split_proposal_requires_protocol_branch(self) -> None:
        flipped = protocol()
        flipped["protocol_id"] = "PROTO-ML1M-TEMPORAL-FULL-001"
        flipped["split"] = {
            "strategy": "temporal_user_holdout",
            "ratio": [0.8, 0.1, 0.1],
            "group_by": "user",
        }
        result = guard(action=proposal(planned=flipped))
        self.assertEqual("ILLEGAL", result["action_legality"]["development_verdict"])
        self.assertTrue(result["affected_claim_scope"]["protocol_branch_required"])
        self.assertEqual("CROSS_PROTOCOL_BRANCH_ONLY", result["claim_ceiling"]["level"])

    def test_sampled_evaluation_observation_cannot_update_full_sort_claim(self) -> None:
        flipped = protocol()
        flipped["protocol_id"] = "PROTO-ML1M-RANDOM-SAMPLED-001"
        flipped["evaluation_candidate_universe"] = {
            "mode": "sampled",
            "sample_size": 100,
        }
        result = guard(observed=observation(observed=flipped))
        self.assertEqual(
            "EXCLUDE_FROM_CURRENT_CLAIM",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertIn(
            "GUARD_OBSERVED_PROTOCOL_FLIP",
            result["evidence_admissibility"]["reason_codes"],
        )
        self.assertEqual("CROSS_PROTOCOL_BRANCH_ONLY", result["claim_ceiling"]["level"])

    def test_training_sampling_and_evaluation_sampling_are_distinct_axes(self) -> None:
        training_flip = protocol()
        training_flip["protocol_id"] = "PROTO-ML1M-NO-TRAIN-NEG-FULL-001"
        training_flip["training_sampling"] = {"mode": "none"}
        result = guard(action=proposal(planned=training_flip), observed=None)
        paths = {item["path"] for item in result["protocol_diagnostics"]}
        self.assertTrue(any("/training_sampling" in path for path in paths))
        self.assertFalse(any("/evaluation_candidate_universe" in path for path in paths))

    def test_smoke_pass_is_executability_only(self) -> None:
        result = guard(
            observed=observation(kind="INTERFACE_SMOKE", status="SMOKE_PASS")
        )
        self.assertEqual(
            "RECORD_EXECUTABILITY_ONLY",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertEqual("EXECUTABILITY_ONLY", result["claim_ceiling"]["level"])

    def test_runtime_blocker_is_not_a_mechanism_result(self) -> None:
        result = guard(observed=observation(status="RUNTIME_BLOCKED"))
        self.assertEqual(
            "RECORD_RUNTIME_BLOCKER_ONLY",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertEqual("RUNTIME_BLOCKER_ONLY", result["claim_ceiling"]["level"])

    def test_partial_artifact_identity_is_quarantined(self) -> None:
        result = guard(observed=observation(identity="PARTIAL"))
        self.assertEqual(
            "QUARANTINE_PROVENANCE_INCOMPLETE",
            result["evidence_admissibility"]["development_disposition"],
        )
        self.assertEqual(
            "PROVENANCE_INCOMPLETE_DIAGNOSTIC", result["claim_ceiling"]["level"]
        )

    def test_duplicate_observation_is_excluded(self) -> None:
        current = evidence()
        current["observation_ids"] = ["OBS-001"]
        result = evaluate_evidence_guard(
            claim=claim(),
            protocol=protocol(),
            current_evidence=current,
            action_proposal=proposal(),
            observation=observation(),
        )
        self.assertIn(
            "GUARD_DUPLICATE_OBSERVATION",
            result["evidence_admissibility"]["reason_codes"],
        )
        self.assertEqual("NONE", result["affected_claim_scope"]["current_claim_effect"])

    def test_nonfinite_metric_fails_closed_without_exception(self) -> None:
        bad = observation()
        bad["metrics"] = {"ndcg": math.nan}
        result = guard(observed=bad)
        self.assertEqual("INCONCLUSIVE", result["action_legality"]["development_verdict"])
        self.assertEqual("QUARANTINE_INPUT", result["router_directive"])
        self.assertEqual(
            "GUARD_NONCANONICAL_INPUT", result["protocol_diagnostics"][0]["code"]
        )

    def test_router_batch_filters_protocol_flip_without_ranking_candidates(self) -> None:
        matched = proposal()
        flipped_protocol = copy.deepcopy(protocol())
        flipped_protocol["protocol_id"] = "PROTO-SAMPLED"
        flipped_protocol["evaluation_candidate_universe"] = {
            "mode": "sampled",
            "sample_size": 100,
        }
        flipped = proposal(planned=flipped_protocol)
        flipped["proposal_id"] = "PROPOSAL-FLIPPED"
        batch = guard_router_candidates(
            claim=claim(),
            protocol=protocol(),
            current_evidence=evidence(),
            proposals=[matched, flipped],
        )
        self.assertEqual(["PROPOSAL-001"], batch["eligible_proposal_ids"])
        self.assertEqual(2, batch["candidate_count"])
        self.assertFalse(batch["may_update_claim_state"])
        self.assertFalse(batch["may_update_accepted_evidence_history"])


if __name__ == "__main__":
    unittest.main()
