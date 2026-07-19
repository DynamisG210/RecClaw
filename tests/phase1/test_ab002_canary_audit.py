from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recclaw_phase1.ab002_canary_audit import _arm_artifacts


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _defer_feedback() -> list[dict[str, object]]:
    return [
        {
            "event": "evidence_guard_development_feedback",
            "phase": "PRECHECK",
            "hook_mode": "active",
            "precheck_recommendation": "REVISE_BEFORE_RUN",
            "original_decision": "PENDING_ORIGINAL_EXECUTION",
            "memory_channel": "PROPOSAL_REVISION_FEEDBACK",
            "next_iteration_effect": "REQUEST_PROPOSAL_REVISION",
            "expose_to_next_iteration": True,
            "may_authorize_execution": False,
            "may_update_primary_search_memory": False,
            "formal_acceptance": False,
        }
    ]


class AB002CanaryAuditTests(unittest.TestCase):
    def _treatment_root(self, temporary: str, *, ordinary: bool = False) -> Path:
        root = Path(temporary)
        repetition = root / "runs" / "treatment" / "search_seed_9001"
        pilot = repetition / "outputs" / "pilot"
        _write_jsonl(pilot / "agent_memory.jsonl", [{"event": "implementation_result"}])
        _write_jsonl(pilot / "candidate_proposals.jsonl", [{"proposal_id": "p1"}])
        _write_json(pilot / "candidate_search_tree.json", {})
        _write_json(pilot / "experience_summary.json", {})
        budget = [{"purpose": "smoke"}]
        if ordinary:
            budget.append({"purpose": "ordinary"})
        _write_jsonl(repetition / "candidate_execution_budget.jsonl", budget)
        return root

    def test_treatment_all_guard_deferred_may_have_no_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._treatment_root(temporary)
            result = _arm_artifacts(
                root, "treatment", guard_feedback=_defer_feedback()
            )
        self.assertEqual("GUARD_DEFERRED_ALL", result["ordinary_execution_mode"])
        self.assertEqual(0, result["result_row_count"])
        self.assertNotIn("results", result["artifact_paths"])

    def test_missing_results_without_exact_guard_deferral_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._treatment_root(temporary)
            with self.assertRaisesRegex(ValueError, "ordinary training"):
                _arm_artifacts(root, "treatment", guard_feedback=[])

    def test_crash_relabelled_precheck_does_not_justify_zero_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._treatment_root(temporary)
            feedback = _defer_feedback()
            feedback[0]["memory_channel"] = "DIAGNOSTIC_FEEDBACK"
            feedback[0]["next_iteration_effect"] = "INGEST_DIAGNOSTIC_FEEDBACK_ONLY"
            feedback[0]["diagnostic_blocker_signal"] = True
            with self.assertRaisesRegex(ValueError, "ordinary training"):
                _arm_artifacts(root, "treatment", guard_feedback=feedback)

    def test_missing_results_with_ordinary_budget_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._treatment_root(temporary, ordinary=True)
            with self.assertRaisesRegex(ValueError, "artifacts missing"):
                _arm_artifacts(
                    root, "treatment", guard_feedback=_defer_feedback()
                )


if __name__ == "__main__":
    unittest.main()
