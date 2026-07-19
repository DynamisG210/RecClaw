from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import jsonschema

from recclaw_core.exploration.evidence_guard import evaluate_evidence_guard


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Phase1SchemaTests(unittest.TestCase):
    def test_guard_envelope_and_result_validate(self) -> None:
        contract = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/experiment_contract.json").read_text()
        )
        envelope = {
            "claim": contract["claim"],
            "protocol": contract["protocol"],
            "current_evidence": contract["current_evidence"],
            "action_proposal": {
                "proposal_id": "P-1",
                "action_family": "RUN_OFFLINE_TOPN",
                "target_claim_id": contract["claim"]["claim_id"],
                "protocol_id": contract["protocol"]["protocol_id"],
                "planned_protocol": contract["protocol"],
                "target_model": contract["claim"]["target_model"],
                "comparator": contract["claim"]["comparator"],
                "seed_count": 1,
                "seed_ids": ["2026"],
                "purpose": "development check",
            },
            "observation": None,
        }
        input_schema = json.loads(
            (PROJECT_ROOT / "schemas/evidence_guard_envelope.schema.json").read_text()
        )
        output_schema = json.loads(
            (PROJECT_ROOT / "schemas/evidence_guard_result.schema.json").read_text()
        )
        jsonschema.validate(envelope, input_schema)
        result = evaluate_evidence_guard(**envelope)
        jsonschema.validate(result, output_schema)

    def test_all_five_milestone_records_validate(self) -> None:
        schema = json.loads(
            (PROJECT_ROOT / "schemas/phase1_milestone_record.schema.json").read_text()
        )
        records = sorted((PROJECT_ROOT / "records/phase1").glob("*.json"))
        self.assertEqual(5, len(records))
        for path in records:
            with self.subTest(path=path.name):
                jsonschema.validate(json.loads(path.read_text()), schema)

    def test_ab002_contract_validates(self) -> None:
        schema = json.loads(
            (PROJECT_ROOT / "schemas/phase1_ab002_contract.schema.json").read_text()
        )
        contract = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/experiment_contract.json").read_text()
        )
        jsonschema.validate(contract, schema)

    def test_pre_canary_policies_bind_exact_implementation_bytes(self) -> None:
        def digest(relative: str) -> str:
            return hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()

        broker = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/common_llm_broker_spec.json").read_text()
        )
        run_policy = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/neutral_run_audit_policy.json").read_text()
        )
        outcome = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/outcome_audit_policy.json").read_text()
        )
        self.assertTrue(broker["frozen_before_canary"])
        self.assertTrue(run_policy["frozen_before_canary"])
        self.assertTrue(outcome["frozen_before_canary"])
        self.assertEqual(
            digest("src/recclaw_phase1/paired_llm_broker.py"), broker["implementation_sha256"]
        )
        self.assertEqual(
            digest("src/recclaw_phase1/neutral_outcome_auditor.py"),
            run_policy["implementation_file_sha256"],
        )
        self.assertEqual(
            digest("src/recclaw_phase1/neutral_outcome_auditor.py"),
            outcome["auditor_implementation_file_sha256"],
        )
        self.assertEqual(
            digest("src/recclaw_phase1/ab002_final_analysis.py"),
            outcome["analysis_implementation_sha256"],
        )
        contract = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/experiment_contract.json").read_text()
        )
        canary = contract["canary"]
        for field, relative in {
            "preflight_implementation_sha256": "src/recclaw_phase1/ab002_preflight.py",
            "audit_implementation_sha256": "src/recclaw_phase1/ab002_canary_audit.py",
            "orchestrator_implementation_sha256": "src/recclaw_phase1/ab002_canary_orchestrator.py",
            "controlled_probes_implementation_sha256": "src/recclaw_phase1/ab002_canary_probes.py",
        }.items():
            self.assertEqual(digest(relative), canary[field])


if __name__ == "__main__":
    unittest.main()
