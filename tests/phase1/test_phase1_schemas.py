from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema

from recclaw_core.exploration.evidence_guard import evaluate_evidence_guard


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Phase1SchemaTests(unittest.TestCase):
    def test_guard_envelope_and_result_validate(self) -> None:
        contract = json.loads(
            (PROJECT_ROOT / "configs/phase1/ab002/evidence_guard_contract.json").read_text()
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
            (PROJECT_ROOT / "configs/phase1/ab002/evidence_guard_contract.json").read_text()
        )
        jsonschema.validate(contract, schema)


if __name__ == "__main__":
    unittest.main()
