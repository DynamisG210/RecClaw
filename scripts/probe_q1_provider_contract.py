#!/usr/bin/env python3
"""Run one non-research Provider conditional-schema qualification probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (  # noqa: E402
    API_CONFIG,
    MODEL,
    _write_new_json,
)
from recclaw_core.experiments.helix_abc_v1.idea_quality import (  # noqa: E402
    Q1_CONDITIONAL_SENTINEL_PROMPT,
    Q1_PROPOSAL_TOKEN_CEILING,
    derive_q1_conditional_contract_sentinel_schema,
    q1_provider_contract_static_matrix,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiCanaryBrokerV1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    args = parser.parse_args()
    probe_root = args.probe_root.resolve()
    if probe_root.exists():
        raise RuntimeError(f"contract probe root already exists: {probe_root}")
    probe_root.mkdir(parents=True)
    schema = derive_q1_conditional_contract_sentinel_schema()
    schema_path = probe_root / "CONDITIONAL_SENTINEL_SCHEMA.json"
    schema_sha256 = _write_new_json(schema_path, schema)
    matrix_sha256 = _write_new_json(
        probe_root / "STATIC_CONTRACT_MATRIX.json",
        q1_provider_contract_static_matrix(),
    )
    broker = LabApiCanaryBrokerV1(
        probe_root / "physical_attempt_01",
        schema_path=schema_path,
        config_path=API_CONFIG,
        model=MODEL,
        max_total_tokens_per_call=Q1_PROPOSAL_TOKEN_CEILING,
        timeout_ms=900_000,
        release_manifest_path=None,
    )
    try:
        call = broker.call_with_session(
            logical_call_id="q1-provider-contract:conditional-sentinel",
            proposal_generation_session_id=(
                "q1-provider-contract:conditional-sentinel-session"
            ),
            prompt=Q1_CONDITIONAL_SENTINEL_PROMPT,
            expected_proposal_count=1,
            max_total_tokens=Q1_PROPOSAL_TOKEN_CEILING,
        )
        physical = {
            "status": "SUCCESS",
            "http_status": 200,
            "request_digest": call.request_digest,
            "response_digest": call.response_digest,
            "receipt_digest": None,
            "returned_model": call.returned_model,
            "latency_ms": call.latency_ms,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "billed_tokens": call.total_tokens,
        }
    except CanaryBrokerError as error:
        outcome = error.outcome.to_dict() if error.outcome is not None else None
        receipt = error.receipt.to_dict() if error.receipt is not None else None
        physical = {
            "status": "FAILED",
            "http_status": (
                receipt.get("exit_code_or_NONE")
                if isinstance(receipt, dict)
                else None
            ),
            "request_digest": (
                receipt.get("start_record_digest")
                if isinstance(receipt, dict)
                else None
            ),
            "response_digest": None,
            "receipt_digest": (
                receipt.get("receipt_digest") if isinstance(receipt, dict) else None
            ),
            "outcome_digest": (
                outcome.get("outcome_digest") if isinstance(outcome, dict) else None
            ),
            "failure_class": (
                outcome.get("failure_class") if isinstance(outcome, dict) else None
            ),
            "error_type": type(error).__name__,
            "latency_ms": error.wall_time_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "billed_tokens": 0,
        }
    finally:
        broker.close()
    result = {
        "schema": "recclaw.research-line.q1-provider-contract-probe.v1",
        "probe_kind": "SYNTHETIC_CONDITIONAL_SCHEMA_SENTINEL",
        "research_semantics": False,
        "research_candidate_generated": False,
        "model": MODEL,
        "endpoint_identity_source": "LAB_API_BROKER_RELEASE_V1",
        "tools": [],
        "token_ceiling": Q1_PROPOSAL_TOKEN_CEILING,
        "temperature": 0.0,
        "strict": True,
        "physical_calls": 1,
        "retries": 0,
        "schema_sha256": schema_sha256,
        "static_matrix_sha256": matrix_sha256,
        "physical": physical,
        "decision": (
            "CONDITIONAL_COMPOSITION_UNSUPPORTED"
            if physical["http_status"] == 400
            else "ROOT_CAUSE_NOT_LOCALIZED"
        ),
        "held_out_reads": 0,
        "outcome_fields_consumed": [],
        "implementer_calls": 0,
        "qualifier_calls": 0,
        "gpu_runs": 0,
    }
    digest = _write_new_json(probe_root / "CONTRACT_PROBE_RECEIPT.json", result)
    print(json.dumps({"receipt_sha256": digest, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
