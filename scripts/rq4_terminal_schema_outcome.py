"""Restore terminal schema rejection from the original broker receipt.

A schema-invalid paid response consumes its producer opportunity; it is not an
unresolved network request. No database row or successful output is rewritten.
"""
from dataclasses import replace
from pathlib import Path
import json
import sqlite3

def restore_terminal_schema(outcomes, home, campaign_id):
    round_id=json.loads((Path(home).parent/"PROSPECTIVE_CONTRACT.json").read_text())["next_round"]
    restored=[]
    for outcome in outcomes:
        if outcome.failure_code not in ("PROVIDER_TRANSPORT_FAILURE", "SEALED_PROVIDER_REQUEST_UNAVAILABLE"):
            restored.append(outcome)
            continue
        root=Path(home)/"search/campaign/provider_calls/proposals"/outcome.producer_role
        evidence=[]
        for path in root.glob("physical_attempt_*/broker.sqlite3"):
            with sqlite3.connect("file:"+str(path)+"?mode=ro",uri=True) as db:
                rows=db.execute("select logical_call_id,request_digest,status,error_type,error_detail_json,receipt_json,closure_receipt_json from calls").fetchall()
            for logical,request,status,error,detail,receipt,closure in rows:
                receipt=json.loads(receipt or '{}');closure=json.loads(closure or '{}')
                if (logical.startswith(campaign_id+":") and f":round-{round_id}:" in logical
                    and logical.endswith(":"+outcome.producer_role+":proposal")
                    and status=="FAILED" and error=="SCHEMA_VALIDATION_FAILURE"
                    and receipt.get("http_status")==200
                    and receipt.get("request_digest")==request
                    and closure.get("start_record_digest")==request
                    and closure.get("provider_request_confirmation")=="RESPONSE_RECEIVED"
                    and closure.get("timed_out") is False):
                    evidence.append(dict(broker=str(path),logical_call_id=logical,
                                         request_digest=request,status=status,
                                         error_type=error,error_detail=json.loads(detail)))
        if len(evidence)!=1:
            restored.append(outcome)
            continue
        proof=evidence[0]
        facts=dict(outcome.resolution_facts)
        facts.update(failure_code="PROVIDER_RESPONSE_REJECTED",
                     failure_taxonomy="SCIENTIFIC_INCONCLUSIVE",
                     failure_stage="PROVIDER_RESPONSE_SCHEMA",
                     original_failure_code=outcome.failure_code,
                     terminal_broker_evidence=proof)
        provenance=dict(outcome.provenance)
        provenance.update(terminal_schema_restored=True,terminal_broker_evidence=proof,
                          original_failure_code=outcome.failure_code)
        restored.append(replace(outcome,failure_code="PROVIDER_RESPONSE_REJECTED",
            failure_detail="Original paid response failed schema validation; no redraw.",
            resolution_facts=facts,provenance=provenance))
    return tuple(restored)
