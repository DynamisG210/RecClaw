"""Cold native failure provenance bridge; no native receipt is rewritten.

Campaign composition must explicitly consume the returned composite response.
This module does not enable completion for existing frozen searches.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
from declaration_provider_v2 import digest


class NativeFailureBridge:
    def __init__(self, completion, *, frozen_session_id):
        self.completion = completion
        self.frozen_session_id = frozen_session_id
        self.root = Path(completion.settings["resource_root"]).resolve()

    def resolve(self, result, *, call_root, wire_bytes, response_bytes):
        if result.call is not None:
            return None  # already successful native calls never need completion
        if (result.failure or {}).get("failure_class") != "SCHEMA_VALIDATION_FAILURE":
            raise ValueError("Only an actual native schema failure may enter this bridge")
        sealed = result.sealed_request
        if not isinstance(sealed, dict): raise ValueError("Native sealed request missing")
        physical_root = Path(call_root).resolve()/('physical_attempt_%02d' % result.attempts[-1]["ordinal"])
        physical_root.relative_to(self.root)  # original owner scope, not caller-chosen output namespace
        db = physical_root/"broker.sqlite3"
        con = sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM calls WHERE logical_call_id=?",
                               (sealed["logical_call_id"],)).fetchall()
        finally: con.close()
        if len(rows) != 1: raise ValueError("One original native row is required")
        row = dict(rows[0]); receipt = json.loads(row["receipt_json"])
        if row["status"] != "FAILED" or row["error_type"] != "SCHEMA_VALIDATION_FAILURE":
            raise ValueError("Native row is not the preserved schema failure")
        session = row["proposal_generation_session_id"]
        if session != self.frozen_session_id or sealed["proposal_generation_session_id"] != session:
            raise ValueError("Native session differs from the prospective owner binding")
        wire = json.loads(wire_bytes)
        identity = dict(expected_proposal_count=sealed["expected_proposal_count"],
                        proposal_generation_session_id=session, request_payload=wire)
        if (sealed["request_payload"] != wire or digest(identity) != row["request_digest"] or
                sealed["request_digest"] != row["request_digest"] or
                result.failure["request_digest"] != row["request_digest"] or
                receipt["request_digest"] != row["request_digest"]):
            raise ValueError("Sealed request and persisted native failure do not join")
        detail = json.loads(row["error_detail_json"])
        original = json.loads(detail["original_response_content"])
        envelope = json.loads(response_bytes)
        texts = [part["text"] for item in envelope.get("output",[]) if item.get("type")=="message"
                 for part in item.get("content",[]) if part.get("type")=="output_text"]
        if len(texts)!=1 or json.loads(texts[0])!=original or envelope.get("model")!=row["returned_model"]:
            raise ValueError("Original HTTP output and native failure payload differ")
        usage=envelope.get("usage",{})
        if any(usage.get(k)!=row[k] for k in ["input_tokens","output_tokens"]):
            raise ValueError("Original HTTP usage and native row differ")
        label=hashlib.sha256(wire_bytes).hexdigest()
        paid=[e for e in self.completion.ledger.events() if e.get("event")=="SUCCESS" and e.get("label")==label]
        if len(paid)!=1 or any(paid[0].get(k)!=row[k] for k in ["input_tokens","output_tokens"]):
            raise ValueError("Original paid ledger does not bind this native event")
        schema=wire.get("text",{}).get("format",{}).get("schema")
        if schema is None:
            marker="Required response JSON Schema (validated locally):\n"
            prompt=wire.get("input")
            if not isinstance(prompt,str) or prompt.count(marker)!=1:
                raise ValueError("Original wire schema not uniquely available")
            schema=json.loads(prompt.split(marker)[1])
        if len(original["proposals"])!=identity["expected_proposal_count"]:
            raise ValueError("Original proposal count differs")
        return dict(logical_call_id=row["logical_call_id"],session_id=session,original=original,
                    schema=schema,original_raw=response_bytes,
                    original_failure={"native_receipt":receipt,"failure":result.failure,
                        "native_database_sha256":hashlib.sha256(db.read_bytes()).hexdigest(),
                        "original_paid_call_id":paid[0]["call_id"]})

    def complete(self, result, *, call_root, wire_bytes, response_bytes):
        args=self.resolve(result,call_root=call_root,wire_bytes=wire_bytes,response_bytes=response_bytes)
        if args is None:return result
        response=self.completion.complete(**args)
        return dict(authority="PROSPECTIVE_DECLARATION_COMPOSITE_NOT_NATIVE_RECEIPT",
                    response=response,original_native_failure=result,
                    original_generation=args["logical_call_id"],session_id=args["session_id"])
