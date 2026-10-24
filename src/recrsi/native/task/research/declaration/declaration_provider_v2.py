"""Recover declarations from persisted provider responses.
Store the derived declaration separately from the original response and receipt."""
import copy
import fcntl
from email.utils import parsedate_to_datetime
import hashlib
import json
import time
from pathlib import Path

import httpx
from jsonschema import Draft202012Validator

FIELD = "compatibility_requirements"


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save_new(path, value):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != encoded(value):
            raise ValueError("Saved completion identity or wire changed")
        return
    with path.open("xb") as stream:
        stream.write(encoded(value))


def state_write(attempt, **state):
    temporary = attempt/"STATE.tmp"
    temporary.write_bytes(encoded(state))
    temporary.replace(attempt/"STATE.json")


class CompletionPending(RuntimeError):
    """Original completion is still pending; never request a new generation."""


class ScientificCompletionFailed(ValueError):
    pass


class DeclarationProvider:
    channel = "recclaw:declaration_completion"

    def __init__(self, settings, ledger, resources, *, model, http_transport=None):
        if model not in ledger.model_prices:
            raise ValueError("Completion model must belong to the common ledger")
        if Path(settings["api_ledger"]).resolve() != ledger.path.resolve():
            raise ValueError("Declaration completion must share the search ledger")
        self.settings, self.ledger, self.resources = settings, ledger, resources
        self.model, self.http_transport = model, http_transport

    @classmethod
    def from_config(cls, config_path, *, model):
        from openevolve_provider import _settings_and_ledger
        from ro_main_provider import MainROTransport
        settings, ledger = _settings_and_ledger(str(config_path))
        original = MainROTransport(settings, ledger)
        return cls(settings, ledger, original.resources, model=model)

    def complete(self, *, logical_call_id, session_id, original, schema,
                 original_raw, original_failure):
        identity = dict(run_identity=self.settings["run_identity"],
                        logical_call_id=logical_call_id, session_id=session_id)
        attempt = Path(self.settings["resource_root"])/"declaration_completions"/digest(identity)
        attempt.mkdir(parents=True, exist_ok=True)
        with (attempt/"OWNER.lock").open("a+b") as owner:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._complete_locked(logical_call_id=logical_call_id, session_id=session_id,
                original=original, schema=schema, original_raw=original_raw,
                original_failure=original_failure)

    def _complete_locked(self, *, logical_call_id, session_id, original, schema,
                         original_raw, original_failure):
        # Successful or other-kind outputs never enter a paid completion step.
        errors = list(Draft202012Validator(schema).iter_errors(original))
        if not errors:
            return copy.deepcopy(original)
        indices = []
        for error in errors:
            path = list(error.path)
            if not (error.validator == "required" and
                    error.message == "'compatibility_requirements' is a required property" and
                    len(path) == 2 and path[0] == "proposals" and type(path[1]) is int):
                raise ValueError("Original non-declaration failure retained")
            indices.append(path[1])
        indices = sorted(set(indices))
        declaration_schema = schema["properties"]["proposals"]["items"]["properties"][FIELD]
        if not (declaration_schema.get("type") == "array" and
                declaration_schema.get("minItems", 0) >= 1 and
                declaration_schema.get("items", {}).get("type") == "string" and
                declaration_schema["items"].get("enum") and "default" not in declaration_schema):
            raise ValueError("Completion requires the original closed-enum declaration contract")
        if not logical_call_id or not session_id or not original_failure:
            raise ValueError("Original generation and failure provenance are required")
        identity = dict(run_identity=self.settings["run_identity"],
                        logical_call_id=logical_call_id, session_id=session_id)
        binding = dict(original_raw_sha256=hashlib.sha256(original_raw).hexdigest(),
                       scientific_payload_sha256=digest(original), schema_sha256=digest(schema),
                       original_failure_sha256=digest(original_failure))
        # Key excludes response bytes and output paths: changing either cannot
        # buy another completion for the same original native generation.
        root = Path(self.settings["resource_root"])/"declaration_completions"
        root.mkdir(parents=True, exist_ok=True)
        attempt = root/digest(identity)
        attempt.mkdir(exist_ok=True)  # one logical identity, serialized physical attempts
        save_new(attempt/"IDENTITY.json", dict(identity=identity, binding=binding,
            status="LOGICAL_COMPLETION_RESERVED", new_round=0, new_generation=0,
            new_admission=0, native_failure_preserved=True))
        save_new(attempt/"ORIGINAL_FAILURE.json", original_failure)
        (attempt/"ORIGINAL_RESPONSE.bin").write_bytes(original_raw)
        save_new(attempt/"ORIGINAL_PAYLOAD.json", original)
        save_new(attempt/"ORIGINAL_SCHEMA.json", schema)
        request = dict(binding=binding, original_payload=original,
                       missing_indices=indices, allowed_declaration_schema=declaration_schema)
        payload = dict(model=self.model, reasoning={"effort": "medium"},
            max_output_tokens=self.ledger.max_output_tokens, store=False, stream=False,
            text={"format": {"type": "json_object"}},
            instructions=("Return only a JSON object with exactly binding and declarations. "
                "Echo binding exactly. declarations must map each missing index to an honest "
                "nonempty subset of the supplied allowed enum, based on the existing program. "
                "Do not change or propose programs, parents, selection, or other fields. "
                "If a declaration cannot be supported, return an empty array; validation will fail."),
            input=[{"role": "user", "content": encoded(request).decode()}])
        save_new(attempt/"REQUEST.json", payload)
        if (attempt/"TERMINAL_FAILURE.json").exists():
            raise ScientificCompletionFailed("Original paid/local failure retained; no redraw")
        try:
            completion = self._send(payload, attempt)
            if (not isinstance(completion, dict) or set(completion) != {"binding", "declarations"}
                    or completion["binding"] != binding):
                raise ValueError("Completion binding or response fields differ")
            declarations = completion["declarations"]
            if not isinstance(declarations, dict) or set(declarations) != {str(i) for i in indices}:
                raise ValueError("Completion must cover exactly the missing declarations")
            output = copy.deepcopy(original)
            for index in indices:
                Draft202012Validator(declaration_schema).validate(declarations[str(index)])
                output["proposals"][index][FIELD] = copy.deepcopy(declarations[str(index)])
            Draft202012Validator(schema).validate(output)
            projection = copy.deepcopy(output)
            for index in indices:
                del projection["proposals"][index][FIELD]
            if projection != original or digest(projection) != binding["scientific_payload_sha256"]:
                raise ValueError("Scientific payload changed")
            save_new(attempt/"COMPOSITE_VALIDATED.json", dict(identity=identity, binding=binding,
                response=output, original_native_status="SCHEMA_VALIDATION_FAILURE",
                status="DECLARATIONS_COMPLETED_NOT_NATIVE_RECEIPT", new_generation=0,
                new_round=0, new_admission=0))
            return output
        except CompletionPending:
            raise
        except BaseException as error:
            save_new(attempt/"TERMINAL_FAILURE.json", dict(error_type=type(error).__name__,
                retry_allowed=False, original_failure_preserved=True))
            raise

    def _send(self, payload, attempt):
        from se_shared_api_ledger import _load_route_credentials
        state_path = attempt/"STATE.json"
        state = json.loads(state_path.read_bytes()) if state_path.exists() else {}
        if state.get("status") in {"ACCOUNT_PAUSED", "DISPATCH_UNSETTLED"}:
            raise CompletionPending(state["status"]+": original request needs verified reconciliation")
        if time.time() < state.get("retry_not_before", 0):
            raise CompletionPending("Original Retry-After/backoff has not expired")
        cached = attempt/"SUCCESS_RESPONSE.bin"
        if not cached.exists():
            try:
                self.resources.require_capacity()
                estimate = self.ledger.verify_input_budget(payload, channel=self.channel)
            except Exception as error:
                state_write(attempt, status="ZERO_SEND_CAPACITY_PENDING", error_type=type(error).__name__)
                raise CompletionPending("Zero-send admission wait; original logical completion retained") from error
            route = self.settings.get("model_api_configs", {}).get(self.model, self.settings["api_config"])
            credentials = _load_route_credentials(route)
            label = digest(payload)
            with httpx.Client(transport=self.http_transport,
                    timeout=httpx.Timeout(900, connect=30, write=60, pool=30),
                    follow_redirects=False) as client:
                try:
                    call_id = self.ledger.reserve(channel=self.channel, label=label,
                        model=self.model, estimated_input_tokens=estimate)
                except Exception as error:
                    state_write(attempt, status="ZERO_SEND_LEDGER_PENDING", error_type=type(error).__name__)
                    raise CompletionPending("Common ledger admission pending; no dispatch") from error
                physical = attempt/"physical"/call_id
                physical.mkdir(parents=True)
                save_new(physical/"CALL.json", dict(call_id=call_id, request_sha256=label,
                    ledger=str(self.ledger.path)))
                state_write(attempt, status="DISPATCH_UNSETTLED", call_id=call_id)
                usage, response, envelope = None, None, None
                started = time.monotonic()
                try:
                    response = client.post(credentials["base_url"].rstrip("/")+"/responses",
                        content=encoded(payload), headers={"Authorization":"Bearer "+credentials["api_key"],
                                                           "Content-Type":"application/json"})
                    (physical/"RESPONSE.bin").write_bytes(response.content)
                    save_new(physical/"HTTP.json", dict(status_code=response.status_code,
                        retry_after=response.headers.get("Retry-After")))
                    try: envelope=response.json()
                    except ValueError: pass
                    measured=envelope.get("usage") if isinstance(envelope,dict) else None
                    if isinstance(measured,dict) and all(type(measured.get(k)) is int and measured[k]>=0
                            for k in ("input_tokens","output_tokens")):
                        usage=dict(measured,requests=1)
                    response.raise_for_status()
                    if usage is None: raise ValueError("Unknown completion usage on HTTP success")
                except BaseException as error:
                    self.ledger.record("FAILED",call_id=call_id,channel=self.channel,label=label,
                        model=self.model,usage=usage,error=error,elapsed_seconds=time.monotonic()-started)
                    status=response.status_code if response is not None else None
                    if status in {401,402,403}:
                        state_write(attempt,status="ACCOUNT_PAUSED",call_id=call_id,http_status=status)
                        raise CompletionPending("Account/auth pause; no automatic probe") from error
                    transient=(status in {408,429,500,502,503,504} or isinstance(error,
                        (httpx.ConnectError,httpx.ConnectTimeout,httpx.ReadTimeout,httpx.ReadError,
                         httpx.RemoteProtocolError,httpx.WriteError,httpx.WriteTimeout)))
                    if transient:
                        # Raw FAILED stays in the original ledger. A bounded hold is
                        # uncertainty, never a fabricated zero-charge settlement.
                        self._record_infrastructure(call_id,physical,usage,status,type(error).__name__)
                        count=len(list((attempt/"physical").iterdir()))
                        delay=min(60,15*(2**min(count-1,2)))
                        header=response.headers.get("Retry-After") if response is not None else None
                        if header:
                            try: delay=max(delay,float(header))
                            except ValueError:
                                try: delay=max(delay,parsedate_to_datetime(header).timestamp()-time.time())
                                except (ValueError,TypeError,OverflowError): pass
                        state_write(attempt,status="TRANSIENT_PENDING",call_id=call_id,
                            retry_not_before=time.time()+delay,raw_usage="REPORTED" if usage else "UNKNOWN",
                            scientific_opportunity_consumed=False)
                        raise CompletionPending("Same saved wire pending original-owner backoff") from error
                    state_write(attempt,status="NON_TRANSIENT_FAILED",call_id=call_id)
                    raise
                self.ledger.record("SUCCESS",call_id=call_id,channel=self.channel,label=label,
                    model=self.model,usage=usage,elapsed_seconds=time.monotonic()-started)
                cached.write_bytes(response.content)
                state_write(attempt,status="HTTP_SUCCESS_CACHED",call_id=call_id)
        if not self.resources.admit_output():
            raise CompletionPending("Successful response cached; common output admission pending")
        envelope=json.loads(cached.read_bytes())
        if envelope.get("status")!="completed" or envelope.get("model")!=self.model:
            raise ValueError("Completion status/model identity differs")
        messages=[item for item in envelope.get("output",[]) if item.get("type")=="message"]
        if len(messages)!=1 or len(messages[0].get("content",[]))!=1:
            raise ValueError("Expected exactly one completion message")
        content=messages[0]["content"][0]
        if content.get("type")!="output_text":raise ValueError("Expected completion JSON text")
        return json.loads(content["text"])

    def _record_infrastructure(self,call_id,physical,usage,status,error_type):
        with self.ledger._lock:
            events=self.ledger.events()
            failed=next(e for e in reversed(events) if e.get("call_id")==call_id and e["event"]=="FAILED")
            assert failed["channel"]==self.channel
            proof=dict(physical_evidence=str(physical),http_status=status,error_type=error_type,
                raw_usage="REPORTED" if usage else "UNKNOWN",scientific_cost_or_opportunity=False)
            self.ledger._append_unlocked(dict(event="COMPLETION_INFRASTRUCTURE_FAILURE",
                call_id=call_id,timestamp_ns=time.time_ns(),**proof))
            if usage is None:
                prices=self.ledger.model_prices[self.model]
                bound=(self.ledger.max_input_tokens*prices["input"]+
                       self.ledger.max_output_tokens*prices["output"])/1e6
                self.ledger._append_unlocked(dict(event="UNRESOLVED_USAGE_HOLD",call_id=call_id,
                    timestamp_ns=time.time_ns(),reference_hold_usd=bound,usage_reported=False,evidence=proof))
        # Original common ledger retains physical count/cost and hard admission
        # bounds. Infrastructure exclusion is a separate scientific view, never
        # a refund or an increased raw request/monetary ceiling.
