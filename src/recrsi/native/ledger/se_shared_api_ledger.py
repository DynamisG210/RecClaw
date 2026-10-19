"""Fail-closed API usage ledger for Self-EvolveRec shared experiments."""

from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path


SCHEMA = "recclaw.external-comparison.se-shared-api-event.v1"
LUNA_MODEL = "gpt-5.6-luna"
LUNA_WIRE_MODEL = "openai/gpt-5.6-luna"


def _load_route_credentials(path):
    values = {"base_url": [], "api_key": []}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        name, separator, value = raw.strip().partition("=")
        if separator and name.strip() in values:
            parsed = ast.literal_eval(value.strip())
            if isinstance(parsed, str):
                values[name.strip()].append(parsed.strip())
    if not values["base_url"] or len(values["base_url"]) != len(values["api_key"]):
        raise ValueError("invalid alternate Luna credential pairs")
    return {"base_url": values["base_url"][0], "api_key": values["api_key"][0]}


def _canonical_model(model):
    value = str(model)
    if value in (LUNA_MODEL, LUNA_WIRE_MODEL):
        return LUNA_MODEL
    if value in ("gpt-5.6-terra", "openai/gpt-5.6-terra"):
        return "gpt-5.6-terra"
    return value


class BudgetExhausted(RuntimeError):
    pass


class InputBudgetExceeded(RuntimeError):
    pass


class ProviderExecutionPaused(RuntimeError):
    pass


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if not isinstance(value, type) and hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    return str(value)


def _estimate_tokens(value):
    import tiktoken

    encoded = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return len(tiktoken.get_encoding("o200k_base").encode(encoded))


def _http_status(error):
    for value in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def _usage_fields(usage):
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        get = usage.get
    else:
        get = lambda name, default=0: getattr(usage, name, default)
    input_tokens = int(get("input_tokens", get("prompt_tokens", 0)) or 0)
    output_tokens = int(get("output_tokens", get("completion_tokens", 0)) or 0)
    raw_requests = get("requests", 1)
    requests = 1 if raw_requests is None else int(raw_requests)
    return requests, input_tokens, output_tokens


class ApiLedger:
    def __init__(self, path, max_requests, *, input_price_per_million=None, output_price_per_million=None, model_prices=None, normalized_unit_usd=None, max_normalized_u=None, max_input_tokens=48000, max_output_tokens=20000, input_estimate_margin=4096):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_requests = int(max_requests)
        if self.max_requests < 1:
            raise ValueError("max_requests must be positive")
        self.input_price_per_million = None if input_price_per_million is None else float(input_price_per_million)
        self.output_price_per_million = None if output_price_per_million is None else float(output_price_per_million)
        self.model_prices = {
            str(model): {key: float(value) for key, value in prices.items()}
            for model, prices in (model_prices or {}).items()
        }
        self.normalized_unit_usd = (
            None if normalized_unit_usd is None else float(normalized_unit_usd)
        )
        self.max_normalized_u = None if max_normalized_u is None else float(max_normalized_u)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.input_estimate_margin = int(input_estimate_margin)
        if min(self.max_input_tokens, self.max_output_tokens) < 1 or self.input_estimate_margin < 0:
            raise ValueError("token limits must be positive and margin must be nonnegative")
        mixed_pricing = bool(self.model_prices)
        scalar_pricing = (
            self.input_price_per_million is not None
            or self.output_price_per_million is not None
            or self.max_normalized_u is not None
        )
        if mixed_pricing and scalar_pricing and (
            self.input_price_per_million is not None
            or self.output_price_per_million is not None
        ):
            raise ValueError("mixed and scalar pricing cannot be combined")
        if not mixed_pricing and scalar_pricing and (
            self.input_price_per_million is None
            or self.output_price_per_million is None
            or self.max_normalized_u is None
        ):
            raise ValueError("normalized pricing requires input price, output price, and max U")
        if not mixed_pricing and scalar_pricing and (
            self.input_price_per_million <= 0
            or self.output_price_per_million <= 0
            or self.max_normalized_u <= 0
        ):
            raise ValueError("normalized pricing values must be positive")
        if mixed_pricing and (self.normalized_unit_usd is None or self.normalized_unit_usd <= 0):
            raise ValueError("mixed pricing requires a positive normalized unit")
        if mixed_pricing:
            for model, prices in self.model_prices.items():
                if (
                    prices.get("input", 0) <= 0
                    or prices.get("output", 0) <= 0
                    or prices.get("cache_read", 0) <= 0
                    or prices.get("cache_creation", 0) <= 0
                ):
                    raise ValueError(f"invalid model prices: {model}")
        elif scalar_pricing and self.normalized_unit_usd is None:
            self.normalized_unit_usd = self.input_price_per_million + self.output_price_per_million
        self._lock = threading.Lock()

    def events(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _summarize(events):
        calls = {}
        legacy = []
        for event in events:
            call_id = event.get("call_id")
            if call_id is None:
                if event["event"] in {"SUCCESS", "FAILED"}:
                    legacy.append(event)
                continue
            calls.setdefault(call_id, {})[event["event"]] = event
        completed = list(legacy)
        open_reservations = []
        for state in calls.values():
            terminal = state.get("SUCCESS") or state.get("FAILED")
            if terminal is not None:
                completed.append(terminal)
            elif state.get("RESERVED") is not None:
                open_reservations.append(state["RESERVED"])
        return {
            "logical_calls": len(completed),
            "requests": sum(int(event.get("requests", 0)) for event in completed + open_reservations),
            "input_tokens": sum(int(event.get("input_tokens", 0)) for event in completed),
            "output_tokens": sum(int(event.get("output_tokens", 0)) for event in completed),
            "unknown_usage_failures": sum(event["event"] == "FAILED" and not event.get("usage_reported") for event in completed),
            "open_reservations": len(open_reservations),
            "fatal_provider_statuses": sorted(
                {
                    int(event["http_status"])
                    for event in completed
                    if event.get("event") == "FAILED"
                    and event.get("http_status") in {401, 402}
                }
            ),
        }

    def _cost_totals(self, events):
        calls = {}
        legacy = []
        for event in events:
            call_id = event.get("call_id")
            if call_id is None:
                if event.get("event") in {"SUCCESS", "FAILED"}:
                    legacy.append(event)
            else:
                calls.setdefault(call_id, {})[event.get("event")] = event
        completed = legacy + [
            state.get("SUCCESS") or state.get("FAILED")
            for state in calls.values()
            if state.get("SUCCESS") is not None or state.get("FAILED") is not None
        ]
        input_cost = output_cost = 0.0
        by_model = {}
        for event in completed:
            model = event.get("model")
            prices = self.model_prices.get(model) if self.model_prices else None
            if prices is None:
                if self.input_price_per_million is None or self.output_price_per_million is None:
                    raise ValueError(f"missing pricing for model: {model}")
                prices = {
                    "input": self.input_price_per_million,
                    "output": self.output_price_per_million,
                }
            input_tokens = int(event.get("input_tokens", 0) or 0)
            output_tokens = int(event.get("output_tokens", 0) or 0)
            current = by_model.setdefault(model or "legacy", {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            current["input_tokens"] += input_tokens
            current["output_tokens"] += output_tokens
            current["cost_usd"] += (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
            input_cost += input_tokens * prices["input"] / 1_000_000
            output_cost += output_tokens * prices["output"] / 1_000_000
        return input_cost, output_cost, by_model

    def summary(self):
        summary = self._summarize(self.events())
        if self.max_normalized_u is not None:
            input_cost, output_cost, by_model = self._cost_totals(self.events())
            unit_usd = self.normalized_unit_usd
            cost_usd = input_cost + output_cost
            normalized_u = cost_usd / unit_usd
            summary.update(
                {
                    "input_price_per_million_usd": self.input_price_per_million,
                    "output_price_per_million_usd": self.output_price_per_million,
                    "model_prices_usd": self.model_prices or None,
                    "model_costs_usd": by_model,
                    "normalized_unit_usd": unit_usd,
                    "normalized_cost_usd": cost_usd,
                    "normalized_u": normalized_u,
                    "max_normalized_u": self.max_normalized_u,
                    "normalized_budget_reached": normalized_u >= self.max_normalized_u,
                    "cached_input_normalization": "NONCACHED_INPUT_PRICE",
                    "reasoning_token_accounting": "INCLUDED_IN_OUTPUT_TOKENS",
                }
            )
        return summary

    def verify_input_budget(self, value, *, channel):
        estimate = _estimate_tokens(value)
        if estimate + self.input_estimate_margin > self.max_input_tokens:
            raise InputBudgetExceeded(
                f"SE_SHARED_INPUT_BUDGET_EXHAUSTED:{channel}:"
                f"estimate={estimate}+margin={self.input_estimate_margin}/limit={self.max_input_tokens}"
            )
        return estimate

    def reserve(self, *, channel, label, requests=1, estimated_input_tokens=None):
        requests = int(requests)
        if requests < 1:
            raise ValueError("reserved requests must be positive")
        with self._lock:
            events = self.events()
            summary = self._summarize(events)
            if summary["fatal_provider_statuses"]:
                raise ProviderExecutionPaused(
                    f"SE_SHARED_PROVIDER_EXECUTION_PAUSED:HTTP_{summary['fatal_provider_statuses'][0]}"
                )
            if summary["unknown_usage_failures"]:
                raise ProviderExecutionPaused("SE_SHARED_PROVIDER_EXECUTION_PAUSED:UNKNOWN_USAGE_FAILURE")
            if self.max_normalized_u is not None:
                input_cost, output_cost, _ = self._cost_totals(events)
                unit_usd = self.normalized_unit_usd
                cost_usd = input_cost + output_cost
                if cost_usd / unit_usd >= self.max_normalized_u:
                    raise BudgetExhausted(
                        f"SE_SHARED_NORMALIZED_API_BUDGET_EXHAUSTED:{cost_usd / unit_usd:.9f}/{self.max_normalized_u:.9f}U"
                    )
            if summary["requests"] + requests > self.max_requests:
                raise BudgetExhausted(
                    f"SE_SHARED_API_REQUEST_BUDGET_EXHAUSTED:{summary['requests']}+{requests}/{self.max_requests}"
                )
            call_id = uuid.uuid4().hex
            self._append_unlocked(
                {
                    "schema": SCHEMA,
                    "event": "RESERVED",
                    "call_id": call_id,
                    "channel": channel,
                    "label": label,
                    "timestamp_ns": time.time_ns(),
                    "requests": requests,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_input_tokens": estimated_input_tokens,
                    "usage_reported": False,
                    "elapsed_seconds": None,
                    "error_type": None,
                    "error_digest": None,
                }
            )
            return call_id

    def _append_unlocked(self, payload):
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def record(self, event, *, call_id, channel, label, model=None, reserved_requests=1, usage=None, error=None, elapsed_seconds=None):
        requests, input_tokens, output_tokens = _usage_fields(usage)
        if event == "FAILED" and usage is None:
            requests = int(reserved_requests)
        payload = {
            "schema": SCHEMA,
            "event": event,
            "call_id": call_id,
            "channel": channel,
            "label": label,
            "model": model,
            "timestamp_ns": time.time_ns(),
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usage_reported": usage is not None,
            "elapsed_seconds": elapsed_seconds,
            "error_type": type(error).__name__ if error is not None else None,
            "error_digest": hashlib.sha256(str(error).encode("utf-8")).hexdigest() if error is not None else None,
            "http_status": _http_status(error) if error is not None else None,
        }
        with self._lock:
            self._append_unlocked(payload)


_INSTALLED = False
_LEDGER = None
_SYNC_SEMAPHORE = None
_ASYNC_SEMAPHORES = {}


def install_api_accounting(ledger_path, max_requests, concurrency=2, agent_max_turns=3, *, input_price_per_million=None, output_price_per_million=None, model_prices=None, normalized_unit_usd=None, max_normalized_u=None, max_input_tokens=48000, max_output_tokens=20000, input_estimate_margin=4096):
    global _INSTALLED, _LEDGER, _SYNC_SEMAPHORE
    if _INSTALLED:
        return _LEDGER
    ledger = ApiLedger(
        ledger_path,
        max_requests,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        model_prices=model_prices,
        normalized_unit_usd=normalized_unit_usd,
        max_normalized_u=max_normalized_u,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        input_estimate_margin=input_estimate_margin,
    )
    _LEDGER = ledger
    concurrency = int(concurrency)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    agent_max_turns = int(agent_max_turns)
    if agent_max_turns < 1:
        raise ValueError("agent_max_turns must be positive")
    _SYNC_SEMAPHORE = threading.BoundedSemaphore(concurrency)

    import openai
    original_openai = openai.OpenAI

    luna_config_path = os.environ.get("E2_SHARED_LUNA_API_CONFIG", "").strip()
    luna_credentials = (
        _load_route_credentials(luna_config_path) if luna_config_path else None
    )
    luna_wire_model = os.environ.get("E2_SHARED_LUNA_WIRE_MODEL", LUNA_WIRE_MODEL).strip()
    if not luna_wire_model:
        raise ValueError("E2_SHARED_LUNA_WIRE_MODEL must be non-empty")

    class ResponsesProxy:
        def __init__(self, inner, luna_inner=None):
            self._inner = inner
            self._luna_inner = luna_inner

        def create(self, *args, **kwargs):
            label = str(kwargs.get("model", "unknown"))
            model = _canonical_model(label)
            inner = self._luna_inner if model == LUNA_MODEL and self._luna_inner is not None else self._inner
            if model == LUNA_MODEL and self._luna_inner is not None:
                kwargs["model"] = luna_wire_model
            requested_output = int(kwargs.get("max_output_tokens", ledger.max_output_tokens))
            if requested_output > ledger.max_output_tokens:
                raise ValueError(
                    f"SE_SHARED_OUTPUT_LIMIT_EXCEEDED:{requested_output}/{ledger.max_output_tokens}"
                )
            kwargs["max_output_tokens"] = requested_output
            input_estimate = ledger.verify_input_budget(
                {
                    "instructions": kwargs.get("instructions"),
                    "input": kwargs.get("input", args[0] if args else None),
                    "tools": kwargs.get("tools"),
                },
                channel="responses",
            )
            started = time.monotonic()
            with _SYNC_SEMAPHORE:
                call_id = ledger.reserve(
                    channel="responses",
                    label=label,
                    requests=1,
                    estimated_input_tokens=input_estimate,
                )
                try:
                    response = inner.responses.create(*args, **kwargs)
                except Exception as error:
                    ledger.record("FAILED", call_id=call_id, channel="responses", label=label, model=model, error=error, elapsed_seconds=time.monotonic() - started)
                    raise
            ledger.record("SUCCESS", call_id=call_id, channel="responses", label=label, model=model, usage=getattr(response, "usage", None), elapsed_seconds=time.monotonic() - started)
            return response

        def __getattr__(self, name):
            return getattr(self._inner, name)

    class BudgetedOpenAI:
        def __init__(self, *args, **kwargs):
            import httpx

            kwargs.setdefault("max_retries", 0)
            kwargs.setdefault(
                "timeout",
                httpx.Timeout(
                    900.0,
                    connect=30.0,
                    read=900.0,
                    write=60.0,
                    pool=30.0,
                ),
            )
            from relay_retry import sync_client
            if 'http_client' not in kwargs:
                kwargs['http_client'] = sync_client(
                    ledger.path.with_name('EXTERNAL_RELAY_EVENTS.jsonl'), kwargs['timeout'])
            self._inner = original_openai(*args, **kwargs)
            self._inner_luna = None
            if luna_credentials is not None:
                luna_kwargs = dict(kwargs)
                luna_kwargs["api_key"] = luna_credentials["api_key"]
                luna_kwargs["base_url"] = luna_credentials["base_url"].rstrip("/")
                self._inner_luna = original_openai(**luna_kwargs)
            self.responses = ResponsesProxy(self._inner, self._inner_luna)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    openai.OpenAI = BudgetedOpenAI

    original_async_openai = openai.AsyncOpenAI

    class AsyncResponsesProxy:
        def __init__(self, inner, luna_inner=None):
            self._inner = inner
            self._luna_inner = luna_inner

        async def create(self, *args, **kwargs):
            label = str(kwargs.get("model", "unknown"))
            model = _canonical_model(label)
            inner = self._luna_inner if model == LUNA_MODEL and self._luna_inner is not None else self._inner
            if model == LUNA_MODEL and self._luna_inner is not None:
                kwargs["model"] = luna_wire_model
            return await inner.responses.create(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner.responses, name)

    class BudgetedAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            import httpx

            kwargs.setdefault("max_retries", 0)
            kwargs.setdefault(
                "timeout",
                httpx.Timeout(
                    900.0,
                    connect=30.0,
                    read=900.0,
                    write=60.0,
                    pool=30.0,
                ),
            )
            from relay_retry import async_client
            if 'http_client' not in kwargs:
                kwargs['http_client'] = async_client(
                    ledger.path.with_name('EXTERNAL_RELAY_EVENTS.jsonl'), kwargs['timeout'])
            self._inner = original_async_openai(*args, **kwargs)
            self._inner_luna = None
            if luna_credentials is not None:
                luna_kwargs = dict(kwargs)
                luna_kwargs["api_key"] = luna_credentials["api_key"]
                luna_kwargs["base_url"] = luna_credentials["base_url"].rstrip("/")
                self._inner_luna = original_async_openai(**luna_kwargs)
            self.responses = AsyncResponsesProxy(self._inner, self._inner_luna)

        async def close(self):
            await self._inner.close()
            if self._inner_luna is not None:
                await self._inner_luna.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    openai.AsyncOpenAI = BudgetedAsyncOpenAI

    import agents
    original_runner = agents.Runner.run

    async def budgeted_runner(cls, *args, **kwargs):
        loop = asyncio.get_running_loop()
        semaphore = _ASYNC_SEMAPHORES.setdefault(id(loop), asyncio.Semaphore(concurrency))
        label = getattr(args[0], "name", "agent") if args else "agent"
        requested_turns = int(kwargs.get("max_turns", agent_max_turns))
        reserved_turns = min(requested_turns, agent_max_turns)
        kwargs["max_turns"] = reserved_turns
        agent = args[0] if args else kwargs.get("starting_agent")
        model = getattr(agent, "model", None)
        agent_settings = getattr(agent, "model_settings", None)
        if getattr(agent_settings, "max_tokens", None) != ledger.max_output_tokens:
            raise ValueError("SE_SHARED_AGENT_OUTPUT_LIMIT_NOT_FROZEN")
        agent_input = kwargs.get("input", args[1] if len(args) > 1 else None)
        input_estimate = ledger.verify_input_budget(
            {
                "agent_name": getattr(agent, "name", None),
                "instructions": getattr(agent, "instructions", None),
                "input": agent_input,
                "output_type": getattr(agent, "output_type", None),
                "tools": getattr(agent, "tools", None),
            },
            channel="agents",
        )
        started = time.monotonic()
        async with semaphore:
            call_id = ledger.reserve(
                channel="agents",
                label=str(label),
                requests=reserved_turns,
                estimated_input_tokens=input_estimate,
            )
            try:
                result = await original_runner(*args, **kwargs)
            except Exception as error:
                ledger.record("FAILED", call_id=call_id, channel="agents", label=str(label), model=model, reserved_requests=reserved_turns, error=error, elapsed_seconds=time.monotonic() - started)
                raise
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        ledger.record("SUCCESS", call_id=call_id, channel="agents", label=str(label), model=model, reserved_requests=reserved_turns, usage=usage, elapsed_seconds=time.monotonic() - started)
        return result

    agents.Runner.run = classmethod(budgeted_runner)
    if hasattr(agents, "set_tracing_disabled"):
        agents.set_tracing_disabled(True)
    _INSTALLED = True
    return ledger
