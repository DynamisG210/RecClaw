"""Retry user-confirmed no-output/no-charge relay faults at the HTTP boundary.

The native agent call stays alive: previous tool/model/worker stages are not replayed.
Unknown transport errors, partial output and reported usage are never exempted.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import httpx

_LOG_LOCK = threading.Lock()


def confirmed_empty_relay_fault(status, body):
    if status not in {400, 429, 500, 502, 503, 504}:
        return False
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None

    def has_output(value):
        if isinstance(value, dict):
            if any(value.get(key) for key in ("usage", "output", "choices", "output_text", "partial_output")):
                return True
            return any(has_output(child) for child in value.values())
        return isinstance(value, list) and any(has_output(child) for child in value)

    if has_output(payload) or "data: " in text:
        return False
    lowered = text.lower()
    if status == 502:
        return any(marker in lowered for marker in ('502 bad gateway', 'upstream connect error',
            'upstream error: do request failed', 'http/2 protocol_error'))
    if status == 504:
        return (isinstance(payload, dict) and bool(payload.get("error"))) or any(
            marker in lowered for marker in ("gateway timeout", "gateway time-out", "504 gateway"))
    if status == 503:
        return any(marker in lowered for marker in ("servers overloaded", "service temporarily unavailable"))
    if status == 400:
        return "unknown provider for model gpt-5.6-terra" in lowered
    if status == 429:
        return (any(f"all credentials for model {model} are cooling down" in lowered
                    for model in ("gpt-5.6-terra", "gpt-5.6-luna"))
                or (isinstance(payload, dict) and isinstance(payload.get('error'), dict)
                    and payload['error'].get('code') == 'session_affinity_cooldown'
                    and 'the credential bound to this session is rate limited' in lowered
                    and 'the session was held rather than moved to preserve prompt caching' in lowered))
    return "upstream error: do request failed" in lowered or "http/2 protocol_error" in lowered


def record_fault(path, request, response, body, attempt, elapsed):
    # Never log authorization headers, prompt bodies, raw provider bodies or URLs.
    event = {
        "event": "CONFIRMED_EXTERNAL_EMPTY_RETRY",
        "http_status": response.status_code, "attempt": attempt,
        "request_body_sha256": hashlib.sha256(request.content).hexdigest(),
        "error_body_sha256": hashlib.sha256(body).hexdigest(),
        "timestamp_ns": time.time_ns(), "elapsed_seconds": elapsed,
        "scientific_tokens": 0, "scientific_requests": 0,
        "basis": "user-confirmed relay error semantics; no output or usage present",
    }
    with _LOG_LOCK, Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class RelayRetryTransport(httpx.BaseTransport):
    def __init__(self, log_path, *, inner=None, sleep=time.sleep):
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0)
        self.log_path = log_path
        self.sleep = sleep

    def handle_request(self, request):
        request.read()  # SDK JSON request is replayable; never rebuild its content.
        attempt = 0
        while True:
            started = time.monotonic()
            response = self.inner.handle_request(request)
            if response.status_code not in {400, 429, 500, 502, 503, 504}:
                return response
            body = response.read()
            if not confirmed_empty_relay_fault(response.status_code, body):
                return response
            attempt += 1
            record_fault(self.log_path, request, response, body, attempt, time.monotonic() - started)
            response.close()
            self.sleep(min(60, 5 * attempt))

    def close(self):
        self.inner.close()


class AsyncRelayRetryTransport(httpx.AsyncBaseTransport):
    def __init__(self, log_path, *, inner=None, sleep=asyncio.sleep):
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(retries=0)
        self.log_path = log_path
        self.sleep = sleep

    async def handle_async_request(self, request):
        await request.aread()
        attempt = 0
        while True:
            started = time.monotonic()
            response = await self.inner.handle_async_request(request)
            if response.status_code not in {400, 429, 500, 502, 503, 504}:
                return response
            body = await response.aread()
            if not confirmed_empty_relay_fault(response.status_code, body):
                return response
            attempt += 1
            record_fault(self.log_path, request, response, body, attempt, time.monotonic() - started)
            await response.aclose()
            await self.sleep(min(60, 5 * attempt))

    async def aclose(self):
        await self.inner.aclose()


def sync_client(log_path, timeout):
    return httpx.Client(transport=RelayRetryTransport(log_path), timeout=timeout, follow_redirects=True)


def async_client(log_path, timeout):
    return httpx.AsyncClient(transport=AsyncRelayRetryTransport(log_path), timeout=timeout, follow_redirects=True)
