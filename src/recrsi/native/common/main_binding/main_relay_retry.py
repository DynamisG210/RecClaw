"""Existing empty-relay retry semantics plus the prepared OE run-clock consumer.

No new error classification or retry limit. The identical HTTP request is retried,
and the run-clock exclusion is settled before its next deadline check.
"""
import asyncio
import hashlib
from pathlib import Path
import time

import httpx
from main_relay_source import confirmed_empty_relay_fault, record_fault


class MainRelayTransport(httpx.AsyncBaseTransport):
    def __init__(self, log_path, resources, call_id, *, inner=None, sleep=asyncio.sleep):
        self.log_path, self.resources, self.call_id = log_path, resources, call_id
        self.inner = inner if inner is not None else httpx.AsyncHTTPTransport(retries=0)
        self.sleep = sleep

    async def handle_async_request(self, request):
        await request.aread()
        attempt = 0
        while True:
            started = time.monotonic()
            started_ns = self.resources.clock.now_ns()
            response = await self.inner.handle_async_request(request)
            if response.status_code not in {400,429,500,502,503,504}:
                return response
            body = await response.aread()
            if not confirmed_empty_relay_fault(response.status_code, body):
                return response
            attempt += 1
            record_fault(self.log_path, request, response, body, attempt, time.monotonic()-started)
            owner = self.call_id()
            evidence = dict(physical_log=str(self.log_path), http_status=response.status_code,
                attempt=attempt, error_body_sha256=hashlib.sha256(body).hexdigest(),
                request_body_sha256=hashlib.sha256(request.content).hexdigest(), call_id=owner)
            self.resources.pause_empty_relay(owner, started_ns, evidence)
            await response.aclose()
            await self.sleep(min(60,5*attempt))
            # Do not check deadline first and then try to refund an outage after
            # a sticky stop. New normal activity resumes the clock here.
            self.resources.clock.resume(owner=owner)
            from resource_stop import SearchResourceStop
            try:
                self.resources.require_capacity()
            except SearchResourceStop as stop:
                stop.confirmed_external_empty_evidence = evidence
                raise

    async def aclose(self):
        await self.inner.aclose()


def async_client(log_path, timeout, *, resources, call_id):
    return httpx.AsyncClient(transport=MainRelayTransport(Path(log_path), resources, call_id),
                            timeout=timeout, follow_redirects=True)
