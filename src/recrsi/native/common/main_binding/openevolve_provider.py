"""Prepared MAIN OpenEvolve LLM ABI on the common endpoint2 Responses path.

For the prepared serial (parallel_evaluations=1) consumer. Native ensemble,
prompts, diff generation and archive selection stay upstream-owned. This copy
implements common API/active/GPU/genuine admission with the prepared evaluator.
Supervisor-recorded external pauses, interrupted-job recovery and the six native
launch configurations still need integration before formal MAIN execution.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time


@lru_cache(maxsize=1)
def _settings_and_ledger(config_path):
    settings = json.loads(Path(config_path).read_text())
    sys.path.insert(0, settings["ledger_module_dir"])
    from main_api_ledger import MainApiLedger
    ledger = MainApiLedger(settings["api_ledger"], **settings["api_limits"])
    return settings, ledger


def endpoint2_client(model_config):
    """Picklable upstream init_client: resolve trusted configuration in child."""
    settings, ledger = _settings_and_ledger(os.environ["E1_RQ1_PROVIDER_CONFIG"])
    return Endpoint2LLM(model_config, settings, ledger)


class Endpoint2LLM:
    channel = "openevolve:generate"

    def __init__(self, config, settings, ledger):
        if config.name not in {"gpt-5.6-luna", "gpt-5.6-terra"}:
            raise ValueError("The common comparison model pool is Terra/Luna")
        if config.reasoning_effort != "medium" or config.max_tokens != ledger.max_output_tokens:
            raise ValueError("OpenEvolve generation profile differs from common inputs")
        self.model = config.name
        self.system_message = config.system_message
        self.settings = settings
        self.ledger = ledger
        from main_resources import MainResources
        self.resources = MainResources(settings['resource_root'], settings['run_identity'],
                                       settings['active_wall_path'], ledger)
        self.clock = self.resources.clock

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context(self.system_message,
                    [{"role": "user", "content": prompt}], **kwargs)

    async def generate_with_context(self, system_message, messages, **kwargs):
        response = await self.generate_response(system_message, messages, **kwargs)
        return response.output_text

    async def generate_response(self, system_message, messages, **kwargs):
        """Return this call's settled response; no shared last-response state."""
        import httpx
        from openai import AsyncOpenAI
        from main_relay_retry import async_client
        from se_shared_api_ledger import (
            BudgetExhausted, InputBudgetExceeded, ProviderExecutionPaused,
            _load_route_credentials)
        from resource_stop import SearchResourceStop

        if kwargs.get("max_tokens", self.ledger.max_output_tokens) != self.ledger.max_output_tokens:
            raise ValueError("Cannot change the frozen output allowance")
        if kwargs.get("reasoning_effort", "medium") != "medium":
            raise ValueError("Cannot change the frozen reasoning effort")
        # No temperature/top-p or unsupported Responses seed. The configured
        # native random_seed still controls upstream ensemble/search sampling.
        payload = dict(model=self.model, instructions=system_message, input=messages,
                       max_output_tokens=self.ledger.max_output_tokens,
                       reasoning={"effort": "medium"}, store=False, stream=False)
        channel = self.channel
        label = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.resources.require_capacity()
        route_config = self.settings.get('model_api_configs', {}).get(self.model, self.settings['api_config'])
        credentials = _load_route_credentials(route_config)
        timeout = httpx.Timeout(900, connect=30, read=900, write=60, pool=30)
        # Client setup occurs before reservation; setup errors are not requests.
        async with AsyncOpenAI(api_key=credentials["api_key"],
                base_url=credentials["base_url"].rstrip("/"), max_retries=0,
                timeout=timeout, http_client=async_client(
                    Path(self.settings["api_ledger"]).with_name("EXTERNAL_RELAY_EVENTS.jsonl"),
                    timeout, resources=self.resources, call_id=lambda: call_id)) as client:
            try:
                self.resources.require_capacity()
                estimate = self.ledger.verify_input_budget(payload, channel=channel)
                call_id = self.ledger.reserve(channel=channel, label=label,
                                             model=self.model,
                                             estimated_input_tokens=estimate)
            except (BudgetExhausted, InputBudgetExceeded, ProviderExecutionPaused) as error:
                raise SearchResourceStop(str(error)) from error
            started = time.monotonic()
            usage = None
            try:
                saved = Path(self.settings['resource_root'])/'provider_outputs'/call_id
                saved.mkdir(parents=True)
                (saved/'REQUEST.json').write_text(json.dumps(payload, sort_keys=True))
                (saved/'ROUTE.json').write_text(json.dumps(dict(model=self.model,
                    credential_config=route_config, url=credentials['base_url'].rstrip('/')+'/responses',
                    request_sha256=label)))
                response = await client.responses.create(**payload)
                raw_usage = response.usage
                if raw_usage is None or any(type(getattr(raw_usage, key, None)) is not int
                        or getattr(raw_usage, key) < 0 for key in ("input_tokens", "output_tokens")):
                    raise RuntimeError("OpenEvolve response has unresolved provider usage")
                usage = raw_usage
                (saved/'RESPONSE.json').write_text(response.model_dump_json())
            except BaseException as error:
                empty = getattr(error, 'confirmed_external_empty_evidence', None)
                if isinstance(error, SearchResourceStop) and empty is not None:
                    self.ledger.record_external_empty_stop(call_id=call_id,
                        model=self.model, evidence=empty)
                else:
                    self.ledger.record("FAILED", call_id=call_id, channel=channel,
                        label=label, model=self.model, usage=usage, error=error,
                        elapsed_seconds=time.monotonic() - started)
                # Preserve unknown/paid failures; never turn provider outages
                # into candidate fitness or repeat completed research stages.
                raise SearchResourceStop("OpenEvolve provider interrupted; inspect preserved ledger") from error
            self.ledger.record("SUCCESS", call_id=call_id, channel=channel,
                label=label, model=self.model, usage=usage,
                elapsed_seconds=time.monotonic() - started)
            if not self.resources.admit_output():
                # Preserve returned text as late evidence, but never feed it to
                # upstream generation/selection or call the model again here.
                late_path = Path(self.settings['api_ledger']).parent / 'late_provider_outputs'
                late_path.mkdir(exist_ok=True)
                (late_path / f'{call_id}.json').write_text(json.dumps(dict(
                    call_id=call_id, model=self.model, output_text=response.output_text,
                    admitted=False, usage=dict(input_tokens=usage.input_tokens,
                                               output_tokens=usage.output_tokens))))
                raise SearchResourceStop('MAIN late provider output retained but not admitted')
            if not response.output_text:
                raise ValueError("Paid OpenEvolve response contains no candidate text")
            return response
