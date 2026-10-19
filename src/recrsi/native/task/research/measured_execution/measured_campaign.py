"""Existing campaign round owner, manifest consumer and conservative actual ledger."""
from contextlib import contextmanager
import json,time
from pathlib import Path
from unittest.mock import patch
from measured_selector import save_once

def install_conservative_ledger():
    from main_api_ledger import MainApiLedger
    if getattr(MainApiLedger,'_current_resource_policy',False):
        raise RuntimeError('Install measured hard accounting before the old failed-call filter')
    # Retain the original common ledger's known failures, in-flight reservations,
    # fatal statuses and UNKNOWN holds in admission. No refund or larger cap.
    MainApiLedger._current_resource_policy=True

@contextmanager
def campaign_scope(composition,manifest,manifest_sha,transport_module):
    from recclaw_core.research_line import provider
    from campaign_manifest_consumer import ManifestCampaignConsumer
    from manifest_round_owner import ManifestRoundOwner
    from campaign_round_owner import PendingRound
    from native_provider_capture import capture_native_transport
    class ReservedConsumer(ManifestCampaignConsumer):
        def _consume(self,owner,budget_key,result,native_outputs):
            total=owner.proposal_output_token_ceiling_total_per_context
            if total is None:raise ValueError('Missing finite original role ceiling')
            with patch.object(owner,'proposal_output_token_ceiling_total_per_context',total-3000):
                return super()._consume(owner,budget_key,result,native_outputs)
    consumer=ReservedConsumer.from_manifest(manifest,expected_sha256=manifest_sha)
    campaign=composition.campaign
    owner=ManifestRoundOwner(campaign,consumer,provider);owner.transport_module=transport_module
    cls=type(campaign);original=cls.run_round;inside=False
    def run(current,*args,**kw):
        nonlocal inside
        if current is not campaign or inside:return original(current,*args,**kw)
        while True:
            inside=True
            try:result=owner.run_once()
            finally:inside=False
            if not isinstance(result,PendingRound):return result
            root=consumer.bridge.root/'declaration_completions'
            states=[json.loads(p.read_bytes()) for p in root.glob('*/STATE.json')]
            terminal=[s for s in states if s.get('status') in {'ACCOUNT_PAUSED','DISPATCH_UNSETTLED'}]
            terminal += [s for s in states if s.get('status')=='ZERO_SEND_LEDGER_PENDING']
            if terminal:
                # Preserve same native STARTED boundary for evidence-led coordination.
                # No auto account probing, blind dispatch or false legal budget completion.
                save_once(campaign.root/'OWNER_EVIDENCE_REQUIRED.json',dict(states=terminal,
                    pending=result.record_path,automatic_retry=False))
                raise SystemExit('Saved pending round requires dispatch/account evidence reconciliation')
            retry=max([float(s.get('retry_not_before',0)) for s in states]+[0])
            time.sleep(min(60,max(5,retry-time.time())))
    # All timers remain raw. With no active-wall limit, excluding overlapping
    # per-request outages is unnecessary and could misstate aggregate compute.
    with patch.object(cls,'run_round',run):
        yield
