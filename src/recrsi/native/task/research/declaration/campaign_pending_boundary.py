"""Prospective pending boundary for the observed native producer Exception catch.
No original campaign or entrypoint imports this module. Pending is control flow,
not a ProducerOutcome, a successful native receipt, or a scientific opportunity.
"""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path
from campaign_declaration_consumer import CampaignDeclarationConsumer
from declaration_provider_v3 import CompletionPending
from declaration_provider_v2 import digest
from se_shared_api_ledger import ProviderExecutionPaused

class DeclarationCampaignPending(BaseException):
    def __init__(self,path,sha256):
        self.path=str(path);self.sha256=sha256
        super().__init__('Declaration infrastructure pending; preserve started round: '+self.path)

def load_pending_trace(path, *, expected_sha256, run_identity, session_id):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Pending receipt changed')
    record=json.loads(raw)
    if record['run_identity']!=run_identity or record['session_id']!=session_id:
        raise ValueError('Pending receipt belongs to another campaign')
    if record['schema']!='GRAPH_DECLARATION_CAMPAIGN_PENDING_V1':raise ValueError('Unknown pending receipt')
    return record['native_trace']

class PendingCampaignConsumer(CampaignDeclarationConsumer):
    @contextmanager
    def install(self,module):
        with super().install(module):
            finalize=module._finalize_provider_result
            def with_pending(owner,**kwargs):
                try:return finalize(owner,**kwargs)
                except (CompletionPending,ProviderExecutionPaused) as error:
                    result=kwargs['result']
                    if kwargs['kind']!='research_producer' or result.call is not None:
                        raise
                    if owner.session_id!=self.bridge.frozen_session_id:raise ValueError('Pending owner differs')
                    trace=module._call_trace(kind=kwargs['kind'],config_identity=owner.config_identity,
                        logical_call_id=kwargs['logical_call_id'],session_id=owner.session_id,
                        prompt=kwargs['prompt'],result=result)
                    # Native trace remains FAILED, with original sealed-request identity.
                    identity=dict(run_identity=self.bridge.completion.settings['run_identity'],
                        logical_call_id=result.sealed_request['logical_call_id'],session_id=owner.session_id)
                    attempt=self.bridge.root/'declaration_completions'/digest(identity)
                    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [attempt/'STATE.json',attempt/'REQUEST.json'] if p.exists()}
                    record=dict(schema='GRAPH_DECLARATION_CAMPAIGN_PENDING_V1',
                        run_identity=identity['run_identity'],session_id=owner.session_id,
                        logical_call_id=identity['logical_call_id'],budget_key=kwargs['budget_key'],
                        cause_class=type(error).__name__,native_trace=trace,completion_snapshot=pins,
                        scientific_outcome_created=False,advance_round=False,refund_opportunity=False,
                        automatic_account_probe=False)
                    blob=(json.dumps(record,sort_keys=True,ensure_ascii=True,indent=2)+'\n').encode()
                    sha=hashlib.sha256(blob).hexdigest()
                    path=attempt/'campaign_pending'/(sha+'.json');path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():
                        if path.read_bytes()!=blob:raise ValueError('Pending receipt identity collision')
                    else:
                        with path.open('xb') as f:f.write(blob)
                    raise DeclarationCampaignPending(path,sha) from error
            module._finalize_provider_result=with_pending
            try:yield self
            finally:module._finalize_provider_result=finalize
