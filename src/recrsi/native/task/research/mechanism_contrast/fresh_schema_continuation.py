"""Same settled-schema continuation policy, adopted at a fresh campaign boundary."""
import fcntl,json
from pathlib import Path

def install():
    import validation_boundary,campaign_declaration_consumer,non_declaration_consumer_1215
    from settled_schema_continuation_v2_1215 import settled_trace
    original=validation_boundary.ContractBoundary
    class FreshSettledBoundary(original):
        def __init__(self,output):
            super().__init__(output);self.journal=Path(output)/'SCHEMA_CONTINUATION_FAILURES.jsonl'
        def require_new_request(self):
            super().require_new_request()
            if self.journal.exists():
                from resource_stop import SearchResourceStop
                current=json.loads((self.path.parent/'campaign/CAMPAIGN_STATE.json').read_bytes())['state']['next_round_index']
                for row in map(json.loads,self.journal.read_text().splitlines()):
                    if row['round_index']>=current:raise SearchResourceStop('SCHEMA_CONTINUATION_NATIVE_FAILURE_NOT_SETTLED')
                    settled_trace(self.path.parent,row['round_index'],row['detail']['logical_call_id'])
        def record(self,stage,detail):
            if stage!='provider_schema':return super().record(stage,detail)
            assert detail['failure_class']=='SCHEMA_VALIDATION_FAILURE'
            current=json.loads((self.path.parent/'campaign/CAMPAIGN_STATE.json').read_bytes())['state']['next_round_index']
            with self.journal.open('a',encoding='utf8') as f:
                fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(dict(stage=stage,detail=detail,round_index=current,policy='settled-schema-continuation-v2-fresh',failure_retained=True,new_round_refund=False),ensure_ascii=True)+'\n');f.flush()
    validation_boundary.ContractBoundary=FreshSettledBoundary
    campaign_declaration_consumer.CampaignDeclarationConsumer.install=non_declaration_consumer_1215.CampaignDeclarationConsumer.install
