"""Prospective common continuation after a settled native schema failure."""
import hashlib,json,fcntl,pickle
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def settled_trace(output,round_index,logical_call_id):
    record=pickle.loads((Path(output)/'campaign'/f'ROUND_{round_index:02d}_CHECKPOINT.pkl').read_bytes())
    matches=[t for t in record.result.provider_traces if t.get('logical_call_id')==logical_call_id]
    assert len(matches)==1
    t=matches[0]
    assert (t['receipt']['status']=='FAILED' and t['failure']['failure_class']=='SCHEMA_VALIDATION_FAILURE') or t.get('consumer_status')=='DECLARATION_COMPOSITE_VALIDATED'
    return t
def boundary_class(original, approved_stop_sha256):
    class SettledSchemaBoundary(original):
        def __init__(self,output):
            super().__init__(output)
            self.hard_path=Path(output)/'SCHEMA_CONTINUATION_NONSCHEMA_STOP.json'
            self.journal=Path(output)/'SCHEMA_CONTINUATION_FAILURES.jsonl'
        def require_new_request(self):
            from resource_stop import SearchResourceStop
            if self.hard_path.exists():
                raise SearchResourceStop('SCHEMA_CONTINUATION_NONSCHEMA_CONTRACT_STOP')
            if not self.path.exists() or sha(self.path)!=approved_stop_sha256:
                raise SearchResourceStop('SCHEMA_CONTINUATION_ORIGINAL_STOP_IDENTITY_CHANGED')
            if self.journal.exists():
                current=json.loads((self.path.parent/'campaign/CAMPAIGN_STATE.json').read_bytes())['state']['next_round_index']
                for row in map(json.loads,self.journal.read_text().splitlines()):
                    if row['round_index']>=current:
                        raise SearchResourceStop('SCHEMA_CONTINUATION_NATIVE_FAILURE_NOT_SETTLED')
                    settled_trace(self.path.parent,row['round_index'],row['detail']['logical_call_id'])
        def record(self,stage,detail):
            if stage!='provider_schema':
                hard=original(self.path.parent)
                hard.path=self.hard_path
                return hard.record(stage,detail)
            assert detail['failure_class']=='SCHEMA_VALIDATION_FAILURE'
            round_index=json.loads((self.path.parent/'campaign/CAMPAIGN_STATE.json').read_bytes())['state']['next_round_index']
            with self.journal.open('a',encoding='utf8') as f:
                fcntl.flock(f,fcntl.LOCK_EX)
                f.write(json.dumps(dict(stage=stage,detail=detail,round_index=round_index,policy='settled-schema-continuation-v2',failure_retained=True,new_round_refund=False),ensure_ascii=True)+'\n')
                f.flush()
    return SettledSchemaBoundary

def install(output,certificate_path):
    import validation_boundary,campaign_declaration_consumer,non_declaration_consumer_1215
    cert=json.loads(Path(certificate_path).read_bytes())
    assert cert['status']=='FROZEN_COLD_OWNER_ADOPTION' and cert['output']==str(output)
    manifest=Path(cert['manifest']['path'])
    assert sha(manifest)==cert['manifest']['sha256']
    policy=json.loads(manifest.read_bytes())
    assert policy['policy_revision']=='settled-schema-continuation-v2'
    for p,d in policy['implementation_pins'].items():assert sha(p)==d,p
    assert sha(Path(output)/'CONTRACT_STOP.json')==cert['original_stop']['sha256']
    validation_boundary.ContractBoundary=boundary_class(validation_boundary.ContractBoundary,cert['original_stop']['sha256'])
    campaign_declaration_consumer.CampaignDeclarationConsumer.install=non_declaration_consumer_1215.CampaignDeclarationConsumer.install

