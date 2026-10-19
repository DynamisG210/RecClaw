"""Prospective owner boundary; preserve native STARTED/state while declaration waits."""
from dataclasses import dataclass
from pathlib import Path
import fcntl,hashlib,json
from campaign_pending_boundary import DeclarationCampaignPending,load_pending_trace
from recclaw_core.research_line.campaign import _atomic_write,_write_once

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True)
class PendingRound:
    record_path: str
    record_sha256: str
    round_index: int
    status: str = 'DECLARATION_PENDING'

class DeclarationRoundOwner:
    def __init__(self,campaign,consumer,provider_module):
        self.campaign=campaign;self.consumer=consumer;self.provider_module=provider_module
        self.run_identity=consumer.bridge.completion.settings['run_identity']
        self.session_id=consumer.bridge.frozen_session_id

    def run_once(self):
        c=self.campaign
        if c.producer.session_id!=self.session_id:raise ValueError('Campaign session differs')
        index=c.state.next_round_index
        journal=c.root/'declaration_pending'/c._round_generation_prefix(index)
        journal.mkdir(parents=True,exist_ok=True)
        with (journal/'OWNER.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            head=journal/'HEAD.json'
            if head.exists():
                ref=json.loads(head.read_bytes());p=journal/ref['record_file']
                if sha(p)!=ref['record_sha256']:raise ValueError('Pending owner record changed')
                prior=json.loads(p.read_bytes())
                if prior['status']=='DECLARATION_PENDING':
                    if (prior['run_identity']!=self.run_identity or prior['session_id']!=self.session_id
                        or prior['state_digest']!=c.state.digest or prior['round_index']!=index):
                        raise ValueError('Pending native round identity differs')
                    for path,digest in prior['native_checkpoint_pins'].items():
                        if sha(path)!=digest:raise ValueError('Pending native checkpoint changed')
                    trace=load_pending_trace(prior['pending_path'],expected_sha256=prior['pending_sha256'],
                        run_identity=self.run_identity,session_id=self.session_id)
                    traces=prior['completed_provider_traces']+[trace]
                    c.producer.configure_resume_provider_replay(traces)
            try:
                with self.consumer.install(self.provider_module):result=c.run_round(index)
            except DeclarationCampaignPending as pending:
                # The native runner has already sealed STARTED. It must not seal a failure round.
                if c.state.next_round_index!=index or c.round_checkpoint_path(index).exists():
                    raise ValueError('Pending unexpectedly advanced or sealed native round')
                paths=[c.root/c.state_filename,c.root/c.state_projection_filename,c._started_path(index)]
                record=dict(status='DECLARATION_PENDING',schema='GRAPH_DECLARATION_ROUND_OWNER_V1',
                    run_identity=self.run_identity,session_id=self.session_id,round_index=index,
                    state_digest=c.state.digest,native_checkpoint_pins={str(p):sha(p) for p in paths},
                    pending_path=pending.path,pending_sha256=pending.sha256,
                    completed_provider_traces=list(c.producer.call_traces),
                    scientific_outcome_created=False,advance_round=False)
                return self._save(journal,record,index)
            record=dict(status='NATIVE_ROUND_RETURNED',schema='GRAPH_DECLARATION_ROUND_OWNER_V1',
                run_identity=self.run_identity,session_id=self.session_id,round_index=index,
                state_digest=c.state.digest,native_result_digest=result.digest)
            self._save(journal,record,index)
            return result

    @staticmethod
    def _save(journal,record,index):
        blob=(json.dumps(record,sort_keys=True,ensure_ascii=True,indent=2)+'\n').encode()
        digest=hashlib.sha256(blob).hexdigest();p=journal/(digest+'.json')
        if not _write_once(p,blob) and p.read_bytes()!=blob:raise ValueError('Owner journal collision')
        _atomic_write(journal/'HEAD.json',(json.dumps(dict(record_file=p.name,record_sha256=digest),sort_keys=True)+'\n').encode())
        return PendingRound(str(p),digest,index)
