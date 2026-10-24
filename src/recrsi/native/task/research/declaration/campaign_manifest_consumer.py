"""Join persisted short-storage mappings to sealed broker requests in SQLite."""
import hashlib,json,sqlite3
from pathlib import Path
from campaign_pending_boundary import PendingCampaignConsumer

class ManifestCampaignConsumer(PendingCampaignConsumer):
    @classmethod
    def from_manifest(cls,path,*,expected_sha256):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Frozen consumer manifest changed')
        manifest=json.loads(raw)
        for name,expected in manifest['runtime_pins'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Frozen consumer runtime changed')
        return super().from_manifest(path,expected_sha256=expected_sha256)

    def _evidence(self,result):
        root=self.bridge.root
        sealed=result.sealed_request
        ordinal=result.attempts[-1]['ordinal']
        matches=[]
        for path in sorted((root/'broker_call_paths').glob('*.json')):
            mapping=json.loads(path.read_bytes())
            relative=mapping['original_relative_path']
            key=hashlib.sha256(relative.encode()).hexdigest()
            if path.stem!=key or mapping['broker_relative_path']!='broker_calls/'+key:
                raise ValueError('Native short-storage mapping changed')
            original=(root/relative).resolve();original.relative_to(root)
            call=root/mapping['broker_relative_path']
            db=call/('physical_attempt_%02d'%ordinal)/'broker.sqlite3'
            if not db.exists():continue
            wal=db.with_name(db.name+'-wal')
            if wal.exists() and wal.stat().st_size:
                raise ValueError('Native broker has unsettled SQLite writes')
            con=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
            try:
                rows=con.execute('SELECT request_digest,proposal_generation_session_id FROM calls WHERE logical_call_id=?',(sealed['logical_call_id'],)).fetchall()
            finally:con.close()
            if rows:
                if rows!=[(sealed['request_digest'],sealed['proposal_generation_session_id'])]:
                    raise ValueError('Native mapping and sealed identity disagree')
                matches.append(call)
        if len(matches)!=1:raise ValueError('Exactly one native persisted call mapping is required')
        # Actual wire lookup stays in the original consumer; replace only its
        # incorrect logical-id-to-directory assumption with the verified mapping.
        evidence=super()._evidence(result)
        evidence['call_root']=matches[0]
        return evidence
