"""Capture exact transport bytes after the common ledger records native SUCCESS."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from declaration_provider_v3 import save_new

@contextmanager
def capture_native_transport(module):
    original=module.MainROTransport.send
    async def send(transport,request,timeout):
        before={e.get('call_id') for e in transport.ledger.events()}
        wire=bytes(request.data)
        raw=await original(transport,request,timeout)
        label=hashlib.sha256(wire).hexdigest()
        rows=[e for e in transport.ledger.events() if e.get('event')=='SUCCESS'
              and e.get('label')==label and e.get('call_id') not in before]
        if len(rows)!=1:raise ValueError('Native output capture needs one new common-ledger SUCCESS')
        root=Path(transport.settings['resource_root'])/'durable_provider_responses'
        root.mkdir(parents=True,exist_ok=True)
        call=rows[0]['call_id']
        for suffix,blob in [('request',wire),('response',raw)]:
            p=root/(call+'.'+suffix)
            if p.exists():
                if p.read_bytes()!=blob:raise ValueError('Existing durable transport bytes conflict')
            else:
                with p.open('xb') as f:
                    f.write(blob);f.flush()
                    import os
                    os.fsync(f.fileno())
        save_new(root/(call+'.capture.json'),dict(call_id=call,request_sha256=label,
            response_sha256=hashlib.sha256(raw).hexdigest(),native_output_unchanged=True))
        return raw
    module.MainROTransport.send=send
    try:yield
    finally:module.MainROTransport.send=original


def install():
    import manifest_round_owner
    manifest_round_owner.capture_native_transport=capture_native_transport
