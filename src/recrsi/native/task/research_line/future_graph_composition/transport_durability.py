"""Persist paid response bytes before SUCCESS accounting and broker handoff."""
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from unittest.mock import patch

VERSION='graph-paid-response-durability-v1'


def persist_paid_response(settings, call_id, label, payload, raw, request_data):
    root=Path(settings['api_ledger']).parent/'durable_provider_responses'
    root.mkdir(exist_ok=True)
    record=dict(version=VERSION,call_id=call_id,request_body_sha256=label,
        response_sha256=hashlib.sha256(raw).hexdigest(),model=payload['model'])
    # Store the exact wire request bytes through the caller, never credentials.
    assert hashlib.sha256(request_data).hexdigest()==label
    for suffix,data in (('.request',request_data),('.response',raw),('.json',json.dumps(record,sort_keys=True).encode())):
        target=root/(call_id+suffix)
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('Conflicting durable response identity')
            continue
        temp=target.with_suffix(target.suffix+'.tmp')
        with temp.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        temp.replace(target)
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def corrected_send(module):
    source=textwrap.dedent(inspect.getsource(module.MainROTransport.send))
    needle="    self.ledger.record('SUCCESS', call_id=call_id, channel=channel, label=label,"
    if source.count(needle)!=1:raise ValueError('Unexpected native transport SUCCESS boundary')
    source=source.replace(needle,"    persist_paid_response(self.settings, call_id, label, payload, raw, request.data)\n"+needle)
    namespace=dict(module.__dict__,persist_paid_response=persist_paid_response)
    exec(compile(source,__file__,'exec'),namespace)
    return namespace['send']


@contextmanager
def durable_transport_scope():
    import ro_main_provider
    with patch.object(ro_main_provider.MainROTransport,'send',corrected_send(ro_main_provider)):
        yield
