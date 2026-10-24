"""Verify checkpoint bytes referenced by an existing NGCF FIT/TEST chain."""
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--chain', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.chain.read_text())
    pins = {}
    for r in source['rows']:
        path, digest = r['checkpoint'], r['checkpoint_sha256']
        assert path not in pins or pins[path] == digest
        pins[path] = digest
    commands = '\n'.join('sha256sum -- '+shlex.quote(path) for path in pins)+'\n'
    proc = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,'bash','-s'], input=commands.encode(), capture_output=True)
    actual = {line[66:]:line[:64] for line in proc.stdout.decode().splitlines() if len(line)>66}
    rows = [dict(slot=r['slot'],checkpoint=r['checkpoint'],expected=r['checkpoint_sha256'],
                 actual=actual.get(r['checkpoint']),match=actual.get(r['checkpoint'])==r['checkpoint_sha256']) for r in source['rows']]
    result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  chain_sha256=hashlib.sha256(a.chain.read_bytes()).hexdigest(),host=a.host,
                  rows=rows,exit_code=proc.returncode,stderr=proc.stderr.decode(),
                  scope='Physical checkpoint byte verification only; does not resolve changed historical search ledgers or cohort selection.',
                  new_api=0,new_fit=0,new_test=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checkpoints=len(pins),matches=sum(r['match'] for r in rows),failures=[r for r in rows if not r['match']],exit_code=proc.returncode)))


if __name__=='__main__':
    main()
