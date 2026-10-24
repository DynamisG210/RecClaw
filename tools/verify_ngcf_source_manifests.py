"""Verify frozen NGCF source, staged copy, and selection-manifest bytes read-only."""
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); rows=json.loads(a.sources.read_text()); pins={}
    for r in rows:
        for key,hkey in [('source','source_sha256'),('staged_source','source_sha256'),('selection_manifest','selection_manifest_sha256')]:
            assert r[key] not in pins or pins[r[key]]==r[hkey]
            pins[r[key]]=r[hkey]
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,'tar -cf - -- '+' '.join(shlex.quote(x) for x in pins)],capture_output=True)
    assert proc.returncode==0,proc.stderr.decode()
    checks=[]; manifests={}
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for path,expected in pins.items():
            raw=archive.extractfile(path.lstrip('/')).read(); actual=hashlib.sha256(raw).hexdigest()
            checks.append(dict(path=path,expected=expected,actual=actual,match=actual==expected))
            if path.endswith('.json'):manifests[path]=json.loads(raw)
    result=dict(sources_sha256=hashlib.sha256(a.sources.read_bytes()).hexdigest(),checks=checks,selection_manifests=manifests,
                rows=rows,new_api=0,new_fit=0,new_test=0,scope='Original frozen18 source searches only, not replacement RO1. Byte matches do not alone validate DEV winner policy or checkpoint bytes.')
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(pins=len(checks),matched=sum(c['match'] for c in checks),mismatches=[c for c in checks if not c['match']],manifests=len(manifests))))


if __name__=='__main__':main()
