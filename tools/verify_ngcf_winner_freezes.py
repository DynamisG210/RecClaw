"""Collect downstream winner freezes, retaining mismatches rather than overwriting pins."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.binding.read_text());ms=d['selection_manifests']
    paths=[m['freeze'] for m in ms.values()]
    proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',a.host,'tar -cf - -- '+' '.join(shlex.quote(x) for x in paths)],capture_output=True)
    assert proc.returncode==0,proc.stderr.decode()
    rows=[]
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as archive:
        for r in d['rows']:
            m=ms[r['selection_manifest']];raw=archive.extractfile(m['freeze'].lstrip('/')).read();f=json.loads(raw);h=hashlib.sha256(raw).hexdigest()
            rows.append(dict(method=r['method'],repeat=r['repeat'],path=m['freeze'],expected=m['freeze_sha256'],actual=h,byte_match=h==m['freeze_sha256'],source_match=f.get('source_sha256',f.get('winner_source_sha256'))==r['source_sha256'],data=f))
    a.out.write_text(json.dumps(dict(binding_sha256=hashlib.sha256(a.binding.read_bytes()).hexdigest(),rows=rows,new_api=0,new_fit=0,new_test=0,scope='Changed freezes remain unresolved until historical bytes or documented amendment recovered. Source agreement is not full freeze agreement.'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),byte_matches=sum(r['byte_match'] for r in rows),source_matches=sum(r['source_match'] for r in rows))))


if __name__=='__main__':main()
