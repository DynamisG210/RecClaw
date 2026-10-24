import argparse,ast,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--verified-source',type=Path,required=True);p.add_argument('--watcher-sha256',required=True);a=p.parse_args();Q=a.verified_source;A=a.root;R=A/'runtime'
receipt=json.loads((Q/'E2_DEADLINE_SIGNAL_TEST.json').read_text());assert receipt['status']=='PASS'
for n in ('owned_e2_signal_v1.py','worker_watch_e2_v1.py'):
 assert hashlib.sha256((Q/n).read_bytes()).hexdigest()==({'worker_watch_e2_v1.py':a.watcher_sha256}.get(n,receipt['source'][n]))
assert not list(A.glob('*/INFRA000002_GPU_ROUND_003_STARTED.json'))
p=R/'worker_watch.py';s=p.read_text();shutil.copyfile(p,R/'worker_watch_before_owned_signal.py')
src=(Q/'worker_watch_e2_v1.py').read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='privileged_e2_signal');function=ast.get_source_segment(src,node)
s=s.replace('def observe(config, pidfd):',function+'\n\ndef observe(config, pidfd):')
needle="            _pidfd_syscall(424,ctypes.c_int(pidfd),ctypes.c_int(signal.SIGKILL),ctypes.c_void_p(0),ctypes.c_uint(0))"
assert needle in s
s=s.replace(needle,"            try:\n                _pidfd_syscall(424,ctypes.c_int(pidfd),ctypes.c_int(signal.SIGKILL),ctypes.c_void_p(0),ctypes.c_uint(0))\n            except PermissionError:\n                privileged_e2_signal(pidfd)\n            except ProcessLookupError:\n                pass")
compile(s,str(p),'exec');p.write_text(s);shutil.copyfile(Q/'owned_e2_signal_v1.py',R/'owned_e2_signal_v1.py')
b=json.loads((A/'EXECUTION_BINDING.json').read_text());b['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();b['files'][str(R/'owned_e2_signal_v1.py')]=receipt['source']['owned_e2_signal_v1.py'];(A/'EXECUTION_BINDING.json').write_text(json.dumps(b,indent=2))
(A/'OWNED_SIGNAL_ADOPTION.json').write_text(json.dumps({'source_receipt':receipt,'source_root':str(Q),'verified_source_watcher_sha256':a.watcher_sha256,'adopted_function':'privileged_e2_signal','retained':'Original RQ4 private budget imports, deadlines, stopped_by_common_limit semantics, settlement','watcher_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'new_fit':0},indent=2))
print('Adopted verified signal helper only; RQ4 deadline semantics unchanged')
