from contextlib import contextmanager,ExitStack
from pathlib import Path
import json,os,sys,time,subprocess,hashlib
E=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(E/'root_runtime'))
from shared_gpu_capacity import gpu_capacity
@contextmanager
def physical_gpu(config,source,kind):
    output=Path(config['output']).resolve()
    assert E in output.parents
    assert set(os.sched_getaffinity(0))==set(config['cpus'])
    driver=config['driver_root'];slots=Path(config['resource_slots'])
    path=output/'PHYSICAL_PLACEMENT.jsonl';waiting=False
    def record(event,**kw):
        with path.open('a') as f:f.write(json.dumps(dict(event=event,time_ns=time.time_ns(),pid=os.getpid(),starttick=Path('/proc/self/stat').read_text().split()[21],kind=kind,source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),**kw))+'\n')
    while True:
        env=dict(os.environ,LD_LIBRARY_PATH=driver+':'+os.environ.get('LD_LIBRARY_PATH',''))
        rows=subprocess.check_output([driver+'/nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],env=env,text=True)
        cards=sorted([(int(r.split(',')[2]),int(r.split(',')[0]),r.split(',')[1].strip()) for r in rows.splitlines()],reverse=True)
        for free,index,uuid in cards:
            if free<9472:continue
            with ExitStack() as stack:
                try:lease=stack.enter_context(gpu_capacity(slots,uuid,driver,9472,str(output)))
                except BlockingIOError:continue
                except RuntimeError as e:
                    if not str(e).startswith('Insufficient capacity:'):raise
                    continue
                placed=dict(config,gpu_uuid=uuid)
                record('CLAIM',gpu_uuid=uuid,cpus=config['cpus'],lease=lease)
                try:yield placed
                finally:record('RELEASE',gpu_uuid=uuid)
                return
        if not waiting:record('WAIT_SHARED_GPU_CAPACITY',gpu_reserved=False);waiting=True
        time.sleep(45)
