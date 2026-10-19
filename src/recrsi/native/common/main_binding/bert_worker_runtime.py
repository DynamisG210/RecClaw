"""BERT physical callback: exact candidate bytes, existing isolation and metering."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess


class BertRuntime:
    def __init__(self, config, resources, baseline):
        self.config, self.resources = config, resources
        self.budget, self.baseline = resources.gpu, baseline
        self.parent_sha = hashlib.sha256(Path(config['parent_source']).read_bytes()).hexdigest()

    def worker_boundary(self, timeout=None):
        from worker_job_time import TimedWorkerBoundary
        cfg = self.config
        root = Path(cfg['output'])
        return TimedWorkerBoundary(Path(cfg['worker_python']), Path(cfg['core']),
            Path(cfg['recbole_root']), Path(cfg['driver_dir']), root/'worker_control',
            cfg['gpu_uuid'], root/'GPU_WORKER_EVENTS.jsonl', self.budget,
            self.resources.clock, cfg['qualification_timeout_seconds'],
            cfg['worker_timeout_seconds'] if timeout is None else timeout)

    def _train(self, candidate_path, source, timeout, *, evaluation_reserved=False):
        from resource_stop import SearchResourceStop
        cfg = self.config
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        if Path(candidate_path).read_bytes().decode('utf-8') != source:
            raise ValueError('Candidate changed before protected execution')
        root = Path(cfg['output']) / 'bert_workers' / digest
        root.mkdir(parents=True, exist_ok=True)
        with (root/'owner.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            candidate = root/'candidate.py'
            if candidate.exists():
                if candidate.read_bytes().decode('utf-8') != source:
                    raise ValueError('Saved candidate bytes differ')
            else:
                candidate.write_text(source, encoding='utf-8')
            work = root/'output'
            work.mkdir(exist_ok=True)
            expected = dict(source_sha256=digest, seed=cfg['train_seed'],
                worker_sha256=hashlib.sha256(Path(cfg['bert_worker']).read_bytes()).hexdigest(),
                recipe_sha256=hashlib.sha256(Path(cfg['training_config']).read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((Path(cfg['data_root'])/'manifest.json').read_bytes()).hexdigest())
            result_path = work/'worker_result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['identity'] != expected:
                    raise ValueError('Saved BERT result belongs to another protocol')
                return result
            if (root/'STARTED.json').exists():
                raise RuntimeError('Interrupted physical job needs its original checkpoint recovery')
            if evaluation_reserved:
                self.resources.require_reserved_evaluation_capacity(digest)
            else:
                self.resources.require_capacity()
            boundary = self.worker_boundary(timeout)
            for name in ('home', 'tmp', 'cache'):
                (work/name).mkdir(exist_ok=True)
            env = dict(PATH=str(Path(cfg['worker_python']).parent)+':/usr/bin:/bin',
                HOME=str(work/'home'), TMPDIR=str(work/'tmp'), XDG_CACHE_HOME=str(work/'cache'),
                LANG='C.UTF-8', LC_ALL='C.UTF-8', CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],
                PYTHONPATH=os.pathsep.join([str(Path(cfg['core'])/'src'), cfg['recbole_root']]),
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            command = [cfg['worker_python'], '-B', cfg['bert_worker'], '--candidate', str(candidate),
                '--recipe', cfg['training_config'], '--data', cfg['data_root'], '--output', str(work),
                '--seed', str(cfg['train_seed'])]
            with (root/'worker.log').open('a') as log:
                process = boundary.popen(command, cwd=work, env=env,
                    readonly=[candidate, Path(cfg['bert_worker']), Path(cfg['training_config']),
                              Path(cfg['data_root']), Path(cfg['recbole_root']), Path(cfg['core'])/'src'],
                    writable=[work], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                (root/'STARTED.json').write_text(json.dumps(dict(worker_pid=process.worker_pid,
                    controller_pid=os.getpid(), command=command, identity=expected)))
                code = process.wait()
            if not result_path.exists():
                raise RuntimeError(f'BERT worker exited {code} without an outcome: {root}')
            result = json.loads(result_path.read_text())
            if result['identity'] != expected or result['source_sha256'] != digest:
                raise ValueError('Protected BERT result identity differs')
            if not self.resources.admit_output():
                raise SearchResourceStop('BERT output retained after the common resource deadline')
            return result
