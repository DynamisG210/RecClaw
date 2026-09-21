"""Common qualified graph evaluation. Native controllers retain search policy.

Owner must bind completed root, scientific recipe and finite per-stage timeouts.
Qualification/training each use existing E2 isolation and persistent GPU watcher.
No fallback, implicit retry, root refit or unmetered Provider exists here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE=Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(config,*,worker_process=True):
    from graph_root_launcher import setup
    if worker_process:
        closure=setup(config)
    else:
        # Native controllers may have separate dependency environments. Exact
        # worker runtime verification happens in the detached worker-python
        # evaluator before any paid physical work, never against controller Python.
        if sha(config['closure'])!=config['closure_sha256']:
            raise ValueError('Worker closure changed')
        closure=read(config['closure'])
        for path,expected in config['assets'].items():
            if sha(path)!=expected: raise ValueError('Graph asset changed: '+path)
        sys.path[:0]=[closure['main_binding'],closure['comparison_root']]
    for key in ('qualification_timeout_seconds','worker_timeout_seconds'):
        value=config.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=21600:
            raise ValueError('Main must freeze a finite post-calibration '+key)
    if config.get('phase') not in ('BOUNDED_GRAPH_DEVELOPMENT','FORMAL_MAIN'):
        raise ValueError('Candidate execution requires a prospectively bound search phase')
    if len(set(config['cpus']))!=4 or not config['gpu_uuid'].startswith('GPU-'):
        raise ValueError('Owner must bind the graph allocation')
    return closure


def resources_for(config):
    from openevolve_provider import _settings_and_ledger
    from main_resources import MainResources
    from active_wall import ActiveWallClock
    settings,api=_settings_and_ledger(config['provider_config'])
    if (settings['run_identity']!=config['run_identity']
            or Path(settings['resource_root']).resolve()!=Path(config['output']).resolve()
            or Path(settings['active_wall_path']).resolve()!=Path(config['active_wall_path']).resolve()):
        raise ValueError('Graph worker and real Provider must share all run resources')
    ActiveWallClock.create(config['active_wall_path'],config['run_identity'])
    return MainResources(config['output'],config['run_identity'],config['active_wall_path'],api)


def timed_boundary(config,closure,resources,source_sha,kind,*,reserved=True):
    from gagc.e2_boundary import E2Boundary
    from worker_job_time import TimedWorkerBoundary
    from worker_watch import append_event
    class Boundary(E2Boundary):
        _durable_watch=TimedWorkerBoundary._durable_watch
        def popen(self,command,**kwargs):
            if reserved: resources.require_reserved_evaluation_capacity(source_sha)
            else: resources.require_capacity()
            key={'qualification':'qualification_timeout_seconds','training':'worker_timeout_seconds',
                 'observation':'observation_timeout_seconds'}[kind]
            requested=config[key]
            if isinstance(requested,bool) or not isinstance(requested,(int,float)) or not math.isfinite(requested) or not 0<requested<=21600:
                raise ValueError('Main must bind the finite '+kind+' timeout')
            token,allowance=resources.gpu.reserve(requested,kind='graph_'+kind)
            process=super().popen(command,**kwargs)
            job=Path(process.args[-1]);started=Path(read(job)['worker_pid_path']).stat().st_mtime_ns
            resources.gpu.started(token,started_ns=started,worker_pid=process.worker_pid,
                control_job=job,gpu_uuid=self.gpu_uuid)
            process.graph_budget_token=token
            job_id=uuid.uuid4().hex
            append_event(self.timing_path,dict(event='START',job_id=job_id,started_ns=started,
                pid=process.worker_pid,gpu_uuid=self.gpu_uuid,cards=1,control_job=str(job)))
            return self._durable_watch(process,token,allowance,requested,started,job_id)
    boundary=Boundary(worker_python=Path(closure['python']),driver_root=Path(config['driver_root']),
        control_root=Path(config['output'])/'candidate_control',
        launcher=Path(closure['comparison_root'])/'helpers/se_namespace_exec.py',
        gpu_uuid=config['gpu_uuid'],cpus=config['cpus'])
    boundary.resource_entry=HERE/'graph_confined_entry.py'
    boundary.resource_budget,boundary.active_clock=resources.gpu,resources.clock
    boundary.timing_path=Path(config['output'])/'GPU_WORKER_EVENTS.jsonl'
    return boundary


def run_stage(config,closure,resources,source,trial,kind,*,reserved=True):
    from resource_stop import SearchResourceStop
    binding=read(config['binding'])
    outer=trial/kind;outer.mkdir(exist_ok=False)
    output=outer/'worker'
    script=(HERE/'graph_qualification_worker.py' if kind=='qualification'
            else Path(config['candidate_worker']).resolve(strict=True))
    command=[closure['python'],'-B','-u',str(script),'--binding',config['binding'],
        '--candidate',str(source),'--output',str(output),'--train-seed',str(config['train_seed']),
        '--device','cuda']
    if kind=='training': command+=['--role','candidate','--expected-source-sha',sha(source),'--execute']
    assets=[HERE/'graph_worker.py',HERE/'graph_candidate_contract.py',script,
        HERE/'graph_confined_entry.py',Path(config['binding']),source,Path(binding['recbole_root'])]
    names=('train','user','item') if kind=='qualification' else ('train','dev','user','item')
    assets += [Path(binding['data_files'][name]['path']) for name in names]
    if kind=='qualification': assets += [HERE/'graph_fixed_mapping.py',Path(binding['catalog_mapping_path'])]
    if kind=='training': assets += [HERE/'graph_checkpoint_verify.py']
    # Landlock file grants permit reads but not Python's parent-directory scan.
    # Expose only this stage's trusted modules as a separate read-only directory.
    runtime=trial/(kind+'_runtime');runtime.mkdir(exist_ok=False)
    modules=[path for path in assets if path.parent==HERE and path.suffix=='.py']
    for path in modules:
        (runtime/path.name).write_bytes(path.read_bytes())
    assets=[path for path in assets if path not in modules]+[runtime]
    command[3]=str(runtime/script.name)
    environment=dict(PATH='/usr/bin:/bin',HOME=str(outer),TMPDIR=str(outer),XDG_CACHE_HOME=str(outer),
        PYTHONHASHSEED=str(config['train_seed']),CUDA_VISIBLE_DEVICES=config['gpu_uuid'],
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    boundary=timed_boundary(config,closure,resources,sha(source),kind,reserved=reserved)
    boundary.resource_entry=runtime/'graph_confined_entry.py'
    with (outer/'worker.log').open('ab') as log:
        process=boundary.popen(command,cwd=outer,env=environment,readonly=assets,writable=[outer],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        code=process.wait()
    if not resources.admit_output():
        raise SearchResourceStop('Graph '+kind+' retained outside common resource limit')
    gpu=resources.gpu.snapshot()
    if gpu['reserved_seconds']:
        raise RuntimeError('Physical graph work has unresolved timing; no new attempt')
    receipt=read(outer/'BOUNDARY.json')
    expected=dict(uid=65534,no_new_privs='1',cpu_affinity=sorted(config['cpus']),
        network='isolated',network_interfaces=['lo'],memory_limit_bytes=32*1024**3,
        torch_allocator_limit_bytes=9*1024**3,torch_threads=4)
    if any(receipt.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Graph candidate physical boundary differs')
    result_path=output/('QUALIFICATION.json' if kind=='qualification' else 'RESULT.json')
    if result_path.exists():
        result=read(result_path)
        if result.get('status')=='INFRASTRUCTURE_FAILURE':
            raise RuntimeError('Trusted graph worker bootstrap failed: '+result['error'])
    elif (output/'FAILURE.json').exists():
        result=dict(read(output/'FAILURE.json'),status='CANDIDATE_FAILURE',failure_stage=kind)
    elif code!=0:
        # The real namespace returned a settled exit; retain timeout/crash as
        # negative evidence, not a metric. Missing exits remain exceptions above.
        result=dict(status='CANDIDATE_FAILURE',failure_stage=kind,
            error='Protected worker exited without a result',exit_code=code,test_accessed=False)
    else:
        raise RuntimeError('Worker returned zero without trusted result; reconcile original job')
    if result['status'] in ('SUCCESS','QUALIFIED'):
        if (code!=0 or result['source_sha256']!=sha(source)
                or result['catalog_mapping_sha256']!=binding['catalog_mapping_sha256']
                or result['train_seed']!=config['train_seed'] or result['test_accessed'] is not False):
            raise RuntimeError('Graph worker output identity differs')
        if kind=='training' and (result['binding_sha256']!=sha(config['binding'])
                or result['stop_reason'] not in ('EARLY_STOP','EPOCH_LIMIT')):
            raise RuntimeError('Incomplete or changed graph training output')
    result['physical_stage']=dict(kind=kind,output=str(output),exit_code=code,
        boundary=str(outer/'BOUNDARY.json'),gpu_ledger=str(resources.gpu.path),
        gpu_seconds=gpu['jobs'][process.graph_budget_token]['gpu_seconds'])
    job=gpu['jobs'][process.graph_budget_token]
    if code and job['gpu_seconds']>=job['reserved_seconds']: result['timed_out']=True
    return result


def evaluate_source(config,source_path=None):
    closure=configure(config)
    resources=resources_for(config)
    from graph_reference_root import reference_root
    from graph_candidate_contract import qualify_source
    from graph_program import primary_score
    from resource_stop import SearchResourceStop
    from evaluation_ledger import EvaluationLimitReached
    baseline,ledger=reference_root(config,resources)
    if source_path is None or sha(source_path)==sha(config['parent_source']):
        return baseline
    digest=sha(source_path)
    saved=ledger.recoverable(digest)
    if saved:
        result,admitted=saved['result'],saved['admitted']
        ledger.finish(saved['token'],valid=result['artifacts']['valid'],
            primary_metric=result['metrics'].get('ndcg_at_5'),admitted=admitted,
            evidence=result['artifacts']['native_evidence'])
        if not admitted: raise SearchResourceStop('Original graph output was not admitted')
        return result
    resources.require_capacity()
    try: token=ledger.reserve(digest)
    except EvaluationLimitReached as error: raise SearchResourceStop(str(error)) from error
    trial=Path(config['output'])/'graph_candidates'/token;trial.mkdir(parents=True,exist_ok=False)
    source=trial/'candidate.py';source.write_bytes(Path(source_path).read_bytes())
    try:
        qualify_source(source.read_text(encoding='utf-8'))
    except (SyntaxError,ValueError) as error:
        outcome=dict(status='CANDIDATE_FAILURE',failure_stage='source_contract',error=str(error),
            test_accessed=False,worker_started=False)
    else:
        try:
            from graph_qualification import qualify_program
            qualification=qualify_program(config,source)
            if qualification['status']=='QUALIFIED':
                outcome=run_stage(config,closure,resources,source,trial,'training')
                outcome['qualification']=qualification
            elif qualification['status']=='CANDIDATE_FAILURE': outcome=qualification
            else: raise RuntimeError('Unknown qualification outcome')
        except SearchResourceStop as stop:
            ledger.finish(token,valid=False,primary_metric=None,admitted=False,
                          evidence=dict(resource_stop=str(stop),trial=str(trial)))
            raise
    outcome['source_sha256']=digest
    outcome['wall_time_secs']=outcome.get('physical_stage',{}).get('gpu_seconds',0.)
    if 'qualification' in outcome:
        outcome['wall_time_secs']+=outcome['qualification'].get('physical_stage',{}).get('gpu_seconds',0.)
    valid=outcome['status']=='SUCCESS'
    score=primary_score(outcome,read(config['binding'])) if valid else None
    admitted=resources.admit_output()
    result=dict(metrics={'combined_score':score,'ndcg_at_5':score} if valid else {'combined_score':-1.},
        artifacts=dict(valid=valid,source_sha256=digest,native_evidence=outcome))
    ledger.ready(token,result,admitted=admitted)
    ledger.finish(token,valid=valid,primary_metric=score,admitted=admitted,evidence=outcome)
    if not admitted: raise SearchResourceStop('Graph result retained outside common admission')
    return result


def command(request,fd):
    return [sys.executable,'-B',str(Path(__file__).resolve()),'--request',str(request),'--lock-fd',str(fd)]


def run_program(config,path=None):
    closure=configure(config,worker_process=False)
    from durable_evaluation import run_native_job
    return run_native_job(config,path,command_factory=lambda request,fd:[closure['python'],'-B',
        str(Path(__file__).resolve()),'--request',str(request),'--lock-fd',str(fd)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--lock-fd',type=int,required=True)
    args=parser.parse_args();os.set_inheritable(args.lock_fd,False)
    request=read(args.request);configure(request['config'])
    from durable_evaluation import write_json
    from resource_stop import SearchResourceStop
    try:
        path=None
        if request['source'] is not None:
            path=args.request.parent/'program.py';path.write_bytes(request['source'].encode('utf-8'))
        value=evaluate_source(request['config'],path)
        result=dict(status='RETURNED',value=value)
    except SearchResourceStop as error: result=dict(status='RESOURCE_STOP',message=str(error))
    except Exception as error: result=dict(status='ERROR',error_type=type(error).__name__,message=str(error))
    write_json(args.request.parent/'result.json',result)


if __name__=='__main__': main()
