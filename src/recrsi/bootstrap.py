"""Prepare task assets and measure the starting model using the native worker."""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .config import NATIVE, read, sha, resolve, initialize
from .prepare import write, pins, WORKER_IDENTITY


def worker_modules():
    source = NATIVE / 'task/search_source'
    sys.path[:0] = [str(source), str(NATIVE / 'common/main_binding'),
                   str(NATIVE / 'common/comparison')]
    import graph_worker
    return graph_worker


def assets(project, data_dir, literature, description):
    """Freeze mapped IDs and fixed negatives from supplied TRAIN/DEV files."""
    import numpy as np
    from recbole.config import Config
    from recbole.data import create_dataset
    import recbole
    project = Path(project).resolve()
    inputs = project / 'inputs'
    if (inputs / 'root_mapping.json').exists():
        raise FileExistsError('Assets are already frozen; use a new project for changed inputs')
    binding = read(inputs / 'binding.json')
    source = inputs / 'root_source.py'
    worker = worker_modules()
    binding['root_source_sha256'] = sha(source)
    binding['recbole_root'] = str(Path(recbole.__file__).parent.parent)
    binding['data_files'] = {}
    for role, suffix in [('train','train.inter'), ('dev','dev.inter'), ('user','user'), ('item','item')]:
        path = (Path(data_dir) / (binding['dataset'] + '.' + suffix)).resolve(strict=True)
        binding['data_files'][role] = dict(path=str(path), sha256=sha(path))
    worker.inspect_binding(binding, execute=True)
    # RecBole owns token ordering; do not approximate it with a separate mapper.
    with tempfile.TemporaryDirectory(prefix='recrsi-map-') as directory:
        staging = Path(directory)
        data = staging / 'data' / binding['dataset']
        data.mkdir(parents=True)
        for item in binding['data_files'].values():
            shutil.copyfile(item['path'], data / Path(item['path']).name)
        values = worker.config_values(binding, staging, 51001, 'cpu')
        argv = sys.argv
        try:
            sys.argv = [argv[0]]
            config = Config(model=worker.load_model(source), dataset=binding['dataset'], config_dict=values)
        finally:
            sys.argv = argv
        dataset = create_dataset(config)
        mapping = {f:list(map(str, dataset.field2id_token[f])) for f in ('user_id','item_id')}
        train_size = dataset.file_size_list[0]
        dev_size = dataset.file_size_list[1]
        train = dataset.inter_feat.iloc[:train_size]
        dev = dataset.inter_feat.iloc[train_size:train_size + dev_size]
        seen = {}
        for _, row in train.iterrows():
            history = seen.setdefault(int(row['user_id']), set())
            history.add(int(row['item_id']))
            if 'item_id_list' in row:
                history.update(map(int, row['item_id_list']))
        rng = np.random.default_rng(binding['evaluation']['seed'])
        users, rows = [], []
        for _, row in dev.iterrows():
            user, target = int(row['user_id']), int(row['item_id'])
            if user in users or target in seen.get(user, set()):
                raise ValueError('Fixed100 requires one unseen DEV target per user')
            excluded = seen.get(user, set()) | {0, target}
            available = np.asarray([i for i in range(1, len(mapping['item_id'])) if i not in excluded])
            if len(available) < 99:
                raise ValueError('Need at least 99 unseen negative items for every DEV user')
            users.append(user)
            rows.append([target, *map(int, rng.choice(available, 99, replace=False))])
    pool = inputs / 'dev_candidates.npz'
    np.savez(pool, users=np.asarray(users), candidates=np.asarray(rows))
    binding['dev_candidate_pool'] = dict(path=str(pool), sha256=sha(pool))
    binding['catalog_mapping_sha256'] = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    documents = read(literature)
    if not documents.get('documents'):
        raise ValueError('Literature requires a nonempty documents array')
    write(inputs / 'root_mapping.json', mapping)
    write(inputs / 'literature.json', documents)
    write(inputs / 'task_description.json', dict(task=description))
    (inputs / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    return dict(users=len(users), items=len(mapping['item_id'])-1, binding=str(inputs / 'binding.json'))


def baseline(project, output, device):
    """Train the user-supplied starting model, then produce search input receipts."""
    project, output = Path(project).resolve(), Path(output).resolve()
    cfg = read(project / 'task.json')
    if type(cfg.get('repeat')) is not int or cfg['repeat'] not in (1, 2, 3):
        raise ValueError('Choose repeat 1, 2, or 3 before measuring the baseline')
    inputs = project / 'inputs'
    env_path = Path(resolve(cfg['environment'], project))
    env = read(env_path)
    worker = worker_modules()
    binding = read(inputs / 'binding.json')
    if binding.get('adapter') != cfg['adapter']:
        raise ValueError('Task and baseline adapters differ')
    source = inputs / 'root_source.py'
    binding['train_seed'] = 51000 + cfg['repeat']
    if device == 'cuda':
        binding['recbole_root'] = resolve(env['recbole_root'], env_path.parent)
    worker.inspect_binding(binding, execute=True)
    if sha(source) != binding['root_source_sha256']:
        raise ValueError('Starting model changed after assets were frozen')
    if (inputs / 'root_calibration.json').exists():
        raise FileExistsError('Baseline already recorded; reuse it or create a new project')
    if output.exists():
        raise FileExistsError('Baseline output exists; retain it and use a new output directory')
    with ExitStack() as stack:
        started_ns = time.time_ns()
        if device == 'cpu':
            output.mkdir(parents=True, exist_ok=False)
            binding_path = Path(write(output / 'BINDING.json', binding))
            # Only the user-provided baseline is executed here, never an LLM candidate.
            import torch
            torch.set_num_threads(4)
            old_cwd = Path.cwd()
            try:
                result = worker.execute(binding, source, output / 'worker', binding['train_seed'], 'cpu', 'root', sha(binding_path))
            finally:
                os.chdir(old_cwd)
            finished_ns = time.time_ns()
            elapsed = (finished_ns-started_ns)/1e9
            physical = dict(jobs={'root':dict(started_ns=started_ns, finished_ns=finished_ns,
                                            gpu_seconds=0., cpu_wall_seconds=elapsed, device='cpu')})
            envelope = dict(status='SUCCESS', result=result, artifacts=dict(native_evidence=dict(
                termination=result['stop_reason'], unresolved_gpu_seconds=0, device='cpu',
                physical_cost=dict(gpu_seconds=0., active_seconds=elapsed))))
            write(output / 'GPU_JOB_BUDGET.json', physical)
            write(output / 'ROOT_CALIBRATION.json', envelope)
        else:
            cpus = set(env['cpus'])
            if len(cpus) != 4 or not cpus <= os.sched_getaffinity(0):
                raise ValueError('Assign four available CPU cores in environment.json')
            slots = Path(resolve(env['resource_slots'], env_path.parent))
            # Same canonical per-core locks as search; no uncoordinated baseline work.
            from .resources import claim_cpus
            stack.enter_context(claim_cpus(slots, cpus))
            previous = os.sched_getaffinity(0)
            os.sched_setaffinity(0, cpus)
            stack.callback(os.sched_setaffinity, 0, previous)
            sys.path.insert(0, str(NATIVE / 'task/root_runtime'))
            from shared_gpu_capacity import gpu_capacity
            driver = resolve(env['driver_root'], env_path.parent)
            stack.enter_context(gpu_capacity(slots, env['gpu_uuid'], driver, 9472, str(output)))
            output.mkdir(parents=True, exist_ok=False)
            binding_path = Path(write(output / 'BINDING.json', binding))
            python = resolve(env['worker_python'], env_path.parent, interpreter=True)
            closure = json.loads(subprocess.check_output([python, '-c', WORKER_IDENTITY], text=True))
            closure.update(main_binding=str(NATIVE / 'common/main_binding'),
                           comparison_root=str(NATIVE / 'common/comparison'), source_files=pins(NATIVE))
            closure_path = write(output / 'CLOSURE.json', closure)
            config = dict(binding=str(binding_path), parent_source=str(source), output=str(output),
                closure=closure_path, closure_sha256=sha(closure_path), assets=pins(NATIVE),
                identity=dict(run=cfg['run_id']+'-baseline', train_seed=binding['train_seed']),
                driver_root=driver, gpu_uuid=env['gpu_uuid'], cpus=sorted(cpus))
            config_path = write(output / 'CONFIG.json', config)
            subprocess.run([python, str(NATIVE / 'task/search_source/graph_root_launcher.py'),
                            '--config', config_path, '--execute'], check=True)
            envelope = read(output / 'ROOT_CALIBRATION.json')
            physical = read(output / 'GPU_JOB_BUDGET.json')
            result = envelope['result']
        from root_contract import validate_root_receipt
        cost = validate_root_receipt(envelope, physical)
    write(inputs / 'root_binding.json', binding)
    # Binding bytes must match the exact bytes measured by the worker.
    (inputs / 'root_binding.json').write_bytes(binding_path.read_bytes())
    write(inputs / 'root_calibration.json', envelope)
    write(inputs / 'root_gpu_ledger.json', physical)
    from .config import native_module
    context = native_module('_recrsi_facts_context', 'task/research_line/graph_native_inputs.py')
    write(inputs / 'facts.json', dict(task_id=binding['task_id'], task=read(inputs / 'task_description.json')['task'],
        simulation_context=context.CONTEXT_RULE, root_source_sha256=sha(source),
        data_sha256={k:v['sha256'] for k,v in binding['data_files'].items()},
        catalog_mapping_sha256=result['catalog_mapping_sha256'], original_model_config=binding['model_config'],
        training=binding['training'], evaluation=binding['evaluation'], selected_root_metrics=result['metrics'],
        root_reference_cost=cost))
    return dict(status='SUCCESS', device=device, metrics=result['metrics'], output=str(output), cost=cost)


def demo(output):
    """Run the documented synthetic SASRec example with a real CPU optimizer."""
    output = Path(output).resolve()
    project = initialize(output / 'project')
    data = output / 'data'
    data.mkdir()
    header = 'user_id:token\titem_id_list:token_seq\titem_id:token\n'
    (data/'cds.train.inter').write_text(header + ''.join(
        f'u{u}\t'+' '.join(f'i{j}' for j in range(1,i))+f'\ti{i}\n' for u in range(1,5) for i in range(2,11)))
    (data/'cds.dev.inter').write_text(header + ''.join(
        f'u{u}\t'+' '.join(f'i{j}' for j in range(1,11))+f'\ti{10+u}\n' for u in range(1,5)))
    (data/'cds.user').write_text('user_id:token\n'+''.join(f'u{u}\n' for u in range(1,5)))
    (data/'cds.item').write_text('item_id:token\n'+''.join(f'i{i}\n' for i in range(1,121)))
    literature = output/'literature.json'
    write(literature, dict(documents=[dict(title='Self-Attentive Sequential Recommendation', year=2018)]))
    path = project/'inputs/binding.json'
    binding = read(path)
    binding['training'].update(epochs=2, patience=2, batch_size=16)
    binding['model_config'].update(hidden_size=8, inner_size=16, n_layers=1, MAX_ITEM_LIST_LENGTH=10)
    path.write_text(json.dumps(binding, indent=2))
    assets(project, data, literature, 'Synthetic sequential recommendation installation example')
    report = baseline(project, output/'baseline', 'cpu')
    report['example'] = 'synthetic CPU training and checkpoint reload'
    write(output/'DEMO_RESULT.json', report)
    return report
