"""Resume the existing retain R13 candidate after its binding-only AST repair.

Uses original physical qualification/evaluator and original finite ledgers.
The native raw failed round remains immutable; recovery is a linked sidecar.
No producer or implementation service is called.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
import py_compile
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main(home, original):
    recovery = home/'engineering_recovery'
    receipt_path = recovery/'RETAIN_R13_EXECUTION.json'
    receipt = dict(pid=os.getpid(), birth_ticks=Path('/proc/self/stat').read_text().split()[21],
                   command=sys.argv, stage='WAITING_PAIR_LOCK', started_ns=time.time_ns(),
                   new_provider_calls=0, round=13, arm='retain')
    with receipt_path.open('x') as handle:
        json.dump(receipt, handle, indent=2)

    def save(stage, **values):
        receipt.update(stage=stage, updated_ns=time.time_ns(), **values)
        temp = receipt_path.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2, default=str))
        temp.replace(receipt_path)

    deadline = time.monotonic()+86400
    try:
        with (home/'PAIR_EXECUTION.lock').open('a+b') as pair:
            while True:
                try:
                    fcntl.flock(pair, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Finite pair-lock wait exhausted')
                    time.sleep(5)
            cfg = read(home/'retain/EVALUATOR_CONFIG.json')
            allocation = read(home/'ALLOCATION.json')
            assert allocation['phase_round_end'] == 13
            assert allocation['search_seed'] == 762002
            sys.path[:0] = [str(home), str(home/'runtime'), str(original),
                           str(Path(cfg['task_root'])/'controller_adapter'),
                           *read(original/'ORIGINAL_RUNTIME_PATHS.json')]
            # Pin task-native modules before evaluator.configure changes sys.path.
            import auto_search_space as native
            import e2_ro_search_space_adapter
            from morec_ro_execution import search_adapter
            from e2_budget_binding import bind_config
            bind_config(cfg)
            from admission import admitted
            save('WAITING_SHARED_ADMISSION')
            with admitted(home, cfg, read(home/'SHARED_ADMISSION.json')) as placement:
                cfg.update(cpus=placement['cpus'], gpu_uuid=placement['gpu_uuid'])
                os.sched_setaffinity(0, set(cfg['cpus']))
                from morec_main_evaluator import resources_for, reference_root, run_qualification, run_files
                resources = resources_for(cfg)
                _, ledger = reference_root(cfg, resources)
                before_api = resources.api.summary()
                assert ledger.snapshot()['genuine_count'] == 0, 'Never replay a completed FIT'
                repair = read(recovery/'RETAIN_R13_REPAIR_V2.json')
                candidate = Path(repair['repaired'])
                original_candidate = Path(repair['source'])
                assert hashlib.sha256((candidate/'main_code.py').read_bytes()).hexdigest() == repair['repaired_sha256']
                assert hashlib.sha256((original_candidate/'main_code.py').read_bytes()).hexdigest() == repair['original_raw_sha256']
                from e2_shared_contract import REQUIRED_PROGRAM_FILES
                for name in REQUIRED_PROGRAM_FILES:
                    if name != 'main_code.py':
                        assert (candidate/name).read_bytes() == (original_candidate/name).read_bytes()
                    py_compile.compile(str(candidate/name), doraise=True)
                checkpoint = home/'retain/search/campaign/ROUND_13_PREPARED_CHECKPOINT.pkl'
                prepared = pickle.loads(checkpoint.read_bytes())['prepared']
                selected = prepared.innovation.selected_outcome
                assert selected.spec.digest == 'a753f12cb0c33852e9e77d57ea1522c3529ad4003da645597b7337d900001ea7'
                seeded = pickle.loads((home/'retain_INITIAL_STATE.pkl').read_bytes())
                parents = list(seeded.candidate_root_by_capability.values())
                assert len(parents) == 1
                inputs = read(original/'long/INPUTS.json')
                space = native.load_search_space_package(Path(inputs['space']['package_root']),
                    expected_source=native.SearchSpaceSource.AUTO_GENERATED,
                    expected_digest=inputs['space']['digest'], require_frozen=False)
                adapter = search_adapter(cfg, space)
                from ro_shared_controller_adapter import E2SharedQualifier
                adapter.initial_root = Path(parents[0])
                adapter.qualifier = E2SharedQualifier(adapter.initial_root)
                audit = adapter.qualifier.require_legal(candidate)
                mechanism = adapter._prepare_native_innovation(selected)['mechanism_program']
                adapter._conceptual_target_audit(candidate, mechanism)
                smoke_root = recovery/'retain_r13_smoke'
                smoke_script = Path(cfg['task_root'])/'controller_adapter/run_ro_shared_native_qualification_smoke.py'
                command = [cfg['worker_python'], '-B', '-u', str(smoke_script),
                    '--workspace', str(Path(cfg['task_root']).parent.parent),
                    '--candidate-root', str(candidate), '--output-dir', str(smoke_root),
                    '--train-seed', str(cfg['train_seed'])]
                save('QUALIFICATION_RUNNING', placement=placement, mutation_audit=audit,
                     selected_spec_digest=selected.spec.digest,
                     prepared_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     repair=repair, qualification_command=command)
                completed = run_qualification(cfg, command, timeout=900)
                if completed.returncode != 0:
                    raise RuntimeError('Qualification failed: '+(completed.stderr or completed.stdout)[-4000:])
                smoke = read(smoke_root/'RO_SHARED_QUALIFICATION_SMOKE_RESULT.json')
                assert smoke['status'] == 'SUCCESS' and smoke['completed_epochs'] == 1
                assert isinstance(smoke['training_batches'], int) and smoke['training_batches'] > 0
                assert smoke['test_accessed'] is False
                save('FIT_RUNNING', qualification=smoke)
                files = {name:(candidate/name).read_text() for name in REQUIRED_PROGRAM_FILES}
                result = run_files(cfg, files)
                after_api = resources.api.summary()
                assert after_api == before_api, 'Unexpected provider ledger change'
                save('NATIVE_RECOVERY_RETURNED', result=result, api=after_api,
                     evaluation=ledger.snapshot(), gpu=resources.gpu.snapshot(),
                     test_accessed=False, scientific_opportunity_index=1,
                     raw_failed_round_preserved=True,
                     campaign_state_integration='LINKED_RECOVERY_SIDECAR_NOT_YET_PROJECTED')
    except BaseException as error:
        save('ENGINEERING_RECOVERY_FAILED', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('home', type=Path)
    parser.add_argument('original', type=Path)
    args = parser.parse_args()
    main(args.home, args.original)
