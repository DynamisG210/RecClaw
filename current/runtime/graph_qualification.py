"""One durable train-only qualification per source, shared by every consumer.

RO calls this before making its qualification receipt. The common evaluator
reuses that exact result; no duplicate batch, extra genuine score or free work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def physical_qualification(job, folder):
    from graph_candidate_evaluator import configure, resources_for, run_stage
    from graph_reference_root import reference_root
    from graph_candidate_contract import qualify_source
    config = job['graph_config']
    closure = configure(config)
    resources = resources_for(config)
    _, ledger = reference_root(config, resources)
    source = folder / 'candidate.py'
    source.write_bytes(job['source'].encode('utf-8'))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    # Before RO materialization there is no genuine reservation; during normal
    # evaluation the same source owns one. The last reserved slot stays legal.
    reserved = any(row['source_sha256'] == digest for row in ledger.snapshot()['pending'].values())
    if reserved:
        resources.require_reserved_evaluation_capacity(digest)
    else:
        resources.require_capacity()
    try:
        qualify_source(job['source'])
    except (SyntaxError, ValueError) as error:
        return dict(status='CANDIDATE_FAILURE', failure_stage='source_contract',
                    error=str(error), source_sha256=digest, test_accessed=False, worker_started=False)
    result = run_stage(config, closure, resources, source, folder, 'qualification', reserved=reserved)
    result['source_sha256'] = digest
    return result


def qualify_program(config, source_path):
    from graph_candidate_evaluator import configure
    closure = configure(config, worker_process=False)
    from durable_evaluation import run_native_job
    source = Path(source_path).read_bytes().decode('utf-8')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    job = dict(output=str(Path(config['output']) / 'graph_qualifications' / digest),
               graph_config=config, source=source)
    return run_native_job(job, command_factory=lambda request, fd: [closure['python'], '-B',
        str(Path(__file__).resolve()), '--request', str(request), '--lock-fd', str(fd)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    os.set_inheritable(args.lock_fd, False)
    job = json.loads(args.request.read_bytes())['config']
    from graph_candidate_evaluator import configure
    configure(job['graph_config'])
    from durable_evaluation import write_json
    from resource_stop import SearchResourceStop
    try:
        value = physical_qualification(job, args.request.parent)
        result = dict(status='RETURNED', value=value)
    except SearchResourceStop as error:
        result = dict(status='RESOURCE_STOP', message=str(error))
    except Exception as error:
        result = dict(status='ERROR', error_type=type(error).__name__, message=str(error))
    write_json(args.request.parent / 'result.json', result)


if __name__ == '__main__':
    main()
