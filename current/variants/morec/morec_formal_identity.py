"""Exact prospective MoRec MAIN identities; no run starts or result reuse."""
from copy import deepcopy

METHODS = ('ro', 'rh', 'se', 'deep', 'openevolve', 'shinka')
ENVIRONMENT = 'MoRec_text_initialized_SASRec_Amazon_CDs'
PHASE = 'FORMAL_RQ1'
EVAL_SEED = 72026


def identity_for(method, repeat):
    if method not in METHODS or type(repeat) is not int or not 1 <= repeat <= 8:
        raise ValueError('MoRec MAIN requires one of six methods and repeat 1..8')
    return dict(environment=ENVIRONMENT, method=method, phase=PHASE, repeat=repeat,
                run=f'{method}-{repeat:02d}', search_seed=41000+repeat,
                train_seed=51000+repeat)


def formal_contract(template, repeat):
    identity = identity_for('ro', repeat)
    contract = deepcopy(template)
    contract['bootstrap_seed'] = 91025+repeat
    contract['repeat_policy']['search_seeds'] = [identity['search_seed']]
    contract['research_provenance']['comparison_status'] = 'PROSPECTIVE_MOREC_FORMAL_RQ1'
    # Signed protocol/amendment identity, baseline, six axes, generation allowance,
    # mutable regions and all evaluation restrictions are inherited unchanged.
    return contract


def validate_formal_launch(launch, config):
    identity = identity_for('ro', launch.get('repeat'))
    expected = dict(search_seed=identity['search_seed'], training_seed=identity['train_seed'],
                    autospace_seed=91025+identity['repeat'])
    if any(type(launch.get(key)) is not int or launch[key] != value for key, value in expected.items()):
        raise ValueError('MoRec formal launch seed pairing differs from MAIN section 3')
    if (config.get('run_identity') != identity or config.get('train_seed') != identity['train_seed']
            or config.get('eval_seed') != EVAL_SEED):
        raise ValueError('MoRec formal evaluator identity or train/eval seed mismatch')


def formal_task_contract(template, repeat):
    identity = identity_for('ro', repeat)
    contract = deepcopy(template)
    if contract.get('eval_seed') != EVAL_SEED:
        raise ValueError('Do not change the frozen common evaluation seed')
    contract.update(search_seed=identity['search_seed'], train_seed=identity['train_seed'])
    return contract


def cohort_rows():
    rows = []
    for repeat in range(1, 9):
        for method in METHODS:
            identity = identity_for(method, repeat)
            rows.append(dict(identity=identity, eval_seed=EVAL_SEED,
                             autospace_seed=91025+repeat if method == 'ro' else None,
                             output_relative=f'runs/{identity["run"]}',
                             required_root_train_seed=identity['train_seed'],
                             root_status='NOT_PREPARED_OR_EXECUTED',
                             execution_status='NOT_STARTED'))
    return rows
