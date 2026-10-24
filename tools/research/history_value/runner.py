"""Native continuation composition using independent manifests and seeded states."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def compose(home, original, arm, preflight, placement=None, engineering_resume=False):
    cfg = read(home/arm/'EVALUATOR_CONFIG.json')
    allocation = read(home/'ALLOCATION.json')
    phase_round_end = allocation.get('phase_round_end', 15)
    if not 13 <= phase_round_end <= 15:
        raise ValueError('Phase must remain within the preregistered three-opportunity pair')
    input_contract = None
    if (home/'INITIAL_INPUT_CONTRACT_V2.json').exists():
        from initial_input_contract import bind_initial_prompt, check_fresh_recovery
        input_contract = read(home/'INITIAL_INPUT_CONTRACT_V2.json')
        check_fresh_recovery(Path(cfg['output']), input_contract)
    if placement is not None:
        cfg.update(cpus=placement['cpus'], gpu_uuid=placement['gpu_uuid'])
    launch = read(home/arm/'LAUNCH.json')
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(home/'runtime'), str(original), str(original/'runtime'),
                   str(Path(cfg['task_root'])/'controller_adapter'),
                   *read(original/'ORIGINAL_RUNTIME_PATHS.json')]
    from e2_budget_binding import bind_config
    bind_config(cfg)
    provider_cfg = read(cfg['provider_config'])
    os.environ.update(RECCLAW_RECBOLE_ROOT=cfg['task_root'],
        RECCLAW_PYTHON_EXECUTABLE=cfg['worker_python'],
        RECCLAW_API_CONFIG=provider_cfg['api_config'], E2_EXECUTION_MODE='MAIN',
        E2_SHARED_TRAIN_SEED=str(cfg['train_seed']), E2_SHARED_SEARCH_SEED=str(launch['search_seed']),
        E2_SHARED_CANDIDATE_PYTHON=cfg['worker_python'], E1_RQ1_PROVIDER_CONFIG=cfg['provider_config'],
        E2_SHARED_CONTROLLER_SLOT_CEILING='15', TMPDIR=str(root))
    from original_run_morec_ro import research_config
    import auto_search_space as native
    import e2_ro_search_space_adapter as adapter_module
    if Path(adapter_module.__file__).resolve() != Path(cfg['task_root'])/'controller_adapter/e2_ro_search_space_adapter.py':
        raise ValueError('Wrong original task adapter import')
    from morec_main_evaluator import resources_for, reference_root
    from morec_ro_execution import MorecROExecution, search_adapter
    from ro_main_provider import native_ro_provider
    from learning_components import FrozenSearchPolicy
    from recclaw_core.research_line.standalone import compose_standalone_campaign
    from recclaw_core.experiments.helix_abc_v1 import lab_api_broker
    from run_native_json import include_schema, json_transport
    # Match the original entry's final import precedence after native imports.
    sys.path.insert(0, str(original))
    spec = importlib.util.spec_from_file_location('history_parent_caller', original/'parent_caller.py')
    parent_caller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent_caller)
    inputs = read(original/'long/INPUTS.json')
    space = native.load_search_space_package(Path(inputs['space']['package_root']),
        expected_source=native.SearchSpaceSource.AUTO_GENERATED,
        expected_digest=inputs['space']['digest'], require_frozen=False)
    resources = resources_for(cfg)
    _, ledger = reference_root(cfg, resources)
    config = replace(research_config(launch, cfg), round_count=15,
                     provider_maximum_physical_attempts=1,
                     close_exhausted_no_metric_slot=True,
                     execution_purpose='HISTORY_VALUE_SAME_INCUMBENT')
    from recclaw_core.research_line.standalone import _source_baseline_identity
    def bind_identity(state):
        memory = dict(state.context.scientific_memory)
        common = dict(memory.get('global_memory', {}))
        common['standalone_identity'] = _source_baseline_identity(config)
        memory['global_memory'] = common
        return replace(state, context=replace(state.context, scientific_memory=memory))
    state_path = root/'campaign/CAMPAIGN_STATE.pkl'
    if state_path.exists() and not (root/'STAGE_STARTED.json').exists():
        # Engineering migration of the just-created, zero-call initial state:
        # the native resume identity is common metadata, not research history.
        if resources.api.summary()['requests'] != 0 or ledger.snapshot()['genuine_count'] != 0:
            raise ValueError('Initial identity migration requires zero new work')
        current = pickle.loads(state_path.read_bytes())
        fixed = bind_identity(current)
        if fixed != current:
            backup = root/'PRE_IDENTITY_BINDING_STATE.pkl'
            if not backup.exists():
                backup.write_bytes(state_path.read_bytes())
            state_path.write_bytes(pickle.dumps(fixed, protocol=pickle.HIGHEST_PROTOCOL))
        # Keep the native projection paired with its pickle, including recovery
        # after a prior interrupted initial metadata write.
        from recclaw_core.research_line.campaign import canonical_json_bytes, canonical_value
        projection = canonical_value({'schema': 'recclaw.research-line.campaign-state-projection.v1',
                                      'state_digest': fixed.digest, 'state': fixed.to_dict()})
        state_path.with_suffix('.json').write_bytes(canonical_json_bytes(projection) + b'\n')
    adapter = search_adapter(cfg, space)
    seeded = pickle.loads((home/(arm+'_INITIAL_STATE.pkl')).read_bytes())
    historical = pickle.loads((original/'long/search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes()).state_before
    incumbent_id = seeded.frontier['incumbent_candidate_id']
    incumbent = next(c for c in (*seeded.carryover_open_candidates, *historical.carryover_open_candidates)
                     if c.candidate_id == incumbent_id)
    adapter.initial_root = Path(seeded.candidate_root_by_capability[incumbent.capability_ref])
    adapter.qualifier = type(adapter.qualifier)(adapter.initial_root)
    execution = MorecROExecution(cfg)
    original_identity = lab_api_broker.LabApiCanaryBrokerV1.sealed_request_identity
    def identity(broker, **kw):
        return json_transport(original_identity(broker, **kw))
    if preflight:
        def deny(event, args):
            if event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen', 'os.system'):
                raise RuntimeError('Composition preflight forbids provider and worker execution')
        sys.addaudithook(deny)
    with (root/'SEARCH_OWNER.lock').open('a+b') as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.enter_context(parent_caller.bind(adapter))
        # The incumbent is a construction source, not an eligible old search
        # candidate. Keep it available even when carryover search is empty.
        historical = pickle.loads((original/'long/search/campaign/ROUND_13_CHECKPOINT.pkl').read_bytes()).state_before
        incumbent_id = seeded.frontier['incumbent_candidate_id']
        # The exact incumbent was resolved from the continued state above.
        original_bind = adapter.bind_implementation_sources
        def bind_sources(*, candidates, candidate_roots):
            parents = tuple(candidates)
            if not any(c.candidate_id == incumbent_id for c in parents):
                parents += (incumbent,)
            original_bind(candidates=parents, candidate_roots=candidate_roots)
        stack.enter_context(patch.object(adapter, 'bind_implementation_sources', bind_sources))
        # Make the same incumbent available before the first producer projection,
        # rather than relying on a later implementation-source binding.
        bind_sources(candidates=(), candidate_roots=seeded.candidate_root_by_capability)
        recovery = root/'FIRST_PAID_BATCH_RECOVERY.pkl'
        if recovery.exists():
            from recclaw_core.research_line import campaign as campaign_module
            paid = pickle.loads(recovery.read_bytes())
            native_round = campaign_module.run_research_round
            def recover_first_batch(*args, **kwargs):
                context = kwargs.get('context', args[0] if args else None)
                if context.round_index == 13:
                    assert context.digest == paid.context_digest
                    kwargs['prepared_round'] = None
                    with patch.dict(native_round.__globals__, {'produce_research_specs': lambda *a, **k: paid.producer_outcomes}):
                        return native_round(*args, **kwargs)
                return native_round(*args, **kwargs)
            stack.enter_context(patch.object(campaign_module, 'run_research_round', recover_first_batch))
        call = stack.enter_context(native_ro_provider(cfg['provider_config']))
        stack.enter_context(patch.object(lab_api_broker.LabApiCanaryBrokerV1,
                                        'sealed_request_identity', identity))
        def provider_call(**kw):
            if input_contract is not None and 'prompt' in kw:
                from initial_input_contract import CONTEXT_MARKER
                if CONTEXT_MARKER in kw['prompt']:
                    kw['prompt'] = bind_initial_prompt(kw['prompt'], input_contract)
            return call(**include_schema(kw))
        composition = compose_standalone_campaign(config,
            resume=(root/'campaign/CAMPAIGN_STATE.pkl').exists(),
            provider_call=provider_call,
            launch=execution, search_space_adapter=adapter)
        policy = FrozenSearchPolicy(composition).install()
        stack.callback(policy.close)
        sources = {name: (adapter.initial_root/name).read_text() for name in
                   ('main_code.py', 'models.py', 'diagnosis_tools.py',
                    'self_evolverec_interface.py', 'morec_initialization.py')}
        actual_producer = getattr(composition.provider, '_producer', composition.provider)
        actual_producer.proposal_context_appendix += ('\nIdentical initial incumbent source:\n'
                                                     + json.dumps(sources, sort_keys=True))
        receipt = root/'INITIAL_STATE_ADOPTED.json'
        if not receipt.exists():
            if composition.campaign.state.next_round_index != 1:
                raise ValueError('Cannot replace an advanced campaign state')
            raw = (home/(arm+'_INITIAL_STATE.pkl')).read_bytes()
            state = bind_identity(pickle.loads(raw))
            if state.campaign_id != launch['campaign_id']:
                raise ValueError('Seeded state campaign mismatch')
            composition.campaign._state = state
            composition.campaign._persist_state()
            receipt.write_text(json.dumps({'sha256': hashlib.sha256(raw).hexdigest(),
                'next_round': state.next_round_index, 'arm': arm,
                'api_calls': resources.api.summary()['requests'],
                'training_jobs': len(resources.gpu.snapshot()['jobs'])}, indent=2))
        composition.campaign._persist_state()
        if preflight:
            state = composition.campaign.state
            assert state.active_profile.campaign_id == state.context.campaign_id
            assert state.active_profile.profile_ref == state.context.active_profile_ref
            assert state.active_profile.profile_digest == state.context.active_profile_digest
            return {'stage': 'NATIVE_CAMPAIGN_STATE_PERSISTED', 'arm': arm,
                    'next_round': composition.campaign.state.next_round_index,
                    'api': resources.api.summary(), 'evaluation': ledger.snapshot()}
        if placement is None:
            raise RuntimeError('Physical admission missing')
        with (root/('ENGINEERING_RESUME_STARTED.json' if engineering_resume else 'STAGE_STARTED.json')).open('x') as started:
            json.dump({'pid': os.getpid(), 'birth_ticks': Path('/proc/self/stat').read_text().split()[21],
                       'next_round': composition.campaign.state.next_round_index,
                       'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, started)
        os.sched_setaffinity(0, set(cfg['cpus']))
        from resource_stop import SearchResourceStop
        original_round = composition.campaign.run_round
        def bounded_round(*args, **kwargs):
            record = original_round(*args, **kwargs)
            if getattr(record, 'status', None) == 'INCOMPLETE':
                raise RuntimeError('Incomplete native opportunity persisted; diagnose before another paid generation')
            return record
        composition.campaign.run_round = bounded_round
        try:
            records = composition.run(round_count=max(0, phase_round_end+1-composition.campaign.state.next_round_index))
            result = {'stage': 'NATIVE_STAGE_RETURNED', 'rounds': len(records)}
        except SearchResourceStop as error:
            result = {'stage': 'RESOURCE_STOP', 'reason': str(error)}
        result.update(api=resources.api.summary(), evaluation=ledger.snapshot(),
                      gpu=resources.gpu.snapshot(), test_accessed=False)
        (root/'STAGE_RESULT.json').write_text(json.dumps(result, default=str, indent=2))
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('home', type=Path)
    p.add_argument('original', type=Path)
    p.add_argument('arm', choices=('retain', 'reset'))
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--engineering-resume', action='store_true')
    a = p.parse_args()
    if a.preflight:
        result = compose(a.home, a.original, a.arm, True)
    else:
        from admission import admitted
        with (a.home/'PAIR_EXECUTION.lock').open('a+b') as pair_lock:
            fcntl.flock(pair_lock, fcntl.LOCK_EX)
            with admitted(a.home, read(a.home/a.arm/'EVALUATOR_CONFIG.json'),
                          read(a.home/'SHARED_ADMISSION.json')) as placement:
                result = compose(a.home, a.original, a.arm, False, placement, a.engineering_resume)
    print(json.dumps(result, default=str))
