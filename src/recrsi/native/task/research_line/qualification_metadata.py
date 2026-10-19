"""Carry one source-bound candidate mechanism recipe across qualification/train.

Install after the existing entry.setup imports runtime_fix. Does not edit its
source, change worker config, regenerate proposals or replay physical work.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def carry_mechanism_recipe(original, config, source):
    raw = Path(source).read_bytes()
    source_digest = hashlib.sha256(raw).hexdigest()
    request = (Path(config['output']) / 'graph_qualifications' / source_digest /
               'native_jobs/root/request.json')
    if not request.is_file():
        return original(config, source)
    saved = json.loads(request.read_bytes())['config']
    cached = saved['graph_config']
    recipe = cached.get('candidate_mechanism_recipe')
    if recipe is None:
        return original(config, source)

    from score_binding import bind_source_contract
    from graph_autospace import digest
    from open_mechanism import VERSION
    if (recipe.get('version') != VERSION
            or digest(recipe['definition']) != recipe.get('definition_digest')):
        raise ValueError('Cached candidate mechanism definition identity differs')
    effect = recipe['definition']['score_effect']
    entries = [] if effect['parameterization'] == 'UNCHANGED' else [dict(
        primitive_id='candidate-mechanism:' + recipe['definition_digest'],
        score_parameterization=effect['parameterization'],effect_phase=effect['phase'])]
    required = dict(selected_spec_digest=recipe['spec_digest'], selected_primitive_ids=[], entries=entries)
    bound = bind_source_contract(raw, required)
    if bound != cached.get('qualification_score_contract'):
        raise ValueError('Cached candidate mechanism/score/source binding differs')
    if 'candidate_mechanism_recipe' in config and config['candidate_mechanism_recipe'] != recipe:
        raise ValueError('Current candidate mechanism recipe differs from qualified recipe')
    if 'qualification_score_contract' in config and config['qualification_score_contract'] != bound:
        raise ValueError('Current score contract differs from qualified contract')

    # Add exactly the already-qualified recipe. Keep every scientific/operational
    # config field in the original equality check, including this restored field.
    # Remove an already-present score contract only to avoid the legacy early
    # return; original must still verify source, full config and score identity.
    carried = {k: v for k, v in config.items() if k != 'qualification_score_contract'}
    carried['candidate_mechanism_recipe'] = recipe
    return original(carried, source)


@contextmanager
def qualification_metadata(runtime_fix):
    original = runtime_fix.qualified_config
    def qualified(config, source):
        return carry_mechanism_recipe(original, config, source)
    with patch.object(runtime_fix, 'qualified_config', qualified):
        yield


@contextmanager
def entry_metadata(entry):
    """Same fix for fresh repeats 02/03 through their unchanged entry.main CLI."""
    setup = entry.setup
    def configured(*args, **kwargs):
        module, inputs = setup(*args, **kwargs)
        import runtime_fix
        execute = module.execute
        def run(*execute_args, **execute_kwargs):
            with qualification_metadata(runtime_fix):
                return execute(*execute_args, **execute_kwargs)
        module.execute = run
        return module, inputs
    with patch.object(entry, 'setup', configured):
        yield
