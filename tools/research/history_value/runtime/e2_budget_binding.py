"""Private E2 budget declaration shared by native controllers and evaluator children.

No asset/seed selection, provider routing, or scientific search-policy change.
Only pinned declarations can bind fresh E2 identities; completed runs cannot widen.
"""
import hashlib
import json
import math
import os
from pathlib import Path


def read_budget(identity=None):
    path=Path(os.environ['RECCLAW_E2_BUDGET_MANIFEST'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != os.environ['RECCLAW_E2_BUDGET_SHA256']:
        raise ValueError('Budget changed')
    data=json.loads(path.read_text())
    allocation_path=Path(data['allocation_path'])
    if hashlib.sha256(allocation_path.read_bytes()).hexdigest()!=data['allocation_sha256']:
        raise ValueError('Independent allocation changed')
    allocation=json.loads(allocation_path.read_text())
    if allocation['allocation_id'] not in ('history_value_same_incumbent_pair_20261024_v1',
                                         'history_value_same_incumbent_pair_20261024_v2'):
        raise ValueError('Wrong independent allocation')
    who=data['consumer']
    if allocation['allocation_id'].endswith('_v2'):
        if who['search_seed'] != allocation['search_seed'] or who['run'] != allocation['campaign_prefix']+'-'+who['arm']:
            raise ValueError('V2 seed or campaign does not match its independent allocation')
    if data['schema']!='recrsi.history-value-budget.v1' or who['arm'] not in ('retain','reset'):
        raise ValueError('Wrong history study arm')
    if identity is not None and identity!=who:
        raise ValueError('Wrong identity')
    expected=dict(genuine=allocation['max_valid_candidate_fits_per_arm'],
        reference_api_usd=allocation['api_usd_cap_per_arm'],
        execution_seconds=allocation['gpu_seconds_cap_per_arm'],
        active_seconds=allocation['queue_admission_window_seconds'])
    if data['caps']!=expected or data['space_reference'] is not None:
        raise ValueError('Independent continuation resources differ')
    data['manifest_sha256']=digest
    return data


def bind_config(config):
    os.environ['RECCLAW_E2_BUDGET_MANIFEST']=str(Path(config['e2_budget_manifest']).resolve())
    os.environ['RECCLAW_E2_BUDGET_SHA256']=config['e2_budget_sha256']
    data=read_budget(config['run_identity'])
    if Path(config['output']).resolve()!=Path(data['output']).resolve():
        raise ValueError('E2 independent output differs')
    return data


def limit(identity,key,explicit=None):
    value=read_budget(identity)['caps'][key]
    if explicit is not None and explicit!=value:raise ValueError('E2 consumer ceiling differs')
    return value


def setup(data):
    source=data['space_reference']
    if source is None:return None
    factor=1
    return dict(budget_manifest_sha256=data['manifest_sha256'],source_evidence=source['source_evidence'],
        known_reference_api_usd=source['known_reference_api_usd']*factor,
        unknown_reference_hold_usd=source['unknown_reference_hold_usd']*factor,
        gpu_seconds=source['execution_seconds']*factor,active_seconds=source['active_seconds']*factor,
        physical_execution_repeated=False,physical_new_provider_calls=0,allocation=factor)


def carry_space(state,identity):
    expected=setup(read_budget(identity))
    if expected is not None:
        if 'autospace_reference_cost' in state and state['autospace_reference_cost']!=expected:
            raise ValueError('E2 space reference changed on resume')
        state['autospace_reference_cost']=expected
    elif 'autospace_reference_cost' in state:
        raise ValueError('RH cannot inherit RO space reference')


def api_ledger(base,settings):
    data=read_budget(settings['run_identity']);caps=data['caps'];limits=settings['api_limits']
    if (limits['normalized_unit_usd']!=14 or limits['max_normalized_u']!=caps['reference_api_usd']/14
            or limits['max_input_tokens']!=128000 or limits['max_output_tokens']!=20000
            or limits['input_estimate_margin']!=4096):
        raise ValueError('E2 provider budget or original MoRec token bound differs')
    if Path(settings['resource_root']).resolve()!=Path(data['output']).resolve():
        raise ValueError('E2 provider points at another output')
    expected_limits=dict(data['original_api_limits'],max_normalized_u=caps['reference_api_usd']/14)
    if limits!=expected_limits:raise ValueError('Original MoRec prices/request/token inputs changed')
    carry=setup(data)
    if carry is None:return base(settings['api_ledger'],**limits)
    expected=dict(event='RQ5_SPACE_REFERENCE',**carry)
    class Ledger(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            with self._lock:
                events=self.events();old=[e for e in events if e['event']=='RQ5_SPACE_REFERENCE']
                if old:
                    if old!=[expected]:raise ValueError('E2 API setup changed')
                elif events:raise ValueError('Setup must precede first E2 request')
                else:self._append_unlocked(expected)
        def _cost_totals(self,events):
            if [e for e in events if e['event']=='RQ5_SPACE_REFERENCE']!=[expected]:
                raise ValueError('E2 API setup missing or duplicated')
            a,b,models=super()._cost_totals(events)
            return a+carry['known_reference_api_usd']+carry['unknown_reference_hold_usd'],b,models
        def summary(self):
            result=super().summary()
            result['carried_space_reference']=carry
            result['new_request_reference_usd']=result['normalized_cost_usd']-carry['known_reference_api_usd']-carry['unknown_reference_hold_usd']
            result['reference_total_includes_unknown_setup_hold']=True
            return result
    return Ledger(settings['api_ledger'],**limits)


def prompt_limits(identity):
    c=read_budget(identity)['caps']
    return (f"At most {c['genuine']} new candidate evaluations and "
            f"{c['reference_api_usd']} reference USD for this arm, including "
            f"failed and unresolved calls; {c['execution_seconds']} GPU seconds. "
            "Historical source and space are reused without new setup calls.")
