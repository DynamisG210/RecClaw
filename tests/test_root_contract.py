import copy
import pytest
from recrsi.config import native_module

validate=native_module('_test_root_contract','task/search_source/root_contract.py').validate_root_receipt


@pytest.fixture
def receipt():
    return dict(status='SUCCESS',result=dict(role='root',checkpoint_reload_verified=True),
                artifacts=dict(native_evidence=dict(termination='EPOCH_LIMIT',unresolved_gpu_seconds=0,
                    physical_cost=dict(gpu_seconds=2.,active_seconds=3.)))), dict(jobs={'root':dict(finished_ns=1,gpu_seconds=2.)})


@pytest.mark.parametrize('broken',['missing_receipt','unsettled','cost_mismatch','checkpoint','nonfinite'])
def test_rejects_incomplete_baseline(receipt,broken):
    envelope,physical=copy.deepcopy(receipt)
    if broken=='missing_receipt':envelope.pop('artifacts')
    elif broken=='unsettled':physical['jobs']['root']['finished_ns']=None
    elif broken=='cost_mismatch':physical['jobs']['root']['gpu_seconds']=1.
    elif broken=='checkpoint':envelope['result']['checkpoint_reload_verified']=False
    else:envelope['artifacts']['native_evidence']['physical_cost']['gpu_seconds']=float('nan')
    with pytest.raises(ValueError):validate(envelope,physical)
