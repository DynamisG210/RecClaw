"""Sequential input and prediction boundaries, independent of recommendation quality."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from recrsi.config import initialize, native_module, sha


def test_default_project_is_sasrec_cds(tmp_path):
    root = initialize(tmp_path/'project')
    task = json.loads((root/'task.json').read_text())
    binding = json.loads((root/'inputs/binding.json').read_text())
    assert task['adapter'] == binding['adapter'] == 'recbole-sequential-fixed100-v1'
    assert (binding['line'], binding['dataset']) == ('sasrec', 'cds')
    assert binding['root_source_sha256'] == sha(root/'inputs/root_source.py')
    contract = native_module('_seq_source_contract','task/search_source/graph_candidate_contract.py')
    assert contract.qualify_source((root/'inputs/root_source.py').read_text())['status'] == 'ELIGIBLE_SOURCE'
    assert (root/'inputs/RecBole-LICENSE.txt').is_file()


@pytest.mark.parametrize('invalid', ['future', 'duplicate', 'ordering', 'length'])
def test_rejects_invalid_sequence_boundaries(tmp_path, invalid):
    worker = native_module('_seq_worker_contract','task/search_source/graph_worker.py')
    header='user_id:token\titem_id_list:token_seq\titem_id:token\n'
    train=tmp_path/'cds.train.inter';dev=tmp_path/'cds.dev.inter'
    train.write_text(header+'u1\ti1\ti2\nu1\ti1 i2\ti3\n')
    dev.write_text(header+'u1\ti1 i2 i3\ti4\n')
    b={'model_config':{'MAX_ITEM_LIST_LENGTH':3},'data_files':{'train':{'path':str(train)},'dev':{'path':str(dev)}}}
    assert worker.inspect_sequences(b) == {'u1':['i1','i2','i3']}
    if invalid=='future': dev.write_text(header+'u1\ti1 i2 i4\ti5\n')
    elif invalid=='duplicate': dev.write_text(header+'u1\ti1 i2 i3\ti4\nu1\ti1 i2 i3\ti5\n')
    elif invalid=='ordering': train.write_text(header+'u1\ti1\ti2\nu1\ti2 i1\ti3\n')
    else: b['model_config']['MAX_ITEM_LIST_LENGTH']=2
    with pytest.raises(ValueError):worker.inspect_sequences(b)


def test_sequence_scores_follow_pool_user_order_without_labels(tmp_path, monkeypatch):
    torch=pytest.importorskip('torch')
    from recbole.data.interaction import Interaction
    import numpy as np
    import recrsi
    source=Path(recrsi.__file__).parent/'native/task/search_source'
    monkeypatch.syspath_prepend(str(source))
    evaluator=native_module('_seq_score','task/research/measured_execution/evaluation_metric_contract_v1/runtime/sampled_eval_accelerated.py')
    class Model:
        USER_ID='user_id'; ITEM_ID='item_id'; ITEM_SEQ='item_id_list'; ITEM_SEQ_LEN='item_length'; n_items=105
        def eval(self):pass
        def full_sort_predict(self, batch):
            assert set(batch.interaction)=={'user_id','item_id_list','item_length'}
            target=batch[self.ITEM_SEQ][:,0]+1
            return torch.nn.functional.one_hot(target, self.n_items).float()
    pool=tmp_path/'pool.npz'
    np.savez(pool, users=[2,1], candidates=np.tile(np.arange(1,101),(2,1)))
    inter=Interaction({'user_id':torch.tensor([1,2]),'item_id':torch.tensor([6,10]),
        'item_id_list':torch.tensor([[5],[9]]),'item_length':torch.tensor([1,1])})
    data=SimpleNamespace(_dataset=SimpleNamespace(inter_feat=inter))
    metrics=evaluator.evaluate_dev(Model(),{'device':'cpu'},data,{'path':str(pool),'sha256':sha(pool)})
    assert metrics['ndcg@5']==metrics['hit@5']==1.


def test_discovery_abi_matches_selected_task(monkeypatch):
    import recrsi
    native=Path(recrsi.__file__).parent/'native/task/research_line'
    monkeypatch.syspath_prepend(str(native))
    monkeypatch.syspath_prepend(str(native/'future_graph_composition'))
    from abi_guidance import abi_scope
    import independent_discovery as discovery
    old=discovery.ABI
    with abi_scope({'adapter':'recbole-sequential-fixed100-v1'}):
        assert 'POINTWISE' in discovery.ABI['input_type']
        assert 'item_id_list' in discovery.ABI['interaction_fields']
        assert 'neg_item_id' not in discovery.ABI['interaction_fields']
        assert 'InputType.PAIRWISE' not in discovery.DISCOVERY_INSTRUCTION
    assert discovery.ABI is old
    with abi_scope({'adapter':'recbole-graph-fixed100-v1'}):
        assert 'PAIRWISE' in discovery.ABI['input_type']
