"""CPU software fixture: real optimizer, fixed-100 scoring and checkpoint reload.

This synthetic fixture is not a research run or evidence of recommendation quality.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


def test_native_worker_train_evaluate_reload(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    import numpy as np
    import recrsi
    native=Path(recrsi.__file__).parent/'native'
    source=native/'task/search_source'
    measurement=native/'task/research/measured_execution'
    composition=native/'task/research_line/future_graph_composition'
    for p in (source,measurement,composition):monkeypatch.syspath_prepend(str(p))
    if os.environ.get('RECBole_ROOT'):monkeypatch.syspath_prepend(os.environ['RECBole_ROOT'])
    spec=importlib.util.spec_from_file_location("sampled_eval",measurement/"evaluation_metric_contract_v1/runtime/sampled_eval_accelerated.py")
    sampled=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampled)
    monkeypatch.setitem(sys.modules,"sampled_eval",sampled)
    import graph_worker_measured as worker
    import torch
    torch.set_num_threads(1)
    data=tmp_path/'input';data.mkdir()
    train='user_id:token\titem_id:token\n'+''.join(f'u{u}\ti{i}\n' for u in range(1,5) for i in range(1,11))
    dev='user_id:token\titem_id:token\n'+''.join(f'u{u}\ti{10+u}\n' for u in range(1,5))
    texts={'train':train,'dev':dev,'user':'user_id:token\n'+''.join(f'u{u}\n' for u in range(1,5)),
           'item':'item_id:token\n'+''.join(f'i{i}\n' for i in range(1,121))}
    files={}
    for role,text in texts.items():
        p=data/('electronics.'+({'train':'train.inter','dev':'dev.inter'}.get(role,role)))
        p.write_text(text);files[role]={'path':str(p),'sha256':worker.digest(p)}
    model=tmp_path/'model.py'
    model.write_text('''import torch
from torch import nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
class FreshCandidateModel(GeneralRecommender):
    input_type=InputType.PAIRWISE
    def __init__(self,config,dataset):
        super().__init__(config,dataset)
        self.users=nn.Embedding(self.n_users,8)
        self.items=nn.Embedding(self.n_items,8)
    def calculate_loss(self,interaction):
        u=self.users(interaction[self.USER_ID])
        p=self.items(interaction[self.ITEM_ID])
        n=self.items(interaction[self.NEG_ITEM_ID])
        return -torch.nn.functional.logsigmoid((u*(p-n)).sum(-1)).mean()
    def predict(self,interaction):
        return (self.users(interaction[self.USER_ID])*self.items(interaction[self.ITEM_ID])).sum(-1)
    def full_sort_predict(self,interaction):
        return self.users(interaction[self.USER_ID])@self.items.weight.T
''')
    pool=data/'pool.npz'
    np.savez(pool,users=np.arange(1,5),candidates=np.tile(np.arange(1,101),(4,1)))
    binding=dict(status='FROZEN',line='ngcf',task_id='NGCF_EL_NATIVE_SEARCH_20261013',dataset='electronics',
        data_files=files,root_source_sha256=worker.digest(model),model_config={},
        training=dict(epochs=2,patience=1,eval_step=1,batch_size=16,learning_rate=.01,weight_decay=0.),
        evaluation=dict(primary_metric='NDCG@5',metrics=['NDCG','Hit','Recall','MRR','Precision'],topk=[5,10],batch_size=4096,candidate_universe='FIXED_100_UNIFORM',seed=1),
        dev_candidate_pool=dict(path=str(pool),sha256=worker.digest(pool)))
    worker.inspect_binding(binding,execute=True)
    monkeypatch.chdir(tmp_path)
    result=worker.execute(binding,model,tmp_path/'attempt',123,'cpu','root',hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest())
    assert result['status']=='SUCCESS'
    assert result['epochs_completed']==2
    assert result['checkpoint_verification']['exact'] is True
    assert result['checkpoint_verification']['optimizer_steps']==0
    assert result['test_accessed'] is False
    assert len((tmp_path/'attempt/LEARNING_CURVE.jsonl').read_text().splitlines())==2
    assert (tmp_path/'attempt/TRAIN_SIGNAL.jsonl').is_file()
    assert worker.digest(Path(result['checkpoint']))==result['checkpoint_sha256']
