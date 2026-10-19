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


@pytest.mark.parametrize("custom", [False, True, "sequential"])
def test_native_worker_train_evaluate_reload(tmp_path, monkeypatch, custom):
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
    sequential = custom == 'sequential'
    if sequential:
        header = 'user_id:token\titem_id_list:token_seq\titem_id:token\n'
        texts['train'] = header + ''.join(f'u{u}\t' + ' '.join(f'i{j}' for j in range(1,i)) + f'\ti{i}\n' for u in range(1,5) for i in range(2,11))
        texts['dev'] = header + ''.join(f'u{u}\t' + ' '.join(f'i{j}' for j in range(1,11)) + f'\ti{10+u}\n' for u in range(1,5))
    dataset = 'cds' if sequential else 'my_catalog' if custom else 'electronics'
    files={}
    for role,text in texts.items():
        p=data/(dataset+'.'+({'train':'train.inter','dev':'dev.inter'}.get(role,role)))
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
    if sequential:
        model.write_bytes((native/'templates/sasrec.py').read_bytes())
    pool=data/'pool.npz'
    np.savez(pool,users=np.arange(1,5),candidates=np.tile(np.arange(1,101),(4,1)))
    binding=dict(status='FROZEN',line='ngcf',task_id='NGCF_EL_NATIVE_SEARCH_20261013',dataset='electronics',
        data_files=files,root_source_sha256=worker.digest(model),model_config={},
        training=dict(epochs=2,patience=1,eval_step=1,batch_size=16,learning_rate=.01,weight_decay=0.),
        evaluation=dict(primary_metric='NDCG@5',metrics=['NDCG','Hit','Recall','MRR','Precision'],topk=[5,10],batch_size=4096,candidate_universe='FIXED_100_UNIFORM',seed=1),
        dev_candidate_pool=dict(path=str(pool),sha256=worker.digest(pool)))
    if custom:
        binding.update(adapter='recbole-graph-fixed100-v1', task_id='my-task-v1', dataset=dataset, line='my_pairwise_model')
    if custom:
        mapping = {"user_id": ["[PAD]"] + [f"u{u}" for u in range(1,5)],
                   "item_id": ["[PAD]"] + [f"i{i}" for i in range(1,121)]}
        binding.update(catalog_mapping_sha256=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),
                       metric_binding_version="graph.fixed100.ndcg5.hr5.se-style.v1")
        binding["evaluation"].update(seed=72026, secondary_metric="Hit@5")
    if sequential:
        template = json.loads((native/'templates/sasrec-cds-binding.json').read_text())
        binding.update(adapter='recbole-sequential-fixed100-v1', line='sasrec', task_id='sasrec-cds',
                       metric_binding_version=template['metric_binding_version'], model_config=template['model_config'])
        binding['model_config'].update(hidden_size=8, inner_size=16, n_layers=1, MAX_ITEM_LIST_LENGTH=10)
        # RecBole maps target IDs before sequence aliases for this supplied dataset.
        mapping['item_id'] = ['[PAD]'] + [f'i{i}' for i in range(2,15)] + ['i1'] + [f'i{i}' for i in range(15,121)]
        binding['catalog_mapping_sha256'] = hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
        # Target i11..i14 maps to10..13; negative IDs15..113 exclude all TRAIN items.
        np.savez(pool, users=np.arange(4,0,-1), candidates=np.asarray([[9+u]+list(range(15,114)) for u in range(4,0,-1)]))
        binding['dev_candidate_pool']['sha256'] = worker.digest(pool)
    binding_path = tmp_path / "baseline-binding.json"
    binding_path.write_text(json.dumps(binding))
    worker.inspect_binding(binding,execute=True)
    monkeypatch.chdir(tmp_path)
    result=worker.execute(binding,model,tmp_path/'attempt',51001,'cpu','root',worker.digest(binding_path))
    assert result['status']=='SUCCESS'
    assert result['epochs_completed']==2
    assert result['checkpoint_verification']['exact'] is True
    assert result['checkpoint_verification']['optimizer_steps']==0
    assert result['test_accessed'] is False
    assert len((tmp_path/'attempt/LEARNING_CURVE.jsonl').read_text().splitlines())==2
    assert (tmp_path/'attempt/TRAIN_SIGNAL.jsonl').is_file()
    assert worker.digest(Path(result['checkpoint']))==result['checkpoint_sha256']

    if sequential:
        import graph_qualification_worker as qualification
        qualification_binding = dict(binding, catalog_mapping_path=str(tmp_path/'attempt/ID_MAPPING.json'))
        qualified = qualification.run(qualification_binding, model, tmp_path/'qualification', 51001, 'cpu')
        assert qualified['status'] == 'QUALIFIED'
        assert qualified['validation_rows_accessed'] is False
        assert qualified['optimizer_steps'] == 0

    if custom:
        # Assemble a user task from this actual CPU measurement. The dummy
        # host paths/CPU ledger below are inspection fixtures, never run inputs.
        from recrsi.config import initialize, check_config, native_module
        from recrsi.prepare import prepare
        import subprocess
        project = initialize(tmp_path / "project", adapter="recbole-graph-fixed100-v1")
        cfg = json.loads((project / "task.json").read_text())
        runtime = json.loads((project / "environment.json").read_text())
        ctx = native_module("_worker_test_context", "task/research_line/graph_native_inputs.py")
        assets = {"binding":binding_path, "root_binding":binding_path,
                  "root_source":model, "root_mapping":tmp_path/"attempt/ID_MAPPING.json"}
        records = dict(root_calibration={"status":"SUCCESS", "result":result},
            root_gpu_ledger={"software_fixture_cpu_only":True},
            facts=dict(task_id=binding["task_id"], task="Synthetic custom task integration fixture",
                root_source_sha256=worker.digest(model), data_sha256={k:v["sha256"] for k,v in files.items()},
                catalog_mapping_sha256=result["catalog_mapping_sha256"], simulation_context=ctx.CONTEXT_RULE,
                original_model_config=binding["model_config"], training=binding["training"], evaluation=binding["evaluation"],
                selected_root_metrics=result["metrics"], root_reference_cost={"scope":"CPU software fixture"}),
            literature={"documents":[{"title":"Synthetic fixture"}]})
        for key, value in records.items():
            assets[key] = project / (key + ".json")
            assets[key].write_text(json.dumps(value))
        cfg["adapter"] = binding["adapter"]
        cfg["task"] = {k:str(v) for k,v in assets.items()}
        (project / "task.json").write_text(json.dumps(cfg))
        private_api = project / "private.api-config"
        private_api.write_text("base_url: https://example.invalid/v1\napi_key: fixture-not-a-key\n")
        import recbole
        runtime.update(worker_python=sys.executable, recbole_root=str(Path(recbole.__file__).parent.parent),
                       driver_root=str(project), resource_slots=str(project), api_config=str(private_api))
        (project / "environment.json").write_text(json.dumps(runtime))
        assert check_config(project / "task.json")["task_id"] == binding["task_id"]
        prepared = prepare(project / "task.json", tmp_path / "prepared")
        manifest = json.loads((prepared / "TASK.json").read_text())
        assert manifest["evidence_class"] == "USER_CONFIGURED_SOFTWARE_RUN"
        inspected = subprocess.run([sys.executable,"-m","recrsi","inspect",str(prepared)],
                                   text=True, capture_output=True)
        assert inspected.returncode == 0, inspected.stderr
        report = json.loads(inspected.stdout)
        assert report["identity"]["environment"] == binding["task_id"]
        assert report["identity"]["phase"] == "USER_RESEARCH"
        assert report["evidence_class"] == "USER_CONFIGURED_SOFTWARE_RUN"
