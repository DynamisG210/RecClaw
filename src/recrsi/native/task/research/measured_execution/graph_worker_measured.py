"""Native task worker: train, evaluate and verify checkpoints inside the worker boundary."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def inspect_sequences(binding):
    """Require chronological TRAIN prefixes and one next-item DEV target per user."""
    import csv
    maximum = binding['model_config']['MAX_ITEM_LIST_LENGTH']
    if type(maximum) is not int or maximum < 1:
        raise ValueError('MAX_ITEM_LIST_LENGTH must be a positive integer')
    histories = {}
    dev_users = set()
    for role in ('train', 'dev'):
        with open(binding['data_files'][role]['path'], encoding='utf-8', newline='') as stream:
            rows = csv.DictReader(stream, delimiter='\t')
            if set(rows.fieldnames or []) != {'user_id:token', 'item_id_list:token_seq', 'item_id:token'}:
                raise ValueError('Sequential input requires user_id, item_id_list and item_id columns')
            for row in rows:
                user, target = row['user_id:token'], row['item_id:token']
                prefix = row['item_id_list:token_seq'].split()
                if not user or not target or not prefix or len(prefix) > maximum:
                    raise ValueError('Sequential input requires nonempty bounded prefixes and IDs')
                if role == 'train':
                    if user in histories and prefix != histories[user][-maximum:]:
                        raise ValueError('TRAIN rows must be consecutive chronological prefixes per user')
                    histories[user] = histories.get(user, prefix) + [target]
                else:
                    if user in dev_users or user not in histories or prefix != histories[user][-maximum:]:
                        raise ValueError('DEV requires one TRAIN-derived prefix per user')
                    if target in histories[user]:
                        raise ValueError('Fixed100 DEV target must be unseen in TRAIN')
                    dev_users.add(user)
    if not histories or not dev_users:
        raise ValueError('Sequential TRAIN and DEV must be nonempty')
    return histories


def inspect_binding(binding, *, execute=False):
    if execute and binding["status"] != "FROZEN":
        raise ValueError("Freeze the task binding before execution")
    if binding.get("adapter") in ("recbole-graph-fixed100-v1", "recbole-sequential-fixed100-v1"):
        for key in ("task_id", "dataset", "line"):
            value = binding.get(key)
            if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
                raise ValueError("Custom graph task requires a safe explicit " + key)
        if binding["task_id"] == "NGCF_EL_NATIVE_SEARCH_20261013":
            raise ValueError("Custom tasks must use their own identity, not the historical experiment ID")
    else:
        if binding.get("adapter") is not None:
            raise ValueError("Unsupported task adapter")
        if binding["line"] not in ("ngcf", "lightgcnpp"):
            raise ValueError("Unknown independent graph line")
        if binding["task_id"] != "NGCF_EL_NATIVE_SEARCH_20261013" or binding["dataset"] != "electronics":
            raise ValueError("Declare recbole-graph-fixed100-v1 for a custom task")
    if set(binding["data_files"]) != {"train", "dev", "user", "item"}:
        raise ValueError("Only the four original search files are permitted")
    for role, item in binding["data_files"].items():
        expected = {"train": binding["dataset"]+".train.inter", "dev": binding["dataset"]+".dev.inter",
                    "user": binding["dataset"]+".user", "item": binding["dataset"]+".item"}[role]
        if Path(item["path"]).name != expected or digest(item["path"]) != item["sha256"]:
            raise ValueError("Search asset identity mismatch: " + role)
    recipe = binding["training"]
    if recipe["epochs"] < 1 or recipe["patience"] < 1 or recipe["eval_step"] < 1:
        raise ValueError("Explicit positive training horizon and stopping settings required")
    metric, cutoff = binding["evaluation"]["primary_metric"].split("@")
    if metric not in binding["evaluation"]["metrics"] or int(cutoff) not in binding["evaluation"]["topk"]:
        raise ValueError("Primary metric must be part of the fixed full-sort evaluator")
    if binding["evaluation"]["candidate_universe"] != "FIXED_100_UNIFORM":
        raise ValueError("Graph task uses full-sort evaluation")
    # Prevent model hyperparameters from overriding trusted scientific inputs.
    allowed = {"embedding_size", "n_layers", "alpha", "beta", "gamma", "reg_weight",
               "hidden_size_list", "message_dropout"}
    if binding.get("adapter") == "recbole-sequential-fixed100-v1":
        allowed = {"n_layers", "n_heads", "hidden_size", "inner_size", "hidden_dropout_prob",
                   "attn_dropout_prob", "hidden_act", "layer_norm_eps", "initializer_range",
                   "loss_type", "MAX_ITEM_LIST_LENGTH"}
        if binding["model_config"].get("loss_type") != "CE":
            raise ValueError("Sequential adapter requires the frozen CE training loss")
    if not set(binding["model_config"]).issubset(allowed):
        raise ValueError("Unexpected model config field outside model-only parameters")
    if binding.get("adapter") == "recbole-sequential-fixed100-v1":
        inspect_sequences(binding)
    return {"task_id": binding["task_id"], "status": binding["status"],
            "epochs": recipe["epochs"], "patience": recipe["patience"],
            "primary_metric": binding["evaluation"]["primary_metric"]}


def config_values(binding, output, seed, device):
    """All training/evaluation choices come from this task, never old E1 defaults."""
    output = Path(output)
    t, e = binding["training"], binding["evaluation"]
    values = dict(binding["model_config"],
        data_path=str(output / "data"), benchmark_filename=["train", "dev", "dev"],
        load_col={"inter": ["user_id", "item_id"], "user": ["user_id"], "item": ["item_id"]},
        USER_ID_FIELD="user_id", ITEM_ID_FIELD="item_id", NEG_PREFIX="neg_",
        field_separator="\t", user_inter_num_interval="[0,inf)", item_inter_num_interval="[0,inf)",
        filter_inter_by_user_or_item=False, rm_dup_inter=None,
        epochs=t["epochs"], stopping_step=t["patience"], eval_step=t["eval_step"],
        train_batch_size=t["batch_size"], eval_batch_size=e["batch_size"],
        learner="adam", learning_rate=t["learning_rate"], weight_decay=t["weight_decay"],
        train_neg_sample_args={"distribution": "uniform", "sample_num": 1,
                               "alpha": 1.0, "dynamic": False, "candidate_num": 0},
        eval_args={"split": {"RS": [0.8, 0.1, 0.1]}, "group_by": "user", "order": "RO", "mode": "full"},
        metrics=e["metrics"], topk=e["topk"], valid_metric=e["primary_metric"], valid_metric_bigger=True,
        metric_decimal_place=8, checkpoint_dir=str(output / "checkpoints"),
        dataset_save_path=str(output / "unused_dataset_cache.pth"),
        dataloaders_save_path=str(output / "unused_dataloader_cache.pth"),
        save_dataset=False, save_dataloaders=False,
        seed=seed, reproducibility=True, shuffle=True, worker=0,
        use_gpu=device == "cuda", gpu_id="0", device=device, show_progress=False,
        enable_amp=False, enable_scaler=False, log_wandb=False)
    if binding.get("adapter") == "recbole-sequential-fixed100-v1":
        values.update(LIST_SUFFIX="_list", ITEM_LIST_LENGTH_FIELD="item_length",
                      seq_separator=" ", alias_of_item_id=["item_id_list"],
                      train_neg_sample_args=None)
        values["load_col"]["inter"] = ["user_id", "item_id_list", "item_id"]
        values["eval_args"]["order"] = "TO"
    return values


def load_model(source):
    spec = importlib.util.spec_from_file_location("graph_candidate", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # RecBole pickles Config.model_class in checkpoints
    spec.loader.exec_module(module)
    return module.FreshCandidateModel


def clear_cache(model):
    model.restore_user_e = model.restore_item_e = None


class GraphMemoryPeak:
    """CUDA allocator peaks, including setup, every epoch and checkpoint reload."""
    def __init__(self, torch, output, source, binding_sha, seed):
        self.torch, self.output = torch, output
        self.allocated = self.reserved = 0
        self.seconds = 0.0
        self.events = 0
        self.complete = False
        boundary = Path(os.environ["SE_E2_OS_RECEIPT"])
        self.identity = dict(namespace_pid=os.getpid(), namespace_ppid=os.getppid(),
            gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"], logical_device=0,
            source_sha256=digest(source), binding_sha256=binding_sha, train_seed=seed,
            os_boundary_path=str(boundary), os_boundary_sha256=digest(boundary),
            worker_source_sha256=digest(Path(__file__)),
            process_mapping_basis="Join exact native control_job command, GPU_WORKER_EVENTS worker_pid/GPU UUID and physical OS boundary; namespace PID is not assumed to be the NVML host PID.")
        self.begin()

    def begin(self):
        started = time.monotonic()
        self.torch.cuda.reset_peak_memory_stats(0)
        self.seconds += time.monotonic() - started

    def end(self, phase, epoch=None, complete=False):
        started = time.monotonic()
        cuda = self.torch.cuda
        event = dict(phase=phase, epoch=epoch, utc_ns=time.time_ns(),
            max_allocated_bytes=cuda.max_memory_allocated(0),
            max_reserved_bytes=cuda.max_memory_reserved(0),
            allocated_bytes=cuda.memory_allocated(0), reserved_bytes=cuda.memory_reserved(0))
        self.allocated = max(self.allocated, event["max_allocated_bytes"])
        self.reserved = max(self.reserved, event["max_reserved_bytes"])
        self.events += 1
        self.complete = complete
        with (self.output / "GPU_MEMORY_PHASES.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        self.seconds += time.monotonic() - started
        write_json(self.output / "GPU_MEMORY_PEAK.json", self.summary())

    def summary(self):
        return dict(schema="graph.full-fit-cuda-allocator-peak.v1", identity=self.identity,
            status="COMPLETE" if self.complete else "IN_PROGRESS", phase_events=self.events,
            max_allocated_bytes=self.allocated, max_reserved_bytes=self.reserved,
            telemetry_seconds_excluding_summary_write=self.seconds,
            timing_basis="Native active/GPU/wall costs include telemetry; epoch timers exclude end-of-phase observation. Concurrent host workloads are recorded separately.",
            allocator_allowed_bytes=9 * 1024 ** 3,
            unmeasured="Non-PyTorch CUDA/context memory and NVML host-process total; no automatic reservation reduction.")


"""Sample an existing TRAIN loss/backward/update; never run an extra one."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from unittest.mock import patch

SAMPLE_EPOCHS = (0, 4, 19, 49, 99, 199, 299)

def stats(value):
    import torch
    x = value.detach().float().reshape(-1)
    if not x.numel(): return {'elements': 0}
    return {'elements': x.numel(), 'mean': float(x.mean()),
            'rms': float(x.square().mean().sqrt()), 'min': float(x.min()),
            'max': float(x.max()), 'nonzero': int(torch.count_nonzero(x))}

@contextmanager
def observe_step(model, optimizer, output, source_sha, epoch):
    """Observe the first real optimizer step in seven fixed epochs, no RNG use."""
    if epoch not in SAMPLE_EPOCHS:
        yield
        return
    import torch
    import torch.nn.functional as F
    record = dict(schema='graph.actual-train-signal.v2', source_sha256=source_sha,
                  epoch=epoch, extra_forward=0, extra_backward=0, extra_update=0,
                  scope='One original TRAIN minibatch; observations are not causal ablations.',
                  functional_calls=[], module_calls={}, parameter_groups={})
    hooks=[]; before={}; done=False; in_loss=False
    original_loss=model.calculate_loss
    def loss(*args, **kwargs):
        nonlocal in_loss
        if done: return original_loss(*args, **kwargs)
        in_loss=True
        try: result=original_loss(*args, **kwargs)
        finally: in_loss=False
        parts=result if isinstance(result,tuple) else (result,)
        record['returned_loss_components']=[float(x.detach()) for x in parts]
        record['loss_semantics']='Use exact source; tuple position does not assign ranking/regularization semantics.'
        return result
    def function(name, original):
        def call(x,*args,**kwargs):
            result=original(x,*args,**kwargs)
            if in_loss and not done and x.requires_grad and x.ndim <= 1 and x.numel() <= 4096:
                row=dict(function=name, operand=stats(x), semantic='Source-bound operand, not automatically a ranking margin')
                record['functional_calls'].append(row)
                def grad(g):
                    row['actual_operand_gradient']=stats(g)
                    return None
                hooks.append(x.register_hook(grad))
            return result
        return call
    def module_hook(name):
        def call(module,args,result):
            if in_loss and not done:
                row=record['module_calls'].setdefault(name,dict(type=type(module).__name__,calls=0))
                row['calls']+=1
                if torch.is_tensor(result): row['last_output']=stats(result)
        return call
    for name,module in model.named_modules():
        if name and isinstance(module,(torch.nn.Linear,torch.nn.Embedding,torch.nn.Dropout,torch.nn.ReLU,torch.nn.LeakyReLU,torch.nn.Tanh)):
            hooks.append(module.register_forward_hook(module_hook(name)))
    def pre(opt,args,kwargs):
        if done:return
        with torch.no_grad():
            for name,p in model.named_parameters():
                row=dict(parameter=stats(p), gradient=None if p.grad is None else stats(p.grad))
                record['parameter_groups'][name]=row
                if p.grad is not None:
                    before[name]=(p.detach().clone(),p.grad.detach().clone())
    def post(opt,args,kwargs):
        nonlocal done
        if done:return
        with torch.no_grad():
            for name,p in model.named_parameters():
                if name not in before:continue
                old,g=before[name];delta=p.detach()-old
                row=record['parameter_groups'][name];row['actual_update']=stats(delta)
                # Chunk-free float32 summaries avoid double-size embedding copies.
                dot=float((-g*delta).sum());den=float(g.norm()*delta.norm())
                row['update_cosine_negative_gradient']=dot/den if den else None
        record['optimizer']='Actual existing optimizer state; gradient scale does not predict Adam update.'
        with Path(output).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=True,allow_nan=False)+'\n')
        done=True;before.clear()
        for h in hooks:h.remove()
    step_hooks=[optimizer.register_step_pre_hook(pre),optimizer.register_step_post_hook(post)]
    try:
        with patch.object(model,'calculate_loss',loss),patch.object(F,'softplus',function('softplus',F.softplus)),patch.object(F,'logsigmoid',function('logsigmoid',F.logsigmoid)):
            yield
    finally:
        for h in hooks+step_hooks:h.remove()

def summarize(path, source_sha):
    import hashlib
    path=Path(path)
    if not path.exists():return {'status':'NOT_MEASURED'}
    blob=path.read_bytes();rows=[json.loads(x) for x in blob.splitlines() if x]
    if any(r['source_sha256']!=source_sha for r in rows):raise ValueError('Foreign TRAIN source')
    def extent(values):
        values=[x for x in values if x is not None]
        return dict(first=values[0],last=values[-1],min=min(values),max=max(values)) if values else None
    groups={}
    for name in sorted({k for r in rows for k in r['parameter_groups']}):
        entries=[r['parameter_groups'][name] for r in rows if name in r['parameter_groups']]
        groups[name]={kind:{key:extent([e.get(kind,{}).get(key) for e in entries if e.get(kind)]) for key in ('rms','nonzero')} for kind in ('parameter','gradient','actual_update')}
    return dict(status='MEASURED_ORIGINAL_TRAIN',path=str(path),sha256=hashlib.sha256(blob).hexdigest(),
                epochs=[r['epoch'] for r in rows],parameter_groups=groups,
                functional_samples=[dict(epoch=r['epoch'],calls=r['functional_calls']) for r in rows],
                module_types_seen=sorted({v['type'] for r in rows for v in r['module_calls'].values()}),
                limits='Sampled executed paths; absence is not global inactivity. No new counterfactual training or measured causal effect.')


def execute(binding, source, output, seed, device, role, binding_sha):
    """One worker attempt only; restart/retry and cumulative budgets belong to main."""
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.data.interaction import Interaction
    from recbole.trainer import Trainer
    from recbole.utils import init_seed

    output.mkdir(parents=True, exist_ok=False)  # never overwrite/replay old attempt
    started = time.monotonic()
    os.chdir(output)  # RecBole event/log files stay in this new writable attempt
    data = output / "data" / binding["dataset"]
    data.mkdir(parents=True)
    for item in binding["data_files"].values():
        shutil.copyfile(item["path"], data / Path(item["path"]).name)
    model_class = load_model(source)
    if binding.get('adapter') == 'recbole-sequential-fixed100-v1':
        from recbole.utils import ModelType, InputType
        if model_class.type != ModelType.SEQUENTIAL:
            raise ValueError('Sequential adapter requires SequentialRecommender with POINTWISE inputs')
    values = config_values(binding, output, seed, device)
    # Do not let RecBole's sys.argv parser override a frozen configuration.
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0]]
        config = Config(model=model_class, dataset=binding["dataset"], config_dict=values)
    finally:
        sys.argv = previous_argv
    if binding.get('adapter') == 'recbole-sequential-fixed100-v1' and config['MODEL_INPUT_TYPE'] != InputType.POINTWISE:
        raise ValueError('Sequential CE adapter requires POINTWISE inputs')
    assert config["epochs"] == values["epochs"] and config["stopping_step"] == values["stopping_step"]
    assert config["valid_metric"] == values["valid_metric"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested assigned CUDA worker is unavailable; no CPU fallback")
    memory = GraphMemoryPeak(torch, output, source, binding_sha, seed) if device == "cuda" else None
    init_seed(seed, True)
    dataset = create_dataset(config)
    # Runtime-only batching: match BERT user batching, retain full-sort universe.
    values["eval_batch_size"] = 64 * dataset.item_num
    config["eval_batch_size"] = values["eval_batch_size"]
    train, valid, unused_dev_alias = data_preparation(config, dataset)
    # The third loader is a required RecBole API alias of dev, never a test set.
    del unused_dev_alias
    mapping = {field: list(map(str, dataset.field2id_token[field])) for field in ("user_id", "item_id")}
    mapping_sha = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    expected_mapping = binding.get("catalog_mapping_sha256")
    if expected_mapping is not None and mapping_sha != expected_mapping:
        raise ValueError("Effective catalog mapping differs from the frozen task")
    if role == "candidate" and expected_mapping is None:
        raise ValueError("Candidate runs require the root's frozen catalog mapping")
    if binding.get("adapter") == "recbole-sequential-fixed100-v1":
        from sampled_eval import load_pool
        pool_users, candidates = load_pool(binding["dev_candidate_pool"])
        histories = inspect_sequences(binding)
        item_ids = {token: i for i, token in enumerate(mapping['item_id'])}
        for uid, row in zip(pool_users, candidates):
            if not 0 < int(uid) < len(mapping['user_id']) or int(row.max()) >= len(mapping['item_id']):
                raise ValueError('Sequential candidate pool has out-of-catalog IDs')
            seen = {item_ids[token] for token in histories[mapping['user_id'][int(uid)]]}
            if seen.intersection(map(int, row)):
                raise ValueError('Sequential candidate pool must exclude all TRAIN history and targets')
    write_json(output / "ID_MAPPING.json", mapping)
    init_seed(seed, True)
    model = model_class(config, train._dataset).to(config["device"])
    write_json(output / "BOUNDARY.json", dict(task_id=binding["task_id"], role=role,
        binding_sha256=binding_sha, source_sha256=digest(source), train_seed=seed,
        mapped_users=dataset.user_num, mapped_items=dataset.item_num,
        catalog_mapping_sha256=mapping_sha, train_interactions=len(train._dataset),
        dev_interactions=len(valid._dataset), data_files=binding["data_files"],
        actual_config=values, actual_embedding_shapes={n:list(m.weight.shape) for n,m in model.named_modules() if isinstance(m, torch.nn.Embedding)}, torch_version=torch.__version__, device=str(config["device"]),
        heldout_accessed=False, graph_input="train._dataset only"))

    if memory is not None: memory.end("setup")

    class CurveTrainer(Trainer):
        # Observation only: retain native optimizer, epoch, stopping and evaluator.
        def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
            self.graph_epoch = epoch_idx
            if memory is not None: memory.begin()
            t0 = time.monotonic()
            from acceleration_prototype import TransposeCache
            with TransposeCache(csr=True), observe_step(self.model, self.optimizer, output / "TRAIN_SIGNAL.jsonl", digest(source), epoch_idx):
                loss = super()._train_epoch(train_data, epoch_idx, *args, **kwargs)
            self.graph_train_seconds = time.monotonic() - t0
            self.graph_loss = loss
            if memory is not None: memory.end("training", epoch_idx)
            return loss

        def _valid_epoch(self, valid_data, *args, **kwargs):
            if memory is not None: memory.begin()
            t0 = time.monotonic()
            from sampled_eval import evaluate_dev
            metrics = evaluate_dev(self.model, config, valid_data, binding['dev_candidate_pool'])
            score = metrics[binding['evaluation']['primary_metric'].lower()]
            row = dict(epoch=self.graph_epoch, loss=self.graph_loss, score=score, metrics=metrics,
                       train_seconds=self.graph_train_seconds, validation_seconds=time.monotonic()-t0,
                       elapsed_seconds=time.monotonic()-started)
            with (output / "LEARNING_CURVE.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if memory is not None: memory.end("validation", self.graph_epoch)
            return score, metrics

    trainer = CurveTrainer(config, model)
    if memory is not None: memory.end("trainer_init")
    best_score, metrics = trainer.fit(train, valid, saved=True, show_progress=False)
    epochs_completed = getattr(trainer, "graph_epoch", -1) + 1
    stop_reason = ("EARLY_STOP" if trainer.cur_step > config["stopping_step"]
                   else "EPOCH_LIMIT" if epochs_completed == config["epochs"]
                   else "INCOMPLETE_OR_FIXTURE")
    if not math.isfinite(float(best_score)) or not all(math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError("Nonfinite validation result")
    if memory is not None:
        memory.end("fit_tail")
        memory.begin()
    checkpoint = Path(trainer.saved_model_file)
    saved = torch.load(checkpoint, map_location=config["device"], weights_only=False)
    # Exact checkpoint integrity is checked separately from CUDA sparse-kernel
    # accumulation order; the deterministic probe never re-evaluates dev/test.
    from graph_checkpoint_verify import verify_checkpoint
    probe = train._dataset.inter_feat[:2]
    checkpoint_verification = verify_checkpoint(model_class, config, train._dataset,
                                                model, saved, probe)
    result = dict(status="SUCCESS", task_id=binding["task_id"], role=role, metrics=dict(metrics),
        primary_metric=binding["evaluation"]["primary_metric"], best_score=float(best_score),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), best_epoch=int(saved["epoch"]),
        source_sha256=digest(source), binding_sha256=binding_sha, train_seed=seed,
        catalog_mapping_sha256=mapping_sha, checkpoint_reload_verified=True,
        checkpoint_verification=checkpoint_verification,
        epochs_completed=epochs_completed, max_epochs=config["epochs"],
        early_stop_counter=trainer.cur_step, patience=config["stopping_step"],
        stop_reason=stop_reason, evaluation_seed=binding["evaluation"].get("seed"),
        wall_seconds=time.monotonic()-started, test_accessed=False,
        evidence_class="ROOT_CALIBRATION" if role == "root" else "CANDIDATE_VALIDATION",
        genuine_admission="OWNED_BY_EXTERNAL_COMMON_LEDGER")
    if memory is not None:
        memory.end("checkpoint_reload", complete=True)
        result["gpu_memory_peak"] = memory.summary()
    write_json(output / "RESULT.json", result)
    trainer.tensorboard.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--role", choices=["root", "candidate"], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    binding = json.loads(args.binding.read_bytes())
    source, output = args.candidate.resolve(strict=True), args.output.resolve()
    if digest(source) != args.expected_source_sha:
        raise ValueError("Candidate source identity mismatch")
    if args.role == "root" and args.expected_source_sha != binding["root_source_sha256"]:
        raise ValueError("Root differs from its frozen original baseline")
    inspection = inspect_binding(binding, execute=args.execute)
    if not args.execute:
        print(json.dumps(dict(inspection, executed=False)))
        return
    if output.exists():
        raise FileExistsError("Use a fresh attempt path; existing evidence is preserved")
    sys.path.insert(0, str(Path(binding["recbole_root"]).resolve(strict=True)))
    started = time.monotonic()
    try:
        result = execute(binding, source, output, args.train_seed, args.device, args.role, digest(args.binding))
    except Exception as error:
        # Keep failure evidence, never turn infrastructure failure into score=0.
        # The external ledger owns classification, physical cost and continuation.
        if output.is_dir() and not (output / "RESULT.json").exists():
            write_json(output / "FAILURE.json", dict(status="ERROR", error=repr(error),
                task_id=binding["task_id"], role=args.role, train_seed=args.train_seed,
                wall_seconds=time.monotonic()-started, test_accessed=False))
        raise
    print(json.dumps(result))


if __name__ == "__main__":
    main()
