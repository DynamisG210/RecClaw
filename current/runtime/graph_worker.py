"""Independent graph task worker. Launch inside the existing owner boundary.

No Provider, scheduler, heldout evaluator or generic search controller lives here.
Without --execute this verifies the proposed binding only. Candidate execution
requires the existing external isolation, resource meter and timeout launcher.
"""
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


def inspect_binding(binding, *, execute=False):
    if (binding.get("metric_binding_version") != "graph.fullsort.ndcg5.hr5.v1" or binding["evaluation"]["primary_metric"] != "NDCG@5" or binding["evaluation"]["topk"] != [5, 10] or binding["evaluation"].get("secondary_metric") != "Hit@5"):
        raise ValueError("Prospective full-sort @5 binding required")
    if execute and binding["status"] != "FROZEN":
        raise ValueError("Owner must freeze the new task binding before execution")
    if binding["line"] not in ("ngcf", "lightgcnpp"):
        raise ValueError("Unknown independent graph line")
    if not binding["task_id"] or binding["dataset"] != "ml-1m":
        raise ValueError("Expected explicit new task identity and existing ML1M split")
    if set(binding["data_files"]) != {"train", "dev", "user", "item"}:
        raise ValueError("Only the four original search files are permitted")
    for role, item in binding["data_files"].items():
        expected = {"train": "ml-1m.train.inter", "dev": "ml-1m.dev.inter",
                    "user": "ml-1m.user", "item": "ml-1m.item"}[role]
        if Path(item["path"]).name != expected or digest(item["path"]) != item["sha256"]:
            raise ValueError("Search asset identity mismatch: " + role)
    recipe = binding["training"]
    if recipe["epochs"] < 1 or recipe["patience"] < 1 or recipe["eval_step"] < 1:
        raise ValueError("Explicit positive training horizon and stopping settings required")
    metric, cutoff = binding["evaluation"]["primary_metric"].split("@")
    if metric not in binding["evaluation"]["metrics"] or int(cutoff) not in binding["evaluation"]["topk"]:
        raise ValueError("Primary metric must be part of the fixed full-sort evaluator")
    if binding["evaluation"]["candidate_universe"] != "FULL_SORT":
        raise ValueError("Graph task uses full-sort evaluation")
    # Prevent model hyperparameters from overriding trusted scientific inputs.
    allowed = {"embedding_size", "n_layers", "alpha", "beta", "gamma", "reg_weight",
               "hidden_size_list", "message_dropout"}
    if not set(binding["model_config"]).issubset(allowed):
        raise ValueError("Unexpected model config field outside model-only parameters")
    return {"task_id": binding["task_id"], "status": binding["status"],
            "epochs": recipe["epochs"], "patience": recipe["patience"],
            "primary_metric": binding["evaluation"]["primary_metric"]}


def config_values(binding, output, seed, device):
    """All training/evaluation choices come from this task, never old E1 defaults."""
    output = Path(output)
    t, e = binding["training"], binding["evaluation"]
    return dict(binding["model_config"],
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


def load_model(source):
    spec = importlib.util.spec_from_file_location("graph_candidate", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # RecBole pickles Config.model_class in checkpoints
    spec.loader.exec_module(module)
    return module.FreshCandidateModel


def clear_cache(model):
    model.restore_user_e = model.restore_item_e = None


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
    values = config_values(binding, output, seed, device)
    # Do not let RecBole's sys.argv parser override a frozen configuration.
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0]]
        config = Config(model=model_class, dataset=binding["dataset"], config_dict=values)
    finally:
        sys.argv = previous_argv
    assert config["epochs"] == values["epochs"] and config["stopping_step"] == values["stopping_step"]
    assert config["valid_metric"] == values["valid_metric"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested assigned CUDA worker is unavailable; no CPU fallback")
    init_seed(seed, True)
    dataset = create_dataset(config)
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
    write_json(output / "ID_MAPPING.json", mapping)
    init_seed(seed, True)
    model = model_class(config, train._dataset).to(config["device"])
    write_json(output / "BOUNDARY.json", dict(task_id=binding["task_id"], role=role,
        binding_sha256=binding_sha, source_sha256=digest(source), train_seed=seed,
        mapped_users=dataset.user_num, mapped_items=dataset.item_num,
        catalog_mapping_sha256=mapping_sha, train_interactions=len(train._dataset),
        dev_interactions=len(valid._dataset), data_files=binding["data_files"],
        actual_config=values, torch_version=torch.__version__, device=str(config["device"]),
        heldout_accessed=False, graph_input="train._dataset only"))

    class CurveTrainer(Trainer):
        # Observation only: retain native optimizer, epoch, stopping and evaluator.
        def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
            self.graph_epoch = epoch_idx
            t0 = time.monotonic()
            loss = super()._train_epoch(train_data, epoch_idx, *args, **kwargs)
            self.graph_train_seconds = time.monotonic() - t0
            self.graph_loss = loss
            return loss

        def _valid_epoch(self, valid_data, *args, **kwargs):
            t0 = time.monotonic()
            score, metrics = super()._valid_epoch(valid_data, *args, **kwargs)
            row = dict(epoch=self.graph_epoch, loss=self.graph_loss, score=score, metrics=metrics,
                       train_seconds=self.graph_train_seconds, validation_seconds=time.monotonic()-t0,
                       elapsed_seconds=time.monotonic()-started)
            with (output / "LEARNING_CURVE.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            return score, metrics

    trainer = CurveTrainer(config, model)
    best_score, metrics = trainer.fit(train, valid, saved=True, show_progress=False)
    epochs_completed = getattr(trainer, "graph_epoch", -1) + 1
    stop_reason = ("EARLY_STOP" if trainer.cur_step > config["stopping_step"]
                   else "EPOCH_LIMIT" if epochs_completed == config["epochs"]
                   else "INCOMPLETE_OR_FIXTURE")
    if not math.isfinite(float(best_score)) or not all(math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError("Nonfinite validation result")
    checkpoint = Path(trainer.saved_model_file)
    saved = torch.load(checkpoint, map_location=config["device"], weights_only=False)
    # Verify model-only checkpoint reload on identical user probes without another
    # validation metric or touching the heldout evaluator.
    model.load_state_dict(saved["state_dict"])
    clear_cache(model)
    model.eval()
    probe = Interaction({model.USER_ID: train._dataset.inter_feat[model.USER_ID][:2].to(config["device"])})
    with torch.no_grad():
        before = model.full_sort_predict(probe).detach().clone()
    restored = model_class(config, train._dataset).to(config["device"])
    restored.load_state_dict(saved["state_dict"])
    clear_cache(restored)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(before, restored.full_sort_predict(probe), rtol=1e-6, atol=1e-7)
    result = dict(status="SUCCESS", task_id=binding["task_id"], role=role, metrics=dict(metrics),
        primary_metric=binding["evaluation"]["primary_metric"], best_score=float(best_score),
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), best_epoch=int(saved["epoch"]),
        source_sha256=digest(source), binding_sha256=binding_sha, train_seed=seed,
        catalog_mapping_sha256=mapping_sha, checkpoint_reload_verified=True,
        epochs_completed=epochs_completed, max_epochs=config["epochs"],
        early_stop_counter=trainer.cur_step, patience=config["stopping_step"],
        stop_reason=stop_reason, evaluation_seed=binding["evaluation"].get("seed"),
        wall_seconds=time.monotonic()-started, test_accessed=False,
        evidence_class="ROOT_CALIBRATION" if role == "root" else "CANDIDATE_VALIDATION",
        genuine_admission="OWNED_BY_EXTERNAL_COMMON_LEDGER")
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
