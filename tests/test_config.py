"""Input contract tests; fixture receipts are synthetic, never research evidence."""
import json
from pathlib import Path
import pytest
from recrsi.config import ADAPTER, TASK_PATHS, ENV_PATHS, initialize, load_config, check_config, sha, native_module


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def task(tmp_path):
    root = initialize(tmp_path / "project")
    inputs = root / "inputs"
    inputs.mkdir()
    for key in TASK_PATHS:
        (inputs / (key + (".py" if key == "root_source" else ".json"))).write_text("{}")
    env = json.loads((root / "environment.json").read_text())
    for key in ENV_PATHS:
        p = root / key
        if key in ("worker_python", "api_config"):
            p.write_text("private fixture")
        else:
            p.mkdir()
        env[key] = key
    dump(root / "environment.json", env)
    files = {}
    for role, suffix in (("train", "train.inter"), ("dev", "dev.inter"), ("user", "user"), ("item", "item")):
        p = inputs / ("catalog." + suffix)
        p.write_text("fixture")
        files[role] = dict(path=p.name, sha256=sha(p))
    pool = inputs / "pool.npz"
    pool.write_bytes(b"synthetic contract fixture, not a usable candidate pool")
    binding = dict(adapter=ADAPTER, status="FROZEN", task_id="custom-task", dataset="catalog", line="my_model",
        data_files=files, root_source_sha256=sha(inputs / "root_source.py"),
        catalog_mapping_sha256=sha(inputs / "root_mapping.json"), model_config={},
        training=dict(epochs=2, patience=1, eval_step=1),
        evaluation=dict(primary_metric="NDCG@5", secondary_metric="Hit@5", metrics=["NDCG", "Hit"],
                        topk=[5,10], candidate_universe="FIXED_100_UNIFORM", seed=72026),
        dev_candidate_pool=dict(path=pool.name, sha256=sha(pool)), metric_binding_version="graph.fixed100.ndcg5.hr5.se-style.v1")
    dump(inputs / "binding.json", binding)
    dump(inputs / "root_binding.json", binding)
    dump(inputs / "root_calibration.json", dict(status="SUCCESS", result=dict(task_id="custom-task", train_seed=51001,
        source_sha256=binding["root_source_sha256"], catalog_mapping_sha256=binding["catalog_mapping_sha256"],
        binding_sha256=sha(inputs / "root_binding.json"), primary_metric="NDCG@5", test_accessed=False, metrics={"ndcg@5":.2,"hit@5":.4})))
    native = native_module("_config_test_native", "task/research_line/graph_native_inputs.py")
    dump(inputs / "facts.json", dict(task_id="custom-task", simulation_context=native.CONTEXT_RULE,
        data_sha256={k:v["sha256"] for k,v in files.items()}, catalog_mapping_sha256=binding["catalog_mapping_sha256"],
        root_source_sha256=binding["root_source_sha256"], task="Synthetic input contract fixture",
        original_model_config=binding["model_config"], training=binding["training"], evaluation=binding["evaluation"], selected_root_metrics={"ndcg@5":.2,"hit@5":.4}))
    dump(inputs / "literature.json", {"documents":[{"title":"fixture"}]})
    return root / "task.json"


def test_paths_are_relative_to_their_own_config(task, tmp_path, monkeypatch):
    root = task.parent
    env = json.loads((root / "environment.json").read_text())
    separate = root / "machines"
    separate.mkdir()
    for key in ENV_PATHS:
        env[key] = "../" + env[key]
    dump(separate / "host.json", env)
    config = json.loads(task.read_text())
    config["environment"] = "machines/host.json"
    dump(task, config)
    monkeypatch.chdir(tmp_path)
    settings = load_config(task)
    assert Path(settings["worker_python"]) == root / "machines" / ".." / "worker_python"
    assert Path(settings["facts"]) == root / "inputs/facts.json"
    assert set(settings["configuration_hashes"]) == {"task", "environment"}
    assert check_config(task)["status"] == "INPUTS_VALIDATED_NO_EXECUTION"


def test_environment_variable_path_and_missing_variable(task, monkeypatch):
    envpath = task.parent / "environment.json"
    env = json.loads(envpath.read_text())
    env["api_config"] = "${RECRSI_PRIVATE}/api_config"
    dump(envpath, env)
    monkeypatch.setenv("RECRSI_PRIVATE", str(task.parent))
    assert Path(load_config(task)["api_config"]) == task.parent / "api_config"
    monkeypatch.delenv("RECRSI_PRIVATE")
    with pytest.raises(ValueError, match="Unset path environment variable"):
        load_config(task)


@pytest.mark.parametrize("change, message", [
    ("data", "asset identity"), ("seed", "actual matching"),
    ("recipe", "protocol mismatch"), ("pool", "pool digest"),
    ("unknown", "requires exactly"), ("adapter", "Supported schema/adapter"),
])
def test_rejects_configuration_and_evidence_mismatch(task, change, message):
    inputs = task.parent / "inputs"
    if change == "data":
        (inputs / "catalog.train.inter").write_text("changed")
    elif change == "pool":
        (inputs / "pool.npz").write_bytes(b"changed")
    elif change == "seed":
        path = inputs / "root_calibration.json"
        value = json.loads(path.read_text()); value["result"]["train_seed"] = 999
        dump(path, value)
    elif change == "recipe":
        path = inputs / "binding.json"
        value = json.loads(path.read_text()); value["training"]["epochs"] = 9
        dump(path, value)
    else:
        value = json.loads(task.read_text())
        value["budget" if change == "unknown" else "adapter"] = "unsupported"
        dump(task, value)
    with pytest.raises(ValueError, match=message):
        check_config(task)


def test_init_never_overwrites_existing_directory(tmp_path):
    root = initialize(tmp_path / "project")
    before = (root / "task.json").read_bytes()
    with pytest.raises(FileExistsError):
        initialize(root)
    assert (root / "task.json").read_bytes() == before


def test_check_does_not_launch_or_disclose_credentials(task, monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        pytest.fail("check must not launch processes")
    monkeypatch.setattr(subprocess, "Popen", fail)
    report = json.dumps(check_config(task))
    assert "private fixture" not in report
    assert "api_config" not in report


def test_prepare_rejects_bad_inputs_before_creating_run(task, tmp_path):
    import sys
    if sys.platform != "linux":
        pytest.skip("native preparation is Linux only")
    from recrsi.prepare import prepare
    (task.parent / "inputs/catalog.train.inter").write_text("changed")
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="asset identity"):
        prepare(task, destination)
    assert not destination.exists()


def test_custom_binding_cannot_reuse_historical_identity(task):
    path = task.parent / "inputs/binding.json"
    value = json.loads(path.read_text())
    value["task_id"] = "NGCF_EL_NATIVE_SEARCH_20261013"
    dump(path, value)
    with pytest.raises(ValueError, match="historical experiment ID"):
        check_config(task)
