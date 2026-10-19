"""User-owned task inputs and machine environment, separate from the method."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

SCHEMA = "recrsi.task.v1"
ADAPTER = "recbole-graph-fixed100-v1"
TASK_PATHS = ("binding", "facts", "literature", "root_calibration", "root_binding",
              "root_gpu_ledger", "root_mapping", "root_source")
ENV_PATHS = ("worker_python", "recbole_root", "driver_root", "api_config", "resource_slots")
ENV_KEYS = set(ENV_PATHS) | {"cpus", "gpu_uuid"}
NATIVE = Path(__file__).parent / "native"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(value, base, *, interpreter=False):
    if not isinstance(value, str) or not value:
        raise ValueError("A nonempty input path is required")
    def variable(match):
        name = match[1]
        if name not in os.environ:
            raise ValueError("Unset path environment variable: " + name)
        return os.environ[name]
    path = Path(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", variable, value)).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.exists():
        raise FileNotFoundError(path)
    # Preserve venv symlinks; resolving them changes installed dependencies.
    return str(path.absolute() if interpreter else path.resolve(strict=True))


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " requires exactly: " + ", ".join(sorted(keys)))


def load_config(path):
    path = Path(path).resolve(strict=True)
    raw = read(path)
    if "schema" in raw:
        exact_keys(raw, {"schema", "adapter", "run_id", "repeat", "task", "environment"}, "task config")
        if raw["schema"] != SCHEMA or raw["adapter"] != ADAPTER:
            raise ValueError("Supported schema/adapter: " + SCHEMA + " / " + ADAPTER)
        exact_keys(raw["task"], TASK_PATHS, "task inputs")
        env_path = Path(resolve(raw["environment"], path.parent))
        env = read(env_path)
        exact_keys(env, ENV_KEYS, "environment config")
        settings = {"run_id": raw["run_id"], "repeat": raw["repeat"], "adapter": ADAPTER}
        settings.update({k: resolve(v, path.parent) for k, v in raw["task"].items()})
        settings.update({k: resolve(env[k], env_path.parent, interpreter=k == "worker_python") for k in ENV_PATHS})
        settings.update({k: env[k] for k in ("cpus", "gpu_uuid")})
        config_hashes = {"task": sha(path), "environment": sha(env_path)}
    else:
        # v1.0 input files remain supported, with the same fixed protocol.
        exact_keys(raw, set(TASK_PATHS) | ENV_KEYS | {"run_id", "repeat"}, "legacy task config")
        settings = dict(raw, adapter="ngcf-electronics-legacy")
        settings.update({k: resolve(raw[k], path.parent, interpreter=k == "worker_python") for k in TASK_PATHS + ENV_PATHS})
        config_hashes = {"task": sha(path)}
    if not isinstance(settings["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", settings["run_id"]):
        raise ValueError("run_id must contain only letters, digits, underscores and hyphens")
    if type(settings["repeat"]) is not int or settings["repeat"] not in (1, 2, 3):
        raise ValueError("The fixed method profile supports repeats 1, 2, 3")
    cpus = settings["cpus"]
    if not isinstance(cpus, list) or len(cpus) != 4 or any(type(c) is not int or c < 0 for c in cpus) or len(set(cpus)) != 4:
        raise ValueError("Assign four distinct nonnegative CPU cores")
    if not isinstance(settings["gpu_uuid"], str) or not settings["gpu_uuid"].startswith("GPU-"):
        raise ValueError("An explicit GPU UUID is required")
    for key in TASK_PATHS + ("worker_python", "api_config"):
        if not Path(settings[key]).is_file():
            raise ValueError(key + " must be a file")
    for key in ("recbole_root", "driver_root", "resource_slots"):
        if not Path(settings[key]).is_dir():
            raise ValueError(key + " must be a directory")
    settings["configuration_hashes"] = config_hashes
    return settings


def native_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, NATIVE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(settings):
    """Read and hash inputs only. No API, imports of user code, or worker launch."""
    binding = read(settings["binding"])
    if settings["adapter"] == ADAPTER:
        if binding.get("adapter") != ADAPTER:
            raise ValueError("A custom task binding must explicitly declare adapter=" + ADAPTER)
    elif binding.get("adapter") is not None:
        raise ValueError("Use the split task schema for a custom adapter binding")
    base = Path(settings["binding"]).parent
    binding["data_files"] = {k: dict(v, path=resolve(v["path"], base)) for k, v in binding["data_files"].items()}
    binding["dev_candidate_pool"] = dict(binding["dev_candidate_pool"], path=resolve(binding["dev_candidate_pool"]["path"], base))
    worker = native_module("_recrsi_input_worker", "task/research/measured_execution/graph_worker_measured.py")
    worker.inspect_binding(binding, execute=True)
    contract = native_module("_recrsi_metric_contract", "task/search_source/graph_metric_contract.py")
    contract.require_binding(binding)
    if binding["evaluation"]["seed"] != 72026:
        raise ValueError("This method profile retains evaluation seed 72026")
    if sha(settings["root_source"]) != binding["root_source_sha256"]:
        raise ValueError("Input digest mismatch: root_source")
    mapping_digest = hashlib.sha256(json.dumps(read(settings["root_mapping"]), sort_keys=True).encode()).hexdigest()
    if mapping_digest != binding["catalog_mapping_sha256"]:
        raise ValueError("Catalog mapping identity mismatch")
    pool = binding["dev_candidate_pool"]
    if sha(pool["path"]) != pool["sha256"]:
        raise ValueError("Fixed DEV candidate pool digest mismatch")
    root_binding = read(settings["root_binding"])
    for key in ("adapter", "task_id", "line", "dataset", "root_source_sha256", "catalog_mapping_sha256", "training", "evaluation", "model_config", "metric_binding_version"):
        if binding.get(key) != root_binding.get(key):
            raise ValueError("Baseline/task protocol mismatch: " + key)
    if {k:v["sha256"] for k,v in binding["data_files"].items()} != {k:v["sha256"] for k,v in root_binding["data_files"].items()} or pool["sha256"] != root_binding["dev_candidate_pool"]["sha256"]:
        raise ValueError("Baseline/task data or candidate pool mismatch")
    root = read(settings["root_calibration"])
    result = root["result"]
    expected = dict(task_id=binding["task_id"], train_seed=51000 + settings["repeat"],
                    source_sha256=binding["root_source_sha256"], catalog_mapping_sha256=binding["catalog_mapping_sha256"],
                    binding_sha256=sha(settings["root_binding"]), primary_metric="NDCG@5", test_accessed=False)
    if root["status"] != "SUCCESS" or any(result.get(k) != v for k,v in expected.items()):
        raise ValueError("Baseline must be an actual matching source/data/protocol/seed measurement without TEST access")
    metrics = result.get("metrics", {})
    for key in ("ndcg@5", "hit@5"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Baseline requires finite measured " + key)
    facts = read(settings["facts"])
    inputs = native_module("_recrsi_inputs", "task/research_line/graph_native_inputs.py")
    if facts["task_id"] != binding["task_id"] or facts["simulation_context"] != inputs.CONTEXT_RULE or facts["data_sha256"] != {k:v["sha256"] for k,v in binding["data_files"].items()} or facts["catalog_mapping_sha256"] != binding["catalog_mapping_sha256"]:
        raise ValueError("Task facts must match data, mapping and the TRAIN-only information boundary")
    if facts.get("root_source_sha256") != binding["root_source_sha256"] or not facts.get("task"):
        raise ValueError("Task facts require the actual root source identity and task description")
    for fact, field in (("original_model_config", "model_config"), ("training", "training"), ("evaluation", "evaluation")):
        if facts.get(fact) != binding[field]:
            raise ValueError("Task facts differ from the executed " + field)
    if facts.get("selected_root_metrics") != result.get("metrics"):
        raise ValueError("Task facts must use the actual baseline metrics")
    if not read(settings["literature"]).get("documents"):
        raise ValueError("Supply literature documents for the task")
    return binding


def check_config(path):
    settings = load_config(path)
    binding = validate_inputs(settings)
    return dict(status="INPUTS_VALIDATED_NO_EXECUTION", adapter=settings["adapter"], task_id=binding["task_id"],
                dataset=binding["dataset"], configuration_hashes=settings["configuration_hashes"],
                scope="Input files and protocol identities only; API reachability, GPU, sandbox privileges and worker dependencies are not probed.")


def initialize(destination):
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    task = dict(schema=SCHEMA, adapter=ADAPTER, run_id="my-research-01", repeat=1,
                task={k:"inputs/" + k + (".py" if k == "root_source" else ".json") for k in TASK_PATHS},
                environment="environment.json")
    environment = {k:"/path/to/" + k for k in ENV_PATHS}
    environment.update(cpus=[0,1,2,3], gpu_uuid="GPU-REPLACE-WITH-ASSIGNED-DEVICE-UUID")
    for name, value in (("task.json", task), ("environment.json", environment)):
        (destination/name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return destination
