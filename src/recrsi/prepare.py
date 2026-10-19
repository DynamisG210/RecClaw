"""Bind the final native engine to explicit task inputs and a fresh run directory."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .config import load_config, validate_inputs

VERSION = "recrsi-native-1.2.1"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return str(path)

def pins(root):
    return {str(p): sha(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix in (".py", ".txt", ".json") and "__pycache__" not in p.parts}

WORKER_IDENTITY = r"""
import importlib.metadata as metadata
import hashlib,json,sys
from pathlib import Path
rows=[]
for name in ('torch','numpy','scipy','recbole','jsonschema','rfc8785'):
 d=metadata.distribution(name);r=dict(name=d.metadata['Name'],version=d.version)
 for filename,key in [('RECORD','record_sha256'),('METADATA','metadata_sha256')]:
  data=d.read_text(filename);r[key]=None if data is None else hashlib.sha256(data.encode()).hexdigest()
 rows.append(r)
print(json.dumps(dict(python=sys.executable,python_version=sys.version,python_binary_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),dependencies=rows)))
"""

def prepare(config_path, destination):
    if sys.platform != "linux":
        raise RuntimeError("The native execution boundary requires Linux")
    config_path = Path(config_path).resolve(strict=True)
    settings = load_config(config_path)
    original_binding = validate_inputs(settings)
    inputs = {name: settings[name] for name in (
        "binding", "facts", "literature", "root_calibration", "root_binding",
        "root_gpu_ledger", "root_mapping", "root_source", "worker_python",
        "recbole_root", "driver_root", "api_config", "resource_slots")}
    repeat = settings["repeat"]
    facts = read(inputs["facts"])
    worker = json.loads(subprocess.check_output([inputs["worker_python"],"-c",WORKER_IDENTITY], text=True))
    # Copy only the released code, never an old campaign or paid response store.
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Prepare requires a new run directory; inspect/resume the existing one explicitly")
    native = Path(__file__).parent/"native"
    shutil.copytree(native, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    task = destination/"task"
    source = task/"search_source"
    branch = task/"research_line"
    quality = task/"research"
    method = quality/"mechanism_contrast"
    measurement = quality/"measured_execution"
    declaration = quality/"declaration"
    common = destination/"common/main_binding"
    comparison = destination/"common/comparison"
    bound = method/("repeat%02d" % repeat)
    run_id = settings["run_id"]
    output = branch/"runs"/run_id
    output.mkdir(parents=True)
    phase = "USER_RESEARCH" if original_binding.get("adapter") else "FORMAL_MAIN"
    identity = dict(environment=original_binding["task_id"], method="ro", phase=phase, repeat=repeat,
                    run=run_id, search_seed=41000+repeat, train_seed=51000+repeat)
    cpus = settings["cpus"]
    binding = dict(original_binding, recbole_root=inputs["recbole_root"])
    # Relative input paths resolve against the supplied binding, never the package.
    base = Path(inputs["binding"]).parent
    def resolve_asset(value):
        path=Path(value).expanduser()
        return str((base/path if not path.is_absolute() else path).resolve(strict=True))
    binding["data_files"] = {k:{**v,"path":resolve_asset(v["path"])} for k,v in binding["data_files"].items()}
    for role, asset in binding["data_files"].items():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("Frozen data changed: "+role)
    binding["catalog_mapping_path"] = inputs["root_mapping"]
    binding["dev_candidate_pool"] = {**binding["dev_candidate_pool"],"path":resolve_asset(binding["dev_candidate_pool"]["path"])}
    binding_path = write(bound/"BINDING.json", binding)
    facts_path = write(bound/"SHARED_FACTS.json", facts)
    literature_path = write(bound/"LITERATURE.json", read(inputs["literature"]))
    worker.update(recbole_root=inputs["recbole_root"], main_binding=str(common), comparison_root=str(comparison),
                  source_files={**pins(destination), **{str(p):sha(p) for p in Path(inputs["recbole_root"]).rglob("*.py")}},
                  dependency_scope="Pinned interpreter, runtime sources and installed dependency metadata")
    closure_path = write(bound/"RUNTIME_CLOSURE.json", worker)
    prices = {"gpt-5.6-luna":dict(input=.2,output=1.2,cache_read=.2,cache_creation=.2),
              "gpt-5.6-terra":dict(input=2.,output=12.,cache_read=2.,cache_creation=2.)}
    provider = dict(endpoint="endpoint2", api_config=inputs["api_config"],
        model_api_configs={model:inputs["api_config"] for model in prices}, ledger_module_dir=str(destination/"ledger"),
        run_identity=identity, resource_root=str(output), active_wall_path=str(output/"ACTIVE_WALL.json"),
        api_ledger=str(output/"API_EVENTS.jsonl"),
        api_limits=dict(model_prices=prices,normalized_unit_usd=14,max_normalized_u=.5,max_requests=sys.maxsize,
                        max_input_tokens=128000,max_output_tokens=20000,input_estimate_margin=4096))
    provider_path = write(bound/"PROVIDER_CONFIG.json",provider)
    evaluator = dict(schema="graph-candidate-evaluator.v1", phase=phase, run_identity=identity, output=str(output),
        active_wall_path=provider["active_wall_path"], provider_config=provider_path, binding=binding_path,
        root_binding=inputs["root_binding"], root_calibration=inputs["root_calibration"], root_calibration_sha256=sha(inputs["root_calibration"]),
        root_gpu_ledger=inputs["root_gpu_ledger"], root_mapping=inputs["root_mapping"], parent_source=inputs["root_source"],
        train_seed=identity["train_seed"], closure=closure_path, closure_sha256=sha(closure_path), assets=pins(destination),
        candidate_worker=str(measurement/"graph_worker_measured.py"), main_binding=str(common), comparison_root=str(comparison),
        worker_python=inputs["worker_python"], core=str(source/"isolated_graph_core"), task_root=str(output/"task"),
        initial_diagnosis=str(source/"graph_diagnosis_tools.py"),
        dataset_manifest_digest=hashlib.sha256(json.dumps(binding["data_files"],sort_keys=True).encode()).hexdigest(),
        runtime_binding_digest=sha(binding_path),runtime_release_digest=hashlib.sha256(json.dumps(pins(native),sort_keys=True).encode()).hexdigest(),
        common_facts_sha256=sha(facts_path), literature_sha256=sha(literature_path),
        qualification_timeout_seconds=None,worker_timeout_seconds=None,observation_timeout_seconds=None,
        gpu_uuid=settings["gpu_uuid"],cpus=cpus,driver_root=inputs["driver_root"],resource_slots=inputs["resource_slots"])
    evaluator_path=write(bound/"EVALUATOR_CONFIG.json",evaluator)
    launch=dict(method="ro",output=str(output),evaluator_config=evaluator_path,phase=phase,repeat=repeat,
                search_seed=identity["search_seed"],training_seed=identity["train_seed"],native_python=sys.executable,
                cpus=cpus,shared_facts=facts_path,literature=literature_path,native_source_sha256=pins(destination),
                autospace_seed=91025+repeat,campaign_id=run_id)
    launch_path=write(bound/"LAUNCH.json",launch)
    limits=dict(native_rounds=36,effective_evaluations=12,selected_attempts_per_round=1,discovery_generations_per_round=1,
                training_admissions=36,reference_usd=7,active_seconds=None,worker_seconds=None,qualification_seconds=None,cumulative_gpu_seconds=None)
    write(branch/"MANIFEST.json",dict(version=VERSION,source=str(source),comparison_protocol=VERSION,
        comparison_arm="evidence_bound_parent",generation_arm="branch",limits=limits,
        mechanical_binding=str(branch/"mechanical_binding.py"),helper=str(destination/"support/qualification/runtime_fix.py"),
        requirements=str(destination/"support/requirements/proposal_requirements.py")))
    write(branch/"private_operations/SUPPORT.json",dict(main_binding=str(common)))
    composition=branch/"future_graph_composition"
    write(composition/"CONFIG.json",dict(files={p.name:sha(p) for p in composition.iterdir() if p.suffix in (".py",".txt")}))
    write(composition/"BUNDLE.json",dict(files={p.name:sha(p) for p in composition.iterdir() if p.suffix in (".py",".txt")}))
    sampled=measurement/"evaluation_metric_contract_v1/runtime/sampled_eval_accelerated.py"
    write(measurement/"evaluation_acceleration_v1/RELEASE.json",dict(status="EQUIVALENT_DEV_ACCELERATION_ACCEPTED",
        runtime_version="graph-eval-batch1024-per-user-metric-v1-20261016",reference_sampled_eval_sha256=sha(source/"sampled_eval.py"),
        sampled_eval=str(sampled),pins={str(sampled):sha(sampled)}))
    write(bound/"CONSUMER_MANIFEST.json",dict(schema="PROSPECTIVE_COLD_DECLARATION_V1",provider_config=provider_path,
        provider_config_sha256=sha(provider_path),run_identity=identity,native_session_id=run_id,completion_model="gpt-5.6-luna",runtime_pins=pins(declaration)))
    write(method/"RELEASE.json",dict(version=VERSION,original_branch=str(branch),original_source=str(source),pins=pins(destination)))
    write(destination/"TASK.json",dict(entry=str(method/"measured_entry.py"),launch=launch_path,repeat=repeat,
        resource_slots=inputs["resource_slots"],cpus=cpus,configuration_sha256=sha(config_path),
        configuration_hashes=settings["configuration_hashes"],adapter=settings["adapter"],
        evidence_class="USER_CONFIGURED_SOFTWARE_RUN" if binding.get("adapter") else "LEGACY_PROTOCOL_RUN",method_version=VERSION))
    return destination
