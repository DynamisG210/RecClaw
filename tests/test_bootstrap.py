"""Public example to baseline receipt to prepared search, using real CPU training."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_demo_prepares_and_delivers_root(tmp_path):
    pytest.importorskip('torch')
    import recbole
    from recrsi.bootstrap import demo, worker_modules
    from recrsi.config import check_config, read
    from recrsi.prepare import prepare
    report = demo(tmp_path/'demo')
    assert report['device']=='cpu' and report['cost']['gpu_seconds']==0
    project=tmp_path/'demo/project'
    config=project/'task.json'
    env=read(project/'environment.json')
    api=tmp_path/'private.api-config'
    api.write_text('base_url: https://example.invalid/v1\napi_key: fixture\n')
    env.update(worker_python=sys.executable, recbole_root=str(Path(recbole.__file__).parent.parent),
               resource_slots=str(tmp_path), driver_root=str(tmp_path), api_config=str(api))
    (project/'environment.json').write_text(json.dumps(env))
    assert check_config(config)['status']=='INPUTS_VALIDATED_NO_EXECUTION'
    prepared=prepare(config,tmp_path/'prepared')
    task=read(prepared/'TASK.json')
    launch=read(task['launch'])
    evaluator=read(launch['evaluator_config'])
    # Exercise the runtime root consumer, not only the preparation validator.
    worker_modules()
    from budget_binding import install_budget
    install_budget()
    from graph_root_launcher import NoProviderRoot
    from active_wall import ActiveWallClock
    from main_resources import MainResources
    from graph_reference_root import reference_root
    ActiveWallClock.create(evaluator['active_wall_path'],evaluator['run_identity'])
    resources=MainResources(evaluator['output'],evaluator['run_identity'],evaluator['active_wall_path'],NoProviderRoot())
    baseline, ledger=reference_root(evaluator,resources)
    assert baseline['metrics']['ndcg_at_5']==report['metrics']['ndcg@5']
    assert reference_root(evaluator,resources)[0]==baseline
    inspected=subprocess.run([sys.executable,'-m','recrsi','inspect',str(prepared)],capture_output=True,text=True)
    assert inspected.returncode==0,inspected.stderr
