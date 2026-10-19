"""E2 endpoint2 JSON-mode transport; the complete native response validator stays intact."""
import json
from pathlib import Path
import runpy
import sys

SCHEMA_MARKER = '\n\nRequired response JSON Schema (validated locally):\n'


def include_schema(kwargs):
    amended = dict(kwargs)
    schema = json.loads(Path(amended['schema_path']).read_text())
    suffix = SCHEMA_MARKER + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
    if not amended['prompt'].endswith(suffix):
        amended['prompt'] += suffix
    return amended


def json_transport(identity):
    payload = identity['request_payload']
    if 'input' in payload:
        payload['text'] = {'format': {'type': 'json_object'}}
    else:
        payload['response_format'] = {'type': 'json_object'}
    return identity


def main():
    workspace = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(workspace / 'external_tasks/e2_se_cds_shared/controller_adapter'),
                   str(workspace / 'src'), str(workspace / 'tools')]
    from ro_shared_controller_adapter import BudgetedProviderCall
    from recclaw_core.experiments.helix_abc_v1.lab_api_broker import LabApiCanaryBrokerV1
    original_call = BudgetedProviderCall.__call__
    original_identity = LabApiCanaryBrokerV1.sealed_request_identity

    def call(self, **kwargs):
        # Attach BEFORE input admission/metering. No unmetered schema context.
        return original_call(self, **include_schema(kwargs))

    def identity(self, **kwargs):
        return json_transport(original_identity(self, **kwargs))

    BudgetedProviderCall.__call__ = call
    LabApiCanaryBrokerV1.sealed_request_identity = identity
    runpy.run_path(str(workspace / 'scripts/run_research_line_standalone.py'), run_name='__main__')


if __name__ == '__main__':
    main()
