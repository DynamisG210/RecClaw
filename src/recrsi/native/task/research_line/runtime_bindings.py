from pathlib import Path
def install(output):
 from broker_storage_path import install as broker
 from autospace_schema_tag import install as schema
 broker(output);schema()
 import continuation_binding
 route=continuation_binding.route_new_evaluators
 continuation_binding.route_new_evaluators=lambda entry:route(Path(__file__).resolve().parent/'current_evaluator_entry.py')
