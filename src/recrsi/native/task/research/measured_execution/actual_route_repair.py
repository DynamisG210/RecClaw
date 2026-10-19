"""Bind the original route directly, after setup's fixed legacy route."""
import importlib.util
from pathlib import Path
N=Path(__file__).resolve().parent
def install():
    from capture_compatibility import install as capture
    capture()
    import measured_context,measured_context_complete,continuation_binding
    measured_context.install=measured_context_complete.install
    spec=importlib.util.spec_from_file_location('_unchanged_evaluator_route',continuation_binding.__file__)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    continuation_binding.route_new_evaluators=lambda unused:original.route_new_evaluators(N/'evaluator_accelerated_entry.py')
