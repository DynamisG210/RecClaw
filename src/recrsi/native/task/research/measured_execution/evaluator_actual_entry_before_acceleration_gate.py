"""Reuse the proven evaluator entry; replace only the trusted runtime copy step."""
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
N=Path(__file__).resolve().parent
B=N.parent.parent/'research_line'
sys.path[:0]=[str(B),str(B/'future_graph_composition'),str(N)]
import current_evaluator_entry,continuation_binding
from evaluator_measured_entry import staged
@contextmanager
def measured_stage(module,corrected):
    with patch.object(module,'run_stage',staged(module)):
        yield
if __name__=='__main__':
    with patch.object(continuation_binding,'evaluator_scope',measured_stage):
        current_evaluator_entry.main()
