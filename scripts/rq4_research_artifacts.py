"""Observe frozen main-code returns; never mutate candidate code, locals, or RNG."""
import json
import sys
from pathlib import Path

class ResearchArtifacts:
    def __init__(self, candidate, output):
        self.filename = str((Path(candidate) / 'main_code.py').resolve())
        self.output = Path(output)
        self.curve = []
        self.best_state = None
        self.best_metrics = None
    def observe(self, frame, event, value):
        if event != 'return' or frame.f_code.co_filename != self.filename:
            return
        if frame.f_code.co_name == '_metrics_from_ranks' and isinstance(value, dict):
            parent = frame.f_back
            if parent and parent.f_code.co_filename == self.filename and 'epoch' in parent.f_locals:
                self.curve.append({'epoch': parent.f_locals['epoch'], 'metrics': dict(value)})
        elif frame.f_code.co_name == 'main':
            self.best_state = frame.f_locals.get('best_state')
            self.best_metrics = frame.f_locals.get('best_metrics')
    def __enter__(self):
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError('Existing profiler would be replaced')
        sys.setprofile(self.observe)
        return self
    def __exit__(self, *error):
        sys.setprofile(self.previous)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'EPOCH_METRICS.json').write_text(json.dumps({'curve': self.curve, 'selected_metrics': self.best_metrics}, indent=2))
        if self.best_state is not None:
            import torch
            torch.save(self.best_state, self.output / 'BEST_CHECKPOINT.pt')
