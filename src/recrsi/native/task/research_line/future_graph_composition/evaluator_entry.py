"""Existing owned evaluator with a separately pinned postfit verifier correction."""
import importlib
import json
from pathlib import Path
import sys
from continuation_binding import evaluator_scope


def main():
    request = json.loads(Path(sys.argv[sys.argv.index('--request') + 1]).read_bytes())
    config = request['config']
    entries = [Path(p) for p in config['assets'] if Path(p).name == 'graph_candidate_evaluator.py']
    if len(entries) != 1:
        raise ValueError('One pinned original graph evaluator required')
    entry = entries[0].resolve(strict=True)
    sys.argv[0] = str(entry)
    sys.path.insert(0, str(entry.parent))
    import graph_candidate_evaluator as evaluator
    evaluator.configure(config)
    # Existing run placement/namespace/watchdog/accounting, without shared edits.
    branch = Path(config['output']).parents[1]
    sys.path.insert(0, str(branch / 'private_operations'))
    import owned_evaluator
    owned_evaluator.install()
    with evaluator_scope(evaluator, Path(__file__).parent / 'graph_checkpoint_verify.py'):
        importlib.import_module(entry.stem).main()


if __name__ == '__main__':
    main()
