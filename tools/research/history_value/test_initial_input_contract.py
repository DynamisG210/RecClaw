import copy
import json
from pathlib import Path
import tempfile
import unittest

from initial_input_contract import (CONTEXT_MARKER, SOURCE_MARKER,
    bind_initial_prompt, check_fresh_recovery, non_history_view)


class InitialContractTests(unittest.TestCase):
    def setUp(self):
        self.context = {'context': {'round': 13, 'role': 'composer', 'digest': 'id'},
            'memory': {'history': ['negative result']}, 'contract': {'seed': 7},
            'state': {'construction_parent_options': [{'candidate_id': 'same-parent'}],
                      'latest_result': {'value': -1}}, 'objective': {'task': 'same-task'}}
        self.prompt = self.make_prompt(self.context)
        self.contract = {'schema': 'recrsi.history-initial-input.v2', 'initial_round': 13,
                         'roles': {'composer': non_history_view(self.prompt)}}

    def make_prompt(self, context, source='same source'):
        return SOURCE_MARKER + json.dumps({'models.py': source}) + '\n' + CONTEXT_MARKER + json.dumps(context)

    def test_empty_options_bound_without_erasing_history(self):
        context = copy.deepcopy(self.context)
        context['state']['construction_parent_options'] = []
        output = bind_initial_prompt(self.make_prompt(context), self.contract)
        self.assertIn('negative result', output)
        self.assertEqual(non_history_view(output), self.contract['roles']['composer'])

    def test_source_or_task_drift_rejected_before_transport(self):
        with self.assertRaises(ValueError):
            bind_initial_prompt(self.make_prompt(self.context, 'other source'), self.contract)
        context = copy.deepcopy(self.context)
        context['contract']['seed'] = 8
        with self.assertRaises(ValueError):
            bind_initial_prompt(self.make_prompt(context), self.contract)

    def test_old_paid_recovery_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'FIRST_PAID_BATCH_RECOVERY.pkl').write_bytes(b'old paid bytes')
            with self.assertRaises(ValueError):
                check_fresh_recovery(root, self.contract)

    def test_later_round_unchanged(self):
        context = copy.deepcopy(self.context)
        context['context']['round'] = 14
        prompt = self.make_prompt(context)
        self.assertEqual(bind_initial_prompt(prompt, self.contract), prompt)


if __name__ == '__main__':
    unittest.main()
