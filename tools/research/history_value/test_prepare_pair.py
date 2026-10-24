import copy
import hashlib
import unittest

from prepare_pair import encoded, prepare


class PairTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            'schema': 'recrsi.history-snapshot.v1',
            'incumbent_files': {'models.py': {'content': 'model',
                'sha256': hashlib.sha256(b'model').hexdigest()}},
            'mechanism_space': {'auto_space_family_v1.json': {'families': []}},
            'context': {'frontier': {'incumbent_ndcg@5': 0.4,
                                     'effect_trajectory': ['negative-result-sentinel']},
                        'scientific_memory': {'task_queue': ['old-queue-sentinel']},
                        'policy': {'router': 'old-router-sentinel'}},
            'provenance': {'checkpoint': 'private-path-sentinel'},
        }

    def test_no_prior_state_leaks_to_reset(self):
        pairs, receipt = prepare(self.snapshot)
        for value in ('negative-result-sentinel', 'old-queue-sentinel',
                      'old-router-sentinel', 'private-path-sentinel'):
            self.assertNotIn(value.encode(), encoded(pairs['reset']))
        self.assertIn(b'negative-result-sentinel', encoded(pairs['retain']))
        self.assertEqual(pairs['retain']['common'], pairs['reset']['common'])
        self.assertFalse(receipt['execution_enabled'])

    def test_changed_source_rejected(self):
        self.snapshot['incumbent_files']['models.py']['content'] = 'different'
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            prepare(self.snapshot)

    def test_history_change_cannot_change_common_input(self):
        changed = copy.deepcopy(self.snapshot)
        changed['context']['policy']['router'] = 'another-history'
        _, a = prepare(self.snapshot)
        _, b = prepare(changed)
        self.assertEqual(a['common_sha256'], b['common_sha256'])
        self.assertNotEqual(a['retained_history_sha256'], b['retained_history_sha256'])


if __name__ == '__main__':
    unittest.main()
