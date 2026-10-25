import json
from pathlib import Path
import tempfile
import unittest

from experiments.mechanism import pilot_abstract_conflicts_v2 as v2


def response(c='abstain'):
    return {'finish_reason': 'stop', 'parsed': {'C': c, 'A': 'a', 'E': 'e', 'I': 'i', 'citations': [],
        'dispute': {'proposition': 'p', 'scope': 's', 'snapshot': '2', 'conflict_type': 'none',
                    'decision_condition': 'd', 'unresolved': True}}}


class RepairTests(unittest.TestCase):
    def test_failure_never_becomes_abstain(self):
        self.assertIsNone(v2.consensus([None, None, None]))
        self.assertIsNone(v2.consensus(['abstain', None, None]))
        self.assertIsNone(v2.consensus(['yes', 'no', 'abstain']))
        self.assertEqual(v2.consensus(['abstain', 'abstain', None]), 'abstain')

    def test_truncated_even_parseable_is_invalid(self):
        row = response('yes')
        row['finish_reason'] = 'length'
        self.assertIsNone(v2.vote(row))
        self.assertIsNone(v2.packet(row, [])['report'])

    def test_dispute_schema_required_for_structured_only(self):
        row = response('yes')
        del row['parsed']['dispute']
        self.assertEqual(v2.vote(row), 'yes')
        self.assertIsNone(v2.vote(row, True))

    def test_bad_citations_and_empty_response_fail_closed(self):
        row = response()
        row['parsed']['citations'] = [{}]
        self.assertIsNone(v2.vote(row))
        self.assertIsNone(v2.vote({'parsed': None, 'finish_reason': 'stop'}))

    def test_full_batch_equal_inputs_limits_and_resume_without_network(self):
        class FakeCaller:
            def __init__(self): self.calls = {}
            def call(self, key, messages, max_tokens):
                row = {**response(), 'key': key, 'messages': messages,
                       'response': '{}', 'usage': {'total_tokens': 1}, 'max_tokens': max_tokens}
                self.calls[key] = row
                return row
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            protocol, cases = v2.setup(directory)
            for name in ['runs', 'commitments', 'initial_rounds']: (directory / name).mkdir()
            caller = FakeCaller()
            for case in cases: v2.execute(directory, case, protocol, caller)
            self.assertEqual(len(caller.calls), 108)
            for case in cases:
                qid = case['question_id']
                for i in range(3):
                    a = caller.calls[f'{qid}/ordinary/{i}']
                    b = caller.calls[f'{qid}/structured/{i}']
                    self.assertEqual(a['messages'][1], b['messages'][1])
                    self.assertEqual(a['max_tokens'], b['max_tokens'])
                    self.assertEqual(a['max_tokens'], 1600)
                v2.execute(directory, case, protocol, caller)
            self.assertEqual(len(caller.calls), 108)
            result = v2.analyze(directory)
            self.assertEqual(result['summary']['heldout_seed']['structured']['correct'], 2)
            # Failed outputs must not score the two gold-abstain questions as correct.
            for path in (directory / 'runs').glob('*.json'):
                row = json.loads(path.read_text())
                for branch in row['branches'].values(): branch['consensus'] = None
                path.write_text(json.dumps(row))
            (directory / 'analysis.json').unlink()
            result = v2.analyze(directory)
            self.assertEqual(result['summary']['heldout_seed']['structured']['correct'], 0)


if __name__ == '__main__': unittest.main()
