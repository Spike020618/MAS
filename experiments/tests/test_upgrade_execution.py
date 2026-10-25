import unittest
from experiments.core.mechanism.upgrade_game import Contract, freeze_state, upgrade_request
from experiments.core.mechanism.upgrade_execution import select, seal_selected
from experiments.mechanism.build_upgrade_history import normalize, walk


class UpgradeExecutionTests(unittest.TestCase):
    def setUp(self):
        self.reports = [{'A': '', 'E': [], 'I': '', 'C': 'yes'} for _ in range(3)]
        self.frozen = freeze_state('Q?', ['yes', 'no'], self.reports)
        self.contract = Contract((0, .2, .2, .4, .2, .4, .4, .6), ((0, .01),) * 3, 1, 'engineering-fixture-not-learned')

    def events(self):
        selection = select(self.frozen, self.contract)
        events = [selection]
        for i, action in enumerate(selection['actions']):
            if action:
                events.append({'type': 'selected_upgrade', 'agent': i,
                               'request_hash': upgrade_request(self.frozen, i)['request_hash'],
                               'response_hash': '0' * 64, 'report': self.reports[i],
                               'finish_reason': 'stop', 'usage': {'total_tokens': 100}})
        return events

    def test_seal_and_fail_closed_ordering(self):
        events = self.events()
        sealed = seal_selected(self.frozen, self.contract, events)
        self.assertEqual(sealed['actions'], [1, 1, 1])
        self.assertEqual(sealed['decision'], 'yes')
        for broken in [events[1:], events[:-1], events + [events[-1]]]:
            with self.assertRaises(ValueError): seal_selected(self.frozen, self.contract, broken)

    def test_invalid_output_cannot_be_silently_accepted(self):
        for key, value in [('finish_reason', 'length'), ('request_hash', 'bad'), ('usage', {}), ('type', 'diagnostic')]:
            events = self.events(); events[1][key] = value
            with self.assertRaises(ValueError): seal_selected(self.frozen, self.contract, events)

    def test_history_normalization_and_base_question(self):
        self.assertEqual(normalize('Ａ B\nC'), 'a b c')
        rows = list(walk({'dataset': 'd', 'original_id': '1', 'base_question': 'Original?',
                          'question': 'Wrapped?', 'label': 'yes'}))
        self.assertEqual({r['text'] for r in rows}, {'original?', 'wrapped?'})
        self.assertTrue(all(r['id'] == '1' for r in rows))


if __name__ == '__main__': unittest.main()
