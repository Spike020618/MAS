import copy
import unittest

from experiments.paired_deliberation.atomic_relations_v4 import reduce_relations
from experiments.paired_deliberation.prepare_commit_calibration import cases
from experiments.paired_deliberation.verified_state import VerificationError


class PolarityWitnessTests(unittest.TestCase):
    def test_constructed_48_labels_reduce_consistently(self):
        rows = cases()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({r['id'] for r in rows}), 48)
        for row in rows:
            result = reduce_relations(row['expected_atom'], row['sources'], row['expected_relations'])
            self.assertEqual(result['verdict'], row['expected_verdict'], row['id'])

    def test_one_positive_quote_cannot_certify_both(self):
        row = next(r for r in cases() if r['mode'] == 'conflict')
        bad = copy.deepcopy(row['expected_relations'])
        bad['relations'][0]['refute_quote'] = bad['relations'][0]['support_quote']
        with self.assertRaisesRegex(VerificationError, 'distinct'):
            reduce_relations(row['expected_atom'], row['sources'], bad)
        bad['relations'][0]['refute_quote'] = ''
        with self.assertRaisesRegex(VerificationError, 'missing'):
            reduce_relations(row['expected_atom'], row['sources'], bad)

    def test_valid_quote_with_wrong_object_is_still_a_model_semantic_risk(self):
        # Structural validation cannot prove meaning. Keep this boundary explicit:
        # a model can select a real quote about another object and pass structure.
        row = next(r for r in cases() if r['mode'] == 'matching')
        bad = copy.deepcopy(row['expected_relations'])
        quote = row['sources']['message:1']['text'].split('. ')[-1]
        bad['relations'][0]['support_quote'] = quote
        self.assertEqual(reduce_relations(row['expected_atom'], row['sources'], bad)['verdict'], 'supported')

    def test_duplicate_sources_and_fabricated_quotes_rejected(self):
        row = cases()[0]
        for mode in ('duplicate', 'fabricated'):
            bad = copy.deepcopy(row['expected_relations'])
            if mode == 'duplicate': bad['relations'].append(copy.deepcopy(bad['relations'][0]))
            else: bad['relations'][0]['support_quote'] = 'Invented source.'
            with self.assertRaises(VerificationError): reduce_relations(row['expected_atom'], row['sources'], bad)


if __name__ == '__main__': unittest.main()
