import unittest

from experiments.paired_deliberation.rank_representation import decode_rank_literal


class RankRepresentationTests(unittest.TestCase):
    def test_exact_saved_list_decodes_without_reordering(self):
        ids = [str(i) for i in range(10)]
        record = {'report': {'answer': repr(ids)}}
        decoded, ops = decode_rank_literal(record, {'kind': 'ranking_v6'}, ids)
        self.assertEqual(decoded['report'], {'ranking': ids})
        self.assertTrue(ops); self.assertIn('answer', record['report'])

    def test_no_unknown_duplicate_padding_truncation_or_code_execution(self):
        pool = [str(i) for i in range(10)]
        for value in (repr(pool[:5]), repr(pool + ['x']), repr(pool[:-1] + ['x']), repr(['1'] * 10),
                      'Top choices: ' + repr(pool), "[__import__('os').getcwd()]", repr(tuple(pool))):
            record = {'report': {'answer': value}}
            decoded, ops = decode_rank_literal(record, {'kind': 'ranking_v6'}, pool)
            self.assertEqual(decoded, record); self.assertFalse(ops)


if __name__ == '__main__': unittest.main()
