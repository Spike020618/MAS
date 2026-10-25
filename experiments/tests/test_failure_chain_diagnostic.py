import tempfile
import hashlib
from pathlib import Path
import unittest

from experiments.mechanism.diagnose_failure_chain import diagnose,vote
from experiments.mechanism.sprint_source_pipeline import canonical_page,repair_sources


class FailureChainTests(unittest.TestCase):
    def test_observed_counts_and_unknowns_preserved(self):
        result=diagnose()
        self.assertEqual(result['dev_corrected'],5)
        self.assertEqual(result['dev_harmed'],5)
        self.assertEqual(result['dev_D_correct'],result['dev_R_correct'])
        self.assertEqual(result['locked_same_action_vote_changes'],3)
        self.assertEqual(len(result['development_rows']),12)
        self.assertTrue(all(a['source_sufficiency']=='unjudged_not_inferred_from_self_report'
            for row in result['development_rows'] for a in row['agents']))

    def test_table_extraction_and_canonical_dedup(self):
        with tempfile.TemporaryDirectory() as temp:
            data=b'<html><p>A general description of measured objects in a test collection.</p><table><tr><th>Object</th><th>Size</th></tr><tr><td>Example specimen alpha</td><td>123 millimetres measured length</td></tr></table></html>'
            digest=hashlib.sha256(data).hexdigest(); path=Path(temp)/(digest+'.html');path.write_bytes(data)
            source={'snapshot_path':str(path),'url':'https://en.wikipedia.org/wiki/Example_page','title':'Example','id':'S1'}
            result=repair_sources('What is the measured size of specimen alpha?',
                [source,{**source,'url':'https://en.wikipedia.org/wiki/Example%20page#section'}],Path(temp))
            self.assertEqual(len(result),1)
            self.assertIn('123 millimetres',result[0]['text'])
            self.assertIn('tr',result[0]['selected_block_types'])

    def test_majority_not_correct_report_oracle(self):
        self.assertEqual(vote(['yes','no','no']),'no')
        self.assertEqual(vote(['yes','no','abstain']),'abstain')


if __name__=='__main__':unittest.main()
