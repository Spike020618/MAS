import json
from pathlib import Path
import tempfile
import unittest

from experiments.core.mechanism.sequence_likelihood import requests_for_case, score_tables, sum_token_logprobs, checkpoint_digest
from experiments.mechanism.run_equilibrium_reference import run, readout, evaluate


CASE = {'id':'synthetic:1', 'question':'What is 2 + 2?',
        'candidates':{'a':'4', 'b':'5'}, 'background_correction':False}


def fixture(case=CASE):
    # Constructed numbers test software; never count as model evidence.
    rows = {}
    for slot, req in requests_for_case(case).items():
        value = -.2 if slot in ('g_positive/a','g_negative/b','d/a/correct','d/b/incorrect') else -2.
        rows[slot] = {**req, 'token_logprobs':[value]}
    return rows


class PipelineTests(unittest.TestCase):
    def test_no_labels_in_scoring_schema(self):
        with self.assertRaises(ValueError): requests_for_case({**CASE, 'label':'a'})

    def test_complete_score_and_target_binding(self):
        rows = fixture()
        score_tables(CASE, rows)
        rows['g_positive/a']['target'] = ' 5'
        with self.assertRaises(ValueError): score_tables(CASE, rows)
        with self.assertRaises(ValueError): score_tables(CASE, {})

    def test_invalid_token_values(self):
        for values in ([], [float('nan')], [True], [.1], [None]):
            with self.assertRaises(ValueError): sum_token_logprobs(values)
        self.assertEqual(sum_token_logprobs([-.2,-.3]), -.5)

    def test_background_correction_is_explicit(self):
        case = {**CASE, 'background_correction':True}
        rows = fixture(case)
        ids, g, d = score_tables(case, rows)
        self.assertEqual(ids, ('a','b'))
        self.assertAlmostEqual(g[0][0], 1.8)
        self.assertAlmostEqual(d[0][0], 1.8)

    def test_all_six_readouts_and_evaluation(self):
        r = readout(CASE, fixture())
        self.assertEqual(set(r['answers']), {'G','D','SC','MI','ER-G','ER-D'})
        self.assertTrue(all(v=='a' for v in r['answers'].values()))
        result = evaluate({'synthetic:1':r}, {'synthetic:1':'a'})
        self.assertEqual(result['metrics']['ER-G']['correct'],1)
        self.assertTrue(all(v['holm_p']==1 for v in result['paired'].values()))
        with self.assertRaises(ValueError): evaluate({'synthetic:1':r}, {})

    def test_pipeline_resume_seal_and_modified_input_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cases, scores, labels = root/'cases.json', root/'scores.json', root/'labels.json'
            cases.write_text(json.dumps([CASE]))
            scores.write_text(json.dumps({CASE['id']:fixture()}))
            labels.write_text(json.dumps({CASE['id']:'a'}))
            first = run(cases, root/'out', scores_path=scores, labels_path=labels)
            sealed = (root/'out/sealed_readouts.json').read_bytes()
            self.assertEqual(first, run(cases, root/'out', scores_path=scores, labels_path=labels))
            self.assertEqual(sealed, (root/'out/sealed_readouts.json').read_bytes())
            cases.write_text(json.dumps([{**CASE, 'question':'changed'}]))
            with self.assertRaises(ValueError): run(cases, root/'out', scores_path=scores)

    def test_checkpoint_digest_detects_change(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'config.json'; p.write_text('{}')
            first = checkpoint_digest(root)
            p.write_text('{"x":1}')
            self.assertNotEqual(first, checkpoint_digest(root))


if __name__ == '__main__': unittest.main()
