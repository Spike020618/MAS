import json
import unittest
from experiments.paired_deliberation.screen_domestic import choose,ranking_request
from experiments.paired_deliberation.training_evidence_view import attach
from experiments.tests.test_design_v6 import evidence


class DomesticScreenTests(unittest.TestCase):
    def test_label_blind_selection_is_order_independent_and_bounded(self):
        ids=['case:'+str(i) for i in range(100)]
        self.assertEqual(choose(ids,'recommendation'),choose(list(reversed(ids)),'recommendation'))
        self.assertEqual(len(choose(ids,'recommendation')),8)
        self.assertEqual(len(set(choose(ids,'recommendation'))),8)
    def test_information_treatment_preserves_raw_evidence_and_output_allowance(self):
        e=evidence();request=ranking_request(e)
        view={'kind':'historical_training_prediction_evidence','current_case_labels_available':False}
        offered=attach(request,view,'independent')
        raw=json.loads(request['messages'][1]['content']);treated=json.loads(offered['messages'][1]['content'])
        self.assertEqual(raw['evidence'],treated['evidence'])
        self.assertEqual(raw['schema'],treated['schema'])
        self.assertNotIn('training_prediction_evidence',raw)
        self.assertEqual(offered['max_tokens'],request['max_tokens'])


if __name__=='__main__':unittest.main()
