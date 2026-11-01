import collections,json,re,unittest
from pathlib import Path
import numpy as np
from experiments.paired_deliberation.native_review_calibration_data import OUT,GAIN,review_partition,read_fit_targets,select_pairs

class ReviewPartitionTests(unittest.TestCase):
    def test_conversation_level_split_never_separates_two_prefixes(self):
        ids=['12:2','12:8','39:4','40:1'];split=review_partition(ids)
        for part in split.values():self.assertEqual('12:2'in part,'12:8'in part)
        self.assertEqual(set(split['review-cal-fit'])|set(split['review-cal-diag']),set(ids));self.assertFalse(set(split['review-cal-fit'])&set(split['review-cal-diag']))
    def test_diagnostic_label_payload_never_accessed(self):
        class Diagnostic(dict):
            def __getitem__(self,key):
                if key=='dialog':raise AssertionError('diagnostic response accessed')
                return super().__getitem__(key)
        raw=[{'conv_id':'fit','dialog':[{'role':'Seeker'},{'role':'Recommender','movies':['film','film']}]},Diagnostic(conv_id='diag')]
        labels=read_fit_targets(raw,['fit:1'],{'film':7});self.assertEqual(labels,{'fit:1':[7,7]})
    def test_no_gold_pool_insertion_and_shortage_is_explicit(self):
        ids=['q:2'];pool=np.arange(1,13,dtype=np.int32);offsets=np.array([0,12]);mu=np.linspace(1,.1,12);members=np.array([mu,mu*.9,mu*1.1])
        selected,_,shortage=select_pairs(ids,ids,{'q:2':[99]},pool,offsets,mu,members)
        self.assertTrue(shortage);self.assertEqual(len(selected),1);self.assertEqual(selected[0]['winner'],'tie')
        self.assertIn(int(selected[0]['a']),pool);self.assertIn(int(selected[0]['b']),pool)
        self.assertGreaterEqual(selected[0]['a_position'],3);self.assertLess(selected[0]['a_position'],10);self.assertGreaterEqual(selected[0]['b_position'],10)

class SavedReviewDataLeakageTests(unittest.TestCase):
    def setUp(self):
        if not(OUT/'audit.json').exists():self.skipTest('fold-serving review package not staged')
        self.inputs=json.loads((OUT/'pair64.inputs.json').read_text());self.labels=json.loads((OUT/'pair64.labels.json').read_text())
    def test_complete_quota_and_no_training_label_or_class_in_serving(self):
        self.assertEqual(len(self.inputs),64);self.assertEqual(collections.Counter(r['winner'] for r in self.labels),{'a':24,'b':24,'tie':16})
        self.assertEqual(len({r['id'].rsplit(':',1)[0]for r in self.inputs}),64)
        forbidden={'winner','stratum','observed_a_count','observed_b_count','observed_count','label'}
        def check(obj):
            if isinstance(obj,dict):
                self.assertFalse(set(obj)&forbidden)
                for value in obj.values():check(value)
            elif isinstance(obj,list):
                for value in obj:check(value)
        for row in self.inputs:
            check(row);self.assertRegex(row['task_id'],r'^review-fit:\d+:\d+:\d+:\d+$')
            self.assertTrue(set(row['pair'])<=set(row['blind_pool_native_ids']))
        self.assertEqual(len(json.loads((OUT/'fit48.inputs.json').read_text())),48)
    def test_evidence_history_is_fold80_only_and_diag_remains_disjoint(self):
        fit={str(x)for x in json.loads((GAIN/'fold80.conversations.json').read_text())};hold={str(x)for x in json.loads((GAIN/'hold20.conversations.json').read_text())}
        actual={str(c['conv_id'])for c in json.loads((OUT/'fold-serving-data/train_data.json').read_text())};self.assertEqual(actual,fit);self.assertFalse(actual&hold)
        split=json.loads((OUT/'review_split_commitment.json').read_text())['partition'];diag={r.rsplit(':',1)[0]for r in split['review-cal-diag']['ids']}
        for row in self.inputs:
            self.assertNotIn(row['id'].rsplit(':',1)[0],diag)
            for source in row['sources'].values():
                if source['kind']=='historical_prediction_basis':self.assertIn(source['origin_case_id'].rsplit(':',1)[0],fit)
    def test_calibrated_forecasts_are_prediction_only_with_three_members(self):
        for row in self.inputs:
            cards=[s for sid,s in row['sources'].items()if sid.startswith('forecast:calibrated:')];self.assertEqual(len(cards),2)
            for card in cards:
                forecast=json.JSONDecoder().raw_decode(card['text'].split(': ',1)[1])[0]
                self.assertEqual(len(forecast['member_predicted_counts']),3)
                self.assertAlmostEqual(float(np.std(forecast['member_predicted_counts'])),forecast['three_member_stability_sd'])
                self.assertFalse(card['semantic_entailment_certified']);self.assertFalse(card['current_user_fact_authority'])
if __name__=='__main__':unittest.main()
