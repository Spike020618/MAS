"""Local-artifact inference isolation/replay; skipped without trained artifacts."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from experiments.paired_deliberation.native_calibrated_challenge import DATA,RESULT as CALIBRATOR,CalibratedChallengeModel
from experiments.paired_deliberation.native_calibrated_transfer import RESULT as TRANSFER,PrecomputedCalibratedCounts

class FrozenCalibratedServingTests(unittest.TestCase):
    def setUp(self):
        if not (CALIBRATOR/'fitting_complete.json').exists():self.skipTest('calibrated trained artifact not present')
    def test_frozen_score_pool_replays_without_any_future_targets(self):
        ids=json.loads((DATA/'hold20.ids.json').read_text());diag=json.loads((CALIBRATOR/'split_commitment.json').read_text())['diagnostic']['ids'][0];q=ids.index(diag)
        items=json.loads((DATA/'native_item_ids.json').read_text());row=json.loads((DATA/'hold20.inputs.json').read_text())[q]
        values=[np.load(DATA/name,mmap_mode='r')[q] for name in ('sparse/hold20.native_scores.npy','sparse/hold20.native_ranks.npy','kbrd/hold20.native_logits.npy','kbrd/hold20.native_ranks.npy')]
        model=CalibratedChallengeModel();first=model.score_pool(row,items,*values)
        poisoned=copy.deepcopy(row);poisoned['targets']=[-999,-999,-999];second=model.score_pool(poisoned,items,*values)
        self.assertEqual(first['expected_counts'],second['expected_counts'])
        offsets=np.load(CALIBRATOR/'candidate_offsets.npy',mmap_mode='r');p=np.load(CALIBRATOR/'hold20.expected_count.npy',mmap_mode='r');sl=slice(offsets[q],offsets[q+1])
        self.assertTrue(np.allclose([r[1] for r in first['rows']],p[sl],rtol=0,atol=1e-12))
        self.assertFalse(first['actual_lm_review_effect_estimated']);self.assertFalse(first['semantic_truth_certified'])
    def test_transfer_distinguishes_unsupported_items_from_estimated_zero(self):
        if not (TRANSFER/'artifact_complete.json').exists():self.skipTest('transfer artifact not present')
        serving=PrecomputedCalibratedCounts();cid=serving.ids[0];q=serving.row_index[cid]
        supported=np.flatnonzero(np.isfinite(serving.count[q]));unsupported=np.flatnonzero(~np.isfinite(serving.count[q]))
        ids=[str(serving.items[supported[0]]),str(serving.items[unsupported[0]])];out=serving.score_pool(cid,ids)
        self.assertTrue(out['rows'][0][3]);self.assertFalse(out['rows'][1][3]);self.assertIsNone(out['rows'][1][1])
        self.assertIn(ids[0],out['expected_counts']);self.assertNotIn(ids[1],out['expected_counts'])
    def test_all_shared_materials_have_scores_without_query_labels(self):
        if not (TRANSFER/'new64.inputs.json').exists():self.skipTest('public inference materials not present')
        row=json.loads((TRANSFER/'new64.inputs.json').read_text())[0];pool={m['movie_id'] for m in row['candidates']}
        self.assertNotIn('targets',row);self.assertEqual(pool,set(row['calibrated_expected_counts']))
        self.assertTrue(row['all_methods_must_receive_same_scores'])
        for source in row['sources'].values():
            self.assertFalse(source['current_user_fact_authority']);self.assertFalse(source['semantic_truth_certified'])
            self.assertNotEqual(source['origin_case_id'],row['id']);self.assertEqual(source['query_case_id'],row['id'])
    def test_strong_rrf_alignment_keeps_identical_teachers_and_uses_stronger_baseline(self):
        path=TRANSFER/'rrf_alignment_audit.json'
        if not path.exists():self.skipTest('RRF reconciliation not staged')
        audit=json.loads(path.read_text());p=audit['teacher_parity']
        self.assertEqual(p['sparse_top100_equal_rows'],p['query_count']);self.assertEqual(p['kbrd_top100_equal_rows'],p['query_count'])
        self.assertEqual(p['sparse_reported_score_max_abs_difference'],0)
        self.assertEqual(audit['strongest_legal_rrf_variant'],'full_native__native_id')

if __name__=='__main__':unittest.main()
