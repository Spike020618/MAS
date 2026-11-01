"""Synthetic scorer-only fixtures: not model predictions or an online release."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.calibration_d2 import build
from experiments.paired_deliberation.validate_d2 import score


class ScoringTests(unittest.TestCase):
    def test_semantic_cohort_remains_a_list_when_counting_fresh_receipts(self):
        _,labels=build()
        thresholds={'new_semantic_joint_accuracy_min':.90,'each_soft_purpose_joint_accuracy_min':.875,'new_semantic_invalid_max':2}
        plan={'plan_hash':'synthetic-scorer-only','thresholds':thresholds,'maximum_physical_attempts':144,
              'maximum_estimated_cny':3.0,'verifier':{'allowed_returned_models':['fake-model'],
               'input_cny_per_million':2.0,'output_cny_per_million':8.0},'scope':'unit test only'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'labels.json').write_text(json.dumps(labels));(root/'predictions').mkdir();(root/'slots').mkdir()
            for label in labels:
                rejected=label['expected_rejected']
                result={'authority_rejected':rejected,'sentinel_preserved':True,'hard_exclusion_active':label['expected_hard']}
                if not rejected:result['verification']={'verdict':label['expected_verdict'],'atom':label['expected_scope'],'verification_failed':False}
                (root/'predictions'/(label['id']+'.json')).write_text(json.dumps({'result':result,'integrity':fingerprint(result),'plan_hash':plan['plan_hash']}))
            (root/'usage.json').write_text(json.dumps({'usage':{'usage_complete':True}}))
            raw={'raw':{'status_code':200,'body':json.dumps({'model':'fake-model','usage':{'prompt_tokens':10,'completion_tokens':5}})}}
            (root/'slots/fixture.raw.json').write_text(json.dumps(raw))
            with patch('experiments.paired_deliberation.validate_d2.verify',return_value=plan),patch('experiments.paired_deliberation.validate_d2.provenance',return_value={'missing':[],'unexpected_slots':[],'expected_receipts':1}):
                result=score(root,write=False)
            self.assertEqual(result['new_semantic_n'],64);self.assertEqual(result['new_semantic_correct'],64)
            self.assertEqual(result['physical']['new_responses'],1)
            self.assertEqual(result['regression_n'],4)
            self.assertEqual(result['purpose_summary']['current_preference']['n'],32)


if __name__=='__main__':unittest.main()
