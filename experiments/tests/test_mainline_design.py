import copy
import json
import unittest
from pathlib import Path
from experiments.mechanism.validate_mainline_design import validate


class MainlineDesignTests(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'mechanism/matrices/mechanism_mainline_redesign_v1.json'
        self.config=json.loads(path.read_text())

    def test_design_does_not_claim_collection_ready(self):
        result=validate(self.config)
        self.assertTrue(result['design_consistency_pass'])
        self.assertFalse(result['ready_for_paid_collection'])

    def test_task_specific_tools_and_execution_contract_leak_are_rejected(self):
        for field in ['task_specific_checkers_allowed','contract_visible_in_execution_prompt']:
            config=copy.deepcopy(self.config)
            config['fixed_environment'][field]=True
            self.assertFalse(validate(config)['design_consistency_pass'])

    def test_oracle_selection_and_split_overlap_are_rejected(self):
        self.config['equilibrium']['per_question_gold_oracle_contract_selection_allowed']=True
        self.config['data']['calibration_question_ids']=['d:q']
        self.config['data']['test_question_ids']=['d:q']
        result=validate(self.config)
        self.assertFalse(result['design_consistency_pass'])
        self.assertTrue(any('overlap' in x for x in result['violations']))


if __name__=='__main__':unittest.main()
