import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_deliberation.engineering_d2 import prepare,store,read_saved
from experiments.paired_deliberation.preflight_d2 import VERIFIER


class EngineeringProtocolTests(unittest.TestCase):
    def test_complete_physical_graph_and_budget_are_fixed_before_provider_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            validation=Path(directory)/'validation';validation.mkdir()
            (validation/'release_score.json').write_text('{}')
            with patch('experiments.paired_deliberation.engineering_d2.require_engineering_release',return_value=({'plan_hash':'fixture','verifier':VERIFIER},{})):
                plan=prepare(Path(directory)/'engineering',validation)
            self.assertEqual(plan['expected_paths'],30)
            self.assertEqual(plan['max_actor_physical_attempts_each'],39)
            self.assertEqual(plan['max_verifier_physical_attempts'],84)
            self.assertLessEqual(plan['verifier_full_batch_upper_cny'],3.0)
            self.assertFalse(list(Path(directory).rglob('*.raw.json')))
    def test_saved_results_are_bound_to_exact_engineering_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json'
            store(path,{'state':'preserved'},'original-plan')
            self.assertEqual(read_saved(path,'original-plan'),{'state':'preserved'})
            with self.assertRaisesRegex(ValueError,'changed'):read_saved(path,'different-plan')


if __name__=='__main__':unittest.main()
