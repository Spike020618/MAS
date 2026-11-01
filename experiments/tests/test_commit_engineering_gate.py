import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_deliberation.run_commit_engineering import check_release, prepare


class EngineeringGateTests(unittest.TestCase):
    def test_failed_joint_gate_cannot_even_prepare_engineering(self):
        with tempfile.TemporaryDirectory() as directory:
            validation=Path(directory)/'validation';validation.mkdir()
            (validation/'release_score.json').write_text(json.dumps({'contract_version':'commit_v4',
                'plan_hash':'p','pass':False,'gates':{'relations':False}}))
            output=Path(directory)/'engineering'
            with patch('experiments.paired_deliberation.run_commit_engineering.verify_plan',return_value={'plan_hash':'p'}):
                with self.assertRaisesRegex(RuntimeError,'semantic gate failed'):
                    prepare(output,validation,Path(directory)/'evidence')
            self.assertFalse(output.exists())

    def test_v3_release_cannot_pass_v4_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'release_score.json').write_text(json.dumps({'contract_version':'atomic_v3','plan_hash':'p',
                'pass':True,'gates':{'all':True}}))
            with patch('experiments.paired_deliberation.run_commit_engineering.verify_plan',return_value={'plan_hash':'p'}):
                with self.assertRaisesRegex(RuntimeError,'semantic gate failed'):check_release(root)


if __name__=='__main__':unittest.main()
