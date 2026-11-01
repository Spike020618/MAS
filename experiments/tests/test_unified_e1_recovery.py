import fcntl,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.consensus_recommendation.common import fingerprint
from experiments.resume_unified_e1 import recover

class FrozenE1RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        rows=[{'id':'task'}];plan={'bounded_tasks_hash':fingerprint(rows),'task_ids':['task'],'source_sha256':{}}
        (root/'plan.json').write_text(json.dumps(plan));(root/'inputs.json').write_text(json.dumps(rows));(root/'started.json').write_text('{"pid_at_first_start":123}')
    def test_inspect_does_not_import_runtime_or_change_first_start(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);self.fixture(root);before=(root/'started.json').read_bytes()
            with patch('experiments.paired_deliberation.native_optimization_runtime.Runtime')as api:
                status=recover(root);api.assert_not_called()
            self.assertEqual(status['network_calls_in_inspect'],0);self.assertEqual((root/'started.json').read_bytes(),before)
    def test_existing_collector_blocks_second_resume_before_api(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);self.fixture(root)
            with (root/'run.lock').open('a+')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(recover(root)['collector_lock_busy'])
                with self.assertRaisesRegex(RuntimeError,'still owns'):recover(root,True)
    def test_completed_generation_is_score_only_and_keeps_original_pid(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);self.fixture(root);(root/'generation_complete.json').write_text('{}')
            with patch('experiments.evidence_consensus.runner.score')as score,patch('experiments.paired_deliberation.native_optimization_runtime.Runtime')as api:
                result=recover(root,True);score.assert_called_once_with(root);api.assert_not_called()
            self.assertEqual(result['new_api_calls'],0);self.assertEqual(json.loads((root/'started.json').read_text())['pid_at_first_start'],123)
