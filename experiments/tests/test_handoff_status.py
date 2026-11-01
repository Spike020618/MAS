import hashlib,json,tempfile,unittest
from pathlib import Path
from experiments.paired_deliberation.handoff_status import inspect_run

class HandoffStatusTests(unittest.TestCase):
    def test_unknown_intent_and_source_drift_block_resume_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d);r=w/'run';r.mkdir();s=r/'slots';s.mkdir()
            (r/'plan.json').write_text(json.dumps({'source_sha256':{'example.py':'missing'}}))
            (s/'a.intent.json').write_text('{}')
            before={str(p):p.read_bytes() for p in w.rglob('*') if p.is_file()}
            result=inspect_run(r,w)
            self.assertFalse(result['automatic_resume_ready'])
            self.assertEqual(result['source_drift'],['example.py'])
            self.assertEqual(result['providers'][0]['unknown_requests_do_not_resend'],['a.intent.json'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in w.rglob('*') if p.is_file()})
    def test_receipt_and_seal_are_distinct_from_resumable_work(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'plan.json').write_text('{}');s=r/'slots';s.mkdir()
            (s/'x.intent.json').write_text('{}');(s/'x.raw.json').write_text('{}')
            self.assertTrue(inspect_run(r,r)['automatic_resume_ready'])
            (r/'generation_complete.json').write_text('{}')
            self.assertEqual(inspect_run(r,r)['state'],'sealed_unscored')
            self.assertFalse(inspect_run(r,r)['automatic_resume_ready'])
            (r/'summary.json').write_text('{}')
            self.assertEqual(inspect_run(r,r)['state'],'scored')
