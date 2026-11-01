import json,tempfile,unittest
from pathlib import Path
from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.native_review_diagnostic_receipt_audit import known_usage

class ReceiptAuditTests(unittest.TestCase):
    def test_invalid_schema_still_charged_from_raw_usage(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);scope=root/'qwen_plus';(scope/'cases').mkdir(parents=True);(scope/'slots').mkdir()
            req={'kind':'test'};h=fingerprint(req)
            (scope/'cases/case_0.json').write_text(json.dumps({'task_id':'t','request_hash':h,'valid':False}))
            raw={'status_code':200,'body':json.dumps({'usage':{'prompt_tokens':1000,'completion_tokens':50}})}
            (scope/'slots'/('request_'+h[:32]+'.raw.json')).write_text(json.dumps({'request':req,'raw':raw,'integrity':fingerprint(raw)}))
            r=known_usage(root,'qwen_plus','t',0)
            self.assertTrue(r['known_usage']);self.assertFalse(r['event']['valid'])
            self.assertAlmostEqual(r['configured_known_cny'],.0009)
    def test_unknown_is_not_claimed_zero_cost(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);p=root/'qwen_flash/cases';p.mkdir(parents=True)
            (p/'case_0.json').write_text(json.dumps({'task_id':'t','request_hash':'a'*64}))
            r=known_usage(root,'qwen_flash','t',0)
            self.assertFalse(r['known_usage']);self.assertEqual(r['http_status'],'unknown')
