import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_deliberation.domestic_transport import DomesticTransport
from experiments.paired_deliberation.requests_v5 import RequestsV5


class DomesticTransportTests(unittest.TestCase):
    def request(self):return RequestsV5.envelope('smoke',{'schema':{'ok':True}},'Return JSON.',max_tokens=32)
    def test_pending_request_counts_towards_hard_cap(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'MAS_GLM_API_KEY':'fixture-secret'}):
            transport=DomesticTransport('glm_flash',directory,max_attempts=1)
            Path(directory,'pending.intent.json').write_text(json.dumps({'request':self.request()}))
            with self.assertRaisesRegex(RuntimeError,'cap reached'):transport.preflight(self.request())
    def test_price_reservation_and_output_limit_are_checked_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'MAS_DASHSCOPE_API_KEY':'fixture-secret'}):
            transport=DomesticTransport('qwen_flash',directory,max_cny=0.00000001)
            with self.assertRaisesRegex(RuntimeError,'cap reached'):transport.preflight(self.request())
            request=self.request();request['max_tokens']=100000
            with self.assertRaises(ValueError):transport.preflight(request)
    def test_key_sent_only_to_allowlisted_origin_and_redacted_from_response(self):
        class Response:status_code=200; text='fixture-secret'
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'MAS_GLM_API_KEY':'fixture-secret'}):
            transport=DomesticTransport('glm_flash',directory)
            with patch('requests.post',return_value=Response()) as post:
                result=transport(self.request())
            self.assertNotIn('fixture-secret',str(result))
            self.assertEqual(post.call_args[0][0],'https://open.bigmodel.cn/api/paas/v4/chat/completions')
            self.assertFalse(post.call_args[1]['allow_redirects'])


if __name__=='__main__':unittest.main()
