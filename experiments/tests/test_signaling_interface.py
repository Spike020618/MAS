import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.mechanism import probe_signaling_interface as probe


def response():
    return {'http_status':200,'body':json.dumps({'model':'fixture','usage':{'total_tokens':12},
        'choices':[{'finish_reason':'stop','message':{'content':'A'},'logprobs':{'content':[{
        'token':'A','top_logprobs':[{'token':'A','logprob':-.3},{'token':'B','logprob':-2.}]}]}}]})}


class SignalingInterfaceTests(unittest.TestCase):
    def test_request_excludes_qwen(self):
        with self.assertRaises(ValueError):probe.request('ali_qwen','x','generate_positive')
        req=probe.request('deepseek','fixture','generate_positive')
        self.assertNotIn('api_key',req)
        self.assertEqual(req['thinking'],{'type':'disabled'})

    def test_extraction_preserves_mass_before_conditional_normalization(self):
        parsed=probe.extract({'response':response()})
        self.assertTrue(parsed['usable'])
        self.assertLess(parsed['candidate_mass_total'],1)
        self.assertAlmostEqual(sum(parsed['conditional_probabilities'].values()),1)

    def test_missing_probabilities_and_format_fail_closed(self):
        for kind in ('missing','spaces','truncated'):
            raw=response();body=json.loads(raw['body']);c=body['choices'][0]
            if kind=='missing':c['logprobs']=None
            elif kind=='spaces':c['message']['content']=' A'
            else:c['finish_reason']='length'
            raw['body']=json.dumps(body)
            self.assertFalse(probe.extract({'response':raw})['usable'])

    def test_mock_resume_no_new_calls(self):
        cfg={b:{'api_key':'fixture','model':'fixture'} for b in probe.BACKENDS}
        with tempfile.TemporaryDirectory() as tmp,patch.object(probe,'OUT',Path(tmp)),\
             patch('experiments.core.llm.LLM_BACKENDS',cfg),patch.object(probe,'transport',return_value=response()) as transport,\
             patch('builtins.print'):
            first=probe.run();second=probe.run()
            self.assertEqual(first,second)
            self.assertEqual(transport.call_count,8)
            self.assertEqual(first['http_attempts'],8)


if __name__=='__main__':unittest.main()
