import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.core.mechanism.scored_signaling import compare,decode_score
from experiments.mechanism import run_signaling_pilot as run


def response(codes='AB'):
    return {'http_status':200,'body':json.dumps({'model':'deepseek-flash',
        'system_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501','usage':{'total_tokens':10},
        'choices':[{'finish_reason':'stop','message':{'content':'A'},'logprobs':{'content':[{
            'token':'A','top_logprobs':[{'token':c,'logprob':math.log(1/len(codes))} for c in codes]}]}}]})}


class ScoredSignalingTests(unittest.TestCase):
    def test_multichoice_comparison_and_no_labels(self):
        r=compare(['a','b','c'],(.6,.3,.1),(.1,.3,.6),((.8,.2),(.5,.5),(.2,.8)))
        self.assertEqual(set(r['answers']),set(run.NAMES))
        self.assertEqual(r['answers']['generation'],'a')
        self.assertTrue(r['game_result']['converged_to_tolerance'])
        self.assertNotIn('gold',compare.__code__.co_varnames)

    def test_missing_mass_no_repair(self):
        self.assertTrue(decode_score(response('ABCD'),list('ABCD'))['usable'])
        self.assertFalse(decode_score(response('AB'),list('ABCD'))['usable'])
        self.assertFalse(decode_score({'http_status':500,'body':'bad'},list('AB'))['usable'])

    def test_explicit_coding_reversal(self):
        c={'question':'q','choices':['yes','no']}
        normal=run.request(c,'g_positive','m');reverse=run.request(c,'g_positive_reverse','m')
        self.assertEqual(json.loads(normal['messages'][1]['content'])['code_to_registered_choice']['A'],'yes')
        self.assertEqual(json.loads(reverse['messages'][1]['content'])['code_to_registered_choice']['A'],'no')
        self.assertNotIn('gold',normal)

    def test_partial_pool_keeps_available_control(self):
        c={'question':'q','choices':['a','b']}
        good=decode_score(response(),list('AB'));bad={'usable':False,'tokens':10}
        slots={key:good for key in ['g_positive','g_negative','g_positive_reverse','g_negative_reverse','d/a','d/b']}
        slots['g_negative']=bad
        r=run.decisions(c,slots)
        self.assertEqual(r['answers']['game'],'abstain')
        self.assertFalse(r['details'])

    def test_full_mock_collection_resumes_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';source=Path(tmp)/'source';out.mkdir();source.mkdir()
            cases=[{'id':'fixture:0','question':'q','choices':['a','b']}]
            (out/'public_cases.json').write_text(json.dumps(cases))
            (source/'private_labels.json').write_text(json.dumps([{'id':'fixture:0','label':'a'}]))
            cfg={'deepseek':{'model':'fixture'}}
            with patch.object(run,'OUT',out),patch.object(run,'SOURCE',source),patch.object(run,'prepare'),\
                 patch('experiments.core.llm.LLM_BACKENDS',cfg),\
                 patch.object(run.interface,'transport',return_value=response()) as transport,patch('builtins.print'):
                first=run.collect();second=run.collect()
                self.assertEqual(first,second)
                self.assertEqual(transport.call_count,6)
                self.assertEqual(first['http_attempts'],6)
                self.assertEqual(first['metrics']['game']['required_scoring_calls'],4)


if __name__=='__main__':unittest.main()
