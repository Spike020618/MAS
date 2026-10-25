import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.core.mechanism import targeted_challenge as m
from experiments.mechanism import run_targeted_challenge as run

CASE={'id':'quality:fixture','article_id':'fixture','article':'A source with a fact.',
      'article_sha256':'fixture','question':'Which option?', 'choices':dict(zip('ABCD',['one','two','three','four']))}


class ChallengeTests(unittest.TestCase):
    def test_closed_candidate_and_public_whitelist(self):
        with self.assertRaises(ValueError):m.request(CASE,'E','author',{},[])
        req=m.request({**CASE,'gold_label':'SECRET'},'A','author',{},[])
        self.assertNotIn('SECRET',json.dumps(req))
        self.assertEqual(json.loads(req['messages'][1]['content'])['registered_candidate'],'A')

    def test_balanced_loss_blocks_always_accept_or_reject(self):
        records=[{'id':'q','truth':int(i==0),'x':'reject'} for i in range(4)]
        self.assertEqual(m.balanced_errors(records,'x')['question_macro_balanced_error'],.5)
        for r in records:r['x']='accept'
        self.assertEqual(m.balanced_errors(records,'x')['question_macro_balanced_error'],.5)
        for r in records:r['x']='abstain'
        self.assertEqual(m.balanced_errors(records,'x')['question_macro_balanced_error'],1.)

    def test_private_diagnostics_not_forwarded(self):
        message=m.message({'argument':'a','truth':True,'utility':3,
                           'citations':[{'citation':{'chunk_id':'article','quote':'q'},'literal_source_match':True}]},'author')
        self.assertEqual(set(message),{'argument','citations'})
        self.assertNotIn('literal_source_match',json.dumps(message))

    def test_adaptive_graph_and_zero_call_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def transport(req):
                data=json.loads(req['messages'][1]['content']);instruction=req['messages'][0]['content']
                calls.append(data)
                if 'strongest source-grounded' in instruction:value={'argument':'arg','citations':[]}
                elif 'SINGLE decisive' in instruction:value={'challenge':'Which fact supports this?','query':'fact'}
                elif 'verdict (accept/reject/abstain)' in instruction:value={'verdict':'reject','reason':'unsupported','citations':[]}
                else:value={'answer':'A','reasoning':'fixture','citations':[]}
                return {'http_status':200,'body':json.dumps({'model':'deepseek-flash',
                    'system_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501','usage':{'total_tokens':10},
                    'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})}
            run.STOP.clear()
            with patch.object(run,'OUT',Path(tmp)),patch.object(run,'transport',side_effect=transport),patch('builtins.print'):
                first=run.collect(CASE)
                self.assertEqual(len(calls),37)
                # First candidate: generic restatement has no challenge;
                # adaptive reply receives it; final comparison retains candidate.
                self.assertNotIn('challenge',calls[5]['history'])
                self.assertIn('challenge',calls[7]['history'])
                self.assertNotIn('previous_reviews',calls[6]['history'])
                self.assertEqual(calls[8]['registered_candidate'],calls[9]['registered_candidate'])
                with patch.object(run,'transport',side_effect=AssertionError('no new calls')):
                    self.assertEqual(first,run.collect(CASE))


if __name__=='__main__':unittest.main()
