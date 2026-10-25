from fractions import Fraction as F
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.core.mechanism.verification_risk_game import solve_risk_game
from experiments.mechanism import run_verification_crossed_dev as run


def response(author=False):
    content='A short unverified proposal.' if author else json.dumps({'verdict':'accept','reason':'fixture'})
    return {'http_status':200,'body':json.dumps({'model':'deepseek-flash',
        'system_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501','usage':{'total_tokens':10},
        'choices':[{'finish_reason':'stop','message':{'content':content}}]})}


class VerificationCrossedTests(unittest.TestCase):
    def test_invariant_blind_two_policy_game_has_no_strict_mixing_gain(self):
        for constant in (F(1,10),F(1,2),F(4,5)):
            for a in (F(1,10),F(3,5)):
                for b in (F(1,5),F(9,10)):
                    r=solve_risk_game(((constant,constant),(a,b)))
                    self.assertEqual(r['value'],r['best_pure_worst_loss'])

    def test_third_honest_protocol_can_supply_real_nonconstant_row(self):
        r=solve_risk_game((('1/2','1/2'),('1/10','7/10'),('3/5','1/5')))
        self.assertLess(r['value'],r['best_pure_worst_loss'])

    def test_blind_does_not_receive_proposal_and_all_roles_omit_truth(self):
        c={'question':'q','choices':['yes','no']}
        blind=run.request(c,'yes','independent','m')
        self.assertNotIn('untrusted_proposal',json.loads(blind['messages'][1]['content']))
        for role in ('author','holistic_audit','claim_audit'):
            req=run.request(c,'yes',role,'m','untrusted')
            self.assertNotIn('truth',json.loads(req['messages'][1]['content']))
            self.assertEqual(req['backend'],'deepseek')

    def test_abstain_and_bad_format_retained_as_error(self):
        row={'response':response()};b=json.loads(row['response']['body'])
        b['choices'][0]['message']['content']=json.dumps({'verdict':'abstain','reason':'unknown'})
        row['response']['body']=json.dumps(b)
        self.assertEqual(run.decode(row)['verdict'],'invalid')

    def test_complete_mock_is_resumable_and_blind_reused_not_recalled(self):
        c={'id':'fixture:q','question':'q','choices':['a','b']}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';src=Path(tmp)/'src';out.mkdir();src.mkdir()
            (src/'public_cases.json').write_text(json.dumps([c]))
            (src/'result.json').write_text(json.dumps({'records':[{'id':c['id'],'gold':'a'}]}))
            (src/'source_verification.json').write_text('{}')
            def transport(req):return response('response_format' not in req)
            with patch.object(run,'SOURCE',src),patch.object(run,'OUT',out),\
                 patch('experiments.core.llm.LLM_BACKENDS',{'deepseek':{'model':'fixture'}}),\
                 patch.object(run,'transport',side_effect=transport) as api,patch('builtins.print'):
                # Leave-one-out requires at least two question groups.
                c2={**c,'id':'fixture:q2'}
                (src/'public_cases.json').write_text(json.dumps([c,c2]))
                (src/'result.json').write_text(json.dumps({'records':[{'id':x['id'],'gold':'a'} for x in [c,c2]]}))
                first=run.collect();second=run.collect()
                self.assertEqual(first,second);self.assertEqual(api.call_count,24)
                self.assertEqual(first['logical_trials'],24)
                self.assertFalse(first['gate_passed_for_independent_test'])


if __name__=='__main__':unittest.main()
