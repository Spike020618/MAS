import json
import unittest
from experiments.mechanism.verification_verdict_sensitivity import explicit_verdict


def row(value,finish='stop'):
    return {'response':{'http_status':200,'body':json.dumps({'choices':[{'finish_reason':finish,
        'message':{'content':json.dumps(value)}}]})}}


class VerdictSensitivityTests(unittest.TestCase):
    def test_tie_policy_can_explain_game_vs_pure_difference(self):
        from fractions import Fraction as F
        from experiments.mechanism.audit_verification_ties import best_pure
        from experiments.core.mechanism.verification_risk_game import solve_risk_game
        losses=((F(1,2),F(1,2)),(F(1,12),F(1,12)),(F(1,12),F(1,12)))
        self.assertEqual(best_pure(losses),1)
        self.assertEqual(best_pure(losses,True),2)
        self.assertEqual(solve_risk_game(losses)['verifier_policy'],(F(0),F(0),F(1)))

    def test_only_explicit_metadata_case_recovered(self):
        self.assertEqual(explicit_verdict(row({'type':'json_object','verdict':'accept','reason':'r'})),('accept',True))
        self.assertEqual(explicit_verdict(row({'verdict':'reject','reason':'r'})),('reject',False))

    def test_unknown_keys_missing_verdict_and_truncation_not_repaired(self):
        for value in ({'type':'system','verdict':'accept','reason':'r'},
                      {'type':'json_object','reason':'accept'},
                      {'type':'json_object','verdict':'accept','reason':'r','unexpected':1}):
            self.assertEqual(explicit_verdict(row(value)),('invalid',False))
        self.assertEqual(explicit_verdict(row({'type':'json_object','verdict':'accept','reason':'r'},'length')),('invalid',False))


if __name__=='__main__':unittest.main()
