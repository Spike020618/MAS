from fractions import Fraction as F
import unittest

from experiments.core.mechanism.verification_risk_game import calibration_matrix,solve_risk_game,select_verifier


def trials():
    rows=[]
    for q in ('q1','q2'):
        for claim,label in (('correct',1),('wrong1',0),('wrong2',0)):
            for v in ('strict','lenient'):
                for a in ('neutral','assertive'):
                    rows.append({'question_id':q,'claim_id':claim,'verifier':v,'attack':a,
                        'truth':label,'truth_kind':'official_answer','truth_ref':'fixture-only/'+q,
                        'verdict':'reject' if v=='strict' else 'accept'})
    return rows


class VerificationRiskGameTests(unittest.TestCase):
    def test_always_accept_reject_not_rewarded_for_agreement(self):
        matrix=calibration_matrix(trials(),['strict','lenient'],['neutral','assertive'])
        self.assertEqual(matrix['losses'],((F(1,2),F(1,2)),)*2)

    def test_macro_question_prevents_many_wrong_candidates_dominating(self):
        rows=trials()
        for row in list(rows):
            if row['question_id']=='q1' and row['claim_id']=='wrong2':rows.append({**row,'claim_id':'wrong3'})
        self.assertEqual(calibration_matrix(rows,['strict','lenient'],['neutral','assertive'])['losses'],((F(1,2),F(1,2)),)*2)

    def test_truth_changes_and_missing_pairs_rejected(self):
        for mode in ('truth','missing','agreement'):
            rows=trials()
            if mode=='truth':rows[0]['truth']=0
            elif mode=='missing':rows.pop()
            else:rows[0]['truth_kind']='peer_agreement'
            with self.assertRaises(ValueError):calibration_matrix(rows,['strict','lenient'],['neutral','assertive'])

    def test_invalid_is_error_not_convenient_rejection(self):
        rows=trials()
        for row in rows:row['verdict']='invalid'
        self.assertEqual(calibration_matrix(rows,['strict','lenient'],['neutral','assertive'])['losses'],((F(1),F(1)),)*2)

    def test_complementary_fixture_exact_equilibrium(self):
        r=solve_risk_game(((F(1,10),F(7,10)),(F(7,10),F(1,10))))
        self.assertEqual(r['verifier_policy'],(F(1,2),F(1,2)))
        self.assertEqual(r['attacker_policy'],(F(1,2),F(1,2)))
        self.assertEqual(r['value'],F(2,5));self.assertEqual(r['duality_gap'],0)

    def test_strong_single_verifier_is_not_forced_into_mixture(self):
        r=solve_risk_game((('1/10','1/10'),('3/10','4/10')))
        self.assertEqual(r['verifier_policy'],(F(1),F(0)))
        self.assertEqual(r['value'],F(1,10))

    def test_degenerate_and_rectangular_games(self):
        for matrix in (((0,0,0),(0,0,0)),((1,),(0,)),(('1/2','1/3','2/3'),('1/4','3/4','1/2'))):
            self.assertEqual(solve_risk_game(matrix)['duality_gap'],0)

    def test_float_payoffs_and_enumeration_overrun_fail(self):
        with self.assertRaises(ValueError):solve_risk_game(((.1,.2),))
        with self.assertRaises(ValueError):solve_risk_game(((0,1),(1,0)),max_vertices=1)

    def test_deployment_samples_protocol_not_voting_or_labels(self):
        self.assertEqual(select_verifier(('a','b'),('1/3','2/3'),.2),'a')
        self.assertEqual(select_verifier(('a','b'),('1/3','2/3'),.7),'b')
        self.assertNotIn('truth',select_verifier.__code__.co_varnames)
        with self.assertRaises(ValueError):select_verifier(('a','b'),('1/3','2/3'),1.)


if __name__=='__main__':unittest.main()
