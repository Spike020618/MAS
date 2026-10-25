import unittest
from experiments.core.mechanism.evidence_settlement import settle_evidence
from experiments.mechanism.solve_multifollower import solve_multifollower


class EvidenceGameTests(unittest.TestCase):
    def test_duplicate_source_splits_budget_and_unauthorized_quote_earns_zero(self):
        citation={'source_hash':'s','exact_quote_valid':True,'support_status':'supported'}
        reports=[{'agent_id':a,'cost':1,'authorized_source_hashes':['s'],
                  'citations':[citation,citation]} for a in ['a','b']]
        reports.append({'agent_id':'c','cost':0,'authorized_source_hashes':[], 'citations':[citation]})
        result=settle_evidence(reports,3)
        self.assertEqual(result['paid'],3)
        self.assertEqual(result['agents']['a']['credit'],1.5)
        self.assertEqual(result['agents']['c']['credit'],0)

    def test_quote_provenance_alone_is_not_support(self):
        result=settle_evidence([{'agent_id':'a','cost':1,'authorized_source_hashes':['s'],
            'citations':[{'source_hash':'s','exact_quote_valid':True,'support_status':'insufficient'}]}],3)
        self.assertEqual(result['paid'],0)

    def test_nash_excludes_joint_profile_with_profitable_deviation(self):
        game={'agents':['a','b'],'actions':{'a':['L','R'],'b':['L','R']},'contracts':[
            {'id':'contract','payoffs':[
                {'profile':['L','L'],'followers':[0,0],'leader':0},
                {'profile':['R','L'],'followers':[.5,0],'leader':1},
                {'profile':['L','R'],'followers':[0,.5],'leader':1},
                {'profile':['R','R'],'followers':[-.25,-.25],'leader':1}]}]}
        result=solve_multifollower(game)
        self.assertEqual({tuple(r['profile']) for r in result['contracts'][0]['pure_nash']},
                         {('L','R'),('R','L')})

    def test_missing_counterfactual_profiles_rejected(self):
        with self.assertRaises(ValueError):
            solve_multifollower({'agents':['a'],'actions':{'a':['L','R']},'contracts':[
                {'id':'c','payoffs':[{'profile':['L'],'followers':[0],'leader':0}]}]})

    def test_no_pure_equilibrium_is_not_fabricated(self):
        table=[]
        for a in ['L','R']:
            for b in ['L','R']:
                payoff=1 if a==b else -1
                table.append({'profile':[a,b],'followers':[payoff,-payoff],'leader':0})
        result=solve_multifollower({'agents':['a','b'],
            'actions':{'a':['L','R'],'b':['L','R']},'contracts':[{'id':'matching_pennies','payoffs':table}]})
        self.assertEqual(result['contracts_without_pure_equilibrium'],['matching_pennies'])
        self.assertIsNone(result['solutions']['optimistic'])


if __name__=='__main__':unittest.main()
