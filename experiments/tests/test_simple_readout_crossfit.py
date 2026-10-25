import unittest
from experiments.mechanism.audit_simple_readout_crossfit import fit,predict


class SimpleReadoutTests(unittest.TestCase):
    def training(self):
        return [{'gold':'yes','turns':[{'agent':0,'route':'strong','answer':'yes'},
                                     {'agent':1,'route':'weak','answer':'no'}]} for _ in range(5)]

    def test_reliability_is_fitted_only_on_training(self):
        model=fit(self.training())
        turns=[{'agent':0,'route':'strong','answer':'yes'},{'agent':1,'route':'weak','answer':'no'}]
        p=predict(turns,model)
        self.assertEqual(p['route_reliability_weighted'],'yes')
        self.assertEqual(p['binary_unweighted'],'neutral')
        self.assertNotIn('gold',predict.__code__.co_varnames)

    def test_unseen_routes_and_no_vote_abstain(self):
        m=fit(self.training())
        self.assertEqual(predict([{'agent':0,'route':'new','answer':'yes'}],m)['route_reliability_weighted'],'neutral')

    def test_no_domain_features(self):
        m=fit(self.training())
        self.assertEqual(set(m),{'route_weights','best_agent','training_route_counts','training_agent_counts'})


if __name__=='__main__':unittest.main()
