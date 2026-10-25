import unittest
from experiments.mechanism.audit_information_readout_gap import fit,predict,unique_max,SLOTS


class GapTests(unittest.TestCase):
    def test_empty_and_tie_abstain(self):
        self.assertEqual(unique_max({}),'abstain')
        self.assertEqual(unique_max({'a':1,'b':1}),'abstain')

    def test_single_selection_uses_training_correctness_then_cost(self):
        slots={k:{'answer':'yes','tokens':100 if k=='fast1' else 200} for k in SLOTS}
        policy=fit({'train':slots},{'train':'yes'})
        self.assertEqual(policy['chosen_single'],'fast1')
        test={k:{'answer':'no'} for k in SLOTS}
        self.assertEqual(predict(test,policy)['train_selected_single'],'no')

    def test_five_report_majority_requires_three_votes(self):
        policy={'chosen_single':'fast0','weights':{k:1. for k in SLOTS}}
        slots={k:{'answer':v} for k,v in zip(SLOTS,['a','a','b','c','d'])}
        self.assertEqual(predict(slots,policy)['five_report_majority'],'abstain')
        slots['fast2']['answer']='a'
        self.assertEqual(predict(slots,policy)['five_report_majority'],'a')


if __name__=='__main__':unittest.main()
