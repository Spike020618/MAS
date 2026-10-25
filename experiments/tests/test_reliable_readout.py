import unittest
from experiments.core.mechanism.reliable_readout import calibrate,decide


class ReliableReadoutTests(unittest.TestCase):
    def training(self):
        return [{'id': str(i), 'gold':'yes','choices':['yes','no'],
                 'reports':[{'backend':'good','answer':'yes'}, {'backend':'weak','answer':'yes' if i<3 else 'no'}]} for i in range(6)]

    def test_weighted_minority_can_win_without_gold(self):
        model=calibrate(self.training(),['good','weak'])
        reports=[{'backend':'good','answer':'yes'},{'backend':'weak','answer':'no'},{'backend':'weak','answer':'no'}]
        self.assertEqual(decide(reports,['yes','no'])['answer'],'no')
        self.assertEqual(decide(reports,['yes','no'],model)['answer'],'yes')
        self.assertNotIn('gold',decide.__code__.co_varnames)

    def test_tie_and_missing_reports_abstain(self):
        self.assertEqual(decide([],['yes','no'])['answer'],'abstain')
        self.assertEqual(decide([{'backend':'x','answer':'yes'},{'backend':'x','answer':'no'}],['yes','no'])['answer'],'abstain')

    def test_duplicate_training_question_rejected(self):
        rows=self.training()
        with self.assertRaises(ValueError):calibrate(rows+[rows[0]],['good','weak'])

    def test_multichoice_and_unknown_backend(self):
        model=calibrate(self.training(),['good','weak'])
        self.assertEqual(decide([{'backend':'good','answer':'b'}],['a','b','c'],model)['answer'],'b')
        with self.assertRaises(ValueError):decide([{'backend':'new','answer':'a'}],['a','b'],model)


if __name__=='__main__':unittest.main()
