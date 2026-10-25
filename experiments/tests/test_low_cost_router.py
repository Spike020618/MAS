import unittest
from experiments.mechanism.replay_low_cost_router import upgrade,vote


class RouterTest(unittest.TestCase):
    def test_unanimous(self):
        self.assertFalse(upgrade(['yes']*3))
        self.assertTrue(upgrade(['yes','yes','no']))
        self.assertTrue(upgrade(['yes','yes',None]))
        self.assertTrue(upgrade(['abstain']*3))

    def test_no_majority(self):
        self.assertFalse(upgrade(['yes','yes',None],'no_majority'))
        self.assertTrue(upgrade(['yes','no',None],'no_majority'))
        self.assertTrue(upgrade(['abstain']*3,'no_majority'))

    def test_unknown_not_assumed_correct(self):
        self.assertEqual(vote(['yes','no',None],['yes','no']),'abstain')
        self.assertEqual(vote(['yes','yes',None],['yes','no']),'yes')

    def test_no_domain_or_gold_input(self):
        with self.assertRaises(TypeError):upgrade(['yes']*3,label='yes')
        with self.assertRaises(ValueError):upgrade(['yes']*3,'domain_specific')
