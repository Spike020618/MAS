import unittest
from experiments.core.mechanism.token_signaling_adapter import game_from_probe,SLOTS


def slots():
    return {k:{'usable':True,'conditional_probabilities':{'A':p,'B':1-p}} for k,p in
            zip(SLOTS,(.8,.3,.9,.2))}


class TokenSignalingAdapterTests(unittest.TestCase):
    def test_explicit_normalizations(self):
        raw=game_from_probe(slots());cross=game_from_probe(slots(),normalization='contrastive')
        self.assertEqual(raw.generator_reference[0],(.8,1-.8))
        self.assertNotEqual(raw.generator_reference,cross.generator_reference)
        self.assertTrue(cross.solve()['converged_to_tolerance'])

    def test_missing_and_invalid_inputs_fail(self):
        for field in ('missing','unusable','zero','nan','extra'):
            data=slots()
            if field=='missing':del data[SLOTS[0]]
            if field=='unusable':data[SLOTS[0]]['usable']=False
            if field=='zero':data[SLOTS[0]]['conditional_probabilities']={'A':0.,'B':1.}
            if field=='nan':data[SLOTS[0]]['conditional_probabilities']['A']=float('nan')
            if field=='extra':data['gold']='A'
            with self.assertRaises(ValueError):game_from_probe(data)


if __name__=='__main__':unittest.main()
