import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import correct_sequential_allocation_costs as c


class CostCorrectionTests(unittest.TestCase):
    def test_source_only_gate_does_not_pay_for_unused_answers(self):
        slots=[{'fixed_cost':{'tokens':100,'calls':1},'adaptive_cost':{'tokens':100,'calls':1}},
            {'fixed_cost':{'tokens':120,'calls':2},'adaptive_cost':{'tokens':220,'calls':3}},
            {'fixed_cost':{'tokens':140,'calls':3},'adaptive_cost':{'tokens':340,'calls':5}}]
        per_case={cid:{mode:copy.deepcopy(slots) for mode in c.r.MODES} for cid in ('x','y')}
        cells={}
        for mode in c.r.MODES:
            cells[mode]={'splits':{}}
            for split,n in (('development',1),('confirmation',1),('all',2)):
                cells[mode]['splits'][split]={'fixed_stages':[{'em_correct':q*n,'tokens':t*n,'calls':calls*n} for q,t,calls in ((0,100,1),(1,120,2),(1,140,3))],
                    'simple_policies':{'no_new_documents':{'stages':[1]*n,'em_correct':n,'tokens':220*n,'calls':3*n},
                        'abstain_only':{'stages':[0]*n,'em_correct':0,'tokens':100*n,'calls':n}},
                    'oracle':{'adaptive_cost':{'best_with_fixed_s2_token_and_call_budget':{'correct':n}}}}
        original={'per_case':per_case,'cells':cells,'gate':{}}
        with tempfile.TemporaryDirectory() as tmp,patch.object(c.r,'OUT',Path(tmp)),patch.object(c,'COMMITMENT',Path(tmp)/'commit.json'),patch.object(c,'verify'),patch.object(c.r,'verify'),patch('builtins.print'):
            c.freeze(c.r.OUT/'result.json',original);before=(c.r.OUT/'result.json').read_bytes()
            c.freeze(c.COMMITMENT,{});c.freeze(c.r.OUT/'splits.json',{'x':'development','y':'confirmation'})
            c.analyze();result=c.r.p.base.read(c.r.OUT/'corrected_analysis.json')
            self.assertEqual(before,(c.r.OUT/'result.json').read_bytes())
            for mode in c.r.MODES:
                cell=result['cells'][mode]['splits']['all']
                self.assertEqual(cell['simple_policies']['no_new_documents']['tokens'],240)
                self.assertEqual(cell['simple_policies']['no_new_documents']['calls'],4)
                self.assertEqual(cell['simple_policies']['abstain_only']['tokens'],200)


if __name__=='__main__':unittest.main()
