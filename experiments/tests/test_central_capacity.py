import copy
import json
import unittest

from experiments.paired_deliberation.central_capacity import shared_initial, request
from experiments.paired_deliberation.commit_engine import run_method
from experiments.paired_deliberation.contracts_v6 import OWNERS, initial_request
from experiments.paired_deliberation.budget_v6 import AttemptClient
from experiments.tests.test_design_v6 import Provider, LIMITS, evidence, claim, report


class CentralCapacityTests(unittest.TestCase):
    def test_one_call_generates_three_sections_with_same_nine_claim_capacity(self):
        class P(Provider):
            def complete(self,slot,req):
                if req['kind']=='initial_v6':
                    self.calls.append((slot,copy.deepcopy(req)))
                    obj={'reports':{owner:report([claim('c'+str(i)) for i in range(1,4)]) for owner in OWNERS}}
                    return {'report':obj,'usage':{'prompt_tokens':10,'completion_tokens':5},'model':'offline-fixture','elapsed_ms':1}
                return super().complete(slot,req)
        e,p=evidence(),P();limits={**LIMITS,'per_call_completion_tokens':4096,'total_completion_tokens':20*4096,'total_prompt_tokens':20*64000}
        shared=shared_initial(e,AttemptClient(p,limits),'central')
        self.assertEqual(len(shared['calls']),1)
        self.assertEqual(sum(len(r['claims']) for r in shared['initial'].values()),9)
        self.assertEqual(request(e)['max_tokens'],sum(initial_request(e,owner)['max_tokens'] for owner in OWNERS))
        result=run_method(e,'central',AttemptClient(p,limits),'central_run',shared,max_rounds=1)
        self.assertEqual(result['usage']['attempts'],4) # central initial, select, patch, final; catalog check free HTTP.
        self.assertEqual(result['events'][0]['action'],'observe_retract')

    def test_each_section_still_enforces_three_claim_limit(self):
        class P:
            def complete(self,slot,req):
                return {'report':{'reports':{o:report([claim('c'+str(i)) for i in range(4)]) for o in OWNERS}}}
        with self.assertRaises(ValueError):shared_initial(evidence(),P(),'invalid')


if __name__=='__main__':unittest.main()
