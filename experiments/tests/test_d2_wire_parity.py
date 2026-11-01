import unittest
from experiments.paired_deliberation import runner_d2,runner_d2_live
from experiments.paired_deliberation.budget_d2 import JointBudget
from experiments.tests.test_d2_budget_coordination import LIMITS,PROFILES
from experiments.tests.test_d2_runner import FixtureActor,FixtureJudge
from experiments.tests.test_d2_contract_state import registry
from experiments.paired_deliberation.contracts_v6 import METHODS


class WireParityTests(unittest.TestCase):
    def test_wire_revision_preserves_scripted_five_condition_algorithm(self):
        r=registry();initial_actor=FixtureActor(r)
        shared=runner_d2.shared_initial(r,runner_d2.clients(JointBudget(LIMITS,PROFILES),
            {'actor':initial_actor,'verifier':FixtureJudge()})['actor'],'shared')
        for method in METHODS:
            a=runner_d2.run_method(r,method,{'actor':FixtureActor(r),'verifier':FixtureJudge()},LIMITS,PROFILES,'base_'+method,shared,max_rounds=1)
            b=runner_d2_live.run_method(r,method,{'actor':FixtureActor(r),'verifier':FixtureJudge()},LIMITS,PROFILES,'wire_'+method,shared,max_rounds=1)
            for key in ('state','final','schedule','stop','rejected','tool_events'):
                self.assertEqual(a[key],b[key],(method,key))
            self.assertEqual(a['usage']['attempts'],b['usage']['attempts'])
            self.assertGreaterEqual(b['usage']['reserved_prompt_upper'],a['usage']['reserved_prompt_upper'])


if __name__=='__main__':unittest.main()
