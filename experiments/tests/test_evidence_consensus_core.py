import copy, unittest
from experiments.evidence_consensus.schema import validate_aeic
from experiments.evidence_consensus.state import new_state, register_report, apply_patch, initialize_decision
from experiments.evidence_consensus.actions import fetch_source, verify_source, BudgetLedger, BudgetExceeded


def task(kind='qa'):
    return {'id':'t1','kind':kind,'prompt':'visible question','options':['yes','no'],
            'candidates':['a','b','c','d'], 'initial':'abstain' if kind=='qa' else ['a','b','c','d'],
            'source_catalog':{'s1':{'source_id':'s1','text':'visible passage','scope':{'task_id':'t1','subject':'current_user'},'allowed_uses':['prediction_basis','current_user_fact']},
                              's2':{'source_id':'s2','text':'OTHER user likes a','scope':{'task_id':'t1','subject':'other_user'},'allowed_uses':['prediction_basis'],'current_user_fact_authority':False}}}


def report():
    return {'A':[{'id':'claim1','text':'a hypothesis','scope':{'task_id':'t1'}}], 'E':['s1'],
            'I':[{'claim_id':'claim1','source_id':'s1','relation':'support'}],
            'C':{'proposed_patch':{'expected_version':0,'operation':'set_answer','value':'yes','evidence_ids':['s1'],'preserve_ids':[]}}}


class UnifiedStateTests(unittest.TestCase):
    def test_real_source_expansion_then_reuse_has_no_new_information(self):
        s=new_state(task());self.assertEqual(s['visible_sources'],{})
        first=fetch_source(s,'s1');second=fetch_source(s,'s1')
        self.assertEqual(first['new_sources'],1);self.assertEqual(second['new_sources'],0)
        self.assertEqual(len(s['visible_sources']),1);self.assertEqual(s['version'],0)
        self.assertFalse(verify_source(s,'s1',model_opinion={'supported':True})['semantic_truth_certified'])

    def test_source_mutation_or_foreign_task_rejected(self):
        s=new_state(task());fetch_source(s,'s1');s['visible_sources']['s1']['text']='tampered'
        with self.assertRaises(ValueError):fetch_source(s,'s2')
        bad=task();bad['source_catalog']['s1']['scope']['task_id']='other'
        with self.assertRaises(ValueError):new_state(bad)

    def test_aeic_requires_fetched_bound_evidence_and_scoped_claim(self):
        s=new_state(task())
        with self.assertRaises(ValueError):validate_aeic(report(),s)
        fetch_source(s,'s1');validate_aeic(report(),s)
        wrong=report();wrong['I'][0]['source_id']='s2'
        with self.assertRaises(ValueError):validate_aeic(wrong,s)
        wrong=report();wrong['A'][0]['scope']['task_id']='wrong'
        with self.assertRaises(ValueError):validate_aeic(wrong,s)

    def test_patch_is_versioned_preserves_unrelated_claims_and_rejects_atomically(self):
        s=new_state(task());fetch_source(s,'s1');s=register_report(s,report());before=copy.deepcopy(s)
        patch=report()['C']['proposed_patch'];patch['preserve_ids']=['claim1']
        out=apply_patch(s,patch);self.assertTrue(out['accepted']);self.assertEqual(out['state']['decision'],'yes')
        self.assertEqual(out['state']['claims'],s['claims']);self.assertEqual(s,before)
        stale=apply_patch(out['state'],patch);self.assertFalse(stale['accepted']);self.assertEqual(stale['state'],out['state'])
        bad=copy.deepcopy(patch);bad['value']='outside option'
        self.assertEqual(apply_patch(s,bad)['state'],s)
        forbidden={'expected_version':0,'operation':'update_claim','target_id':'claim1','claim_version':0,'value':'changed','evidence_ids':['s1'],'preserve_ids':['claim1']}
        self.assertFalse(apply_patch(s,forbidden)['accepted'])

    def test_other_user_evidence_cannot_become_current_user_fact(self):
        s=new_state(task());fetch_source(s,'s2')
        p={'expected_version':0,'operation':'set_answer','value':'yes','evidence_ids':['s2'],'evidence_purpose':'current_user_fact'}
        self.assertFalse(apply_patch(s,p)['accepted']);p['evidence_purpose']='prediction_basis'
        self.assertTrue(apply_patch(s,p)['accepted'])

    def test_recommendation_swap_only_changes_named_positions(self):
        s=new_state(task('rec'));fetch_source(s,'s1')
        p={'expected_version':0,'operation':'swap_candidate','a':'b','b':'d','evidence_ids':['s1']}
        out=apply_patch(s,p);self.assertTrue(out['accepted']);self.assertEqual(out['state']['decision'],['a','d','c','b'])
        p['b']='outside';self.assertFalse(apply_patch(s,p)['accepted'])
        p={'expected_version':0,'operation':'replace_ranking','value':['c','a'],'evidence_ids':['s1']}
        self.assertFalse(apply_patch(s,p)['accepted'])
        self.assertEqual(apply_patch(s,p)['state'],s)
        ablation=task('rec');ablation['metadata']={'allow_full_rewrite_ablation':True}
        abl_state=new_state(ablation);fetch_source(abl_state,'s1')
        self.assertEqual(apply_patch(abl_state,p)['state']['decision'],['c','a','b','d'])

    def test_initial_generation_is_not_repeatable_patch_bypass(self):
        s=new_state(task());s=initialize_decision(s,'no');self.assertEqual(s['decision'],'no')
        with self.assertRaises(ValueError):initialize_decision(s,'yes')
        p={'expected_version':0,'operation':'abstain'}
        self.assertEqual(apply_patch(s,p)['state'],s)


class TaskBudgetTests(unittest.TestCase):
    def test_reservation_blocks_parallel_overallocation_and_failed_usage_is_charged(self):
        b=BudgetLedger(2,100,1);b.reserve('one',tokens=70,cny=.7)
        with self.assertRaises(BudgetExceeded):b.reserve('two',tokens=40,cny=.4)
        b.settle('one',tokens=20,cny=.2) # actual returned error usage still charged
        b.reserve('two',tokens=80,cny=.8)
        self.assertEqual(b.snapshot()['used_or_reserved']['tokens'],100)
        b.settle('two',tokens=30,cny=.3)
        self.assertEqual(b.snapshot()['used_or_reserved']['calls'],2)
        with self.assertRaises(BudgetExceeded):b.reserve('three',tokens=0,cny=0)

    def test_unknown_usage_is_not_zero_and_receipts_are_idempotent(self):
        b=BudgetLedger(2,100,1);b.reserve('one',tokens=80,cny=.8)
        b.settle('one',tokens=0,cny=0,known=False)
        self.assertEqual(b.snapshot()['unknown_requests'],['one'])
        self.assertEqual(b.snapshot()['used_or_reserved']['tokens'],80)
        b.settle('one',tokens=0,cny=0,known=False)
        with self.assertRaises(ValueError):b.settle('one',tokens=1,cny=0,known=False)
        with self.assertRaises(ValueError):b.reserve('one',tokens=80,cny=.8)

    def test_actual_overage_is_recorded_before_stop(self):
        b=BudgetLedger(1,100,1);b.reserve('one',tokens=100,cny=1)
        with self.assertRaises(BudgetExceeded):b.settle('one',tokens=101,cny=1)
        self.assertEqual(b.snapshot()['used_or_reserved']['tokens'],101)
