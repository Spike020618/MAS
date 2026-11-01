import copy
import json
import unittest
from experiments import bound_opportunity_probe as b
from experiments.consensus_mechanism import core as c

class BindingTests(unittest.TestCase):
    def test_all_constructed_bindings_are_valid(self):
        for row in b.fixtures(): self.assertTrue(b.validate_binding(row))
    def test_wrong_maintenance_state_rejected(self):
        row = b.fixtures()[2]; row['state']['decision'] = 'no'
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_already_correct_recovery_rejected(self):
        row = b.fixtures()[0]; row['state']['decision'] = 'yes'
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_unadjudicated_rejected(self):
        row = b.fixtures()[0]; row['adjudication']['status'] = 'pending'
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_unknown_bound_source_rejected(self):
        row = b.fixtures()[0]; row['sources']['A'] = ['missing']
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_natural_opportunity_requires_bound_span_witness(self):
        row = b.fixtures()[0]; row['adjudication']['status'] = 'supported'
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_test_partition_cannot_be_label_selected(self):
        row = b.fixtures()[0]; row['adjudication']['status'] = 'supported'
        row['task']['metadata']['acquisition_partition'] = 'test'
        row['adjudication']['span_ids'] = list(c.spans(row['task']))[:1]
        with self.assertRaises(ValueError): b.validate_binding(row)
    def test_basis_all_combinations(self):
        for cited, assigned, expected in [([],['A'],'none'),(['P'],['A'],'initial'),
                (['A'],['A','B'],'assigned'),(['A','P'],['A'],'joint'),
                (['A','B'],['A','B'],'assigned')]:
            self.assertEqual(b.revision_basis([{'source_id': s} for s in cited], assigned), expected)
    def test_initial_only_revision_reaches_commit(self):
        row = b.fixtures()[0]; task = row['task']; state = copy.deepcopy(row['state'])
        c.expose(task, state, ['A','B'])
        span = next(k for k,v in c.spans(task).items() if v['source_id']=='A')
        report = {'target_id':'target','verdict':'support','span_ids':[span], 'rationale':'VX17 approved explicitly',
                  'patch':{'operation':'set_answer','value':'yes'},
                  'decision_effect':{'choice':'revise','reason':'The record explicitly approves VX17','candidate_pair':None}}
        class API:
            def call(self, slot, req): return {'report':report,'usage':{},'model':'fixture'}
        after, event = b.invoke(API(),'test',task,state,[],{'target_id':'target','source_ids':['B']},'linked')
        self.assertTrue(event['valid']); self.assertTrue(event['accepted'])
        self.assertEqual(event['revision_basis'],'initial'); self.assertEqual(after['decision'],'yes')
    def test_exact_state_and_action_bindings_no_initial_generation(self):
        row=b.fixtures()[1]; calls=[]
        class API:
            def call(self, slot, req):
                payload=json.loads(req['messages'][1]['content']); calls.append(payload)
                raise RuntimeError('fixture unavailable')
        result=b.run_case(row,API(),0)
        self.assertEqual(len(calls),8)
        self.assertTrue(all(p['phase'] in ('direct','review') for p in calls))
        self.assertTrue(all(p['current_decision']=='no' for p in calls))
        for branch in b.BRANCHES:
            x,y=(result['arms'][branch+':'+m] for m in ('ordinary','linked'))
            self.assertEqual(x['exposed_state_hash'],y['exposed_state_hash'])
        self.assertEqual(result['arms']['AB:linked']['action']['source_ids'],['A','B'])
        self.assertEqual(row['state']['visible'],[])
    def test_failed_events_cannot_create_opportunity_signal(self):
        class API:
            def call(self,*args): raise RuntimeError('fixture unavailable')
        rows=b.fixtures(); cases=[b.run_case(r,API(),i) for i,r in enumerate(rows)]
        summary=b.summarize(rows,cases)
        self.assertEqual(summary['tasks'],5); self.assertFalse(summary['promotion_gate'])
        self.assertFalse(b.budget_eligibility(summary,2,3)['eligible'])
    def test_no_scarcity_blocks_budget_experiment(self):
        s={'methods':{'linked':{'valid':8,'events':8,'positive_evidence_tasks':1,'action_difference_tasks':1}}}
        self.assertFalse(b.budget_eligibility(s,3,3)['eligible'])
        self.assertTrue(b.budget_eligibility(s,2,3)['eligible'])
        self.assertFalse(b.budget_eligibility(s,0,3)['eligible'])

if __name__=='__main__': unittest.main()

class TypedReviewTests(unittest.TestCase):
    def test_claim_verdict_and_answer_option_explicitly_separated(self):
        row=b.fixtures()[3]
        req=b.review_request(row['task'],row['state'],[],{'target_id':'target','source_ids':[]})
        text=req['messages'][0]['content']
        self.assertIn('ONLY support, refute, or unclear',text)
        self.assertIn('NEVER a verdict',text)
        self.assertIn('legal even with unclear verdict and no citations',text)
    def test_uncertainty_revision_without_fabricated_evidence_commits(self):
        row=b.fixtures()[0]
        report={'target_id':'target','verdict':'unclear','rationale':'No records about VX17 are visible',
                'span_ids':[],'patch':{'operation':'set_answer','value':'not_enough_info'},
                'decision_effect':{'choice':'revise','reason':'No support for yes or no','candidate_pair':None}}
        class API:
            def call(self,*args):return {'report':report,'usage':{},'model':'fixture'}
        after,event=b.invoke(API(),'uncertain',row['task'],row['state'],[],{'target_id':'target','source_ids':[]},'linked')
        self.assertTrue(event['valid']);self.assertEqual(after['decision'],'not_enough_info')
        self.assertEqual(event['revision_basis'],'none')
