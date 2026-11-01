import copy
import json
import unittest
from experiments import calibrated_action_value as v
from experiments.consensus_mechanism import core as c
from experiments.bound_opportunity_probe import fixtures

class ValueTests(unittest.TestCase):
    def test_validity_and_signed_outcomes_are_separate(self):
        outcome=v.OutcomeModel().fit([{'x':1}]*3,[-1,0,1])
        p=outcome.probs([{'x':1}])[0]
        self.assertAlmostEqual(sum(p.values()),1.)
        self.assertGreater(p[-1],0);self.assertGreater(p[1],0)
        valid=v.OutcomeModel().fit([{'x':1}]*2,[0,0])
        self.assertEqual(valid.probs([{'x':1}])[0],{0:1.})
    def test_missing_class_no_fabricated_positive_floor(self):
        model=v.OutcomeModel().fit([{'x':1}]*3,[0,0,0])
        self.assertEqual(model.probs([{'x':2}])[0],{0:1.})
    def test_features_ignore_private_labels_and_identifier(self):
        row=fixtures()[1];t=row['task'];t['metadata']['given_claim']='Record VX17 is approved.'
        a=v.features(t,row['state'],['A','B']);copy_t=copy.deepcopy(t)
        copy_t['id']='another-id';copy_t['metadata']['gold']='no';copy_t['metadata']['acquisition_partition']='test'
        self.assertEqual(a,v.features(copy_t,row['state'],['A','B']))
    def test_invalid_prediction_has_zero_expected_gain(self):
        row=fixtures()[1];t=row['task'];t['metadata']['given_claim']='Record VX17 is approved.'
        bundle={'outcome':v.OutcomeModel().fit([{}],[1]),'validity':v.OutcomeModel().fit([{}],[0]),'cost_weight':.1}
        value=v.action_values(t,row['state'],[{'target_id':'target','source_ids':['A']}],bundle)[0]
        self.assertEqual(value['gains'],[0.,0.,0.])
    def test_compact_actor_maps_native_relation_to_local_patch(self):
        row=fixtures()[3];row['task']['options']=['supported','refuted','not_enough_info']
        raw={'decision':'not_enough_info','claim':'No evidence about VX17','span_ids':[]}
        requests=[]
        class API:
            def call(self,slot,req):
                requests.append(req);return {'report':raw,'usage':{},'model':'fixture'}
        after,event=v.actor(API(),'test',row['task'],row['state'],[])
        self.assertTrue(event['valid']);self.assertEqual(after['decision'],'not_enough_info')
        self.assertEqual(event['raw_report']['decision'],'not_enough_info')
        self.assertEqual(event['report']['verdict'],'unclear')
        self.assertNotIn('gold',json.dumps(requests))

if __name__=='__main__':unittest.main()

class FitIsolationTests(unittest.TestCase):
    def test_invalid_outcomes_excluded_and_no_test_vault_required(self):
        import tempfile
        from pathlib import Path
        from experiments.consensus_recommendation.common import create,fingerprint
        tasks=[];cases=[];labels={}
        for i in range(4):
            tid='fixture:'+str(i);partition='fit'if i<3 else'calibration'
            tasks.append({'id':tid,'metadata':{'acquisition_partition':partition}})
            labels[tid]={'kind':'qa','answer':'supported'};branches={}
            for j,name in enumerate(v.BRANCHES):
                valid=not(i==0 and name=='B')
                branches[name]={'decision':'supported'if name in ('A','AB')else'refuted',
                  'event':{'valid':valid},'features':{'materials':float(j),'initial:refuted':1.}}
            cases.append({'task_id':tid,'partition':partition,'initial':'refuted','branches':branches})
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            for i,case in enumerate(cases):create(root/'train/cases'/f'case_{i}.json',case)
            create(root/'train/generation_complete.json',{'task_ids':[t['id']for t in tasks],'hashes':[fingerprint(x)for x in cases]})
            create(root/'train.labels.private.json',labels)
            # Deliberately no aggregate vault or test label file exists.
            model=v.fit(root,tasks)
            summary=json.loads((root/'model/summary.json').read_text())
            self.assertEqual(summary['invalid_fit_excluded'],1)
            self.assertEqual(summary['valid_fit_actions'],11)
            self.assertFalse(summary['test_labels_used'])
            rows=json.loads((root/'model/observations.json').read_text())
            self.assertIsNone(next(r for r in rows if not r['valid'])['reward'])
            self.assertEqual(set(model['training_task_ids']),{t['id']for t in tasks})
