import json
import unittest
from experiments import bound_game_bridge as g, bound_opportunity_probe as b

class ConstantModel:
    def __init__(self,value):self.value=value
    def predict(self,features):return [self.value]*len(features)

class BridgeTests(unittest.TestCase):
    def bundle(self):
        return {'model':ConstantModel(.2),'bias':0.,'interaction_model':ConstantModel(.1),'interaction_bias':0.}
    def test_forecasts_deterministic_and_do_not_read_evaluation(self):
        row=b.fixtures()[1]
        expected=g.choose(row['task'],row['state'],row['reports'],self.bundle(),'game',2,2)
        row['evaluation']['answer']='deliberately_wrong_label'
        self.assertEqual(expected,g.choose(row['task'],row['state'],row['reports'],self.bundle(),'game',2,2))
    def test_all_modes_respect_same_resource_domain(self):
        row=b.fixtures()[1]
        for mode in g.MODES:
            chosen,a=g.choose(row['task'],row['state'],[],self.bundle(),mode,1,1)
            self.assertLessEqual(len(chosen),1)
            self.assertLessEqual(len({s for r in chosen for s in r['source_ids']}),1)
            if mode=='game': self.assertLessEqual(a['max_feasible_deviation'],1e-10)
    def test_game_potential_never_exceeds_central_exact_optimum(self):
        row=b.fixtures()[1]
        _,game=g.choose(row['task'],row['state'],[],self.bundle(),'game',2,2)
        _,central=g.choose(row['task'],row['state'],[],self.bundle(),'central',2,2)
        self.assertLessEqual(game['potential'],central['potential']+1e-10)
    def test_real_executor_uses_selected_sources_and_commits(self):
        row=b.fixtures()[1]; requests=[]
        class API:
            def call(self,slot,req):
                payload=json.loads(req['messages'][1]['content']); requests.append(payload)
                return {'report':{'target_id':payload['assigned_action']['target_id'],
                    'verdict':'support','span_ids':list(payload['evidence_spans']),
                    'rationale':'fixture record chain', 'patch':{'operation':'set_answer','value':'yes'},
                    'decision_effect':{'choice':'revise','reason':'Both records establish approval','candidate_pair':None}},
                    'usage':{},'model':'fixture'}
        result=g.execute(row,API(),self.bundle(),'game',0)
        self.assertEqual(result['decision'],'yes');self.assertTrue(result['events'][0]['accepted'])
        self.assertEqual(set(requests[0]['assigned_action']['source_ids']),{'A','B'})
        self.assertEqual(result['calls'],1);self.assertEqual(result['source_reads'],2)
        self.assertNotIn('evaluation',json.dumps(requests)); self.assertEqual(row['state']['decision'],'no')
    def test_failure_remains_failure_and_spends_budget(self):
        row=b.fixtures()[1]
        class API:
            def call(self,*args):raise RuntimeError('fixture unavailable')
        result=g.execute(row,API(),self.bundle(),'game',0)
        self.assertEqual(result['decision'],'no');self.assertFalse(result['events'][0]['valid'])
        self.assertLessEqual(result['calls'],2);self.assertLessEqual(result['source_reads'],2)

if __name__=='__main__': unittest.main()
