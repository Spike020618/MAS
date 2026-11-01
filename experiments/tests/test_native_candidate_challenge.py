import copy,json,unittest
from experiments.consensus_recommendation.common import fingerprint
from experiments.paired_deliberation.native_candidate_challenge import build_targets,consensus_ranking,allocate,apply_response,challenge_request

class OpenCandidateTests(unittest.TestCase):
    def setUp(self):
        self.pool=[str(i) for i in range(100)]
        self.initial=[self.pool[10:20],self.pool[10:20],self.pool[15:25]]
        self.base=consensus_ranking(self.initial,self.pool)
        self.features={m:{'neighbor_score':(100-int(m))/100,'association_score':(100-int(m))/200} for m in self.pool}
        self.targets=build_targets(self.initial,self.pool,self.features)
        self.target=next(t for t in self.targets if t['novel_to_initial_union'])
        self.b=self.target['b']
        self.sources={'train-score:'+self.b:{'kind':'training_aggregate','allowed_use':'prediction_basis','movie_id':self.b,'origin_case_id':'training_aggregate','text':'Train-only neighbor score 0.9; observed recommendation is predictive analogy only.'}}
    def test_candidate_omitted_by_every_initial_agent_actually_enters_top10(self):
        self.assertTrue(all(self.b not in r for r in self.initial));self.assertNotIn(self.b,self.base[:10])
        out=apply_response(self.base,self.target,{'preferred':self.b,'source_id':'train-score:'+self.b},self.sources,'dev:1',fingerprint(self.base))
        self.assertTrue(out['valid']);self.assertIn(self.b,out['ranking'][:10]);self.assertEqual(out['ranking'][:3],self.base[:3])
        changed=[i for i,(a,b) in enumerate(zip(self.base,out['ranking'])) if a!=b]
        self.assertEqual(len(changed),2)
        self.assertFalse(out['current_preference_certified']);self.assertFalse(out['entailment_verified'])
    def test_invalid_source_title_only_and_stale_state_preserve_entire_old_ranking(self):
        for source in [{'kind':'catalog','allowed_use':'catalog_attribute','movie_id':self.b,'text':'Title only'},
            {'kind':'historical_prediction_basis','allowed_use':'prediction_basis','movie_id':self.b,'origin_case_id':'dev:1','text':'Current example future outcome'},
            {'kind':'training_aggregate','allowed_use':'prediction_basis','movie_id':'not-the-challenger','text':'Other item'}]:
            out=apply_response(self.base,self.target,{'preferred':self.b,'source_id':'x'},{'x':source},'dev:1')
            self.assertFalse(out['valid']);self.assertEqual(out['ranking'],self.base)
        out=apply_response(self.base,self.target,{'preferred':self.b,'source_id':'train-score:'+self.b},self.sources,'dev:1','stale')
        self.assertFalse(out['valid']);self.assertEqual(out['ranking'],self.base)
    def test_null_abstention_and_integer_public_ids_preserve_types(self):
        ints=list(map(int,self.base));out=apply_response(ints,self.target,{'preferred':None,'source_id':None},self.sources,'dev:1')
        self.assertTrue(out['valid']);self.assertTrue(out['abstained']);self.assertEqual(out['ranking'],ints)
        out=apply_response(ints,self.target,{'preferred':self.b,'source_id':'train-score:'+self.b},self.sources,'dev:1')
        self.assertTrue(out['valid']);self.assertTrue(all(type(m) is int for m in out['ranking']))
    def test_targets_disjoint_and_feature_mapping_does_not_use_outcomes(self):
        items=[m for t in self.targets for m in (t['a'],t['b'])]
        self.assertEqual(len(items),len(set(items)))
        self.assertTrue(any(t['novel_to_initial_union'] for t in self.targets))
        mutated=copy.deepcopy(self.features)
        for m in mutated:mutated[m]['target_label']=True
        self.assertEqual(build_targets(self.initial,self.pool,mutated),self.targets)
    def test_request_does_not_offer_other_candidates_per_item_cards(self):
        sources={**self.sources,'kg:999':{'kind':'knowledge_graph','allowed_use':'catalog_attribute','movie_id':'999','text':'Other item facts'},
                 'history:x':{'kind':'historical_prediction_basis','allowed_use':'prediction_basis','movie_ids':['999'],'text':'Historical training context'}}
        row={'id':'dev:1','text':'Seeker: request','candidates':[{'movie_id':m,'title':m} for m in self.pool]}
        req=challenge_request(row,self.initial,self.target,1,sources);payload=json.loads(req['messages'][1]['content'])
        self.assertNotIn('kg:999',payload['sources']);self.assertIn('history:x',payload['sources']);self.assertFalse(req['labels_available'])

class AllocationControlTests(unittest.TestCase):
    def test_same_payoff_central_control_exposes_local_equilibrium_trap(self):
        targets=[{'owner':1,'value':1},{'owner':0,'value':.9},{'owner':2,'value':.8}];costs=[[.01]*3 for _ in range(3)]
        game=allocate(targets,costs,'game');central=allocate(targets,costs,'central')
        self.assertEqual(game['max_unilateral_gain'],0);self.assertGreater(central['selected_potential'],game['selected_potential'])
        self.assertEqual(central['selected_max_unilateral_gain'],0)
        for player,q in central['allocation']:self.assertNotEqual(player,targets[q]['owner'])
    def test_high_cost_causes_real_no_call_allocation(self):
        targets=[{'owner':0,'value':.01}];costs=[[.1],[.1],[.1]]
        for mode in ('central','game'):self.assertEqual(allocate(targets,costs,mode)['allocation'],[])

if __name__=='__main__':unittest.main()
