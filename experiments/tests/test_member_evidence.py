import copy,json,unittest
from experiments import member_evidence_data as d
from experiments.consensus_mechanism import core as c

class MemberDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.t=d.read(d.DEFAULT/'tasks.json')[0]
    def state(self):
        state=c.fresh(self.t);c.expose(self.t,state,self.t['metadata']['seed_ids']);return state
    def test_more_candidates_than_budget_and_no_exact_seed_overlap(self):
        self.assertGreater(len(self.t['metadata']['acquisition_ids']),2)
        seed_texts={self.t['source_catalog'][s]['text']for s in self.t['metadata']['seed_ids']}
        self.assertTrue(all(self.t['source_catalog'][s]['text']not in seed_texts for s in self.t['metadata']['acquisition_ids']))
    def test_member_sees_private_seed_and_shared_new_material(self):
        requests=[];state=self.state();sid=self.t['metadata']['acquisition_ids'][0]
        class API:
            def call(self,slot,req):requests.append(json.loads(req['messages'][1]['content']));raise RuntimeError('fixture')
        after,event=d.review(self.t,state,[],1,[sid],API(),'test')
        shown={v['source_id']for v in requests[0]['evidence_spans'].values()}
        self.assertEqual(shown,{self.t['metadata']['seed_ids'][1],sid})
        self.assertEqual(set(after['visible']),set(state['visible'])|{sid})
        self.assertNotIn('gold',json.dumps(requests));self.assertEqual(state['visible'],self.t['metadata']['seed_ids'])
    def test_single_control_gets_all_initial_material(self):
        requests=[]
        class API:
            def call(self,slot,req):requests.append(json.loads(req['messages'][1]['content']));raise RuntimeError('fixture')
        d.review(self.t,self.state(),[],3,[],API(),'test')
        shown={v['source_id']for v in requests[0]['evidence_spans'].values()}
        self.assertEqual(shown,set(self.t['metadata']['seed_ids']))
    def test_context_features_vary_without_agent_id_feature(self):
        reports=[{'decision':a,'claim':'climate evidence','span_ids':[]}for a in ['supported','refuted','not_enough_info']]
        ids=self.t['metadata']['acquisition_ids'][:1]
        a=d.features(self.t,self.state(),reports,0,ids);b=d.features(self.t,self.state(),reports,1,ids)
        self.assertNotEqual(a,b);self.assertFalse(any('reviewer_id'in k for k in a))
if __name__=='__main__':unittest.main()
