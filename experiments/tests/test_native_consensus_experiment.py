import copy,json,unittest
from experiments.paired_deliberation.native_consensus_experiment import review_targets,review_request,readout,rank_request,validate_review
from experiments.paired_deliberation.rapid_negotiation import scores,local_update
from experiments.paired_deliberation.scoped_evidence import validate
from experiments.consensus_recommendation.common import fingerprint

class NativeConsensusTests(unittest.TestCase):
    def setUp(self):
        self.pool=[str(i) for i in range(100)]
        self.initial=[self.pool[:10],self.pool[5:15],self.pool[10:20]]
        self.vectors=[scores(r,self.pool) for r in self.initial]
        self.row={'id':'a:2','text':'Seeker: I want comedy\nRecommender: What about horror?',
                  'candidates':[{'movie_id':m,'title':'Movie '+m} for m in self.pool]}
    def test_game_queries_are_disjoint_cross_boundary_and_dissenting(self):
        targets=review_targets(self.vectors,self.pool,True);self.assertTrue(targets)
        used=[];ranking=list(map(str,readout(self.vectors,self.pool)))
        for t in targets:
            self.assertIn(t['a'],ranking[:10]);self.assertIn(t['b'],ranking[10:20])
            gaps=[v[t['a']]-v[t['b']] for v in self.vectors]
            self.assertLess(min(gaps),0);used.extend([t['a'],t['b']])
        self.assertEqual(len(used),len(set(used)))
    def test_only_two_owned_scores_change_and_source_scope_is_enforced(self):
        t=review_targets(self.vectors,self.pool,True)[0];req=review_request(self.row,self.initial,t,1)
        obj={'preferred':t['b'],'source_id':'turn:0'}
        answer=validate_review(obj,req,self.row);out=local_update(self.vectors,t,answer['preferred'],fingerprint(self.vectors))
        for i in range(3):
            for m in self.pool:
                if i!=t['owner'] or m not in (t['a'],t['b']):self.assertEqual(out[i][m],self.vectors[i][m])
        obj['source_id']='turn:1'
        bound=validate_review(obj,req,self.row)
        self.assertEqual(bound['bound_sources'][0]['allowed_use'],'dialogue_context')
        self.assertFalse(bound['current_preference_certified'])
        obj['current_user_fact']='I like horror'
        with self.assertRaises(ValueError):validate_review(obj,req,self.row)
    def test_abstention_and_unknown_sources(self):
        t=review_targets(self.vectors,self.pool,True)[0];req=review_request(self.row,self.initial,t,1)
        bound=validate_review({'preferred':None,'source_id':None},req,self.row)
        self.assertTrue(bound['abstained']);self.assertEqual(bound['bound_sources'],[])
        for sid in ['turn:999',['turn:0'],'movie:not-present']:
            with self.assertRaises(ValueError):validate_review({'preferred':t['a'],'source_id':sid},req,self.row)
    def test_single_has_only_own_previous_and_no_labels(self):
        req=rank_request(self.row,'recheck',self.initial[0]);payload=json.loads(req['messages'][1]['content'])
        self.assertFalse(req['labels_available']);self.assertNotIn('initial_rankings',payload)
        self.assertNotIn('targets',payload);self.assertEqual(payload['your_previous_ranking'],self.initial[0])

class NativeSourceAliasTests(unittest.TestCase):
    def test_unique_catalog_alias_preserves_exact_source_and_preference(self):
        row={'id':'a','text':'Seeker: comedy','candidates':[{'movie_id':'100','title':'One'},{'movie_id':'200','title':'Two'}]}
        req=review_request(row,[['100','200']]*3,{'a':'100','b':'200'},1)
        bound=validate_review({'preferred':'200','source_id':'100'},req,row)
        self.assertEqual(bound['preferred'],'200');self.assertEqual(bound['bound_sources'][0]['source_id'],'movie:100')
        self.assertEqual(bound['bound_sources'][0]['text'],'Title: One');self.assertFalse(bound['current_preference_certified'])
        self.assertEqual(len(bound['representation_operations']),1)
    def test_ambiguous_numeric_source_never_resolves_to_catalog(self):
        row={'id':'a','text':'Seeker: comedy','candidates':[{'movie_id':'0','title':'One'},{'movie_id':'200','title':'Two'}]}
        req=review_request(row,[['0','200']]*3,{'a':'0','b':'200'},1)
        with self.assertRaises(ValueError):validate_review({'preferred':'200','source_id':'0'},req,row)

if __name__=='__main__':unittest.main()
