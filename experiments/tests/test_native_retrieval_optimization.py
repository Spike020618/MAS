import copy,unittest
from experiments.paired_deliberation.native_retrieval_optimization import enhanced_contexts,NativeRetriever
from experiments.paired_deliberation.native_redial_benchmark import contexts

class NativeRetrievalTests(unittest.TestCase):
    def test_future_movie_and_entity_annotations_never_enter_prefix(self):
        def u(role,text,movies=[],entities=[]):return {'role':role,'text':[text],'movies':movies,'entity':entities}
        raw=[{'conv_id':'a','dialog':[u('Seeker','comedy',['old'],['genre']),u('Seeker','please'),u('Recommender','SECRET',['target'],['future'])]}]
        e={'old':1,'genre':10,'target':2,'future':11}
        row=enhanced_contexts(raw,e)[0]
        self.assertEqual(row['context_movies'],[1]);self.assertEqual(row['context_entities'],[10,1]);self.assertNotIn('SECRET',row['text'])
        self.assertEqual(row['targets'],[2]);self.assertEqual(row['text'],contexts(raw,e)[0]['text'])
    def test_serving_evidence_uses_only_train_targets_not_current_gold(self):
        train=[{'id':'train1:1','text':'Seeker: want comedy','targets':[2],'context_entities':[1],'context_movies':[1]},
               {'id':'train2:1','text':'Seeker: want action comedy','targets':[3],'context_entities':[1],'context_movies':[1]}]
        model=NativeRetriever([1,2,3],{'<http://dbpedia.org/resource/A>':1,'<http://dbpedia.org/resource/B>':2,'<http://dbpedia.org/resource/C>':3}).fit(train)
        row={'id':'new:1','text':'Seeker: want comedy','context_entities':[1,999],'context_movies':[1],'targets':[2]}
        before=model.evidence(row,['2','3']);changed=copy.deepcopy(row);changed['targets']=[3]
        self.assertEqual(before,model.evidence(changed,['2','3']));self.assertFalse(before['current_user_fact_authority'])
        ranking=model.rank(row,{'neighbors50':1,'entity_a1.0_b0.25':.5,'recent_mentions':1})
        self.assertEqual(len(ranking),3);self.assertEqual({r['movie_id'] for r in ranking},{'1','2','3'})
        row['id']='train1:99';evidence=model.evidence(row,['2','3'])
        self.assertTrue(all(x['origin_case_id'].split(':')[0]!='train1' for x in evidence['historical_examples']))
if __name__=='__main__':unittest.main()
