import importlib.util,unittest
import numpy as np
HAS=bool(importlib.util.find_spec('torch'))
if HAS:
    import experiments.paired_deliberation.inspired_strong_baselines as inspired

@unittest.skipUnless(HAS,'isolated KBRD environment required')
class InspiredStrongTests(unittest.TestCase):
    def test_inspired_keeps_rare_relation_and_loops_every_endpoint(self):
        n,r,edges,mapping=inspired.inspired_graph({'a':1,'b':2,'c':3},['a\trare\tb','a\trare\tb','b\tsecond\tc'])
        self.assertEqual((n,r,len(edges)),(4,3,7));self.assertEqual(mapping,{'rare':0,'SELF_LOOP':1,'second':2})
        self.assertEqual({(h,t) for h,t,rel in edges if rel==mapping['SELF_LOOP']},{(1,1),(2,2),(3,3)})
    def test_prefix_entities_and_stable_native_ids_do_not_use_current_response(self):
        raw=[{'dialog':[{'role':'Seeker','text':['comedy'],'movies':['old'],'entity':['genre']},
            {'role':'Recommender','text':['SECRET'],'movies':['new'],'entity':['future']}]}]
        row=inspired.rows(raw,{'old':1,'genre':2,'new':3,'future':4},'train')[0]
        self.assertEqual(row['id'],'inspired:train:0000:1');self.assertEqual(row['context_entities'],[2,1]);self.assertEqual(row['targets'],[3]);self.assertNotIn('SECRET',row['text'])
        self.assertEqual(inspired.fold(row),inspired.fold({'id':'inspired:train:0000:99'}))
    def test_archive_test_and_premature_validation_are_blocked(self):
        old=inspired.ALLOW_VALID;inspired.ALLOW_VALID=False
        try:
            for path in ['/data/crslab_inspired_native/inspired_nltk.zip','/data/test_data.json','/data/crslab_inspired_native/valid_data.json']:
                with self.assertRaises(RuntimeError):inspired.access_hook('open',(path,'r',0))
        finally:inspired.ALLOW_VALID=old
    def test_actual_checkpoint_roundtrip_retains_trained_weights(self):
        import tempfile,torch
        from pathlib import Path
        from experiments.paired_deliberation.native_kbrd_reproduction import Graph,RecommendationOnly
        graph=Graph(3,1,[(i,i,0) for i in range(3)])
        train=[{'context_entities':[0,1],'targets':[2]}]
        with tempfile.TemporaryDirectory() as tmp:
            model,history=inspired.fit_epochs(graph,train,1,Path(tmp),checkpoint_epochs=(1,))
            checkpoint=torch.load(Path(tmp)/'epoch1.pt',map_location='cpu',weights_only=False)
            self.assertIsInstance(checkpoint['model'],dict);self.assertIn('model_name',checkpoint)
            replay=RecommendationOnly(graph);replay.load_state_dict(checkpoint['model'])
            for name,p in model.state_dict().items():torch.testing.assert_close(p,replay.state_dict()[name],rtol=0,atol=0)

    def test_native_rank_mapping_and_ties(self):
        a=np.array([[1.,1.,0.],[0.,2.,2.]],dtype=np.float32);r=inspired.native_rank_matrix(a,np.array([1,0,2]))
        self.assertEqual(r.dtype,np.uint16);self.assertEqual(r.tolist(),[[2,1,3],[3,1,2]])

class InspiredPureScoreReaderTests(unittest.TestCase):
    def test_pool_id_rank_mapping_works_without_torch_or_labels(self):
        import json,tempfile
        from pathlib import Path
        from experiments.paired_deliberation.inspired_predictor_scores import InspiredPredictorScores
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'valid.ids.json').write_text(json.dumps(['inspired:valid:0000:2']));(p/'native_item_ids.json').write_text(json.dumps(list(range(1,1816))))
            (p/'native_catalog.json').write_text(json.dumps([{'movie_id':str(i),'native_entity_name':'Name '+str(i)} for i in range(1,1816)]))
            s=p/'sparse/fixed_redial_transfer';k=p/'kbrd/train_inner_tuned';s.mkdir(parents=True);k.mkdir(parents=True)
            values=np.arange(1815,dtype=np.float32)[None,:];rank=(1815-np.arange(1815,dtype=np.uint16))[None,:]
            np.save(s/'valid.native_scores.npy',values);np.save(s/'valid.native_ranks.npy',rank)
            np.save(k/'valid.native_logits.npy',-values);np.save(k/'valid.native_ranks.npy',np.arange(1,1816,dtype=np.uint16)[None,:]);np.save(k/'valid.full_entity_logsumexp.npy',np.array([7.],dtype=np.float32))
            (k/'summary.json').write_text(json.dumps({'epoch':20,'checkpoint_sha256':'a'*64}))
            for name in ('neighbors50','entity_a1.0_b0.25','recent_mentions'):np.save(p/'sparse'/('valid.'+name+'.npy'),values)
            source=InspiredPredictorScores(p).score_pool('inspired:valid:0000:2',['1815','1'])
            self.assertEqual(source['conversation_id'],'inspired:valid:0000');self.assertFalse(source['current_user_fact_authority'])
            self.assertEqual([x[2] for x in source['rows']],[1,1815]);self.assertEqual([x[5] for x in source['rows']],[1815,1])

if __name__=='__main__':unittest.main()
