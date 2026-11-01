"""Run with the isolated KBRD environment; no training/API in the unit suite."""
import ast,importlib.util,json,unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
HAS=bool(importlib.util.find_spec('torch') and importlib.util.find_spec('torch_geometric'))
if HAS:
    import torch
    from experiments.paired_deliberation.native_kbrd_reproduction import author_graph,native_samples,toy_parity

@unittest.skipUnless(HAS,'isolated .venv-kbrd-reproduction with official PyG required')
class NativeKBRDParityTests(unittest.TestCase):
    def test_official_rgcn_attention_and_full_microbatch_parameter_gradients(self):
        before=torch.get_rng_state().clone();r=toy_parity()
        self.assertTrue(r['pass']);self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertLess(max(r['rgcn_gradient_max_abs'].values()),1e-11)
        self.assertLess(max(r['adam_update_max_abs'].values()),1e-10)
    def test_author_graph_threshold_selfloop_and_duplicate_semantics(self):
        e={'a':0,'b':1};kg={'0':[[7,1]]*501}
        n,r,edges,mapping=author_graph(e,kg)
        self.assertEqual((n,r),(2,1));self.assertEqual(set(edges),{(0,1,0),(1,0,0)});self.assertEqual(mapping,{7:0})
        self.assertEqual(author_graph(e,{'0':[[7,1]]*500})[1],0)
    def test_actual_author_attention_code_and_gradients(self):
        from experiments.paired_deliberation.native_kbrd_reproduction import ROOT,Graph,RecommendationOnly
        source=ROOT/'experiments/results/native_kbrd_reproduction_v1/author_sources/crslab/model/utils/modules/attention.py'
        if not source.exists():self.skipTest('pinned author source not staged')
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SelfAttentionBatch')
        scope={'torch':torch,'nn':torch.nn,'F':torch.nn.functional};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        torch.manual_seed(17);model=RecommendationOnly(Graph(6,1,[(i,i,0) for i in range(6)]),dim=4,bases=2).double();author=scope['SelfAttentionBatch'](4,4).double()
        with torch.no_grad():author.a.copy_(model.attn_a);author.b.copy_(model.attn_b)
        one=torch.randn(6,4,dtype=torch.double,requires_grad=True);two=one.detach().clone().requires_grad_(True);contexts=[[],[0,2],[3,4,5]]
        a=model.encode_user(contexts,one);b=torch.stack([author(two[c]) for c in contexts]);self.assertLess(float((a-b).abs().max().detach()),1e-12)
        weights=torch.randn_like(a);(a*weights).sum().backward();(b*weights).sum().backward()
        for x,y in [(one.grad,two.grad),(model.attn_a.grad,author.a.grad),(model.attn_b.grad,author.b.grad)]:self.assertLess(float((x-y).abs().max()),1e-11)

    def test_actual_author_full_kg_processing_matches_port(self):
        from experiments.paired_deliberation.native_kbrd_reproduction import ROOT,DATA
        source=ROOT/'docs/research/benchmark_protocol_sources_2026-10-27/CRSLab/crslab/data/dataset/redial/redial.py'
        if not source.exists() or not (DATA/'dbpedia_subkg.json').exists():self.skipTest('pinned public dataset source not staged')
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_entity_kg_process')
        scope={'defaultdict':defaultdict};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        entities=json.loads((DATA/'entity2id.json').read_text());kg=json.loads((DATA/'dbpedia_subkg.json').read_text())
        self_obj=SimpleNamespace(n_entity=max(entities.values())+1,entity_kg=kg,id2entity={v:k for k,v in entities.items()})
        author=scope['_entity_kg_process'](self_obj);n,r,edges,mapping=author_graph(entities,kg)
        self.assertEqual(r,author['n_relation']);self.assertEqual(set(edges),set(map(tuple,author['edge'])))

    def test_future_entities_and_duplicate_targets_follow_native_contract(self):
        def u(role,movies,entities):return {'role':role,'movies':movies,'entity':entities}
        raw=[{'conv_id':'a','dialog':[u('Seeker',['past'],['genre']),u('Recommender',['next'],['future']),u('Recommender',['next'],[])]}]
        rows=native_samples(raw,{'past':1,'genre':2,'next':3,'future':4})
        self.assertEqual(len(rows),2);self.assertEqual([r['item'] for r in rows],[3,3])
        self.assertTrue(all(r['context_entities']==[2,1] for r in rows))

class PrecomputedKBRDScoreTests(unittest.TestCase):
    def test_native_column_mapping_and_pool_rank_are_exact_without_labels(self):
        import tempfile,numpy as np
        from experiments.paired_deliberation.native_kbrd_scores import PrecomputedKBRD
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'valid.ids.json').write_text(json.dumps(['conv:1']));(p/'native_item_ids.json').write_text(json.dumps([9,3,77]))
            np.save(p/'valid.native_logits.npy',np.array([[2.,-1.,1.]],dtype=np.float32))
            (p/'serving_metadata.json').write_text(json.dumps({'checkpoint_sha256':'abcd'*16,'selected_epoch':9,'selection_metric':'valid_recall10'}))
            source=PrecomputedKBRD(p).score_pool('conv:1',['77','9'])
            self.assertEqual([r[0] for r in source['rows']],['77','9']);self.assertEqual([r[1] for r in source['rows']],[1.,2.]);self.assertEqual([r[3] for r in source['rows']],[2,1])
            self.assertFalse(source['current_user_fact_authority'])
            with self.assertRaises(KeyError):PrecomputedKBRD(p).score_pool('conv:1',['unknown'])

if __name__=='__main__':unittest.main()
