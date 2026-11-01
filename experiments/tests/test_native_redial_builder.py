import unittest
from experiments.paired_deliberation.native_redial_benchmark import contexts,evaluate
class NativeBuilderTests(unittest.TestCase):
    def test_prefix_merge_duplicates_and_unknown_movies(self):
        def u(role,text,movies=[]):return dict(role=role,text=[text],movies=movies)
        raw=[{'conv_id':'a','dialog':[u('Recommender','initial',['m']),u('Seeker','request'),u('Seeker','more'),u('Recommender','SECRET',['m','unknown']),u('Recommender','FUTURE',['m'])]}]
        rows=contexts(raw,{'m':7})
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['targets'],[7,7])
        self.assertEqual(rows[0]['text'],'Recommender: initial\nSeeker: request more')
    def test_pool_miss_zero_and_duplicate_target_denominator(self):
        s=evaluate([{'targets':[1,1,8]}],[[1,2,3]])
        self.assertEqual(s['item_targets'],3);self.assertAlmostEqual(s['metrics']['recall@1'],2/3)
        self.assertEqual(s['retrieval_recall@100'],2/3)

class AuthorPreprocessingParity(unittest.TestCase):
    def test_actual_released_validation_targets_match_author_functions(self):
        import ast,json
        from types import SimpleNamespace
        from copy import copy
        from experiments.paired_deliberation.native_redial_benchmark import DATA,ROOT
        source=ROOT/'docs/research/benchmark_protocol_sources_2026-10-27/CRSLab/crslab/data/dataset/redial/redial.py'
        if not (DATA/'released/valid_data.json').exists():self.skipTest('public release not staged')
        tree=ast.parse(source.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_merge_conv_data','_augment_and_add')]
        scope={'copy':copy};exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),scope)
        p=DATA/'released'; e=json.loads((p/'entity2id.json').read_text())
        obj=SimpleNamespace(entity2id=e,tok2ind=json.loads((p/'token2id.json').read_text()),unk_token_idx=3,word2id=json.loads((p/'concept2id.json').read_text()))
        raw=json.loads((p/'valid_data.json').read_text()); expected=[]
        for conv in raw:
            turns=scope['_augment_and_add'](obj,scope['_merge_conv_data'](obj,conv['dialog']))
            expected.extend(t['items'] for t in turns if t['role']=='Recommender' and t['items'])
        self.assertEqual([r['targets'] for r in contexts(raw,e)],expected)

if __name__=='__main__':unittest.main()
