import unittest,json
from experiments.paired_deliberation.native_inspired_benchmark import contexts,author_targets,DATA
class NativeInspiredTests(unittest.TestCase):
    def test_prefix_only_role_merge_and_duplicate_response_targets(self):
        def u(role,text,movies):return {'role':role,'text':[text],'movies':movies}
        raw=[{'dialog':[u('Recommender','hello',['old']),u('Seeker','request',[]),u('Seeker','more',[]),u('Recommender','FUTURE',['film']),u('Recommender','SECRET',['film','unknown'])]}]
        rows=contexts(raw,{'old':1,'film':2},'valid')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['id'],'inspired:valid:0000:2')
        self.assertEqual(rows[0]['text'],'Recommender: hello\nSeeker: request more');self.assertEqual(rows[0]['targets'],[2,2])
        self.assertEqual(rows[0]['previous_movie_ids'],[1])
    def test_ids_depend_on_fixed_order_not_future_response_content(self):
        a=[{'dialog':[{'role':'Seeker','text':['q'],'movies':[]},{'role':'Recommender','text':['filmA'],'movies':['a']}]}]
        b=[{'dialog':[{'role':'Seeker','text':['q'],'movies':[]},{'role':'Recommender','text':['filmB'],'movies':['b']}]}]
        x=contexts(a,{'a':1,'b':2},'valid')[0];y=contexts(b,{'a':1,'b':2},'valid')[0]
        self.assertEqual(x['id'],y['id']);self.assertEqual(x['text'],y['text']);self.assertNotEqual(x['targets'],y['targets'])
    def test_all_published_valid_targets_match_original_author_methods(self):
        folder=DATA/'released'
        if not(folder/'valid_data.json').exists():self.skipTest('public release not staged')
        raw=json.loads((folder/'valid_data.json').read_text());entities=json.loads((folder/'entity2id.json').read_text())
        self.assertEqual([r['targets'] for r in contexts(raw,entities,'valid')],author_targets(raw,folder))
if __name__=='__main__':unittest.main()
