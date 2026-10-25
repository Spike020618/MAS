import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from experiments.core.mechanism.document_verification import chunks,retrieve,citations_valid
from experiments.mechanism import run_document_verification as run


class DocumentTests(unittest.TestCase):
    def test_preparation_strips_annotations_and_separates_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'official_fixture.zip';out=root/'results'
            rows=[{'article_id':str(i),'article':'Original document '+str(i),'license':'fixture',
                'questions':[{'question':'Question '+str(i),'options':['a','b','c','d'],
                              'gold_label':1,'writer_label':'SECRET','validation':['SECRET']}]} for i in range(20)]
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('QuALITY.v1.0.1.htmlstripped.dev','\n'.join(json.dumps(r) for r in rows))
            with patch.object(run,'ROOT',root),patch.object(run,'ARCHIVE',archive),patch.object(run,'OUT',out),patch('builtins.print'):
                run.prepare();run.prepare()
            public=json.loads((out/'public_cases.json').read_text())
            self.assertEqual(len({c['article_sha256'] for c in public}),20)
            self.assertNotIn('SECRET',json.dumps(public))
            self.assertEqual(set(public[0]),{'id','article_id','question','choices','article','article_sha256'})

    def test_exact_character_offsets(self):
        text='\n A  fox\tjumped.\n'+('word '*500)
        for row in chunks(text):self.assertEqual(row['text'],text[row['start']:row['end']])

    def test_no_truth_claim_from_literal_match(self):
        windows=[{'id':'p0','text':'The sky is green.'}]
        a=citations_valid([{'chunk_id':'p0','quote':'sky is green'},{'chunk_id':{},'quote':'sky'},None],windows)
        self.assertTrue(a[0]['literal_source_match']);self.assertFalse(a[0]['entailment_verified'])
        self.assertFalse(a[1]['literal_source_match']);self.assertFalse(a[2]['literal_source_match'])

    def test_retrieval_is_label_free_and_bounded(self):
        text='ordinary '*250+'needle target '+'other '*250
        result=retrieve(text,'needle',limit=2)
        self.assertLessEqual(len(result),2)
        self.assertIn('needle',result[0]['text'])
        self.assertEqual(retrieve('', 'needle'),[])

    def test_request_whitelists_public_fields(self):
        case={'question':'Question','choices':{'A':'x','B':'y'},'gold_label':'SECRET','validation':'SECRET'}
        req=run.request(case,'full',[{'id':'article','start':0,'end':3,'text':'abc'}])
        self.assertNotIn('SECRET',json.dumps(req))
        self.assertNotIn('gold_label',json.dumps(req))

    def test_parser_tolerates_metadata_without_scoring_source_as_truth(self):
        value={'answer':'A','reasoning':'r','citations':[{'chunk_id':'article','quote':'abc'}], 'type':'json_object'}
        row={'wall_seconds':.1,'response':{'body':json.dumps({'usage':{'total_tokens':10},
             'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})}}
        result=run.parse(row,'full',[{'id':'article','text':'abc'}])
        self.assertTrue(result['valid']);self.assertEqual(result['answer'],'A')
        self.assertFalse(result['citations'][0]['entailment_verified'])

    def test_real_call_graph_and_zero_call_resume_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            case={'id':'quality:fixture','article_id':'fixture','article':'a source paragraph',
                  'article_sha256':'fixture','question':'Question','choices':{'A':'x','B':'y'}}
            count=[]
            def transport(req):
                count.append(req)
                if req['max_tokens']==400:value={'claim':'a premise','query':'source'}
                else:value={'answer':'A','reasoning':'source','citations':[]}
                return {'http_status':200,'body':json.dumps({'model':'deepseek-flash',
                    'system_fingerprint':'e72d3c7f663d3e1383b00b3884cb4501','usage':{'total_tokens':10},
                    'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})}
            with patch.object(run,'OUT',Path(directory)),patch.object(run,'transport',side_effect=transport),patch('builtins.print'):
                first=run.collect_case(case)
                self.assertEqual(len(count),6)
                with patch.object(run,'transport',side_effect=AssertionError('no network replay')):
                    self.assertEqual(first,run.collect_case(case))
                self.assertEqual(len(count),6)


if __name__=='__main__':unittest.main()
