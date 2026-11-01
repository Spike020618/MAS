from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import replicate_evidence_acquisition as r


class FakeCorpus:
    def __init__(self,path): pass
    def search(self,query,k=2):
        text='Retrieved '+query
        return [{'docid':sha256(text.encode()).hexdigest(),'title':query,'text':text,'text_sha256':sha256(text.encode()).hexdigest()}]
    def close(self): pass


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        text='Initial evidence';self.case={'id':'new:1','question':'Which item?', 'native_documents':[
            {'docid':'original','title':'Initial','text':text,'text_sha256':sha256(text.encode()).hexdigest()}]}
        self.calls=[]

    def transport(self,req):
        self.calls.append(req);prompt=req['messages'][0]['content']
        value={'queries':['first query','second query']} if 'exactly TWO' in prompt else {'query':'next query'} if 'ONE next' in prompt else {'answer':'test','reasoning':'','docids':[]}
        return {'http_status':200,'body':json.dumps({'model':'deepseek-flash','system_fingerprint':r.p.base.FP,
            'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})}

    def test_content_addressing_stable_across_source_aliases(self):
        d=self.case['native_documents'][0]
        first,_=r.canonical([d]);second,mapping=r.canonical([{**d,'docid':'another'},d])
        self.assertEqual(first,second);self.assertEqual(len(next(iter(mapping.values()))),2)

    def test_six_call_dag_and_offline_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(r,'Corpus',FakeCorpus):
            a=r.collect(self.case,self.transport)
            b=r.collect(self.case,lambda req:self.fail('resume called API'))
            self.assertEqual(a,b);self.assertEqual(len(self.calls),6)
            self.assertEqual(set(a),{'0','1'})
            for req in self.calls:
                self.assertNotIn('answer',json.loads(req['messages'][1]['content']))
            for rep in a:self.assertEqual(set(a[rep]),{'native','expanded'})

    def test_bad_plan_fails_without_repair(self):
        def malformed(req):
            body=json.loads(self.transport(req)['body']);body['choices'][0]['message']['content']='{}'
            return {'http_status':200,'body':json.dumps(body)}
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            with self.assertRaisesRegex(RuntimeError,'invalid query plan'):r.collect(self.case,malformed)
            self.assertEqual(len(self.calls),1)

    def test_source_budget_and_forgery(self):
        d=self.case['native_documents'][0]
        with self.assertRaises(ValueError):r.canonical([{**d,'text':'forged'}])
        text='z'*40001
        with self.assertRaises(ValueError):r.canonical([{**d,'text':text,'text_sha256':sha256(text.encode()).hexdigest()}])


if __name__=='__main__':unittest.main()
