from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.mechanism import run_sequential_allocation_screen as r


class FakeCorpus:
    def __init__(self,path):pass
    def search(self,query,k=2):
        text='Source '+query
        return [{'docid':sha256(text.encode()).hexdigest(),'title':query,'text':text,'text_sha256':sha256(text.encode()).hexdigest()}]
    def close(self):pass


class SequentialTests(unittest.TestCase):
    def setUp(self):
        text='Initial material'
        self.case={'id':'test:1','question':'Which item?', 'native_documents':[
            {'docid':'native:1','title':'Initial','text':text,'text_sha256':sha256(text.encode()).hexdigest()}]}
        self.calls=[]
    def transport(self,req):
        self.calls.append(req)
        payload=json.loads(req['messages'][1]['content'])
        if 'ONE next' in req['messages'][0]['content']:value={'query':'query '+str(len(payload['documents']))}
        else:value={'reasoning':'short','docids':[],'answer':'abstain'}
        return {'http_status':200,'body':json.dumps({'model':'deepseek-flash','system_fingerprint':r.p.base.FP,
            'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]})}

    def test_dag_eight_calls_offline_resume_and_label_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(r,'Corpus',FakeCorpus):
            a=r.collect(self.case,self.transport,{'used':0,'cap':5})
            b=r.collect(self.case,lambda req:self.fail('resume API'),{'used':0,'cap':5})
            self.assertEqual(a,b);self.assertEqual(len(self.calls),8)
            self.assertEqual(len(a),3)
            for req in self.calls:self.assertEqual(set(json.loads(req['messages'][1]['content'])),{'question','documents'})
            plans=[json.loads(req['messages'][1]['content']) for req in self.calls if 'ONE next' in req['messages'][0]['content']]
            self.assertGreater(len(plans[1]['documents']),len(plans[0]['documents']))

    def test_budget_and_unknown_attempt_stop_before_network(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            with self.assertRaisesRegex(RuntimeError,'budget'):r.collect(self.case,self.transport,{'used':5,'cap':5})
            self.assertFalse(self.calls)
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)):
            def fail(req):raise TimeoutError()
            with self.assertRaises(RuntimeError):r.collect(self.case,fail,{'used':0,'cap':5})
            with self.assertRaisesRegex(RuntimeError,'unresolved'):r.collect(self.case,self.transport,{'used':0,'cap':5})
            self.assertFalse(self.calls)

    def test_fixed_path_does_not_charge_intermediate_answers(self):
        def row(tokens):return {'response':{'body':json.dumps({'usage':{'total_tokens':tokens}})},'wall_seconds':1}
        states=[{'answers':{'enabled':{'source_key':'a0'}}},
            {'query_plan':{'source_key':'q1'},'answers':{'enabled':{'source_key':'a1'}},'retrieval':{'event':{'wall_seconds':.1}}},
            {'query_plan':{'source_key':'q2'},'answers':{'enabled':{'source_key':'a2'}},'retrieval':{'event':{'wall_seconds':.1}}}]
        rows={k:row(v) for k,v in {'a0':100,'a1':100,'a2':100,'q1':10,'q2':10}.items()}
        self.assertEqual(r.path_account(states,rows,2,'enabled',False)['tokens'],120)
        self.assertEqual(r.path_account(states,rows,2,'enabled',True)['tokens'],320)

    def test_oracle_observation_cost_and_calls_are_not_free(self):
        options=[[(0,0,{'tokens':10,'calls':1}),(1,1,{'tokens':30,'calls':3})]]*2
        values=r.frontier(options)
        feasible=[v for v in values if v['tokens']<=40 and v['calls']<=4]
        self.assertEqual(max(v['correct'] for v in feasible),1)
        self.assertFalse(any(v['correct']==2 and v['tokens']<=40 for v in values))

    def test_mode_is_explicit_and_mcnemar(self):
        a=r.final_request(self.case,[],'disabled');b=r.final_request(self.case,[],'enabled')
        self.assertEqual(a['messages'],b['messages']);self.assertNotIn('temperature',b)
        self.assertEqual(r.mcnemar(4,0),.125);self.assertEqual(r.mcnemar(1,1),1)


if __name__=='__main__':unittest.main()
