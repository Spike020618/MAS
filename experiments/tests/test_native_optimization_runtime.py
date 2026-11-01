import json,tempfile,unittest,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from experiments.paired_deliberation.native_optimization_runtime import Runtime
class OptimizationRuntimeTests(unittest.TestCase):
    def setUp(self):
        pacing=patch.object(Runtime,'_pace',lambda self,cooldown=False:None);pacing.start();self.addCleanup(pacing.stop)
    def payload(self):return {'id':'a','text':'Seeker: comedy','candidates':[{'movie_id':str(i),'title':'Film '+str(i)} for i in range(12)],'candidate_evidence':{},'training_neighbors':[]}
    def test_identical_concurrent_requests_share_one_paid_raw_response(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=Runtime(Path(d),{'v':1});calls=[]
            def fake(req):
                calls.append(True)
                return {'status_code':200,'elapsed_ms':1,'body':json.dumps({'model':'qwen-flash-2025-07-28','usage':{'prompt_tokens':10,'completion_tokens':10},'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'ranking':[str(i) for i in range(10)]})}}]})}
            runtime.provider.transport=fake
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda slot:runtime.rank(slot,self.payload(),'same task'),['one','two']))
            self.assertEqual(len(calls),1);self.assertTrue(all(r['valid'] for r in results))
            self.assertEqual(len(list((Path(d)/'slots').glob('*.raw.json'))),1)
            self.assertEqual(len(list((Path(d)/'logical_calls').glob('*.json'))),2)
    def test_unknown_call_not_repeated_and_both_paths_keep_same_prior(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=Runtime(Path(d),{'v':1});calls=[]
            def timeout(req):calls.append(True);raise RuntimeError('unknown network completion')
            runtime.provider.transport=timeout
            one=runtime.rank('one',self.payload(),'same task');two=runtime.rank('two',self.payload(),'same task')
            self.assertEqual(len(calls),1);self.assertFalse(one['valid']);self.assertFalse(two['valid'])
            self.assertEqual(one['ranking'],two['ranking']);self.assertEqual(one['ranking'],[str(i) for i in range(10)])
if __name__=='__main__':unittest.main()

class QwenJsonPreflightTests(unittest.TestCase):
    def test_missing_json_instruction_is_rejected_before_creating_request(self):
        from experiments.paired_deliberation.transport_d2 import TransportD2
        from experiments.paired_deliberation.preflight_d2 import ACTORS
        with tempfile.TemporaryDirectory() as d:
            t=TransportD2('qwen_flash',ACTORS['qwen_flash'],Path(d),10,1);t.key='offline-test-key'
            req={'messages':[{'role':'system','content':'Return ranking'}],'max_tokens':192,'labels_available':False}
            with self.assertRaisesRegex(ValueError,'requires JSON'):t.preflight(req)
            self.assertEqual(list(Path(d).glob('*.intent.json')),[])
            req['messages'][0]['content']='Return JSON ranking';t.preflight(req)
